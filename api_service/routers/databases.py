import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, SecretStr

from api_service.deps import get_current_user, get_owned_workspace
from shared.db import get_db
from shared.external_db.errors import ConnectionStringError, DatabaseInspectionError
from shared.external_db.models import InspectionResult
from shared.external_db.service import inspect_database
from shared.external_db.uri import ParsedConnection
from shared.models.base import utcnow
from shared.models.database_connection import COLLECTION as DATABASE_CONNECTIONS
from shared.models.database_connection import (
    DatabaseConnection, DatabaseType, RememberOption, as_aware, expiry_for,
)
from shared.models.user import User
from shared.redis_client import get_redis
from shared.secret_box import SecretBoxError, decrypt_secret, encrypt_secret

logger = logging.getLogger("api.databases")

router = APIRouter(tags=["databases"])

MAX_CONNECTIONS_PER_WORKSPACE = 10
TEST_RATE_LIMIT = 20
TEST_RATE_WINDOW_SECONDS = 600
EXPIRED_MESSAGE = "The saved credentials for this connection have expired. Reconnect with the connection string."


class ConnectionTestRequest(BaseModel):
    db_type: DatabaseType
    connection_string: SecretStr


class CreateConnectionRequest(ConnectionTestRequest):
    name: Optional[str] = Field(default=None, max_length=80)
    remember: RememberOption = "30d"


class ReconnectRequest(BaseModel):
    connection_string: SecretStr
    remember: Optional[RememberOption] = None


class UpdateConnectionRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=80)
    remember: Optional[RememberOption] = None


class ConnectionTestResponse(BaseModel):
    ok: bool
    db_type: DatabaseType
    display_uri: Optional[str] = None
    host: Optional[str] = None
    database: Optional[str] = None
    server_version: Optional[str] = None
    db_schema: Optional[dict] = Field(default=None, serialization_alias="schema")
    warnings: list[str] = []
    error: Optional[str] = None


class DatabaseConnectionOut(BaseModel):
    id: str
    workspace_id: str
    name: str
    db_type: DatabaseType
    display_uri: str
    host: str
    database: Optional[str]
    remember: RememberOption
    expires_at: Optional[str]
    status: str
    warnings: list[str]
    last_error: Optional[str]
    object_count: int
    schema_refreshed_at: Optional[str]
    created_by: str
    created_at: str


class DatabaseConnectionDetail(DatabaseConnectionOut):
    db_schema: dict = Field(serialization_alias="schema")


def _iso(value: Optional[datetime]) -> Optional[str]:
    aware = as_aware(value)
    return aware.isoformat() if aware else None


def _out_fields(connection: DatabaseConnection) -> dict:
    return {
        "id": connection.id,
        "workspace_id": connection.workspace_id,
        "name": connection.name,
        "db_type": connection.db_type,
        "display_uri": connection.display_uri,
        "host": connection.host,
        "database": connection.database,
        "remember": connection.remember,
        "expires_at": _iso(connection.expires_at),
        "status": connection.status,
        "warnings": connection.warnings,
        "last_error": connection.last_error,
        "object_count": len(connection.db_schema.get("objects", [])),
        "schema_refreshed_at": _iso(connection.schema_refreshed_at),
        "created_by": connection.created_by,
        "created_at": _iso(connection.created_at),
    }


def _out(connection: DatabaseConnection) -> DatabaseConnectionOut:
    return DatabaseConnectionOut(**_out_fields(connection))


def _detail(connection: DatabaseConnection) -> DatabaseConnectionDetail:
    return DatabaseConnectionDetail(**_out_fields(connection), db_schema=connection.db_schema)


async def _enforce_test_rate_limit(user: User) -> None:
    key = f"database_tests:{user.id}"
    try:
        redis = get_redis()
        attempts = await redis.incr(key)
        if attempts == 1:
            await redis.expire(key, TEST_RATE_WINDOW_SECONDS)
    except Exception:
        logger.exception("database test rate limiter unavailable")
        return
    if attempts > TEST_RATE_LIMIT:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Too many connection attempts. Please wait a few minutes and try again.",
        )


async def _inspect(db_type: str, connection_string: str) -> tuple[ParsedConnection, InspectionResult]:
    try:
        return await inspect_database(db_type, connection_string)
    except (ConnectionStringError, DatabaseInspectionError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None


async def _expire_if_needed(connection: DatabaseConnection) -> DatabaseConnection:
    if connection.status == "expired" or not connection.is_expired():
        return connection
    await get_db()[DATABASE_CONNECTIONS].update_one(
        {"_id": connection.id},
        {"$set": {"encrypted_connection_string": None, "status": "expired", "updated_at": utcnow()}},
    )
    return connection.model_copy(update={"encrypted_connection_string": None, "status": "expired"})


async def get_owned_connection(connection_id: str, user: User) -> DatabaseConnection:
    doc = await get_db()[DATABASE_CONNECTIONS].find_one({"_id": connection_id})
    connection = DatabaseConnection.from_mongo(doc)
    if connection is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Database connection not found")
    await get_owned_workspace(connection.workspace_id, user)
    return await _expire_if_needed(connection)


async def _save_inspection(
    connection: DatabaseConnection, parsed: ParsedConnection, result: InspectionResult,
    connection_string: Optional[str], remember: Optional[str],
) -> DatabaseConnection:
    now = utcnow()
    updates = {
        "display_uri": parsed.display_uri,
        "host": parsed.host_label,
        "database": result.database,
        "status": "connected",
        "db_schema": result.schema_document(),
        "schema_refreshed_at": now,
        "last_tested_at": now,
        "last_error": None,
        "warnings": result.warnings,
        "updated_at": now,
    }
    if connection_string is not None:
        updates["encrypted_connection_string"] = encrypt_secret(connection_string)
        effective_remember = remember or connection.remember
        updates["remember"] = effective_remember
        updates["expires_at"] = expiry_for(effective_remember, now)
    await get_db()[DATABASE_CONNECTIONS].update_one({"_id": connection.id}, {"$set": updates})
    return connection.model_copy(update=updates)


@router.post("/workspaces/{workspace_id}/databases/test", response_model=ConnectionTestResponse)
async def test_connection(workspace_id: str, body: ConnectionTestRequest, user: User = Depends(get_current_user)):
    await get_owned_workspace(workspace_id, user)
    await _enforce_test_rate_limit(user)
    try:
        parsed, result = await inspect_database(body.db_type, body.connection_string.get_secret_value())
    except (ConnectionStringError, DatabaseInspectionError) as exc:
        return ConnectionTestResponse(ok=False, db_type=body.db_type, error=str(exc))
    logger.info(
        "database connection test succeeded: workspace=%s db_type=%s host=%s objects=%d",
        workspace_id, body.db_type, parsed.host_label, len(result.objects),
    )
    return ConnectionTestResponse(
        ok=True, db_type=body.db_type, display_uri=parsed.display_uri, host=parsed.host_label,
        database=result.database, server_version=result.server_version,
        db_schema=result.schema_document(), warnings=result.warnings,
    )


@router.get("/workspaces/{workspace_id}/databases", response_model=list[DatabaseConnectionOut])
async def list_connections(workspace_id: str, user: User = Depends(get_current_user)):
    await get_owned_workspace(workspace_id, user)
    docs = await get_db()[DATABASE_CONNECTIONS].find(
        {"workspace_id": workspace_id}, {"db_schema.objects.fields": 0, "db_schema.objects.foreign_keys": 0},
    ).sort("created_at", -1).to_list(length=MAX_CONNECTIONS_PER_WORKSPACE * 2)
    connections = [await _expire_if_needed(DatabaseConnection.from_mongo(doc)) for doc in docs]
    return [_out(c) for c in connections]


@router.post(
    "/workspaces/{workspace_id}/databases",
    response_model=DatabaseConnectionDetail,
    status_code=status.HTTP_201_CREATED,
)
async def create_connection(workspace_id: str, body: CreateConnectionRequest, user: User = Depends(get_current_user)):
    await get_owned_workspace(workspace_id, user)
    existing = await get_db()[DATABASE_CONNECTIONS].count_documents({"workspace_id": workspace_id})
    if existing >= MAX_CONNECTIONS_PER_WORKSPACE:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"A workspace can have at most {MAX_CONNECTIONS_PER_WORKSPACE} database connections.",
        )
    await _enforce_test_rate_limit(user)
    connection_string = body.connection_string.get_secret_value().strip()
    parsed, result = await _inspect(body.db_type, connection_string)

    try:
        encrypted = encrypt_secret(connection_string)
    except SecretBoxError:
        logger.exception("database connection encryption is not configured")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Secure credential storage is not configured.")

    now = utcnow()
    name = (body.name or "").strip() or result.database or parsed.host_label
    connection = DatabaseConnection(
        workspace_id=workspace_id,
        created_by=user.id,
        name=name[:80],
        db_type=body.db_type,
        encrypted_connection_string=encrypted,
        display_uri=parsed.display_uri,
        host=parsed.host_label,
        database=result.database,
        remember=body.remember,
        expires_at=expiry_for(body.remember, now),
        status="connected",
        db_schema=result.schema_document(),
        schema_refreshed_at=now,
        last_tested_at=now,
        warnings=result.warnings,
        created_at=now,
        updated_at=now,
    )
    await get_db()[DATABASE_CONNECTIONS].insert_one(connection.to_mongo())
    logger.info(
        "database connection saved: id=%s workspace=%s db_type=%s host=%s remember=%s",
        connection.id, workspace_id, body.db_type, parsed.host_label, body.remember,
    )
    return _detail(connection)


@router.get("/databases/{connection_id}", response_model=DatabaseConnectionDetail)
async def get_connection(connection_id: str, user: User = Depends(get_current_user)):
    return _detail(await get_owned_connection(connection_id, user))


@router.post("/databases/{connection_id}/refresh", response_model=DatabaseConnectionDetail)
async def refresh_schema(connection_id: str, user: User = Depends(get_current_user)):
    connection = await get_owned_connection(connection_id, user)
    if connection.encrypted_connection_string is None:
        raise HTTPException(status.HTTP_410_GONE, EXPIRED_MESSAGE)
    await _enforce_test_rate_limit(user)
    try:
        connection_string = decrypt_secret(connection.encrypted_connection_string)
    except SecretBoxError:
        logger.error("could not decrypt credentials for database connection %s", connection.id)
        raise HTTPException(status.HTTP_410_GONE, EXPIRED_MESSAGE)
    try:
        parsed, result = await inspect_database(connection.db_type, connection_string)
    except (ConnectionStringError, DatabaseInspectionError) as exc:
        now = utcnow()
        await get_db()[DATABASE_CONNECTIONS].update_one(
            {"_id": connection.id},
            {"$set": {"status": "failed", "last_error": str(exc), "last_tested_at": now, "updated_at": now}},
        )
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    return _detail(await _save_inspection(connection, parsed, result, None, None))


@router.post("/databases/{connection_id}/reconnect", response_model=DatabaseConnectionDetail)
async def reconnect(connection_id: str, body: ReconnectRequest, user: User = Depends(get_current_user)):
    connection = await get_owned_connection(connection_id, user)
    await _enforce_test_rate_limit(user)
    connection_string = body.connection_string.get_secret_value().strip()
    parsed, result = await _inspect(connection.db_type, connection_string)
    try:
        updated = await _save_inspection(connection, parsed, result, connection_string, body.remember)
    except SecretBoxError:
        logger.exception("database connection encryption is not configured")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Secure credential storage is not configured.")
    return _detail(updated)


@router.patch("/databases/{connection_id}", response_model=DatabaseConnectionOut)
async def update_connection(connection_id: str, body: UpdateConnectionRequest, user: User = Depends(get_current_user)):
    connection = await get_owned_connection(connection_id, user)
    updates: dict = {}
    if body.name is not None:
        name = body.name.strip()
        if not name:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Name cannot be empty")
        updates["name"] = name
    if body.remember is not None:
        updates["remember"] = body.remember
        if connection.encrypted_connection_string is not None:
            updates["expires_at"] = expiry_for(body.remember)
    if updates:
        updates["updated_at"] = utcnow()
        await get_db()[DATABASE_CONNECTIONS].update_one({"_id": connection.id}, {"$set": updates})
        connection = connection.model_copy(update=updates)
    return _out(connection)


@router.delete("/databases/{connection_id}")
async def delete_connection(connection_id: str, user: User = Depends(get_current_user)):
    connection = await get_owned_connection(connection_id, user)
    await get_db()[DATABASE_CONNECTIONS].delete_one({"_id": connection.id})
    logger.info("database connection deleted: id=%s workspace=%s", connection.id, connection.workspace_id)
    return {"ok": True}
