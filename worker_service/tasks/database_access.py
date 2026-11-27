import logging
from datetime import datetime

from shared.models.base import utcnow
from shared.models.database_connection import COLLECTION as DATABASE_CONNECTIONS
from shared.models.database_connection import DatabaseConnection, as_aware
from shared.secret_box import SecretBoxError, decrypt_secret
from shared.workspace_cache import get_cached_db_summaries, set_cached_db_summaries

logger = logging.getLogger("worker.database_access")

MAX_OBJECTS_IN_SUMMARY = 60
MAX_FIELDS_IN_SUMMARY = 50


def _object_names(connection: DatabaseConnection) -> list[str]:
    names = []
    for obj in connection.db_schema.get("objects", [])[:MAX_OBJECTS_IN_SUMMARY]:
        if connection.db_type == "mongodb":
            names.append(obj["name"])
        elif obj.get("schema") and obj.get("schema") not in ("public", connection.database):
            names.append(f"{obj['schema']}.{obj['name']}")
        else:
            names.append(obj["name"])
    return names


def _compact_schema(
    connection: DatabaseConnection, max_objects: int = 40, max_fields: int = MAX_FIELDS_IN_SUMMARY,
) -> dict[str, dict[str, str]]:
    """Build a compact {table_name: {column_name: type}} mapping for direct agent context."""
    schema: dict[str, dict[str, str]] = {}
    for obj in connection.db_schema.get("objects", [])[:max_objects]:
        raw_name = obj.get("name")
        if not raw_name:
            continue
        if connection.db_type == "mongodb":
            table_name = raw_name
        elif obj.get("schema") and obj.get("schema") not in ("public", connection.database):
            table_name = f"{obj['schema']}.{raw_name}"
        else:
            table_name = raw_name

        fields_dict: dict[str, str] = {}
        for f in obj.get("fields", [])[:max_fields]:
            fname = f.get("name")
            ftype = f.get("type", "string")
            if fname:
                fields_dict[fname] = ftype
        schema[table_name] = fields_dict
    return schema


class WorkspaceDatabaseAccess:
    def __init__(self, db, workspace_id: str):
        self.db = db
        self.workspace_id = workspace_id

    async def summaries(self) -> list[dict]:
        cached = await get_cached_db_summaries(self.workspace_id)
        if cached is not None:
            now = utcnow()
            valid: list[dict] = []
            for item in cached:
                expires_raw = item.get("expires_at")
                if expires_raw:
                    try:
                        expires_at = as_aware(datetime.fromisoformat(expires_raw))
                        if expires_at is not None and expires_at <= now:
                            continue
                    except ValueError:
                        pass
                valid.append({k: v for k, v in item.items() if k != "expires_at"})
            return valid

        docs = await self.db[DATABASE_CONNECTIONS].find(
            {"workspace_id": self.workspace_id, "encrypted_connection_string": {"$ne": None}},
            {"db_schema.objects.foreign_keys": 0},
        ).to_list(length=50)
        summaries = []
        cache_items = []
        now = utcnow()
        for doc in docs:
            connection = DatabaseConnection.from_mongo(doc)
            if connection.is_expired(now):
                continue
            summary = {
                "connection_id": connection.id,
                "name": connection.name,
                "db_type": connection.db_type,
                "database": connection.database,
                "objects": _object_names(connection),
                "schema": _compact_schema(connection),
            }
            summaries.append(summary)
            expires_at = as_aware(connection.expires_at)
            cache_items.append({
                **summary,
                "expires_at": expires_at.isoformat() if expires_at else None,
            })
        await set_cached_db_summaries(self.workspace_id, cache_items)
        return summaries

    async def resolve(self, connection_id: str) -> tuple[dict, str]:
        doc = await self.db[DATABASE_CONNECTIONS].find_one(
            {"_id": connection_id, "workspace_id": self.workspace_id},
        )
        connection = DatabaseConnection.from_mongo(doc)
        if connection is None:
            raise LookupError("That database connection doesn't exist in this workspace.")
        if connection.is_expired():
            raise LookupError(
                f"The saved credentials for '{connection.name}' have expired. Ask the user to reconnect it."
            )
        try:
            connection_string = decrypt_secret(connection.encrypted_connection_string)
        except SecretBoxError:
            logger.error("could not decrypt credentials for database connection %s", connection.id)
            raise LookupError(f"The saved credentials for '{connection.name}' can't be used. Ask the user to reconnect it.")
        return {
            "id": connection.id,
            "name": connection.name,
            "db_type": connection.db_type,
            "database": connection.database,
            "db_schema": connection.db_schema,
        }, connection_string
