"""Mongo access shared by api_service and worker_service.

Deliberately thin: `get_db()` returns a motor AsyncIOMotorDatabase and
callers do explicit collection operations (`get_db()["files"].find_one(...)`)
or use the `COLLECTION` constant exported by each model module, e.g.:

    from shared.db import get_db
    from shared.models.file import File, COLLECTION as FILES

    doc = await get_db()[FILES].find_one({"_id": file_id})
    file = File.from_mongo(doc)

`ensure_indexes()` is called once on api_service/worker_service startup.
"""
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from shared.config import get_settings

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        settings = get_settings()
        uri = settings.get("MONGO_URI", "mongodb://localhost:27017")
        _client = AsyncIOMotorClient(uri)
    return _client


def get_db() -> AsyncIOMotorDatabase:
    global _db
    if _db is None:
        settings = get_settings()
        db_name = settings.get("MONGO_DB_NAME", "data_analyzer")
        _db = get_client()[db_name]
    return _db


async def ensure_indexes() -> None:
    db = get_db()
    await db["users"].create_index("google_id", unique=True)
    await db["users"].create_index("email", unique=True)
    await db["workspaces"].create_index("user_id")
    await db["files"].create_index("workspace_id")
    await db["chats"].create_index("workspace_id")
    await db["messages"].create_index("chat_id")
    await db["investigations"].create_index("chat_id")
    await db["charts"].create_index("workspace_id")
    await db["dashboards"].create_index("workspace_id")
    await db["reports"].create_index("workspace_id")
    await db["usage"].create_index("user_id", unique=True)


async def close_client() -> None:
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None
