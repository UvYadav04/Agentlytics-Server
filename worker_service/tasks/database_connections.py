import logging

from shared.db import get_db
from shared.models.base import utcnow
from shared.models.database_connection import COLLECTION as DATABASE_CONNECTIONS
from shared.workspace_cache import invalidate_db_summaries_many

logger = logging.getLogger("worker.database_connections")


async def purge_expired_database_credentials(ctx) -> None:
    now = utcnow()
    db = get_db()
    expired_filter = {
        "expires_at": {"$ne": None, "$lte": now},
        "encrypted_connection_string": {"$ne": None},
    }
    affected = await db[DATABASE_CONNECTIONS].find(expired_filter, {"workspace_id": 1}).to_list(length=5000)
    result = await db[DATABASE_CONNECTIONS].update_many(
        expired_filter,
        {"$set": {"encrypted_connection_string": None, "status": "expired", "updated_at": now}},
    )
    if result.modified_count:
        await invalidate_db_summaries_many(doc.get("workspace_id") for doc in affected)
        logger.info("purged credentials for %d expired database connection(s)", result.modified_count)
