import logging

from shared.db import get_db
from shared.models.base import utcnow
from shared.models.database_connection import COLLECTION as DATABASE_CONNECTIONS

logger = logging.getLogger("worker.database_connections")


async def purge_expired_database_credentials(ctx) -> None:
    now = utcnow()
    result = await get_db()[DATABASE_CONNECTIONS].update_many(
        {"expires_at": {"$ne": None, "$lte": now}, "encrypted_connection_string": {"$ne": None}},
        {"$set": {"encrypted_connection_string": None, "status": "expired", "updated_at": now}},
    )
    if result.modified_count:
        logger.info("purged credentials for %d expired database connection(s)", result.modified_count)
