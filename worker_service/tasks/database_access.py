import logging

from shared.models.database_connection import COLLECTION as DATABASE_CONNECTIONS
from shared.models.database_connection import DatabaseConnection
from shared.secret_box import SecretBoxError, decrypt_secret

logger = logging.getLogger("worker.database_access")

MAX_OBJECTS_IN_SUMMARY = 60


def _object_names(connection: DatabaseConnection) -> list[str]:
    names = []
    for obj in connection.db_schema.get("objects", [])[:MAX_OBJECTS_IN_SUMMARY]:
        names.append(f"{obj['schema']}.{obj['name']}" if obj.get("schema") else obj["name"])
    return names


class WorkspaceDatabaseAccess:
    def __init__(self, db, workspace_id: str):
        self.db = db
        self.workspace_id = workspace_id

    async def summaries(self) -> list[dict]:
        docs = await self.db[DATABASE_CONNECTIONS].find(
            {"workspace_id": self.workspace_id, "encrypted_connection_string": {"$ne": None}},
            {"db_schema.objects.fields": 0, "db_schema.objects.foreign_keys": 0},
        ).to_list(length=50)
        summaries = []
        for doc in docs:
            connection = DatabaseConnection.from_mongo(doc)
            if connection.is_expired():
                continue
            summaries.append({
                "connection_id": connection.id,
                "name": connection.name,
                "db_type": connection.db_type,
                "database": connection.database,
                "objects": _object_names(connection),
            })
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
        }, connection_string
