from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "database_connections"

DatabaseType = Literal["mongodb", "postgresql"]
RememberOption = Literal["always", "30d", "7d", "never"]
ConnectionStatus = Literal["connected", "failed", "expired"]

SESSION_ONLY_TTL = timedelta(hours=12)

REMEMBER_TTL: dict[str, Optional[timedelta]] = {
    "always": None,
    "30d": timedelta(days=30),
    "7d": timedelta(days=7),
    "never": SESSION_ONLY_TTL,
}


def expiry_for(remember: str, now: Optional[datetime] = None) -> Optional[datetime]:
    ttl = REMEMBER_TTL.get(remember)
    if ttl is None:
        return None
    return (now or utcnow()) + ttl


def as_aware(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class DatabaseConnection(MongoModel):
    workspace_id: str
    created_by: str
    name: str
    db_type: DatabaseType
    encrypted_connection_string: Optional[str] = None
    display_uri: str
    host: str
    database: Optional[str] = None
    remember: RememberOption = "30d"
    expires_at: Optional[datetime] = None
    status: ConnectionStatus = "connected"
    db_schema: dict = Field(default_factory=dict)
    schema_refreshed_at: Optional[datetime] = None
    last_tested_at: Optional[datetime] = None
    last_error: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if self.encrypted_connection_string is None:
            return True
        expires_at = as_aware(self.expires_at)
        return expires_at is not None and expires_at <= (now or utcnow())
