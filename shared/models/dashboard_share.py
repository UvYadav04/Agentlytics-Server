from datetime import datetime
from typing import Optional

from pydantic import Field

from shared.models.base import MongoModel, utcnow
from shared.models.database_connection import as_aware

COLLECTION = "dashboard_shares"


class DashboardShare(MongoModel):
    dashboard_id: str
    workspace_id: str
    shared_by: str
    shared_at: datetime = Field(default_factory=utcnow)
    valid_upto: datetime
    shared_to: list[str] = Field(default_factory=list)
    revoked_at: Optional[datetime] = None

    def is_active(self, now: Optional[datetime] = None) -> bool:
        if self.revoked_at is not None:
            return False
        return as_aware(self.valid_upto) > (now or utcnow())
