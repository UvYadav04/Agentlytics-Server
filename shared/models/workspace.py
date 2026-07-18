from datetime import datetime

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "workspaces"


class Workspace(MongoModel):
    user_id: str
    name: str
    created_at: datetime = Field(default_factory=utcnow)
