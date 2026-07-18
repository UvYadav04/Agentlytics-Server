from datetime import datetime

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "charts"


class Chart(MongoModel):
    workspace_id: str
    message_id: str
    title: str = "Untitled chart"
    storage_key: str  # R2 key for the generated chart HTML
    created_at: datetime = Field(default_factory=utcnow)
