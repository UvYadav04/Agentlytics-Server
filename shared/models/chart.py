from datetime import datetime
from typing import Optional

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "charts"


class Chart(MongoModel):
    workspace_id: str
    message_id: Optional[str] = None
    title: str = "Untitled chart"
    storage_key: str  # R2 key for the generated chart HTML
    created_at: datetime = Field(default_factory=utcnow)
