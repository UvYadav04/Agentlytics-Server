from datetime import datetime
from typing import Optional

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "charts"


class Chart(MongoModel):
    workspace_id: str
    # Optional: charts created as part of a real-time dashboard bundle (see
    # shared/models/dashboard.py) aren't tied to the one chat turn that first
    # created them - they get overwritten in place on every refresh, so
    # "which message made this" stops being a meaningful question after the
    # first refresh. Still set at creation time when there is a message.
    message_id: Optional[str] = None
    title: str = "Untitled chart"
    storage_key: str  # R2 key for the generated chart HTML
    created_at: datetime = Field(default_factory=utcnow)
