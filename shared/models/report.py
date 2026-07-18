from datetime import datetime
from typing import Literal, Optional

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "reports"

ReportStatus = Literal["generating", "ready", "failed"]
ReportFormat = Literal["markdown", "csv", "html"]


class Report(MongoModel):
    workspace_id: str
    message_id: str
    title: str = "Untitled report"
    status: ReportStatus = "generating"
    format: ReportFormat = "markdown"
    storage_key: Optional[str] = None  # R2 key, set once status == "ready"
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
