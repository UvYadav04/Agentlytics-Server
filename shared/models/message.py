from datetime import datetime
from typing import Literal, Optional

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "messages"


class Message(MongoModel):
    chat_id: str
    role: Literal["user", "assistant"]
    content: str
    investigation_id: Optional[str] = None
    chart_ids: list[str] = Field(default_factory=list)
    report_id: Optional[str] = None
    csv_file_ids: list[str] = Field(default_factory=list)
    dashboard_ids: list[str] = Field(default_factory=list)
    files_used: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    reply_to: Optional[str] = None
    version: int = 1
    retry_of: Optional[str] = None
    retry_mode: Optional[str] = None
    superseded_by: Optional[str] = None
    superseded_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)
