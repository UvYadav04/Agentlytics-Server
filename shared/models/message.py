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
    created_at: datetime = Field(default_factory=utcnow)
