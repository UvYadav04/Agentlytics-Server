from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "investigations"

InvestigationStatus = Literal["running", "completed", "failed", "cancelled"]


class InvestigationEvent(BaseModel):
    type: str
    message: str
    data: dict = Field(default_factory=dict)
    at: datetime = Field(default_factory=utcnow)


class Investigation(MongoModel):
    chat_id: str
    workspace_id: str
    objective: str
    status: InvestigationStatus = "running"
    events: list[InvestigationEvent] = Field(default_factory=list)
    cancel_requested: bool = False
    final_answer: Optional[str] = None
    started_at: datetime = Field(default_factory=utcnow)
    completed_at: Optional[datetime] = None
