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

    user_id: Optional[str] = None
    file_ids: list[str] = Field(default_factory=list)
    email: Optional[str] = None

    usage_counted: bool = False
    chat_memory_enqueued: bool = False

    retry_count: int = 0
    last_attempt_at: Optional[datetime] = None

    stage: Optional[str] = None
    pending_question: Optional[dict] = None
    questions: list[dict] = Field(default_factory=list)
    error_type: Optional[str] = None
    error_message: Optional[str] = None
