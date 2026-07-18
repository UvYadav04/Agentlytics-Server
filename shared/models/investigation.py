from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "investigations"

InvestigationStatus = Literal["running", "completed", "failed", "cancelled"]


class InvestigationEvent(BaseModel):
    """One entry in Investigation.events[] - also the shape published to the
    Redis pub/sub channel `investigation:{id}` as each event happens. Mongo's
    events[] is the source of truth (SSE replays from here on reconnect);
    Redis pub/sub is just the live tail (see Phase 5 of the build plan)."""

    type: str  # e.g. "status", "tool_call", "delegation", "answer", "cancelled", "error"
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
