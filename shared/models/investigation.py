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

    # Everything worker_service/tasks/reconciliation.py needs to safely re-enqueue this exact
    # run_investigation call if the worker that owned it dies before producing a result - without
    # these, a stuck "running" investigation can be detected but not actually retried. Populated
    # once, at creation, in api_service/routers/chats.py's send_message.
    user_id: Optional[str] = None
    file_ids: list[str] = Field(default_factory=list)
    email: Optional[str] = None

    # Idempotency guards so the reconciliation sweep can safely backfill bookkeeping that a
    # crashed worker never got to (see worker_service/tasks/investigation.py's
    # _finalize_investigation_bookkeeping) without ever double-counting usage or double-enqueuing
    # the chat-memory update if it runs more than once against the same investigation.
    usage_counted: bool = False
    chat_memory_enqueued: bool = False

    # How many times reconciliation has re-dispatched this investigation to a fresh
    # run_investigation job after finding it stuck with no result at all - capped (see
    # reconciliation.py's MAX_AUTO_RETRIES) so a systematically broken request (bad file, bug)
    # doesn't retry forever instead of eventually failing loudly.
    retry_count: int = 0
    # Updated on retry dispatch (in addition to started_at, which stays as the true original
    # request time) so reconciliation measures "how long has THIS attempt been running" rather
    # than re-triggering immediately on every sweep after a retry.
    last_attempt_at: Optional[datetime] = None
