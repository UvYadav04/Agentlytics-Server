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
    # file_ids this specific message's investigation actually read from - straight off
    # OrchestratorResult.files_used (see analyzerEngine/tools/orchestrator/models.py's
    # FinalResultCollector) - shown client-side as "files used" chips so the user can see what
    # this answer was actually based on.
    files_used: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
