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
    # 2-3 suggested next questions (see analyzerEngine/tools/orchestrator/follow_up.py) - only
    # populated for assistant messages that went through a real investigation (Orchestrator or a
    # direct-routed Tabular/Document agent), empty for user messages and for the light-response
    # greeting/no-files path (see api_service/light_investigation.py, which never sets this).
    follow_up_questions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
