from datetime import datetime

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "chats"


class Chat(MongoModel):
    workspace_id: str
    title: str = "New chat"
    created_at: datetime = Field(default_factory=utcnow)

    # Thread-level continuity for the orchestrator - refreshed by
    # worker_service.tasks.investigation after every completed investigation
    # in this chat (never by the orchestrator itself; a fresh
    # OrchestratorAgent is built per job, so nothing survives in memory
    # between messages - see agents/orchestrator/agent.py's
    # _thread_context_brief for how these get folded back into the next
    # investigation's task prompt).
    summary: str = ""
    files_used: list[str] = Field(default_factory=list)
    files_created: list[str] = Field(default_factory=list)
    recent_turns: list[dict] = Field(default_factory=list)
