from datetime import datetime

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "chats"


class Chat(MongoModel):
    workspace_id: str
    title: str = "New chat"
    created_at: datetime = Field(default_factory=utcnow)
    summary: str = ""
    files_used: list[str] = Field(default_factory=list)
    files_created: list[str] = Field(default_factory=list)
    recent_turns: list[dict] = Field(default_factory=list)
