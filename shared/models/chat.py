from datetime import datetime

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "chats"

# Sentinel used both as the field default and as the "hasn't been titled yet" marker that
# worker_service/tasks/chat_title.py checks before auto-titling a chat from its first message -
# and that it never overwrites once the user (or a prior run of that job) has set something else.
DEFAULT_TITLE = "New chat"


class Chat(MongoModel):
    workspace_id: str
    title: str = DEFAULT_TITLE
    created_at: datetime = Field(default_factory=utcnow)
    summary: str = ""
    files_used: list[str] = Field(default_factory=list)
    files_created: list[str] = Field(default_factory=list)
    recent_turns: list[dict] = Field(default_factory=list)
