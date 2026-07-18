from datetime import datetime

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "chats"


class Chat(MongoModel):
    workspace_id: str
    title: str = "New chat"
    created_at: datetime = Field(default_factory=utcnow)
