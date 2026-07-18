from datetime import datetime
from typing import Optional

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "users"


class User(MongoModel):
    google_id: str
    email: str
    name: str
    picture: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
