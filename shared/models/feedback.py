from datetime import datetime

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "feedback"


class Feedback(MongoModel):
    user_id: str
    message: str
    created_at: datetime = Field(default_factory=utcnow)
