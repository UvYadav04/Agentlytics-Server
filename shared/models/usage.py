from datetime import datetime

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "usage"


class Usage(MongoModel):
    user_id: str
    messages_sent: int = 0
    charts_created: int = 0
    reports_created: int = 0
    period_start: datetime = Field(default_factory=utcnow)
