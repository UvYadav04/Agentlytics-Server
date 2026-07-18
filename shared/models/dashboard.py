from datetime import datetime

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "dashboards"


class Dashboard(MongoModel):
    workspace_id: str
    name: str
    chart_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
