from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "dashboards"


class ChartConfig(BaseModel):
    chart_id: str
    name: str
    chart_type: str = "bar"
    title: Optional[str] = None
    label_column: Optional[str] = None
    value_columns: Optional[list[str]] = None
    time_column: Optional[str] = None
    series_column: Optional[str] = None
    value_column: Optional[str] = None
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    z_column: Optional[str] = None
    bins: Optional[int] = None


class Dashboard(MongoModel):
    workspace_id: str
    name: str
    chart_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)

    real_time: bool = False
    file_ids: list[str] = Field(default_factory=list)
    file_ids: list[str] = Field(default_factory=list)
    transform_script: Optional[str] = None
    charts: list[ChartConfig] = Field(default_factory=list)
    global_filters: dict = Field(default_factory=dict)
    layout: list[dict] = Field(default_factory=list)  
    last_refreshed_at: Optional[datetime] = None
