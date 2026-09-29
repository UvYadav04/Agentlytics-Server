from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from shared.models.base import MongoModel, new_id, utcnow

COLLECTION = "dashboards"

DashboardSource = Literal["parquet", "postgresql", "mongodb"]
LIVE_CHART_TYPES = ("bar", "stacked_bar", "line", "area", "pie", "metric", "table")


class ChartAxis(BaseModel):
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    format: Literal["number", "currency", "percent"] = "number"


class ChartConfig(BaseModel):
    chart_id: str = Field(default_factory=new_id)
    name: str
    chart_type: str = "bar"
    title: Optional[str] = None
    description: Optional[str] = None
    label_column: Optional[str] = None
    value_columns: Optional[list[str]] = None
    time_column: Optional[str] = None
    series_column: Optional[str] = None
    value_column: Optional[str] = None
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    z_column: Optional[str] = None
    bins: Optional[int] = None
    axis: ChartAxis = Field(default_factory=ChartAxis)
    labels: dict[str, str] = Field(default_factory=dict)
    aggregation: Optional[str] = None
    query: Optional[dict] = None
    validated_at: Optional[datetime] = None


class TimeWindow(BaseModel):
    column: str
    last: int
    unit: Literal["day", "week", "month", "quarter", "year"]


class DashboardTheme(BaseModel):
    mode: Literal["light", "dark"] = "light"
    palette: Literal["default"] = "default"
    show_legend: bool = True
    show_grid: bool = True


class Dashboard(MongoModel):
    workspace_id: str
    name: str
    description: Optional[str] = None
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: Optional[datetime] = None

    source: DashboardSource = "parquet"
    connection_id: Optional[str] = None
    database: Optional[str] = None
    tables: Optional[list[str]] = None
    time_window: Optional[TimeWindow] = None
    theme: DashboardTheme = Field(default_factory=DashboardTheme)
    message_id: Optional[str] = None

    chart_ids: list[str] = Field(default_factory=list)
    real_time: bool = False
    file_ids: list[str] = Field(default_factory=list)
    transform_script: Optional[str] = None
    charts: list[ChartConfig] = Field(default_factory=list)
    global_filters: dict = Field(default_factory=dict)
    layout: list[dict] = Field(default_factory=list)
    last_refreshed_at: Optional[datetime] = None

    @property
    def is_live_database(self) -> bool:
        return self.source in ("postgresql", "mongodb")
