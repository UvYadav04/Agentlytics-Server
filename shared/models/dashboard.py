from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "dashboards"


class ChartConfig(BaseModel):
    """One chart within a real-time dashboard - everything needed to find that chart's
    fresh output after transform_script re-runs, and to re-render it. Mirrors
    tools.reporting.models.ChartSpec (analyzerEngine's dataclass version of the same
    shape) plus the two fields that only matter once this is persisted: chart_id (which
    Chart doc/storage_key to overwrite) and name (the stable save(..., name=X) key used
    inside transform_script - matched against the sandbox's fresh saved-output list by
    prefix on every refresh, since save() appends a random suffix to the path each run).

    Deliberately a plain BaseModel, not MongoModel - this is an embedded object inside
    Dashboard.charts, not its own Mongo document, so it shouldn't get MongoModel's
    auto-generated `id`/`_id` rename machinery (that would collide confusingly with
    chart_id below)."""

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


class Dashboard(MongoModel):
    workspace_id: str
    name: str
    chart_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)

    # Everything below is empty/default for today's static, manually-curated
    # dashboards (POST /workspaces/{id}/dashboards with existing chart_ids) - a
    # real-time dashboard is the same document shape, just with these populated.
    real_time: bool = False
    # file_ids drives both refresh (resolve each file's current output_ref before
    # re-running transform_script) and the relink flow (swap one file_id for another
    # when the user replaces a data source).
    file_ids: list[str] = Field(default_factory=list)
    # ONE script for the whole dashboard, re-run wholesale on every refresh - see the
    # architecture discussion this was born from: the sandbox runner's `saved` list is
    # populated incrementally by save(), so one chart's block raising doesn't erase
    # save() calls that already ran earlier in the same script (sandbox/runner.py).
    transform_script: Optional[str] = None
    charts: list[ChartConfig] = Field(default_factory=list)
    global_filters: dict = Field(default_factory=dict)
    layout: list[dict] = Field(default_factory=list)  # [{chart_id, x, y, w, h}, ...]
    last_refreshed_at: Optional[datetime] = None
