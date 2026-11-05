import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from shared import usage
from shared.models.dashboard import (
    COLLECTION as DASHBOARDS,
    ChartAxis,
    ChartConfig,
    Dashboard,
    TimeWindow,
)

logger = logging.getLogger("shared.services.dashboard")


@dataclass
class StagedDashboard:
    dashboard_id: str
    workspace_id: str
    title: str
    kind: str  # "live" or "bundle"
    spec: Optional[dict] = None
    connection_id: Optional[str] = None
    manifest_path: Optional[str] = None
    charts_meta: list[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DashboardService:
    """
    Dedicated service to handle dashboard definitions and local staging.
    Prevents large dashboard JSON manifests and multi-chart queries from
    flooding the LLM context window.
    Returns only lightweight metadata (dashboard_id, title, chart count) to the agent.
    When persistence is needed, fetches full data by dashboard_id and saves to MongoDB.
    """

    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = output_dir
        self._staged_dashboards: dict[str, StagedDashboard] = {}

    def stage_live_dashboard(
        self,
        workspace_id: str,
        title: str,
        spec: dict,
        connection_id: Optional[str] = None,
        dashboard_id: Optional[str] = None,
    ) -> dict:
        """
        Stage a live database dashboard definition.
        Returns ONLY compact metadata for LLM context.
        """
        did = dashboard_id or f"live-dashboard-{uuid.uuid4().hex[:12]}"
        charts = spec.get("charts", [])
        staged = StagedDashboard(
            dashboard_id=did,
            workspace_id=workspace_id,
            title=spec.get("title") or title or "Live Dashboard",
            kind="live",
            spec=spec,
            connection_id=connection_id,
            charts_meta=[
                {"title": c.get("title"), "chart_type": c.get("chart_type"), "row_count": c.get("row_count")}
                for c in charts
            ],
        )
        self._staged_dashboards[did] = staged
        logger.info("Staged live dashboard %s (%s) with %d chart(s)", did, staged.title, len(charts))

        return {
            "dashboard_id": did,
            "title": staged.title,
            "database": spec.get("database"),
            "source": spec.get("db_type", "database"),
            "chart_count": len(charts),
            "chart_titles": [c.get("title") for c in charts],
            "status": "staged",
        }

    def stage_bundle_dashboard(
        self,
        workspace_id: str,
        title: str,
        manifest_path: str,
        dashboard_id: Optional[str] = None,
    ) -> dict:
        """Stage a static/computed dashboard bundle manifest."""
        did = dashboard_id or f"dashboard-bundle-{uuid.uuid4().hex[:12]}"
        staged = StagedDashboard(
            dashboard_id=did,
            workspace_id=workspace_id,
            title=title or "Dashboard Bundle",
            kind="bundle",
            manifest_path=manifest_path,
        )
        self._staged_dashboards[did] = staged
        return {
            "dashboard_id": did,
            "title": staged.title,
            "kind": "bundle",
            "status": "staged",
        }

    def get_by_id(self, dashboard_id: str) -> Optional[StagedDashboard]:
        """Fetch full staged dashboard record by ID."""
        return self._staged_dashboards.get(dashboard_id)

    def list_staged(self, workspace_id: Optional[str] = None) -> list[StagedDashboard]:
        """List all staged dashboards, optionally filtered by workspace."""
        if workspace_id:
            return [d for d in self._staged_dashboards.values() if d.workspace_id == workspace_id]
        return list(self._staged_dashboards.values())

    async def persist_dashboard(
        self,
        dashboard_id: str,
        db,
        workspace_id: str,
        message_id: str,
        user_id: str,
        email: Optional[str] = None,
    ) -> Optional[str]:
        """
        Persist a staged live dashboard to MongoDB (db.dashboards).
        Enforces user usage quotas per chart and links message_id.
        """
        staged = self.get_by_id(dashboard_id)
        if staged is None or staged.kind != "live" or not staged.spec:
            logger.warning("persist_dashboard: live dashboard %s not found in staging", dashboard_id)
            return None

        spec = staged.spec
        charts: list[ChartConfig] = []
        for chart in spec.get("charts", []):
            if not await usage.has_chart_capacity(user_id, email):
                logger.info("persist_dashboard: chart limit reached for user %s during dashboard save", user_id)
                break
            value_columns = chart.get("value_columns") or []
            charts.append(
                ChartConfig(
                    name=chart["title"],
                    title=chart["title"],
                    description=chart.get("description"),
                    chart_type=chart["chart_type"],
                    x_column=chart.get("x_column"),
                    label_column=chart.get("x_column"),
                    value_columns=value_columns,
                    value_column=value_columns[0] if len(value_columns) == 1 else None,
                    series_column=chart.get("series_column"),
                    aggregation=chart.get("aggregation"),
                    axis=ChartAxis(**(chart.get("axis") or {})),
                    labels=chart.get("labels") or {},
                    query=chart.get("query"),
                    validated_at=chart.get("validated_at"),
                )
            )
            await usage.increment_charts(user_id)

        if not charts:
            return None

        now = datetime.now(timezone.utc)
        window = spec.get("time_window")
        dashboard = Dashboard(
            workspace_id=workspace_id,
            name=spec.get("title") or staged.title or "Live Dashboard",
            description=spec.get("description"),
            created_by=user_id,
            created_at=now,
            updated_at=now,
            source=spec.get("db_type", "database"),
            connection_id=staged.connection_id,
            database=spec.get("database"),
            tables=spec.get("tables") or [],
            time_window=TimeWindow(**window) if window else None,
            message_id=message_id,
            real_time=True,
            charts=charts,
        )
        await db[DASHBOARDS].insert_one(dashboard.to_mongo())
        logger.info("Successfully persisted live dashboard %s -> %s in MongoDB", dashboard_id, dashboard.id)
        self._staged_dashboards.pop(dashboard_id, None)
        return dashboard.id

    def clear(self) -> None:
        """Clear all staged dashboards."""
        self._staged_dashboards.clear()

