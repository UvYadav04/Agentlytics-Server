import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from shared import usage
from shared.models.chart import COLLECTION as CHARTS, Chart
from shared.models.dashboard import (
    COLLECTION as DASHBOARDS,
    ChartAxis,
    ChartConfig,
    Dashboard,
    TimeWindow,
)
from shared.storage import build_chart_key, get_bucket_name, get_s3_client, new_file_id

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
    Dedicated unified service to handle all dashboard operations across the platform:
    1. build_live_dashboard: builds and stages live database dashboards with validated queries.
    2. stage_live_dashboard: records staged live dashboard metadata.
    3. stage_bundle_dashboard: records staged static/computed dashboard bundle metadata.
    4. persist_dashboard: persists live or bundle dashboards to MongoDB, uploading charts to S3.
    """

    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = output_dir
        self._staged_dashboards: dict[str, StagedDashboard] = {}

    async def build_live_dashboard(
        self,
        connection_id: str,
        request: str,
        database_access: Any = None,
        connection_info: Optional[dict] = None,
        connection_string: Optional[str] = None,
        workspace_id: str = "default",
        on_event: Optional[Callable] = None,
    ) -> dict[str, Any]:
        """
        Build a persistent live database dashboard from a database connection.
        Inspects schema, generates read-only queries for charts, and stages the dashboard.
        """
        conn = connection_info
        conn_str = connection_string

        if (not conn or not conn_str) and database_access is not None:
            try:
                conn, conn_str = await database_access.resolve(connection_id)
            except LookupError as exc:
                return {"error": str(exc), "status": "error"}

        if not conn or not conn_str:
            return {
                "error": f"Database connection '{connection_id}' could not be resolved. Ensure credentials are valid.",
                "status": "error",
            }

        try:
            from tools.dashboard.live_builder import LiveDashboardBuildError, LiveDashboardBuilder
        except ImportError:
            from analyzerEngine.tools.dashboard.live_builder import LiveDashboardBuildError, LiveDashboardBuilder

        builder = LiveDashboardBuilder(on_event=on_event)
        try:
            spec = await builder.build(request, conn, conn_str)
        except LiveDashboardBuildError as exc:
            logger.info("live dashboard build failed for connection %s: %s", connection_id, exc)
            return {"error": str(exc), "status": "error"}

        reference = f"live-dashboard-{uuid.uuid4().hex[:12]}"
        staged_meta = self.stage_live_dashboard(
            workspace_id=workspace_id,
            title=spec.get("title", "Live Dashboard"),
            spec=spec,
            connection_id=conn.get("id") or connection_id,
            dashboard_id=reference,
        )

        return {
            "status": "created",
            "dashboard_id": reference,
            "title": spec.get("title", "Live Dashboard"),
            "database": conn.get("name", "database"),
            "time_window": spec.get("time_window"),
            "chart_count": len(spec.get("charts", [])),
            "charts": [
                {"title": chart["title"], "chart_type": chart["chart_type"], "rows_now": chart.get("row_count")}
                for chart in spec.get("charts", [])
            ],
            "skipped_charts": [d["title"] for d in spec.get("dropped", [])],
        }

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
        Persist a staged live or bundle dashboard to MongoDB (db.dashboards).
        Enforces user usage quotas and links message_id.
        """
        staged = self.get_by_id(dashboard_id)
        if staged is None:
            logger.warning("persist_dashboard: dashboard %s not found in staging", dashboard_id)
            return None

        now = datetime.now(timezone.utc)

        # ---------------------------------------------------------------------
        # Case A: Live database dashboard
        # ---------------------------------------------------------------------
        if staged.kind == "live" and staged.spec:
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

            window = spec.get("time_window")
            db_type = spec.get("db_type") or "mongodb"
            if db_type not in ("postgresql", "mongodb", "parquet"):
                db_type = "mongodb"

            dashboard = Dashboard(
                workspace_id=workspace_id,
                name=spec.get("title") or staged.title or "Live Dashboard",
                description=spec.get("description"),
                created_by=user_id,
                created_at=now,
                updated_at=now,
                source=db_type,
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

        # ---------------------------------------------------------------------
        # Case B: Bundle dashboard
        # ---------------------------------------------------------------------
        if staged.kind == "bundle" and staged.manifest_path and os.path.isfile(staged.manifest_path):
            manifest_path = staged.manifest_path
            folder = os.path.dirname(manifest_path)
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)

            chart_ids: list = []
            chart_configs: list[ChartConfig] = []
            s3 = get_s3_client()
            bucket = get_bucket_name()

            for chart_meta in manifest.get("charts", []):
                if not await usage.has_chart_capacity(user_id, email):
                    break
                html_path = os.path.join(folder, chart_meta.get("html_filename", ""))
                if not os.path.isfile(html_path):
                    continue

                cid = new_file_id()
                key = build_chart_key(workspace_id, cid)
                s3.upload_file(html_path, bucket, key, ExtraArgs={"ContentType": "text/html"})
                chart = Chart(
                    id=cid, workspace_id=workspace_id, message_id=message_id,
                    title=chart_meta.get("title") or "Untitled chart", storage_key=key,
                )
                await db[CHARTS].insert_one(chart.to_mongo())
                await usage.increment_charts(user_id)
                chart_ids.append(cid)
                chart_configs.append(ChartConfig(
                    chart_id=cid,
                    name=chart_meta["name"],
                    chart_type=chart_meta.get("chart_type", "bar"),
                    title=chart_meta.get("title"),
                    label_column=chart_meta.get("label_column"),
                    value_columns=chart_meta.get("value_columns"),
                    time_column=chart_meta.get("time_column"),
                    series_column=chart_meta.get("series_column"),
                    value_column=chart_meta.get("value_column"),
                    x_column=chart_meta.get("x_column"),
                    y_column=chart_meta.get("y_column"),
                    z_column=chart_meta.get("z_column"),
                    bins=chart_meta.get("bins"),
                ))

            if not chart_configs:
                return None

            dashboard = Dashboard(
                workspace_id=workspace_id,
                name=manifest.get("title") or staged.title or "Untitled dashboard",
                chart_ids=chart_ids,
                real_time=True,
                file_ids=manifest.get("file_ids") or [],
                transform_script=manifest.get("transform_script"),
                charts=chart_configs,
                last_refreshed_at=now,
            )
            await db[DASHBOARDS].insert_one(dashboard.to_mongo())
            logger.info("Successfully persisted bundle dashboard %s -> %s in MongoDB", dashboard_id, dashboard.id)
            self._staged_dashboards.pop(dashboard_id, None)
            return dashboard.id

        return None

    def clear(self) -> None:
        """Clear all staged dashboards."""
        self._staged_dashboards.clear()
