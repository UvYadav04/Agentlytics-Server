"""arq job: refresh_dashboard(ctx, dashboard_id).

Re-runs a real-time dashboard's stored transform_script against its dependencies' CURRENT
data (this is what makes a file swap via the relink endpoint take effect - the File lookup
below happens fresh on every refresh, keyed only by file_id - Mongo's own doc _id - never
against a remembered path), matches the sandbox's fresh save() outputs back to each chart by
name, re-renders each matched chart, and overwrites its EXISTING Chart doc's storage_key
content in R2 - same chart_id/URL throughout, so nothing that already links to it breaks and
no new Chart docs (or usage-cap charges) get created on refresh.

Entirely independent of any Investigation - this isn't a chat turn, so there's no SSE event
stream to publish progress to. A chart the script didn't produce a save() for this run (e.g.
because the script raised before reaching it - see sandbox/runner.py's note on why earlier
save() calls still survive a later exception) is left with its last-good content rather than
being blanked out.
"""
import asyncio
import logging
import re
from datetime import datetime, timezone

import pandas as pd

from worker_service import engine_bootstrap  # noqa: F401

from analyzerEngine.sandbox.path_resolver import get_parquet_path
from analyzerEngine.tools.reporting.models import ChartSpec
from analyzerEngine.tools.reporting.reporting_tools import ReportingTools
from analyzerEngine.tools.tabular.sandbox_executor import PythonSandbox, SandboxExecutionError

from shared.db import get_db
from shared.job_timing import log_job_finished, log_job_picked_up
from shared.models.chart import COLLECTION as CHARTS
from shared.models.dashboard import COLLECTION as DASHBOARDS
from shared.models.dashboard import Dashboard
from shared.models.file import COLLECTION as FILES
from shared.storage import get_bucket_name, get_s3_client

logger = logging.getLogger("worker.dashboard_refresh")


def _safe_name(name: str) -> str:
    """Mirror sandbox/path_resolver.py's new_artifact_id sanitization, so a chart's stored
    `name` matches the prefix of whatever file_id save() actually wrote its fresh output
    under."""
    return re.sub(r"[^0-9a-zA-Z_]", "_", str(name))[:40].strip("_") or "result"


async def refresh_dashboard(ctx, dashboard_id: str, requested_at: str | None = None) -> None:
    picked_up_at = log_job_picked_up(logger, ctx, "refresh_dashboard", requested_at=requested_at, dashboard_id=dashboard_id)
    status_for_log = "unknown"
    try:
        db = get_db()
        doc = await db[DASHBOARDS].find_one({"_id": dashboard_id})
        dashboard = Dashboard.from_mongo(doc)
        if dashboard is None:
            logger.warning("refresh_dashboard: dashboard %s no longer exists, skipping", dashboard_id)
            status_for_log = "skipped_missing"
            return
        if not dashboard.real_time or not dashboard.transform_script:
            logger.warning("refresh_dashboard: dashboard %s is not a real-time dashboard, skipping", dashboard_id)
            status_for_log = "skipped_not_realtime"
            return
        if not dashboard.file_ids:
            logger.warning("refresh_dashboard: dashboard %s has no file dependencies, skipping", dashboard_id)
            status_for_log = "skipped_no_files"
            return

        file_docs = await db[FILES].find({"_id": {"$in": dashboard.file_ids}}).to_list(length=len(dashboard.file_ids))
        # Dashboard.file_ids ARE file_ids (Mongo doc _id, == the artifact's file_id per
        # sandbox/path_resolver.py's convention) - no output_ref/path lookup needed at all, just a
        # "ready" filter. sandbox.run() takes {table_name: file_id} directly.
        ready_ids = {f["_id"] for f in file_docs if f.get("status") == "ready"}
        tables = {fid: fid for fid in dashboard.file_ids if fid in ready_ids}
        missing = [fid for fid in dashboard.file_ids if fid not in ready_ids]
        if missing:
            logger.warning(
                "refresh_dashboard: dashboard %s missing a ready file for %s - refreshing with what's available",
                dashboard_id, missing,
            )
        if not tables:
            logger.warning("refresh_dashboard: no ready files to run against for dashboard %s, skipping", dashboard_id)
            status_for_log = "skipped_no_ready_files"
            return

        # Not part of any Investigation (see module docstring), so there's no investigation_id to
        # reuse a sandbox by - each refresh gets its own synthetic id and its container is released
        # immediately after, same one-shot-per-call behavior this had before the persistent-sandbox
        # architecture (see sandbox/sandbox_manager.py). Prefixed so it can never collide with a real
        # investigation_id's own sandbox cache entry.
        sandbox_investigation_id = f"dashboard-refresh-{dashboard_id}"
        # ctx["sandbox_manager"] (built once in worker.py's on_startup), passed explicitly - see
        # investigation.py's matching note on why this must be the exact same instance the rest
        # of the process uses, not a fresh get_manager() lookup that could resolve to a
        # different singleton depending on which import path reached it.
        sandbox_manager = ctx["sandbox_manager"]
        sandbox = PythonSandbox(
            root_dir=engine_bootstrap.PARQUET_ROOT, investigation_id=sandbox_investigation_id,
            manager=sandbox_manager,
        )
        try:
            # PythonSandbox.run() blocks on Docker/UDS calls - keep it off the event loop the
            # same way worker_service/tasks/ingestion.py does for IngestionManager.ingest_file.
            result = await asyncio.to_thread(sandbox.run, dashboard.transform_script, tables, dashboard.workspace_id)
        except SandboxExecutionError as exc:
            logger.error("refresh_dashboard: sandbox failed for dashboard %s: %s", dashboard_id, exc)
            status_for_log = "sandbox_failed"
            return
        finally:
            try:
                await asyncio.to_thread(sandbox_manager.release, sandbox_investigation_id)
            except Exception:
                logger.exception("refresh_dashboard: failed to release sandbox for dashboard %s", dashboard_id)

        saved = result.get("saved") or []
        if result.get("error"):
            logger.warning(
                "refresh_dashboard: transform_script raised for dashboard %s (charts whose save() ran "
                "before the failure may still have refreshed): %s", dashboard_id, result["error"],
            )
        if not saved:
            logger.warning("refresh_dashboard: transform_script produced no saved output for dashboard %s", dashboard_id)
            status_for_log = "no_saved_output"
            return

        chart_docs = await db[CHARTS].find({"_id": {"$in": dashboard.chart_ids}}).to_list(length=len(dashboard.chart_ids))
        storage_key_by_id = {c["_id"]: c["storage_key"] for c in chart_docs}

        s3 = get_s3_client()
        bucket = get_bucket_name()
        refreshed = 0

        for chart in dashboard.charts:
            prefix = f"{_safe_name(chart.name)}_"
            # save()'s file_id is already "{safe_name}_{hex8}" with no path/extension attached
            # (see sandbox/path_resolver.py's new_artifact_id) - a direct prefix match, no more
            # splitting a path apart to get at the basename first.
            match = next((s for s in saved if s["file_id"].startswith(prefix)), None)
            if match is None:
                logger.warning(
                    "refresh_dashboard: no save() output matched chart '%s' (dashboard %s) - leaving its "
                    "current content in place", chart.name, dashboard_id,
                )
                continue

            storage_key = storage_key_by_id.get(chart.chart_id)
            if storage_key is None:
                logger.warning(
                    "refresh_dashboard: chart_id %s referenced by dashboard %s has no Chart doc - skipping",
                    chart.chart_id, dashboard_id,
                )
                continue

            try:
                path = get_parquet_path(engine_bootstrap.PARQUET_ROOT, dashboard.workspace_id, match["file_id"])
                dataframe = pd.read_parquet(path)
                spec = ChartSpec(
                    file_id=match["file_id"], chart_type=chart.chart_type, title=chart.title,
                    name=chart.name, label_column=chart.label_column, value_columns=chart.value_columns,
                    time_column=chart.time_column, series_column=chart.series_column,
                    value_column=chart.value_column, x_column=chart.x_column, y_column=chart.y_column,
                    z_column=chart.z_column,
                )
                section = ReportingTools._render_section(dataframe, spec)
                html = ReportingTools._render_html(section["title"], [section], source_count=1)
                s3.put_object(Bucket=bucket, Key=storage_key, Body=html.encode("utf-8"), ContentType="text/html")
                refreshed += 1
            except Exception:
                logger.exception(
                    "refresh_dashboard: failed to re-render chart '%s' for dashboard %s", chart.name, dashboard_id,
                )

        await db[DASHBOARDS].update_one(
            {"_id": dashboard_id}, {"$set": {"last_refreshed_at": datetime.now(timezone.utc)}},
        )
        logger.info(
            "refresh_dashboard: dashboard %s refreshed (%d/%d charts updated)",
            dashboard_id, refreshed, len(dashboard.charts),
        )
        status_for_log = f"refreshed_{refreshed}_of_{len(dashboard.charts)}"
    finally:
        log_job_finished(logger, "refresh_dashboard", picked_up_at, status=status_for_log, dashboard_id=dashboard_id)
