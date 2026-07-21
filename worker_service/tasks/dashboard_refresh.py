"""arq job: refresh_dashboard(ctx, dashboard_id).

Re-runs a real-time dashboard's stored transform_script against its dependencies' CURRENT
output_refs (this is what makes a file swap via the relink endpoint take effect - the File
lookup below happens fresh on every refresh, never against a remembered output_ref), matches
the sandbox's fresh save() outputs back to each chart by name, re-renders each matched chart,
and overwrites its EXISTING Chart doc's storage_key content in R2 - same chart_id/URL
throughout, so nothing that already links to it breaks and no new Chart docs (or usage-cap
charges) get created on refresh.

Entirely independent of any Investigation - this isn't a chat turn, so there's no SSE event
stream to publish progress to. A chart the script didn't produce a save() for this run (e.g.
because the script raised before reaching it - see sandbox/runner.py's note on why earlier
save() calls still survive a later exception) is left with its last-good content rather than
being blanked out.
"""
import asyncio
import logging
import os
import re
from datetime import datetime, timezone

import pandas as pd

from worker_service import engine_bootstrap  # noqa: F401

from analyzerEngine.tools.reporting.models import ChartSpec
from analyzerEngine.tools.reporting.reporting_tools import ReportingTools
from analyzerEngine.tools.tabular.sandbox_executor import PythonSandbox, SandboxExecutionError

from shared.db import get_db
from shared.models.chart import COLLECTION as CHARTS
from shared.models.dashboard import COLLECTION as DASHBOARDS
from shared.models.dashboard import Dashboard
from shared.models.file import COLLECTION as FILES
from shared.storage import get_bucket_name, get_s3_client

logger = logging.getLogger("worker.dashboard_refresh")


def _safe_name(name: str) -> str:
    """Mirror sandbox/runner.py's save() sanitization exactly, so a chart's stored `name`
    matches the prefix of whatever path save() actually wrote its fresh output to."""
    return re.sub(r"[^0-9a-zA-Z_]", "_", str(name))[:60] or "result"


async def refresh_dashboard(ctx, dashboard_id: str) -> None:
    db = get_db()
    doc = await db[DASHBOARDS].find_one({"_id": dashboard_id})
    dashboard = Dashboard.from_mongo(doc)
    if dashboard is None:
        logger.warning("refresh_dashboard: dashboard %s no longer exists, skipping", dashboard_id)
        return
    if not dashboard.real_time or not dashboard.transform_script:
        logger.warning("refresh_dashboard: dashboard %s is not a real-time dashboard, skipping", dashboard_id)
        return
    if not dashboard.file_ids:
        logger.warning("refresh_dashboard: dashboard %s has no file dependencies, skipping", dashboard_id)
        return

    file_docs = await db[FILES].find({"_id": {"$in": dashboard.file_ids}}).to_list(length=len(dashboard.file_ids))
    output_refs = {
        f["_id"]: f["output_ref"] for f in file_docs if f.get("status") == "ready" and f.get("output_ref")
    }
    missing = [fid for fid in dashboard.file_ids if fid not in output_refs]
    if missing:
        logger.warning(
            "refresh_dashboard: dashboard %s missing a ready file for %s - refreshing with what's available",
            dashboard_id, missing,
        )
    if not output_refs:
        logger.warning("refresh_dashboard: no ready files to run against for dashboard %s, skipping", dashboard_id)
        return

    sandbox = PythonSandbox(root_dir=engine_bootstrap.PARQUET_ROOT)
    try:
        # PythonSandbox.run() blocks on container.wait() - keep it off the event loop the
        # same way worker_service/tasks/ingestion.py does for IngestionManager.ingest_file.
        result = await asyncio.to_thread(sandbox.run, dashboard.transform_script, output_refs, dashboard.workspace_id)
    except SandboxExecutionError as exc:
        logger.error("refresh_dashboard: sandbox failed for dashboard %s: %s", dashboard_id, exc)
        return

    saved = result.get("saved") or []
    if result.get("error"):
        logger.warning(
            "refresh_dashboard: transform_script raised for dashboard %s (charts whose save() ran "
            "before the failure may still have refreshed): %s", dashboard_id, result["error"],
        )
    if not saved:
        logger.warning("refresh_dashboard: transform_script produced no saved output for dashboard %s", dashboard_id)
        return

    chart_docs = await db[CHARTS].find({"_id": {"$in": dashboard.chart_ids}}).to_list(length=len(dashboard.chart_ids))
    storage_key_by_id = {c["_id"]: c["storage_key"] for c in chart_docs}

    s3 = get_s3_client()
    bucket = get_bucket_name()
    refreshed = 0

    for chart in dashboard.charts:
        prefix = f"{_safe_name(chart.name)}_"
        match = next(
            (s for s in saved if os.path.splitext(os.path.basename(s["output_ref"]))[0].startswith(prefix)),
            None,
        )
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
            dataframe = pd.read_parquet(match["output_ref"])
            spec = ChartSpec(
                output_ref=match["output_ref"], chart_type=chart.chart_type, title=chart.title,
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
