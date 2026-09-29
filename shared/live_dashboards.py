import logging
from typing import Optional

from shared.external_db.errors import ConnectionStringError, DatabaseInspectionError
from shared.external_db.queries import QueryResult
from shared.external_db.service import run_dashboard_queries
from shared.external_db.time_window import resolve_window
from shared.models.base import utcnow
from shared.models.dashboard import Dashboard
from shared.models.database_connection import COLLECTION as DATABASE_CONNECTIONS
from shared.models.database_connection import DatabaseConnection
from shared.secret_box import SecretBoxError, decrypt_secret

logger = logging.getLogger("shared.live_dashboards")

UNIT_LABELS = {"day": "day", "week": "week", "month": "month", "quarter": "quarter", "year": "year"}


def window_label(window) -> Optional[str]:
    if window is None:
        return None
    unit = UNIT_LABELS[window.unit]
    return f"Last {window.last} {unit}{'' if window.last == 1 else 's'}"


def _status(status: str, message: Optional[str], dashboard: Dashboard) -> dict:
    return {
        "status": status,
        "message": message,
        "generated_at": utcnow().isoformat(),
        "window": None,
        "charts": [{"chart_id": c.chart_id, "columns": [], "rows": [], "truncated": False, "error": None} for c in dashboard.charts],
    }


async def load_live_data(db, dashboard: Dashboard) -> dict:
    if not dashboard.is_live_database:
        raise ValueError("not a live database dashboard")

    doc = await db[DATABASE_CONNECTIONS].find_one(
        {"_id": dashboard.connection_id, "workspace_id": dashboard.workspace_id},
    )
    connection = DatabaseConnection.from_mongo(doc)
    if connection is None:
        return _status("connection_missing", "The database connection behind this dashboard was removed.", dashboard)
    if connection.is_expired():
        return _status(
            "credentials_expired",
            "The database credentials for this dashboard have expired.",
            dashboard,
        )
    try:
        connection_string = decrypt_secret(connection.encrypted_connection_string)
    except SecretBoxError:
        logger.error("could not decrypt credentials for database connection %s", connection.id)
        return _status("credentials_expired", "The database credentials for this dashboard can't be used.", dashboard)

    window = resolve_window(dashboard.time_window.model_dump()) if dashboard.time_window else None
    queries = [chart.query or {} for chart in dashboard.charts]
    try:
        outcomes = await run_dashboard_queries(
            connection.db_type, connection_string, dashboard.database or connection.database, queries, window,
        )
    except (ConnectionStringError, DatabaseInspectionError) as exc:
        return _status("error", str(exc), dashboard)

    charts = []
    for chart, outcome in zip(dashboard.charts, outcomes):
        if isinstance(outcome, QueryResult):
            charts.append({
                "chart_id": chart.chart_id, "columns": outcome.columns, "rows": outcome.rows,
                "truncated": outcome.truncated, "error": None,
            })
        else:
            charts.append({
                "chart_id": chart.chart_id, "columns": [], "rows": [], "truncated": False,
                "error": str(outcome) or "This chart's query failed.",
            })
    return {
        "status": "ok",
        "message": None,
        "generated_at": utcnow().isoformat(),
        "window": (
            {"start": window.start.isoformat(), "end": window.end.isoformat(), "label": window_label(dashboard.time_window)}
            if window else None
        ),
        "charts": charts,
    }
