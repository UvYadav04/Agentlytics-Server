import logging
import re
import time
from datetime import datetime, timedelta
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api_service.deps import get_current_user, get_owned_file, get_owned_workspace
from shared.db import get_db
from shared.external_db.time_window import validate_window
from shared.job_timing import now_iso
from shared.live_dashboards import load_live_data, window_label
from shared.models.base import utcnow
from shared.models.chart import COLLECTION as CHARTS
from shared.models.dashboard import COLLECTION as DASHBOARDS
from shared.models.dashboard import ChartConfig, Dashboard, DashboardTheme, TimeWindow
from shared.models.dashboard_share import COLLECTION as DASHBOARD_SHARES
from shared.models.dashboard_share import DashboardShare
from shared.models.database_connection import COLLECTION as DATABASE_CONNECTIONS
from shared.models.database_connection import DatabaseConnection, as_aware
from shared.models.user import COLLECTION as USERS
from shared.models.user import User
from shared.models.workspace import COLLECTION as WORKSPACES
from shared.redis_client import get_arq_pool, get_redis
from shared.storage import presign_get

logger = logging.getLogger("api.dashboards")

router = APIRouter(tags=["dashboards"])

LIVE_DATA_RATE_LIMIT = 60
LIVE_DATA_RATE_WINDOW_SECONDS = 60
MAX_SHARE_EMAILS = 25
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class ChartRef(BaseModel):
    id: str
    title: str
    url: str


class ConnectionBrief(BaseModel):
    id: str
    name: str
    db_type: str
    status: str


class LiveChartOut(BaseModel):
    chart_id: str
    title: str
    description: Optional[str]
    chart_type: str
    x_column: Optional[str]
    value_columns: list[str]
    series_column: Optional[str]
    aggregation: Optional[str]
    axis: dict
    labels: dict
    query: Optional[dict] = None
    validated_at: Optional[str] = None


class DashboardOut(BaseModel):
    id: str
    workspace_id: str
    name: str
    description: Optional[str]
    source: str
    database: Optional[str]
    tables: Optional[list[str]]
    time_window: Optional[dict]
    time_window_label: Optional[str]
    theme: dict
    chart_ids: list[str]
    chart_count: int
    created_at: str
    created_by: Optional[str]
    updated_at: Optional[str]
    real_time: bool
    file_ids: list[str]
    last_refreshed_at: str | None


class DashboardDetailOut(DashboardOut):
    charts: list[ChartRef]
    live_charts: list[LiveChartOut]
    connection: Optional[ConnectionBrief]


class LiveChartData(BaseModel):
    chart_id: str
    columns: list[str]
    rows: list[dict]
    truncated: bool
    error: Optional[str]


class LiveDataOut(BaseModel):
    status: Literal["ok", "credentials_expired", "connection_missing", "error"]
    message: Optional[str]
    generated_at: str
    window: Optional[dict]
    charts: list[LiveChartData]


class CreateDashboardRequest(BaseModel):
    name: str
    chart_ids: list[str] = []


class TimeWindowUpdate(BaseModel):
    last: int = Field(ge=1, le=3660)
    unit: Literal["day", "week", "month", "quarter", "year"]


class ThemeUpdate(BaseModel):
    mode: Optional[Literal["light", "dark"]] = None
    show_legend: Optional[bool] = None
    show_grid: Optional[bool] = None


class UpdateDashboardRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    chart_ids: list[str] | None = None
    theme: ThemeUpdate | None = None
    time_window: TimeWindowUpdate | None = None


class RelinkFileRequest(BaseModel):
    old_file_id: str
    new_file_id: str


class CreateShareRequest(BaseModel):
    emails: list[str] = Field(min_length=1, max_length=MAX_SHARE_EMAILS)
    valid_days: int = Field(default=30, ge=1, le=365)


class ShareOut(BaseModel):
    id: str
    dashboard_id: str
    shared_to: list[str]
    shared_at: str
    valid_upto: str
    revoked_at: Optional[str]
    active: bool
    path: str


class SharedByOut(BaseModel):
    name: str
    email: str


class SharedDashboardOut(BaseModel):
    share_id: str
    dashboard_id: str
    name: str
    description: Optional[str]
    source: str
    time_window: Optional[dict]
    time_window_label: Optional[str]
    theme: dict
    charts: list[ChartRef]
    live_charts: list[LiveChartOut]
    shared_by: SharedByOut
    shared_at: str
    valid_upto: str
    is_owner: bool


def _iso(value: Optional[datetime]) -> Optional[str]:
    aware = as_aware(value)
    return aware.isoformat() if aware else None


async def _get_owned_dashboard(dashboard_id: str, user: User) -> Dashboard:
    doc = await get_db()[DASHBOARDS].find_one({"_id": dashboard_id})
    dashboard = Dashboard.from_mongo(doc)
    if dashboard is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dashboard not found")
    await get_owned_workspace(dashboard.workspace_id, user)
    return dashboard


def _out(d: Dashboard) -> DashboardOut:
    return DashboardOut(
        id=d.id, workspace_id=d.workspace_id, name=d.name, description=d.description, source=d.source,
        database=d.database, tables=d.tables,
        time_window=d.time_window.model_dump() if d.time_window else None,
        time_window_label=window_label(d.time_window),
        theme=d.theme.model_dump(), chart_ids=d.chart_ids,
        chart_count=len(d.charts) if d.is_live_database else len(d.chart_ids),
        created_at=_iso(d.created_at), created_by=d.created_by, updated_at=_iso(d.updated_at),
        real_time=d.real_time, file_ids=d.file_ids, last_refreshed_at=_iso(d.last_refreshed_at),
    )


def _live_chart_out(chart: ChartConfig, include_query: bool) -> LiveChartOut:
    return LiveChartOut(
        chart_id=chart.chart_id, title=chart.title or chart.name, description=chart.description,
        chart_type=chart.chart_type, x_column=chart.x_column or chart.label_column,
        value_columns=chart.value_columns or ([chart.value_column] if chart.value_column else []),
        series_column=chart.series_column, aggregation=chart.aggregation,
        axis=chart.axis.model_dump(), labels=chart.labels,
        query=chart.query if include_query else None,
        validated_at=_iso(chart.validated_at) if include_query else None,
    )


async def _chart_refs(dashboard: Dashboard) -> list[ChartRef]:
    if not dashboard.chart_ids:
        return []
    docs = await get_db()[CHARTS].find({"_id": {"$in": dashboard.chart_ids}}).to_list(length=500)
    by_id = {d["_id"]: d for d in docs}
    refs = []
    for chart_id in dashboard.chart_ids:
        d = by_id.get(chart_id)
        if d is not None:
            refs.append(ChartRef(id=d["_id"], title=d["title"], url=presign_get(d["storage_key"])))
    return refs


async def _connection_brief(dashboard: Dashboard) -> Optional[ConnectionBrief]:
    if not dashboard.connection_id:
        return None
    doc = await get_db()[DATABASE_CONNECTIONS].find_one(
        {"_id": dashboard.connection_id, "workspace_id": dashboard.workspace_id},
        {"db_schema": 0},
    )
    connection = DatabaseConnection.from_mongo(doc)
    if connection is None:
        return None
    return ConnectionBrief(
        id=connection.id, name=connection.name, db_type=connection.db_type,
        status="expired" if connection.is_expired() else connection.status,
    )


async def _enforce_live_rate_limit(user: User) -> None:
    key = f"dashboard_live_loads:{user.id}"
    try:
        redis = get_redis()
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, LIVE_DATA_RATE_WINDOW_SECONDS)
    except Exception:
        logger.exception("live dashboard rate limiter unavailable")
        return
    if count > LIVE_DATA_RATE_LIMIT:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many dashboard refreshes. Wait a minute and try again.")


async def _live_data(dashboard: Dashboard, user: User) -> LiveDataOut:
    if not dashboard.is_live_database:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This dashboard isn't connected to a live database.")
    await _enforce_live_rate_limit(user)
    return LiveDataOut(**await load_live_data(get_db(), dashboard))


def _share_out(share: DashboardShare) -> ShareOut:
    return ShareOut(
        id=share.id, dashboard_id=share.dashboard_id, shared_to=share.shared_to,
        shared_at=_iso(share.shared_at), valid_upto=_iso(share.valid_upto),
        revoked_at=_iso(share.revoked_at), active=share.is_active(),
        path=f"/dashboards/shared/{share.id}",
    )


async def _get_accessible_share(share_id: str, user: User) -> tuple[DashboardShare, Dashboard, bool]:
    db = get_db()
    share = DashboardShare.from_mongo(await db[DASHBOARD_SHARES].find_one({"_id": share_id}))
    not_found = HTTPException(status.HTTP_404_NOT_FOUND, "This shared dashboard doesn't exist or is no longer shared with you.")
    if share is None or share.revoked_at is not None:
        raise not_found
    workspace = await db[WORKSPACES].find_one({"_id": share.workspace_id}, {"user_id": 1})
    is_owner = bool(workspace and workspace.get("user_id") == user.id)
    if not is_owner:
        if user.email.strip().lower() not in share.shared_to:
            raise not_found
        if not share.is_active():
            raise HTTPException(status.HTTP_410_GONE, "This share link has expired. Ask the owner to share the dashboard again.")
    dashboard = Dashboard.from_mongo(await db[DASHBOARDS].find_one({"_id": share.dashboard_id}))
    if dashboard is None or dashboard.workspace_id != share.workspace_id:
        raise not_found
    return share, dashboard, is_owner


@router.post("/workspaces/{workspace_id}/dashboards", response_model=DashboardOut)
async def create_dashboard(workspace_id: str, body: CreateDashboardRequest, user: User = Depends(get_current_user)):
    await get_owned_workspace(workspace_id, user)
    dashboard = Dashboard(workspace_id=workspace_id, name=body.name, chart_ids=body.chart_ids, created_by=user.id)
    await get_db()[DASHBOARDS].insert_one(dashboard.to_mongo())
    return _out(dashboard)


@router.get("/workspaces/{workspace_id}/dashboards", response_model=list[DashboardOut])
async def list_dashboards(workspace_id: str, user: User = Depends(get_current_user)):
    await get_owned_workspace(workspace_id, user)
    cursor = get_db()[DASHBOARDS].find({"workspace_id": workspace_id}).sort("created_at", -1)
    docs = await cursor.to_list(length=500)
    return [_out(Dashboard.from_mongo(d)) for d in docs]


@router.get("/dashboards/{dashboard_id}", response_model=DashboardDetailOut)
async def get_dashboard(dashboard_id: str, user: User = Depends(get_current_user)):
    dashboard = await _get_owned_dashboard(dashboard_id, user)
    live_charts = [_live_chart_out(c, include_query=True) for c in dashboard.charts] if dashboard.is_live_database else []
    return DashboardDetailOut(
        **_out(dashboard).model_dump(),
        charts=await _chart_refs(dashboard),
        live_charts=live_charts,
        connection=await _connection_brief(dashboard),
    )


@router.get("/dashboards/{dashboard_id}/data", response_model=LiveDataOut)
async def get_dashboard_data(dashboard_id: str, user: User = Depends(get_current_user)):
    dashboard = await _get_owned_dashboard(dashboard_id, user)
    return await _live_data(dashboard, user)


@router.patch("/dashboards/{dashboard_id}", response_model=DashboardOut)
async def update_dashboard(dashboard_id: str, body: UpdateDashboardRequest, user: User = Depends(get_current_user)):
    dashboard = await _get_owned_dashboard(dashboard_id, user)
    update = {}
    if body.name is not None:
        name = body.name.strip()
        if not name:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Name cannot be empty")
        update["name"] = dashboard.name = name[:120]
    if body.description is not None:
        update["description"] = dashboard.description = body.description.strip()[:300] or None
    if body.chart_ids is not None:
        update["chart_ids"] = dashboard.chart_ids = body.chart_ids
    if body.theme is not None:
        theme = DashboardTheme(**{**dashboard.theme.model_dump(), **body.theme.model_dump(exclude_none=True)})
        update["theme"] = theme.model_dump()
        dashboard.theme = theme
    if body.time_window is not None:
        if dashboard.time_window is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "This dashboard has no time window to change.")
        try:
            validated = validate_window({
                "column": dashboard.time_window.column, "last": body.time_window.last, "unit": body.time_window.unit,
            })
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
        window = TimeWindow(**validated)
        update["time_window"] = window.model_dump()
        dashboard.time_window = window
    if update:
        update["updated_at"] = dashboard.updated_at = utcnow()
        await get_db()[DASHBOARDS].update_one({"_id": dashboard.id}, {"$set": update})
    return _out(dashboard)


@router.delete("/dashboards/{dashboard_id}")
async def delete_dashboard(dashboard_id: str, user: User = Depends(get_current_user)):
    dashboard = await _get_owned_dashboard(dashboard_id, user)
    await get_db()[DASHBOARDS].delete_one({"_id": dashboard.id})
    await get_db()[DASHBOARD_SHARES].delete_many({"dashboard_id": dashboard.id})
    return {"ok": True}


@router.post("/dashboards/{dashboard_id}/refresh")
async def refresh_dashboard(dashboard_id: str, user: User = Depends(get_current_user)):
    request_received_at = now_iso()
    t0 = time.perf_counter()
    logger.info("refresh_dashboard route: request arrived at %s (dashboard_id=%s, user_id=%s)",
                request_received_at, dashboard_id, user.id)

    dashboard = await _get_owned_dashboard(dashboard_id, user)
    if dashboard.is_live_database:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Live database dashboards fetch fresh data every time they're opened.",
        )
    if not dashboard.real_time:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This dashboard isn't real-time - nothing to refresh")

    pool = await get_arq_pool()
    job = await pool.enqueue_job("refresh_dashboard", dashboard_id=dashboard.id, requested_at=request_received_at)
    logger.info(
        "refresh_dashboard route: dashboard %s enqueued as arq job %s at +%.1fms",
        dashboard.id, getattr(job, "job_id", None), (time.perf_counter() - t0) * 1000,
    )
    return {"ok": True}


@router.post("/dashboards/{dashboard_id}/relink", response_model=DashboardOut)
async def relink_dashboard_file(
    dashboard_id: str, body: RelinkFileRequest, user: User = Depends(get_current_user),
):
    request_received_at = now_iso()
    t0 = time.perf_counter()
    logger.info("relink_dashboard_file: request arrived at %s (dashboard_id=%s, user_id=%s)",
                request_received_at, dashboard_id, user.id)

    dashboard = await _get_owned_dashboard(dashboard_id, user)
    if not dashboard.real_time:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This dashboard isn't real-time - nothing to relink")
    if body.old_file_id not in dashboard.file_ids:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "old_file_id is not one of this dashboard's files")

    new_file = await get_owned_file(body.new_file_id, user)
    if new_file.workspace_id != dashboard.workspace_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "new_file_id belongs to a different workspace")
    if new_file.status != "ready":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "new_file_id hasn't finished processing yet")

    updated_file_ids = [body.new_file_id if fid == body.old_file_id else fid for fid in dashboard.file_ids]
    await get_db()[DASHBOARDS].update_one({"_id": dashboard.id}, {"$set": {"file_ids": updated_file_ids}})
    dashboard.file_ids = updated_file_ids
    logger.info("relink_dashboard_file: dashboard %s file_ids updated at +%.1fms",
                dashboard.id, (time.perf_counter() - t0) * 1000)

    pool = await get_arq_pool()
    job = await pool.enqueue_job("refresh_dashboard", dashboard_id=dashboard.id, requested_at=request_received_at)
    logger.info(
        "relink_dashboard_file: dashboard %s enqueued as arq job %s at +%.1fms total",
        dashboard.id, getattr(job, "job_id", None), (time.perf_counter() - t0) * 1000,
    )
    return _out(dashboard)


@router.post("/dashboards/{dashboard_id}/shares", response_model=ShareOut, status_code=status.HTTP_201_CREATED)
async def create_share(dashboard_id: str, body: CreateShareRequest, user: User = Depends(get_current_user)):
    dashboard = await _get_owned_dashboard(dashboard_id, user)
    owner_email = user.email.strip().lower()
    emails: list[str] = []
    for raw in body.emails:
        email = raw.strip().lower()
        if not email:
            continue
        if not EMAIL_PATTERN.match(email):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"'{raw.strip()}' is not a valid email address.")
        if email != owner_email and email not in emails:
            emails.append(email)
    if not emails:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Add at least one email address other than your own.")

    candidates = sorted({*emails, *(raw.strip() for raw in body.emails)})
    docs = await get_db()[USERS].find({"email": {"$in": candidates}}, {"email": 1}).to_list(length=len(candidates) + 5)
    known = {d["email"].strip().lower() for d in docs}
    missing = [e for e in emails if e not in known]
    if missing:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"These people don't have a DataAnalyzer account yet: {', '.join(missing)}",
        )

    now = utcnow()
    share = DashboardShare(
        dashboard_id=dashboard.id, workspace_id=dashboard.workspace_id, shared_by=user.id,
        shared_at=now, valid_upto=now + timedelta(days=body.valid_days), shared_to=emails,
    )
    await get_db()[DASHBOARD_SHARES].insert_one(share.to_mongo())
    logger.info("dashboard %s shared with %d user(s) until %s (share %s)", dashboard.id, len(emails), share.valid_upto, share.id)
    return _share_out(share)


@router.get("/dashboards/{dashboard_id}/shares", response_model=list[ShareOut])
async def list_shares(dashboard_id: str, user: User = Depends(get_current_user)):
    dashboard = await _get_owned_dashboard(dashboard_id, user)
    docs = await get_db()[DASHBOARD_SHARES].find(
        {"dashboard_id": dashboard.id, "revoked_at": None},
    ).sort("shared_at", -1).to_list(length=200)
    return [_share_out(DashboardShare.from_mongo(d)) for d in docs]


@router.delete("/dashboard-shares/{share_id}")
async def revoke_share(share_id: str, user: User = Depends(get_current_user)):
    share = DashboardShare.from_mongo(await get_db()[DASHBOARD_SHARES].find_one({"_id": share_id}))
    if share is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Share not found")
    await _get_owned_dashboard(share.dashboard_id, user)
    await get_db()[DASHBOARD_SHARES].update_one({"_id": share.id}, {"$set": {"revoked_at": utcnow()}})
    return {"ok": True}


@router.get("/shared-dashboards/{share_id}", response_model=SharedDashboardOut)
async def get_shared_dashboard(share_id: str, user: User = Depends(get_current_user)):
    share, dashboard, is_owner = await _get_accessible_share(share_id, user)
    owner = await get_db()[USERS].find_one({"_id": share.shared_by}, {"name": 1, "email": 1}) or {}
    return SharedDashboardOut(
        share_id=share.id, dashboard_id=dashboard.id, name=dashboard.name, description=dashboard.description,
        source=dashboard.source,
        time_window=dashboard.time_window.model_dump() if dashboard.time_window else None,
        time_window_label=window_label(dashboard.time_window),
        theme=dashboard.theme.model_dump(),
        charts=await _chart_refs(dashboard),
        live_charts=[_live_chart_out(c, include_query=False) for c in dashboard.charts] if dashboard.is_live_database else [],
        shared_by=SharedByOut(name=owner.get("name") or "", email=owner.get("email") or ""),
        shared_at=_iso(share.shared_at), valid_upto=_iso(share.valid_upto), is_owner=is_owner,
    )


@router.get("/shared-dashboards/{share_id}/data", response_model=LiveDataOut)
async def get_shared_dashboard_data(share_id: str, user: User = Depends(get_current_user)):
    _, dashboard, _ = await _get_accessible_share(share_id, user)
    return await _live_data(dashboard, user)
