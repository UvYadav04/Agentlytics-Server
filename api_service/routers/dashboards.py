from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api_service.deps import get_current_user, get_owned_workspace
from shared.db import get_db
from shared.models.chart import COLLECTION as CHARTS
from shared.models.dashboard import COLLECTION as DASHBOARDS
from shared.models.dashboard import Dashboard
from shared.models.user import User
from shared.storage import presign_get

router = APIRouter(tags=["dashboards"])


class ChartRef(BaseModel):
    id: str
    title: str
    url: str


class DashboardOut(BaseModel):
    id: str
    workspace_id: str
    name: str
    chart_ids: list[str]
    created_at: str


class DashboardDetailOut(DashboardOut):
    charts: list[ChartRef]


class CreateDashboardRequest(BaseModel):
    name: str
    chart_ids: list[str] = []


class UpdateDashboardRequest(BaseModel):
    name: str | None = None
    chart_ids: list[str] | None = None


async def _get_owned_dashboard(dashboard_id: str, user: User) -> Dashboard:
    doc = await get_db()[DASHBOARDS].find_one({"_id": dashboard_id})
    dashboard = Dashboard.from_mongo(doc)
    if dashboard is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dashboard not found")
    await get_owned_workspace(dashboard.workspace_id, user)
    return dashboard


def _out(d: Dashboard) -> DashboardOut:
    return DashboardOut(id=d.id, workspace_id=d.workspace_id, name=d.name, chart_ids=d.chart_ids,created_at=d.created_at.isoformat())


@router.post("/workspaces/{workspace_id}/dashboards", response_model=DashboardOut)
async def create_dashboard(workspace_id: str, body: CreateDashboardRequest, user: User = Depends(get_current_user)):
    await get_owned_workspace(workspace_id, user)
    dashboard = Dashboard(workspace_id=workspace_id, name=body.name, chart_ids=body.chart_ids)
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
    charts = []
    if dashboard.chart_ids:
        docs = await get_db()[CHARTS].find({"_id": {"$in": dashboard.chart_ids}}).to_list(length=500)
        by_id = {d["_id"]: d for d in docs}
        for chart_id in dashboard.chart_ids:
            d = by_id.get(chart_id)
            if d is None:
                continue
            charts.append(ChartRef(id=d["_id"], title=d["title"], url=presign_get(d["storage_key"])))
    out = _out(dashboard).model_dump()
    return DashboardDetailOut(**out, charts=charts)


@router.patch("/dashboards/{dashboard_id}", response_model=DashboardOut)
async def update_dashboard(dashboard_id: str, body: UpdateDashboardRequest, user: User = Depends(get_current_user)):
    dashboard = await _get_owned_dashboard(dashboard_id, user)
    update = {}
    if body.name is not None:
        update["name"] = body.name
        dashboard.name = body.name
    if body.chart_ids is not None:
        update["chart_ids"] = body.chart_ids
        dashboard.chart_ids = body.chart_ids
    if update:
        await get_db()[DASHBOARDS].update_one({"_id": dashboard.id}, {"$set": update})
    return _out(dashboard)


@router.delete("/dashboards/{dashboard_id}")
async def delete_dashboard(dashboard_id: str, user: User = Depends(get_current_user)):
    dashboard = await _get_owned_dashboard(dashboard_id, user)
    await get_db()[DASHBOARDS].delete_one({"_id": dashboard.id})
    return {"ok": True}
