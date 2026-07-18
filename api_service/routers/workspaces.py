from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api_service.deps import get_current_user, get_owned_workspace
from shared.db import get_db
from shared.models.chart import COLLECTION as CHARTS
from shared.models.chart import Chart
from shared.models.user import User
from shared.models.workspace import COLLECTION as WORKSPACES
from shared.models.workspace import Workspace
from shared.storage import presign_get

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


class ChartSummaryOut(BaseModel):
    id: str
    message_id: str
    title: str
    url: str
    created_at: str


class WorkspaceOut(BaseModel):
    id: str
    name: str
    created_at: str


class CreateWorkspaceRequest(BaseModel):
    name: str


class RenameWorkspaceRequest(BaseModel):
    name: str


def _out(w: Workspace) -> WorkspaceOut:
    return WorkspaceOut(id=w.id, name=w.name, created_at=w.created_at.isoformat())


@router.get("", response_model=list[WorkspaceOut])
async def list_workspaces(user: User = Depends(get_current_user)):
    cursor = get_db()[WORKSPACES].find({"user_id": user.id}).sort("created_at", 1)
    docs = await cursor.to_list(length=200)
    return [_out(Workspace.from_mongo(d)) for d in docs]


@router.post("", response_model=WorkspaceOut)
async def create_workspace(body: CreateWorkspaceRequest, user: User = Depends(get_current_user)):
    workspace = Workspace(user_id=user.id, name=body.name)
    await get_db()[WORKSPACES].insert_one(workspace.to_mongo())
    return _out(workspace)


@router.patch("/{workspace_id}", response_model=WorkspaceOut)
async def rename_workspace(
    workspace_id: str, body: RenameWorkspaceRequest, user: User = Depends(get_current_user)
):
    workspace = await get_owned_workspace(workspace_id, user)
    await get_db()[WORKSPACES].update_one({"_id": workspace.id}, {"$set": {"name": body.name}})
    workspace.name = body.name
    return _out(workspace)


@router.get("/{workspace_id}/charts", response_model=list[ChartSummaryOut])
async def list_charts(workspace_id: str, user: User = Depends(get_current_user)):
    """Right-panel chart/dashboard gallery (build plan Phase 9)."""
    await get_owned_workspace(workspace_id, user)
    cursor = get_db()[CHARTS].find({"workspace_id": workspace_id}).sort("created_at", -1)
    docs = await cursor.to_list(length=200)
    charts = [Chart.from_mongo(d) for d in docs]
    return [
        ChartSummaryOut(
            id=c.id, message_id=c.message_id, title=c.title,
            url=presign_get(c.storage_key), created_at=c.created_at.isoformat(),
        )
        for c in charts
    ]
