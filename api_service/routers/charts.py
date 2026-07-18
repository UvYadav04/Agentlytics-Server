from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api_service.deps import get_current_user, get_owned_workspace
from shared.db import get_db
from shared.models.chart import COLLECTION as CHARTS
from shared.models.chart import Chart
from shared.models.user import User
from shared.storage import presign_get

router = APIRouter(prefix="/charts", tags=["charts"])


class ChartOut(BaseModel):
    id: str
    workspace_id: str
    message_id: str
    title: str
    url: str
    created_at: str


async def _get_owned_chart(chart_id: str, user: User) -> Chart:
    doc = await get_db()[CHARTS].find_one({"_id": chart_id})
    chart = Chart.from_mongo(doc)
    if chart is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chart not found")
    await get_owned_workspace(chart.workspace_id, user)
    return chart


@router.get("/{chart_id}", response_model=ChartOut)
async def get_chart(chart_id: str, user: User = Depends(get_current_user)):
    chart = await _get_owned_chart(chart_id, user)
    return ChartOut(
        id=chart.id, workspace_id=chart.workspace_id, message_id=chart.message_id,
        title=chart.title, url=presign_get(chart.storage_key), created_at=chart.created_at.isoformat(),
    )
