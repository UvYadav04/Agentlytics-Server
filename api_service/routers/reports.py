from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api_service.deps import get_current_user, get_owned_workspace
from shared.db import get_db
from shared.models.report import COLLECTION as REPORTS
from shared.models.report import Report
from shared.models.user import User
from shared.storage import presign_get

router = APIRouter(prefix="/reports", tags=["reports"])


class ReportOut(BaseModel):
    id: str
    workspace_id: str
    message_id: str
    title: str
    status: str
    format: str
    url: str | None
    error: str | None
    created_at: str


async def _get_owned_report(report_id: str, user: User) -> Report:
    doc = await get_db()[REPORTS].find_one({"_id": report_id})
    report = Report.from_mongo(doc)
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found")
    await get_owned_workspace(report.workspace_id, user)
    return report


@router.get("/{report_id}", response_model=ReportOut)
async def get_report(report_id: str, user: User = Depends(get_current_user)):
    report = await _get_owned_report(report_id, user)
    url = presign_get(report.storage_key) if report.storage_key else None
    return ReportOut(
        id=report.id, workspace_id=report.workspace_id, message_id=report.message_id,
        title=report.title, status=report.status, format=report.format, url=url, error=report.error,
        created_at=report.created_at.isoformat(),
    )
