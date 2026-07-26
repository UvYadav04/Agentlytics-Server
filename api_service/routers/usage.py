from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api_service.deps import get_current_user
from shared import usage
from shared.models.user import User

router = APIRouter(tags=["usage"])


class UsageOut(BaseModel):
    messages_sent: int
    messages_limit: int
    messages_per_chat_limit: int
    charts_created: int
    charts_limit: int
    reports_created: int
    reports_limit: int
    chats_created: int
    chats_limit: int
    workspaces_created: int
    workspaces_limit: int


@router.get("/usage", response_model=UsageOut)
async def get_usage(user: User = Depends(get_current_user)):
    u = await usage.get_or_create_usage(user.id)
    return UsageOut(
        messages_sent=u.messages_sent, messages_limit=usage.messages_limit(),
        messages_per_chat_limit=usage.messages_per_chat_limit(),
        charts_created=u.charts_created, charts_limit=usage.charts_limit(),
        reports_created=u.reports_created, reports_limit=usage.reports_limit(),
        chats_created=u.chats_created, chats_limit=usage.chats_limit(),
        workspaces_created=u.workspaces_created, workspaces_limit=usage.workspaces_limit(),
    )
