from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api_service.deps import get_current_user
from shared.db import get_db
from shared.models.feedback import COLLECTION as FEEDBACK
from shared.models.feedback import Feedback
from shared.models.user import User

router = APIRouter(tags=["feedback"])


class FeedbackRequest(BaseModel):
    message: str


@router.post("/feedback")
async def submit_feedback(body: FeedbackRequest, user: User = Depends(get_current_user)):
    feedback = Feedback(user_id=user.id, message=body.message)
    await get_db()[FEEDBACK].insert_one(feedback.to_mongo())
    return {"ok": True}
