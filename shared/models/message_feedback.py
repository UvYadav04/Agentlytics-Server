from datetime import datetime
from typing import Literal, Optional

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "message_feedback"

FeedbackRating = Literal["like", "dislike"]

POSITIVE_REASONS = ("accurate", "helpful", "clear", "good_visuals")

NEGATIVE_REASONS = (
    "incorrect",
    "incomplete",
    "irrelevant",
    "wrong_data",
    "hard_to_understand",
    "too_long",
    "too_short",
    "chart_issue",
    "slow",
    "other",
)

FEEDBACK_REASONS = frozenset(POSITIVE_REASONS + NEGATIVE_REASONS)

MAX_COMMENT_LENGTH = 2000


class MessageFeedback(MongoModel):
    message_id: str
    chat_id: str
    workspace_id: str
    user_id: str
    investigation_id: Optional[str] = None
    reply_to: Optional[str] = None
    message_version: int = 1
    retry_mode: Optional[str] = None
    rating: Optional[FeedbackRating] = None
    reasons: list[str] = Field(default_factory=list)
    comment: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
