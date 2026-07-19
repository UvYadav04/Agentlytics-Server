"""Shadow log of query_router classification decisions.

Written once per incoming chat message (fire-and-forget, see
api_service/routers/chats.py::_schedule_shadow_classification). Nothing
reads this at request time - it exists purely so we can later compute
precision/recall on the fast-path classifier against real traffic before
ever turning short-circuiting on.
"""
from datetime import datetime
from typing import Optional

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "query_shadow_logs"


class QueryShadowLog(MongoModel):
    chat_id: str
    message_id: str
    user_id: str
    query: str
    normalized: str
    tier: str
    intent: Optional[str] = None
    score: float = 0.0
    has_prior_context: bool
    context_gated: bool
    would_shortcircuit: bool
    latency_ms: float
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
