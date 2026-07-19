import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api_service.deps import get_current_user, get_owned_chat, get_owned_investigation, get_owned_workspace
from shared import usage
from shared.db import get_db
from shared.models.chat import COLLECTION as CHATS
from shared.models.chat import Chat
from shared.models.investigation import COLLECTION as INVESTIGATIONS
from shared.models.investigation import Investigation
from shared.models.message import COLLECTION as MESSAGES
from shared.models.message import Message
from shared.models.query_shadow_log import COLLECTION as QUERY_SHADOW_LOGS
from shared.models.query_shadow_log import QueryShadowLog
from shared.models.user import User
from shared.query_router import classify as classify_query
from shared.redis_client import get_arq_pool, get_redis, investigation_channel

logger = logging.getLogger("api.chats")
shadow_logger = logging.getLogger("query_router.shadow")

# Shadow-test only: classification never gates the real request, so it runs
# fire-and-forget. Keep a reference to each task so it isn't garbage
# collected mid-flight (a known asyncio footgun for "unawaited" tasks).
_shadow_tasks: set[asyncio.Task] = set()


def _schedule_shadow_classification(
    *, chat_id: str, message_id: str, user_id: str, query: str, has_prior_context: bool,
) -> None:
    task = asyncio.create_task(
        _shadow_classify_and_log(chat_id, message_id, user_id, query, has_prior_context)
    )
    _shadow_tasks.add(task)
    task.add_done_callback(_shadow_tasks.discard)


async def _shadow_classify_and_log(
    chat_id: str, message_id: str, user_id: str, query: str, has_prior_context: bool,
) -> None:
    try:
        result = classify_query(query, has_prior_context)
        shadow_logger.info(
            "chat=%s message=%s tier=%s intent=%s score=%.3f prior_context=%s "
            "context_gated=%s would_shortcircuit=%s latency_ms=%.2f",
            chat_id, message_id, result.tier, result.intent, result.score,
            has_prior_context, result.context_gated, result.would_shortcircuit, result.latency_ms,
        )
        log = QueryShadowLog(
            chat_id=chat_id,
            message_id=message_id,
            user_id=user_id,
            query=query,
            normalized=result.normalized,
            tier=result.tier,
            intent=result.intent,
            score=result.score,
            has_prior_context=has_prior_context,
            context_gated=result.context_gated,
            would_shortcircuit=result.would_shortcircuit,
            latency_ms=result.latency_ms,
            error=result.error,
        )
        await get_db()[QUERY_SHADOW_LOGS].insert_one(log.to_mongo())
    except Exception:
        # Shadow logging must never affect the real request path.
        shadow_logger.exception("shadow classification failed for message %s", message_id)

router = APIRouter(tags=["chats"])

LIMIT_MESSAGE = (
    "You've used all 20 free messages. Upgrade for more, or check back once your plan resets."
)


class ChatOut(BaseModel):
    id: str
    workspace_id: str
    title: str
    created_at: str


class MessageOut(BaseModel):
    id: str
    chat_id: str
    role: str
    content: str
    investigation_id: str | None
    chart_ids: list[str]
    report_id: str | None
    created_at: str


class CreateChatRequest(BaseModel):
    title: str = "New chat"


class SendMessageRequest(BaseModel):
    content: str


class SendMessageResponse(BaseModel):
    message_id: str
    investigation_id: str | None
    limited: bool = False
    limit_message: str | None = None


def _chat_out(c: Chat) -> ChatOut:
    return ChatOut(id=c.id, workspace_id=c.workspace_id, title=c.title, created_at=c.created_at.isoformat())


def _message_out(m: Message) -> MessageOut:
    return MessageOut(
        id=m.id, chat_id=m.chat_id, role=m.role, content=m.content,
        investigation_id=m.investigation_id, chart_ids=m.chart_ids, report_id=m.report_id,
        created_at=m.created_at.isoformat(),
    )


@router.post("/workspaces/{workspace_id}/chats", response_model=ChatOut)
async def create_chat(workspace_id: str, body: CreateChatRequest, user: User = Depends(get_current_user)):
    await get_owned_workspace(workspace_id, user)
    chat = Chat(workspace_id=workspace_id, title=body.title)
    await get_db()[CHATS].insert_one(chat.to_mongo())
    return _chat_out(chat)


@router.get("/workspaces/{workspace_id}/chats", response_model=list[ChatOut])
async def list_chats(workspace_id: str, user: User = Depends(get_current_user)):
    await get_owned_workspace(workspace_id, user)
    cursor = get_db()[CHATS].find({"workspace_id": workspace_id}).sort("created_at", -1)
    docs = await cursor.to_list(length=500)
    return [_chat_out(Chat.from_mongo(d)) for d in docs]


@router.get("/chats/{chat_id}/messages", response_model=list[MessageOut])
async def list_messages(chat_id: str, user: User = Depends(get_current_user)):
    await get_owned_chat(chat_id, user)
    cursor = get_db()[MESSAGES].find({"chat_id": chat_id}).sort("created_at", 1)
    docs = await cursor.to_list(length=2000)
    return [_message_out(Message.from_mongo(d)) for d in docs]


@router.get("/chats/{chat_id}/active-investigation")
async def active_investigation(chat_id: str, user: User = Depends(get_current_user)):
    """So the frontend can, on chat load, auto-reconnect to a still-running
    investigation instead of showing an idle input (see build plan Phase 5,
    'On chat load')."""
    await get_owned_chat(chat_id, user)
    doc = await get_db()[INVESTIGATIONS].find_one({"chat_id": chat_id, "status": "running"})
    if doc is None:
        return {"investigation_id": None}
    investigation = Investigation.from_mongo(doc)
    return {"investigation_id": investigation.id}


@router.post("/chats/{chat_id}/messages", response_model=SendMessageResponse)
async def send_message(chat_id: str, body: SendMessageRequest, user: User = Depends(get_current_user)):
    chat = await get_owned_chat(chat_id, user)
    # Computed before either message insert below, so it reflects prior
    # turns only - not the message we're about to write.
    has_prior_context = await get_db()[MESSAGES].count_documents({"chat_id": chat_id}) > 0

    if not await usage.has_message_capacity(user.id):
        message = Message(chat_id=chat_id, role="user", content=body.content)
        await get_db()[MESSAGES].insert_one(message.to_mongo())
        return SendMessageResponse(
            message_id=message.id, investigation_id=None, limited=True, limit_message=LIMIT_MESSAGE,
        )

    message = Message(chat_id=chat_id, role="user", content=body.content)
    await get_db()[MESSAGES].insert_one(message.to_mongo())
    # Shadow test only - classifies the query and logs the decision, but
    # never affects routing: the arq enqueue below always runs regardless.
    _schedule_shadow_classification(
        chat_id=chat_id,
        message_id=message.id,
        user_id=user.id,
        query=body.content,
        has_prior_context=has_prior_context,
    )

    investigation = Investigation(chat_id=chat_id, workspace_id=chat.workspace_id, objective=body.content)
    await get_db()[INVESTIGATIONS].insert_one(investigation.to_mongo())

    pool = await get_arq_pool()
    await pool.enqueue_job(
        "run_investigation",
        investigation_id=investigation.id,
        chat_id=chat_id,
        workspace_id=chat.workspace_id,
        user_id=user.id,
        query=body.content,
    )

    return SendMessageResponse(message_id=message.id, investigation_id=investigation.id)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _investigation_stream(investigation_id: str):
    db = get_db()

    doc = await db[INVESTIGATIONS].find_one({"_id": investigation_id})
    if doc is None:
        yield _sse({"type": "error", "message": "Investigation not found"})
        return

    investigation = Investigation.from_mongo(doc)
    for event in investigation.events:
        yield _sse(event.model_dump(mode="json"))

    if investigation.status != "running":
        return

    redis = get_redis()
    pubsub = redis.pubsub()
    await pubsub.subscribe(investigation_channel(investigation_id))
    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=20)
            if message is None:
                yield ": keep-alive\n\n"
                continue
            data = message["data"]
            yield f"data: {data}\n\n"
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                continue
            if parsed.get("type") in ("completed", "cancelled", "error"):
                break
    finally:
        try:
            await pubsub.unsubscribe(investigation_channel(investigation_id))
            await pubsub.aclose()
        except Exception:
            logger.exception("error closing pubsub for investigation %s", investigation_id)


@router.get("/investigations/{investigation_id}/stream")
async def stream_investigation(investigation_id: str, user: User = Depends(get_current_user)):
    await get_owned_investigation(investigation_id, user)
    return StreamingResponse(
        _investigation_stream(investigation_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.post("/investigations/{investigation_id}/cancel")
async def cancel_investigation(investigation_id: str, user: User = Depends(get_current_user)):
    investigation = await get_owned_investigation(investigation_id, user)
    if investigation.status != "running":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Investigation is not running")
    await get_db()[INVESTIGATIONS].update_one({"_id": investigation.id}, {"$set": {"cancel_requested": True}})
    return {"ok": True}
