import asyncio
import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api_service.deps import get_current_user, get_owned_chat, get_owned_investigation, get_owned_workspace
from shared import usage
from shared.db import get_db
from shared.models.chart import COLLECTION as CHARTS
from shared.models.chart import Chart
from shared.models.chat import COLLECTION as CHATS
from shared.models.chat import DEFAULT_TITLE as DEFAULT_CHAT_TITLE
from shared.models.chat import Chat
from shared.models.file import COLLECTION as FILES
from shared.models.file import File
from shared.models.investigation import COLLECTION as INVESTIGATIONS
from shared.models.investigation import Investigation
from shared.models.message import COLLECTION as MESSAGES
from shared.models.message import Message
from shared.models.query_shadow_log import COLLECTION as QUERY_SHADOW_LOGS
from shared.models.query_shadow_log import QueryShadowLog
from shared.models.report import COLLECTION as REPORTS
from shared.models.report import Report
from shared.models.user import User
from shared.query_router import classify as classify_query
from shared.intent_router import route_query_intent_fast
from shared.dummy_files import ensure_dummy_files
from shared.chat_title import generate_title
from api_service.light_investigation import schedule_light_response
from shared.admin import is_admin_email
from shared.job_timing import now_iso
from shared.redis_client import get_arq_pool, get_redis, investigation_channel
from shared.storage import delete_object

logger = logging.getLogger("api.chats")
shadow_logger = logging.getLogger("query_router.shadow")

# Shadow-test only: classification never gates the real request, so it runs
# fire-and-forget. Keep a reference to each task so it isn't garbage
# collected mid-flight (a known asyncio footgun for "unawaited" tasks).
_shadow_tasks: set[asyncio.Task] = set()


def _schedule_shadow_classification(
    *, chat_id: str, message_id: str, user_id: str, query: str,
) -> None:
    task = asyncio.create_task(
        _shadow_classify_and_log(chat_id, message_id, user_id, query)
    )
    _shadow_tasks.add(task)
    task.add_done_callback(_shadow_tasks.discard)


async def _shadow_classify_and_log(
    chat_id: str, message_id: str, user_id: str, query: str,
) -> None:
    try:
        # Shadow test only, so this Mongo round-trip - previously done synchronously in
        # send_message before the real work even started - happens here instead, off the
        # request's critical path. Excludes the message we just inserted (by id, not a
        # before/after ordering trick) so "does this chat have earlier turns" still means
        # exactly what it always did, regardless of when this background task actually runs.
        has_prior_context = await get_db()[MESSAGES].count_documents(
            {"chat_id": chat_id, "_id": {"$ne": message_id}}
        ) > 0
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


# Same "hold a reference so it isn't GC'd mid-flight" pattern as _shadow_tasks above and
# light_investigation.py's _tasks - titling runs entirely in this process now (no arq round
# trip), fired once per chat right after its first message lands.
_title_tasks: set[asyncio.Task] = set()


def _schedule_chat_title(*, chat_id: str, query: str) -> None:
    """Fire-and-forget - titles a brand-new chat off the request's critical path. Calls the same
    kind of small, latency-sensitive DeepInfra call as api_service/light_investigation.py's light
    path (shared/chat_title.generate_title is deliberately shaped the same way - see its
    docstring), run in-process via asyncio.to_thread instead of round-tripping through arq: this
    only ever needs to happen once, right after the first message, with no queue/retry semantics
    worth the extra hop - generate_title() itself never raises and always returns a usable title
    (falling back to a deterministic heuristic), so there's nothing here to retry."""
    task = asyncio.create_task(_generate_and_set_chat_title(chat_id, query))
    _title_tasks.add(task)
    task.add_done_callback(_title_tasks.discard)


async def _generate_and_set_chat_title(chat_id: str, query: str) -> None:
    try:
        # generate_title() wraps a blocking `openai` SDK call - offloaded to a thread so it
        # never stalls this process's event loop (same pattern as route_query_intent_fast below).
        result = await asyncio.to_thread(generate_title, query)
        update = await get_db()[CHATS].update_one(
            {"_id": chat_id, "title": DEFAULT_CHAT_TITLE},
            {"$set": {"title": result.title}},
        )
        logger.info(
            "chat title: chat=%s title=%r matched=%s model=%s fallback=%s latency_ms=%.1f error=%s",
            chat_id, result.title, update.matched_count, result.model, result.fallback,
            result.latency_ms, result.error,
        )
    except Exception:
        # Worst case the chat just keeps its default title - never worth surfacing to the user.
        logger.exception("chat title generation/update failed for chat_id=%s", chat_id)


router = APIRouter(tags=["chats"])

# route_query_intent_fast's own confidence gate (shared/intent_router.py's similarity_threshold/
# margin_threshold) already decides embedding vs. none - nothing left to threshold here.
# run_investigation applies a second, independent safety check on top of that (whether the file
# selection is actually unambiguous for tabular/document routes - see its
# _select_direct_route_files) before a route is actually allowed to skip the Orchestrator.
#
# Two cases skip the arq queue (and the Orchestrator) entirely, straight to
# api_service/light_investigation.py's light-model path: the query classifies as "greeting", or
# the workspace has zero ready files (nothing for Tabular/Document tools to act on regardless of
# what the intent tier thinks the query is about - see send_message below for exactly how these
# two are decided, and why "no files" is its own reason rather than being folded into "greeting").

LIMIT_MESSAGE = (
    "You've used all 20 free messages. Upgrade for more, or check back once your plan resets."
)

CHAT_MESSAGE_LIMIT_MESSAGE = (
    "This chat has reached the 8-message free-tier limit. Start a new chat to keep going."
)

CHAT_LIMIT_MESSAGE = (
    "You've reached the free-tier limit of 2 chats. Continue in one of your existing chats, "
    "or upgrade to start new ones."
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
    csv_file_ids: list[str]
    files_used: list[str]
    follow_up_questions: list[str]
    created_at: str


class CreateChatRequest(BaseModel):
    title: str = DEFAULT_CHAT_TITLE


class UpdateChatRequest(BaseModel):
    title: str


class SendMessageRequest(BaseModel):
    content: str
    # File ids the user explicitly referenced via "@" in the client's message
    # composer (see InputBar.tsx) - passed through to the worker job below.
    file_ids: list[str] = []


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
        csv_file_ids=m.csv_file_ids,
        files_used=m.files_used, follow_up_questions=m.follow_up_questions,
        created_at=m.created_at.isoformat(),
    )


@router.post("/workspaces/{workspace_id}/chats", response_model=ChatOut)
async def create_chat(workspace_id: str, body: CreateChatRequest, user: User = Depends(get_current_user)):
    await get_owned_workspace(workspace_id, user)

    # Free-tier checkpoint: at most 2 chats per user (lifetime, tracked on Usage.chats_created -
    # see shared/usage.py). Checked before insert, same "check before the gated action starts"
    # pattern as send_message's message-capacity check below. Bypassed entirely for admin emails
    # (see shared/admin.py).
    if not await usage.has_chat_creation_capacity(user.id, email=user.email):
        raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED, CHAT_LIMIT_MESSAGE)

    chat = Chat(workspace_id=workspace_id, title=body.title)
    await get_db()[CHATS].insert_one(chat.to_mongo())
    if not is_admin_email(user.email):
        await usage.increment_chats(user.id)
    return _chat_out(chat)


@router.get("/workspaces/{workspace_id}/chats", response_model=list[ChatOut])
async def list_chats(workspace_id: str, user: User = Depends(get_current_user)):
    await get_owned_workspace(workspace_id, user)
    cursor = get_db()[CHATS].find({"workspace_id": workspace_id}).sort("created_at", -1)
    docs = await cursor.to_list(length=500)
    return [_chat_out(Chat.from_mongo(d)) for d in docs]


@router.patch("/chats/{chat_id}", response_model=ChatOut)
async def rename_chat(chat_id: str, body: UpdateChatRequest, user: User = Depends(get_current_user)):
    chat = await get_owned_chat(chat_id, user)
    title = body.title.strip() or DEFAULT_CHAT_TITLE
    await get_db()[CHATS].update_one({"_id": chat.id}, {"$set": {"title": title}})
    chat.title = title
    return _chat_out(chat)


@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, user: User = Depends(get_current_user)):
    """Deleting a chat cascades to everything it produced: its messages, the
    investigations behind them, and the charts/reports those investigations
    generated (all keyed by message_id, so they're never shared with another
    chat - safe to hard-delete outright).

    Files are the one exception: they're workspace-level (see shared/models/file.py -
    no chat_id) and can be reused across chats via Chat.files_used/files_created,
    so we only hard-delete the ones this chat touched that no *other* chat in
    the same workspace still references. Otherwise deleting one chat could
    silently pull a file out from under another chat's history.
    """
    chat = await get_owned_chat(chat_id, user)
    db = get_db()

    message_docs = await db[MESSAGES].find({"chat_id": chat_id}, {"_id": 1}).to_list(length=5000)
    message_ids = [d["_id"] for d in message_docs]

    chart_docs = (
        await db[CHARTS].find({"message_id": {"$in": message_ids}}).to_list(length=5000)
        if message_ids else []
    )
    for doc in chart_docs:
        try:
            delete_object(Chart.from_mongo(doc).storage_key)
        except Exception:
            pass
    if chart_docs:
        await db[CHARTS].delete_many({"_id": {"$in": [d["_id"] for d in chart_docs]}})

    report_docs = (
        await db[REPORTS].find({"message_id": {"$in": message_ids}}).to_list(length=5000)
        if message_ids else []
    )
    for doc in report_docs:
        report = Report.from_mongo(doc)
        if report.storage_key:
            try:
                delete_object(report.storage_key)
            except Exception:
                pass
    if report_docs:
        await db[REPORTS].delete_many({"_id": {"$in": [d["_id"] for d in report_docs]}})

    candidate_file_ids = set(chat.files_used) | set(chat.files_created)
    if candidate_file_ids:
        other_chats = await db[CHATS].find(
            {
                "workspace_id": chat.workspace_id,
                "_id": {"$ne": chat.id},
                "$or": [
                    {"files_used": {"$in": list(candidate_file_ids)}},
                    {"files_created": {"$in": list(candidate_file_ids)}},
                ],
            },
            {"files_used": 1, "files_created": 1},
        ).to_list(length=5000)
        still_referenced: set = set()
        for doc in other_chats:
            still_referenced.update(doc.get("files_used", []))
            still_referenced.update(doc.get("files_created", []))
        orphaned_file_ids = candidate_file_ids - still_referenced

        if orphaned_file_ids:
            file_docs = await db[FILES].find({"_id": {"$in": list(orphaned_file_ids)}}).to_list(length=5000)
            for doc in file_docs:
                file = File.from_mongo(doc)
                try:
                    delete_object(file.storage_key)
                except Exception:
                    pass
                if file.output_ref:
                    try:
                        delete_object(file.output_ref)
                    except Exception:
                        pass
            await db[FILES].delete_many({"_id": {"$in": list(orphaned_file_ids)}})

    await db[INVESTIGATIONS].delete_many({"chat_id": chat_id})
    await db[QUERY_SHADOW_LOGS].delete_many({"chat_id": chat_id})
    await db[MESSAGES].delete_many({"chat_id": chat_id})
    await db[CHATS].delete_one({"_id": chat.id})

    return {"ok": True}


@router.get("/chats/{chat_id}/messages", response_model=list[MessageOut])
async def list_messages(chat_id: str, user: User = Depends(get_current_user)):
    await get_owned_chat(chat_id, user)
    cursor = get_db()[MESSAGES].find({"chat_id": chat_id}).sort("created_at", 1)
    docs = await cursor.to_list(length=2000)
    return [_message_out(Message.from_mongo(d)) for d in docs]


@router.get("/chats/{chat_id}/active-investigation")
async def active_investigation(chat_id: str, user: User = Depends(get_current_user)):

    await get_owned_chat(chat_id, user)
    doc = await get_db()[INVESTIGATIONS].find_one({"chat_id": chat_id, "status": "running"})
    if doc is None:
        return {"investigation_id": None}
    investigation = Investigation.from_mongo(doc)
    return {"investigation_id": investigation.id}


@router.post("/chats/{chat_id}/messages", response_model=SendMessageResponse)
async def send_message(chat_id: str, body: SendMessageRequest, user: User = Depends(get_current_user)):
   
    request_received_at = now_iso()
    t0 = time.perf_counter()
    logger.info(
        "send_message: request arrived at %s (chat_id=%s, user_id=%s)",
        request_received_at, chat_id, user.id,
    )

    chat = await get_owned_chat(chat_id, user)

    chat_has_capacity, user_has_capacity = await asyncio.gather(
        usage.has_chat_message_capacity(chat_id, email=user.email),
        usage.has_message_capacity(user.id, email=user.email),
    )

    if not chat_has_capacity:
        message = Message(chat_id=chat_id, role="user", content=body.content)
        await get_db()[MESSAGES].insert_one(message.to_mongo())
        logger.info(
            "send_message: rejected (per-chat message capacity reached) at +%.1fms (chat_id=%s)",
            (time.perf_counter() - t0) * 1000, chat_id,
        )
        return SendMessageResponse(
            message_id=message.id, investigation_id=None, limited=True,
            limit_message=CHAT_MESSAGE_LIMIT_MESSAGE,
        )

    if not user_has_capacity:
        message = Message(chat_id=chat_id, role="user", content=body.content)
        await get_db()[MESSAGES].insert_one(message.to_mongo())
        logger.info(
            "send_message: rejected (message capacity reached) at +%.1fms (chat_id=%s)",
            (time.perf_counter() - t0) * 1000, chat_id,
        )
        return SendMessageResponse(
            message_id=message.id, investigation_id=None, limited=True, limit_message=LIMIT_MESSAGE,
        )

    message = Message(chat_id=chat_id, role="user", content=body.content)
    investigation = Investigation(
        chat_id=chat_id, workspace_id=chat.workspace_id, objective=body.content,
        user_id=user.id, file_ids=body.file_ids or [], email=user.email,
    )
    db = get_db()
    persist_task = asyncio.gather(
    db[MESSAGES].insert_one(message.to_mongo()),
    db[INVESTIGATIONS].insert_one(investigation.to_mongo()),
)

    
    # "Is this the chat's first message" is decided the same way _shadow_classify_and_log
    # decides "does this chat have prior context": count everything under chat_id EXCLUDING the
    # message we're inserting right now, by its own already-known id. Excluding by id (rather
    # than counting before the insert) is what lets this run inside the same gather as
    # persist_task's insert below without a race - the count is correct regardless of whether
    # that insert has landed yet by the time this query runs.
    # Defensive/self-healing only - the primary, race-free seeding happens at workspace creation
    # (workspaces.py's create_workspace) so sample files normally already exist and are "ready"
    # well before a user's first message. This call is a cheap no-op in that case; it only does
    # real work for a workspace that predates this feature or somehow ended up with zero files.
    route_result, files_doc, prior_message_count, _ = await asyncio.gather(
        asyncio.to_thread(route_query_intent_fast, body.content),
        db[FILES].find_one({"workspace_id": chat.workspace_id, "status": "ready"}, {"_id": 1}),
        db[MESSAGES].count_documents({"chat_id": chat_id, "_id": {"$ne": message.id}}),
        ensure_dummy_files(db, chat.workspace_id),
    )
    has_files = files_doc is not None
    is_first_message = prior_message_count == 0 and chat.title == DEFAULT_CHAT_TITLE
    route = route_result.intent if route_result.intent in ("tabular", "document", "orchestrator") else None

    if route_result.intent == "greeting":
        light_route = "greeting"
    elif not has_files:
        light_route = "no_files"
    else:
        light_route = None

    logger.info(
        "send_message: intent_router method=%s top_intent=%s top_similarity=%.3f margin=%.3f "
        "route=%s light_route=%s has_files=%s router_latency_ms=%.1f at +%.1fms total "
        "(chat_id=%s, error=%s)",
        route_result.method, route_result.top_intent, route_result.top_similarity,
        route_result.margin, route, light_route, has_files, route_result.latency_ms,
        (time.perf_counter() - t0) * 1000, chat_id, route_result.error,
    )

    await persist_task
    logger.info(
        "send_message: user message + investigation %s persisted at +%.1fms (chat_id=%s, message_id=%s)",
        investigation.id, (time.perf_counter() - t0) * 1000, chat_id, message.id,
    )

    if is_first_message:
        _schedule_chat_title(chat_id=chat_id, query=body.content)
        logger.info("send_message: scheduled chat title generation for chat_id=%s (first message)", chat_id)

    if light_route is not None:
        schedule_light_response(
            investigation_id=investigation.id, chat_id=chat_id, workspace_id=chat.workspace_id,
            user_id=user.id, query=body.content, route=light_route,
            requested_at=request_received_at, file_ids=body.file_ids, email=user.email,
        )
        logger.info(
            "send_message: investigation %s dispatched to light-response path (%s) at +%.1fms "
            "total (chat_id=%s)",
            investigation.id, light_route, (time.perf_counter() - t0) * 1000, chat_id,
        )
    else:
        pool = await get_arq_pool()
        job = await pool.enqueue_job(
            "run_investigation",
            investigation_id=investigation.id,
            chat_id=chat_id,
            workspace_id=chat.workspace_id,
            user_id=user.id,
            email=user.email,
            query=body.content,
            file_ids=body.file_ids,
            requested_at=request_received_at,
            route=route,
        )
        logger.info(
            "send_message: investigation %s enqueued as arq job %s at +%.1fms total (chat_id=%s)",
            investigation.id, getattr(job, "job_id", None), (time.perf_counter() - t0) * 1000, chat_id,
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
