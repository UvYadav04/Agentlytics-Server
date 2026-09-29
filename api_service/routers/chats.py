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
from shared.models.database_connection import COLLECTION as DATABASE_CONNECTIONS
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
from shared.chat_summary import generate_chat_summary
from api_service.light_investigation import schedule_light_response
from shared.admin import is_admin_email
from shared.job_timing import now_iso
from shared.models.base import utcnow
from shared.redis_client import get_arq_pool, get_redis, investigation_answer_key, investigation_channel
from shared.storage import delete_object, presign_get

logger = logging.getLogger("api.chats")
shadow_logger = logging.getLogger("query_router.shadow")

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
        shadow_logger.exception("shadow classification failed for message %s", message_id)


_title_tasks: set[asyncio.Task] = set()


def _schedule_chat_title(*, chat_id: str, query: str) -> None:
    task = asyncio.create_task(_generate_and_set_chat_title(chat_id, query))
    _title_tasks.add(task)
    task.add_done_callback(_title_tasks.discard)


async def _generate_and_set_chat_title(chat_id: str, query: str) -> None:
    try:
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
        logger.exception("chat title generation/update failed for chat_id=%s", chat_id)


router = APIRouter(tags=["chats"])


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
    dashboard_ids: list[str]
    files_used: list[str]
    follow_up_questions: list[str]
    created_at: str


class CreateChatRequest(BaseModel):
    title: str = DEFAULT_CHAT_TITLE


class UpdateChatRequest(BaseModel):
    title: str


class SendMessageRequest(BaseModel):
    content: str
    file_ids: list[str] = []
    answer_to: str | None = None


class SendMessageResponse(BaseModel):
    message_id: str | None
    investigation_id: str | None
    limited: bool = False
    limit_message: str | None = None
    answered_question: bool = False


ANSWER_KEY_TTL_SECONDS = 900


async def _deliver_pending_answer(chat_id: str, body: SendMessageRequest) -> str | None:
    answer = body.content.strip()
    if not answer:
        return None
    query = {
        "chat_id": chat_id,
        "status": "running",
        "stage": "awaiting_input",
        "pending_question.expires_at": {"$gt": utcnow()},
    }
    if body.answer_to:
        query["pending_question.question_id"] = body.answer_to
    doc = await get_db()[INVESTIGATIONS].find_one_and_update(
        query, {"$set": {"stage": "answer_received"}}, projection={"_id": 1, "pending_question": 1},
    )
    if doc is None:
        return None
    investigation_id = doc["_id"]
    question_id = (doc.get("pending_question") or {}).get("question_id")
    redis = get_redis()
    key = investigation_answer_key(investigation_id)
    await redis.rpush(key, json.dumps({"question_id": question_id, "answer": answer}))
    await redis.expire(key, ANSWER_KEY_TTL_SECONDS)
    return investigation_id


def _chat_out(c: Chat) -> ChatOut:
    return ChatOut(id=c.id, workspace_id=c.workspace_id, title=c.title, created_at=c.created_at.isoformat())


def _message_out(m: Message) -> MessageOut:
    return MessageOut(
        id=m.id, chat_id=m.chat_id, role=m.role, content=m.content,
        investigation_id=m.investigation_id, chart_ids=m.chart_ids, report_id=m.report_id,
        csv_file_ids=m.csv_file_ids, dashboard_ids=m.dashboard_ids,
        files_used=m.files_used, follow_up_questions=m.follow_up_questions,
        created_at=m.created_at.isoformat(),
    )


@router.post("/workspaces/{workspace_id}/chats", response_model=ChatOut)
async def create_chat(workspace_id: str, body: CreateChatRequest, user: User = Depends(get_current_user)):
    await get_owned_workspace(workspace_id, user)

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


class ChatExportChartOut(BaseModel):
    id: str
    title: str
    url: str


class ChatExportCSVOut(BaseModel):
    id: str
    title: str
    url: str


class ChatExportSummaryOut(BaseModel):
    chat_id: str
    chat_title: str
    turns_count: int
    summary: str
    charts: list[ChatExportChartOut]
    csv_files: list[ChatExportCSVOut]


@router.get("/chats/{chat_id}/export/summary", response_model=ChatExportSummaryOut)
async def export_chat_summary(chat_id: str, user: User = Depends(get_current_user)):
    chat = await get_owned_chat(chat_id, user)
    db = get_db()

    cursor = db[MESSAGES].find({"chat_id": chat_id}).sort("created_at", 1)
    docs = await cursor.to_list(length=2000)

    turns: list[tuple[str, str]] = []
    chart_ids: list[str] = []
    csv_file_ids: list[str] = []
    last_user_content = ""
    for doc in docs:
        message = Message.from_mongo(doc)
        if message.role == "user":
            last_user_content = message.content
            continue
        if message.content and message.content.strip():
            turns.append((last_user_content, message.content))
        for chart_id in message.chart_ids:
            if chart_id not in chart_ids:
                chart_ids.append(chart_id)
        for csv_id in message.csv_file_ids:
            if csv_id not in csv_file_ids:
                csv_file_ids.append(csv_id)

    if not turns:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This chat has no answers yet - nothing to export.")

    summary = generate_chat_summary(turns)

    charts_out: list[ChatExportChartOut] = []
    if chart_ids:
        chart_docs = await db[CHARTS].find({"_id": {"$in": chart_ids}}).to_list(length=len(chart_ids))
        charts_by_id = {d["_id"]: Chart.from_mongo(d) for d in chart_docs}
        for chart_id in chart_ids:
            chart = charts_by_id.get(chart_id)
            if chart is None:
                continue
            charts_out.append(
                ChatExportChartOut(id=chart.id, title=chart.title, url=presign_get(chart.storage_key))
            )

    csv_out: list[ChatExportCSVOut] = []
    if csv_file_ids:
        report_docs = await db[REPORTS].find({"_id": {"$in": csv_file_ids}}).to_list(length=len(csv_file_ids))
        reports_by_id = {d["_id"]: Report.from_mongo(d) for d in report_docs}
        for csv_id in csv_file_ids:
            report = reports_by_id.get(csv_id)
            if report is None or not report.storage_key:
                continue
            csv_out.append(ChatExportCSVOut(id=report.id, title=report.title, url=presign_get(report.storage_key)))

    logger.info(
        "export_chat_summary: chat=%s turns=%d charts=%d csv_files=%d",
        chat_id, len(turns), len(charts_out), len(csv_out),
    )

    return ChatExportSummaryOut(
        chat_id=chat.id, chat_title=chat.title, turns_count=len(turns),
        summary=summary, charts=charts_out, csv_files=csv_out,
    )


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

    answered_investigation_id = await _deliver_pending_answer(chat_id, body)
    if answered_investigation_id is not None:
        logger.info(
            "send_message: delivered answer to investigation %s at +%.1fms (chat_id=%s)",
            answered_investigation_id, (time.perf_counter() - t0) * 1000, chat_id,
        )
        return SendMessageResponse(
            message_id=None, investigation_id=answered_investigation_id, answered_question=True,
        )
    if body.answer_to:
        raise HTTPException(status.HTTP_409_CONFLICT, "This question is no longer waiting for an answer.")

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

    
    route_result, files_doc, connection_doc, prior_message_count, _ = await asyncio.gather(
        asyncio.to_thread(route_query_intent_fast, body.content),
        db[FILES].find_one({"workspace_id": chat.workspace_id, "status": "ready"}, {"_id": 1}),
        db[DATABASE_CONNECTIONS].find_one(
            {"workspace_id": chat.workspace_id, "encrypted_connection_string": {"$ne": None}}, {"_id": 1},
        ),
        db[MESSAGES].count_documents({"chat_id": chat_id, "_id": {"$ne": message.id}}),
        ensure_dummy_files(db, chat.workspace_id),
    )
    has_files = files_doc is not None or connection_doc is not None
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
