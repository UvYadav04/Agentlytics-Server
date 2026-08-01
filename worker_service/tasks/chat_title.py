# NOTE: no longer registered as an arq job (see worker_service/worker.py) - chat titling now runs
# in-process, fire-and-forget, from api_service/routers/chats.py's send_message
# (_schedule_chat_title/_generate_and_set_chat_title) right when the chat's first message lands,
# instead of round-tripping through arq. Left here unregistered rather than deleted; nothing
# imports this module anymore.
import asyncio
import logging

from arq import Retry

from shared.chat_title import generate_title
from shared.db import get_db
from shared.job_timing import log_job_finished, log_job_picked_up
from shared.models.chat import COLLECTION as CHATS
from shared.models.chat import DEFAULT_TITLE

logger = logging.getLogger("worker.chat_title")


async def generate_chat_title(ctx, chat_id: str, query: str, requested_at: str | None = None) -> None:
    """Fire-and-forget companion to send_message: titles a brand-new chat from its first
    message, off the request's critical path - see api_service/routers/chats.py's send_message,
    which enqueues this job only when the message that just landed was the chat's first, in
    parallel with the same asyncio.gather() that already runs intent routing/file lookups there,
    so detecting "first message" costs nothing extra on that path.

    Idempotent and safe to retry: only ever writes a title if the chat is still sitting at
    Chat.DEFAULT_TITLE, both in the pre-check and again as part of the update filter itself, so
    it can never clobber a user's own rename (or a previous run of this same job).
    """
    picked_up_at = log_job_picked_up(
        logger, ctx, "generate_chat_title", requested_at=requested_at, chat_id=chat_id,
    )
    db = get_db()

    try:
        doc = await db[CHATS].find_one({"_id": chat_id}, {"title": 1})
        if doc is None:
            logger.warning("generate_chat_title: chat %s no longer exists - skipping", chat_id)
            return
        if doc.get("title", DEFAULT_TITLE) != DEFAULT_TITLE:
            logger.info(
                "generate_chat_title: chat %s already has a title (%r) - skipping",
                chat_id, doc.get("title"),
            )
            return

        # The DeepInfra call is a blocking `openai` SDK call under the hood - offloaded to a
        # thread so it never stalls this worker's event loop (same pattern used throughout this
        # codebase, e.g. investigation.py's LLM calls, chats.py's route_query_intent_fast).
        result = await asyncio.to_thread(generate_title, query)

        update = await db[CHATS].update_one(
            {"_id": chat_id, "title": DEFAULT_TITLE},
            {"$set": {"title": result.title}},
        )
        logger.info(
            "generate_chat_title: chat %s titled %r (matched=%s, model=%s, fallback=%s, "
            "latency_ms=%.1f, error=%s)",
            chat_id, result.title, update.matched_count, result.model, result.fallback,
            result.latency_ms, result.error,
        )
    except Exception as exc:
        logger.exception(
            "generate_chat_title failed for chat %s (job_try=%s) - retrying", chat_id, ctx.get("job_try"),
        )
        raise Retry(defer=ctx["job_try"] * 5) from exc

    log_job_finished(logger, "generate_chat_title", picked_up_at, chat_id=chat_id)
