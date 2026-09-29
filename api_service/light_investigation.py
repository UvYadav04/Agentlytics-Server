from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from shared import usage
from shared.db import get_db
from shared.light_response import generate_light_reply
from shared.models.investigation import COLLECTION as INVESTIGATIONS
from shared.models.investigation import InvestigationEvent
from shared.models.message import COLLECTION as MESSAGES
from shared.models.message import Message
from shared.redis_client import get_arq_pool, get_redis, investigation_channel

logger = logging.getLogger("api.light_investigation")

MAX_ATTEMPTS = 2
RETRY_STATUS_MESSAGE = "Having trouble generating a reply - retrying..."
FALLBACK_STATUS_MESSAGE = "That's taking longer than expected - bringing in the full assistant..."

_tasks: set[asyncio.Task] = set()


def schedule_light_response(
    *, investigation_id: str, chat_id: str, workspace_id: str, user_id: str, query: str, route: str,
    requested_at: str | None = None, file_ids: list[str] | None = None, email: str | None = None,
) -> None:
    task = asyncio.create_task(
        _run(investigation_id, chat_id, workspace_id, user_id, query, route, requested_at, file_ids or [], email)
    )
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def _append_event(db, investigation_id: str, event_type: str, message: str, data: dict | None = None) -> None:
    event = InvestigationEvent(type=event_type, message=message, data=data or {})
    payload = event.model_dump(mode="json")
    await db[INVESTIGATIONS].update_one({"_id": investigation_id}, {"$push": {"events": payload}})
    try:
        await get_redis().publish(investigation_channel(investigation_id), json.dumps(payload))
    except Exception:
        logger.exception("light_investigation: failed to publish event for investigation %s", investigation_id)


async def _fallback_to_orchestrator(
    db, investigation_id: str, chat_id: str, workspace_id: str, user_id: str, query: str,
    requested_at: str | None, file_ids: list[str], email: str | None = None,
) -> None:
    logger.warning(
        "light_investigation: all %d attempt(s) failed for investigation %s - falling back to the full Orchestrator",
        MAX_ATTEMPTS, investigation_id,
    )
    await _append_event(db, investigation_id, "status", FALLBACK_STATUS_MESSAGE)
    pool = await get_arq_pool()
    await pool.enqueue_job(
        "run_investigation",
        investigation_id=investigation_id, chat_id=chat_id, workspace_id=workspace_id,
        user_id=user_id, query=query, file_ids=file_ids, requested_at=requested_at, route=None,
        email=email,
    )


async def _run(
    investigation_id: str, chat_id: str, workspace_id: str, user_id: str, query: str, route: str,
    requested_at: str | None, file_ids: list[str], email: str | None = None,
) -> None:
    db = get_db()
    result = None
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            result = await asyncio.to_thread(generate_light_reply, query, route)
            if result.content:
                break
            logger.warning(
                "light_investigation: attempt %d/%d failed for investigation %s (%s)",
                attempt, MAX_ATTEMPTS, investigation_id, result.error,
            )
            if attempt < MAX_ATTEMPTS:
                await _append_event(db, investigation_id, "status", RETRY_STATUS_MESSAGE)

        if result is None or not result.content:
            await _fallback_to_orchestrator(
                db, investigation_id, chat_id, workspace_id, user_id, query, requested_at, file_ids, email,
            )
            return

        message = Message(
            chat_id=chat_id, role="assistant", content=result.content, investigation_id=investigation_id,
        )
        await db[MESSAGES].insert_one(message.to_mongo())
        await db[INVESTIGATIONS].update_one(
            {"_id": investigation_id},
            {"$set": {
                "status": "completed", "final_answer": result.content,
                "completed_at": datetime.now(timezone.utc),
            }},
        )
        await usage.increment_messages(user_id)
        await _append_event(
            db, investigation_id, "completed", "Investigation complete.",
            {"message_id": message.id, "chart_ids": [], "report_id": None},
        )
        logger.info(
            "light_investigation: investigation %s completed via %s route (model=%s, latency_ms=%.1f)",
            investigation_id, route, result.model, result.latency_ms,
        )

       
        pool = await get_arq_pool()
        await pool.enqueue_job(
            "update_chat_memory",
            chat_id=chat_id, user_id=user_id, query=query, response=result.content,
            files_used=[], files_created=[], requested_at=requested_at,
        )
    except Exception:
        logger.exception(
            "light_investigation: unhandled failure for investigation %s - attempting fallback", investigation_id,
        )
        try:
            await _fallback_to_orchestrator(
                db, investigation_id, chat_id, workspace_id, user_id, query, requested_at, file_ids, email,
            )
        except Exception:
            logger.exception(
                "light_investigation: fallback enqueue itself failed for investigation %s", investigation_id,
            )
            await _append_event(db, investigation_id, "error", "Something went wrong generating a reply.")
            await db[INVESTIGATIONS].update_one(
                {"_id": investigation_id},
                {"$set": {"status": "failed", "completed_at": datetime.now(timezone.utc)}},
            )
