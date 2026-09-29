import logging
from datetime import timedelta

from shared import usage
from shared.db import get_db
from shared.models.investigation import COLLECTION as INVESTIGATIONS
from shared.models.investigation import Investigation
from shared.models.message import COLLECTION as MESSAGES
from shared.models.message import Message
from worker_service.tasks.investigation import _RETRY_FAILED, _append_event, _now, _with_retries

logger = logging.getLogger("worker.reconciliation")

STUCK_THRESHOLD_MINUTES = 10

MAX_AUTO_RETRIES = 1

FAILURE_MESSAGE = (
    "Something interrupted this investigation before it could finish (the worker handling it was "
    "restarted). Please try asking again."
)

MAX_PER_SWEEP = 200


async def reconcile_stuck_investigations(ctx) -> None:
    db = get_db()
    cutoff = _now() - timedelta(minutes=STUCK_THRESHOLD_MINUTES)

    stuck_docs = await db[INVESTIGATIONS].find({
        "status": "running",
        "$or": [
            {"last_attempt_at": {"$lt": cutoff}},
            {"last_attempt_at": None, "started_at": {"$lt": cutoff}},
        ],
    }).to_list(length=MAX_PER_SWEEP)

    if not stuck_docs:
        return

    logger.warning("reconcile_stuck_investigations: found %d stuck investigation(s)", len(stuck_docs))

    for doc in stuck_docs:
        investigation = Investigation.from_mongo(doc)
        try:
            await _reconcile_one(ctx, db, investigation)
        except Exception:
            logger.exception(
                "reconcile_stuck_investigations: failed to reconcile investigation %s", investigation.id,
            )


async def _reconcile_one(ctx, db, investigation: Investigation) -> None:
    if investigation.cancel_requested:
        await db[INVESTIGATIONS].update_one(
            {"_id": investigation.id, "status": "running"},
            {"$set": {"status": "cancelled", "stage": "cancelled", "completed_at": _now()}},
        )
        return

    message_doc = await db[MESSAGES].find_one(
        {"investigation_id": investigation.id}, {"_id": 1, "content": 1},
    )
    if message_doc is not None:
        await _backfill_bookkeeping(ctx, db, investigation, message_doc)
        return

    if investigation.retry_count >= MAX_AUTO_RETRIES or not investigation.user_id:
        await _give_up(db, investigation)
        return

    await _retry(ctx, db, investigation)


async def _backfill_bookkeeping(ctx, db, investigation: Investigation, message_doc: dict) -> None:
    logger.warning(
        "reconcile_stuck_investigations: investigation %s has message %s but is still "
        "status=running - backfilling the bookkeeping a crashed worker didn't finish",
        investigation.id, message_doc["_id"],
    )
    final_answer = investigation.final_answer or message_doc.get("content", "")

    update = await db[INVESTIGATIONS].update_one(
        {"_id": investigation.id, "status": "running"},
        {"$set": {
            "status": "completed", "stage": "completed",
            "final_answer": final_answer, "completed_at": _now(),
        }},
    )
    if update.matched_count == 0:
        return

    if not investigation.usage_counted and investigation.user_id:
        result = await _with_retries(
            lambda: usage.increment_messages(investigation.user_id),
            description=f"reconciliation usage increment (investigation {investigation.id})",
        )
        if result is not _RETRY_FAILED:
            await db[INVESTIGATIONS].update_one(
                {"_id": investigation.id}, {"$set": {"usage_counted": True}},
            )

    if not investigation.chat_memory_enqueued and investigation.user_id:
        result = await _with_retries(
            lambda: ctx["redis"].enqueue_job(
                "update_chat_memory",
                chat_id=investigation.chat_id, user_id=investigation.user_id,
                query=investigation.objective, response=final_answer,
                files_used=[], files_created=[],
            ),
            description=f"reconciliation chat memory enqueue (investigation {investigation.id})",
        )
        if result is not _RETRY_FAILED:
            await db[INVESTIGATIONS].update_one(
                {"_id": investigation.id}, {"$set": {"chat_memory_enqueued": True}},
            )


async def _retry(ctx, db, investigation: Investigation) -> None:
    logger.warning(
        "reconcile_stuck_investigations: investigation %s stuck with no result - re-dispatching "
        "(attempt %d/%d)",
        investigation.id, investigation.retry_count + 1, MAX_AUTO_RETRIES,
    )
    update = await db[INVESTIGATIONS].update_one(
        {"_id": investigation.id, "status": "running"},
        {"$set": {"last_attempt_at": _now(), "stage": "retrying"}, "$inc": {"retry_count": 1}},
    )
    if update.matched_count == 0:
        return

    await _append_event(db, investigation.id, "status", "Picking this back up after an interruption...")
    await ctx["redis"].enqueue_job(
        "run_investigation",
        investigation_id=investigation.id, chat_id=investigation.chat_id,
        workspace_id=investigation.workspace_id, user_id=investigation.user_id,
        query=investigation.objective, file_ids=investigation.file_ids, email=investigation.email,
    )


async def _give_up(db, investigation: Investigation) -> None:
    logger.error(
        "reconcile_stuck_investigations: giving up on investigation %s after %d retr(y/ies)",
        investigation.id, investigation.retry_count,
    )
    update = await db[INVESTIGATIONS].update_one(
        {"_id": investigation.id, "status": "running"},
        {"$set": {
            "status": "failed", "stage": "failed", "completed_at": _now(),
            "error_type": "worker_crash", "error_message": FAILURE_MESSAGE,
        }},
    )
    if update.matched_count == 0:
        return

    message = Message(
        chat_id=investigation.chat_id, role="assistant", content=FAILURE_MESSAGE,
        investigation_id=investigation.id,
    )
    await db[MESSAGES].insert_one(message.to_mongo())
    await _append_event(db, investigation.id, "error", FAILURE_MESSAGE)
