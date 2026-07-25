"""Millisecond-precision timing helpers for the HTTP request -> arq job handoff.

Every api_service route that enqueues a job (chats.py's send_message, files.py's
confirm_upload, dashboards.py's refresh/relink) captures `now_iso()` the moment it starts
handling the request and passes it through as a `requested_at` job kwarg. Every matching
worker_service task function (run_investigation, run_ingestion, refresh_dashboard) calls
`log_job_picked_up` as its very first line, which logs - in one line, millisecond precision:

  - the wall-clock time this worker actually started running the job
  - job_id/job_try (arq's own retry counter - job_try > 1 means this is a retry)
  - queue_wait_ms: enqueue_job() -> this line, i.e. time spent sitting in Redis waiting for a
    free worker slot (arq puts `enqueue_time` in ctx for every job - no extra plumbing needed)
  - request_to_worker_ms: original HTTP request arrival -> this line - queue_wait_ms PLUS
    whatever the route itself spent (DB writes, etc.) before calling enqueue_job. Only
    available if the route passed requested_at through.

`log_job_finished` pairs with it, logging total in-worker duration (job pickup -> job's own
terminal state) so the full lifecycle - request arrived, enqueued, picked up, finished - is
reconstructable from the logs alone, without needing Loki/Langfuse.
"""
import logging
from datetime import datetime, timezone


def now_iso() -> str:
    """Wall-clock timestamp, millisecond precision, UTC. Call this as the very first line of a
    route handler and pass the result through as a `requested_at` job kwarg."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _parse_iso(ts) -> "datetime | None":
    if ts is None:
        return None
    if isinstance(ts, datetime):
        dt = ts
    else:
        try:
            dt = datetime.fromisoformat(ts)
        except (TypeError, ValueError):
            return None
    # arq's own ctx["enqueue_time"] is already tz-aware (UTC) in practice, but a
    # requested_at string round-tripped through some other path might not be - treat naive as
    # UTC rather than let the subtraction below raise or silently compare wrong types.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def log_job_picked_up(logger: logging.Logger, ctx: dict, job_name: str, requested_at=None, **ids) -> datetime:
    """Call as the first line of every arq job function - see module docstring. Returns the
    picked-up-at datetime so the caller can pass it to log_job_finished later without
    recomputing "now"."""
    picked_up_at = datetime.now(timezone.utc)
    enqueue_time = _parse_iso(ctx.get("enqueue_time"))
    requested_dt = _parse_iso(requested_at)

    queue_wait_ms = (picked_up_at - enqueue_time).total_seconds() * 1000 if enqueue_time else None
    request_to_worker_ms = (picked_up_at - requested_dt).total_seconds() * 1000 if requested_dt else None

    id_str = ", ".join(f"{k}={v}" for k, v in ids.items())
    logger.info(
        "%s: picked up by worker at %s (job_id=%s, try=%s%s%s) - queue_wait=%s, request_to_worker=%s",
        job_name, picked_up_at.isoformat(timespec="milliseconds"),
        ctx.get("job_id"), ctx.get("job_try"),
        ", " if id_str else "", id_str,
        f"{queue_wait_ms:.1f}ms" if queue_wait_ms is not None else "unknown",
        f"{request_to_worker_ms:.1f}ms" if request_to_worker_ms is not None else "unknown",
    )
    return picked_up_at


def log_job_finished(logger: logging.Logger, job_name: str, picked_up_at: datetime, status: str = "done", **ids) -> None:
    """Pairs with log_job_picked_up - call once at the job's own terminal point (success,
    failure, whatever this job tracks) so total in-worker duration (NOT including queue wait,
    which is already logged separately by log_job_picked_up) is visible too."""
    finished_at = datetime.now(timezone.utc)
    duration_ms = (finished_at - picked_up_at).total_seconds() * 1000
    id_str = ", ".join(f"{k}={v}" for k, v in ids.items())
    logger.info(
        "%s: finished at %s (status=%s, duration=%.1fms%s%s)",
        job_name, finished_at.isoformat(timespec="milliseconds"), status, duration_ms,
        ", " if id_str else "", id_str,
    )
