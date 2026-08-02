import logging
from datetime import datetime, timezone

from shared.observability import get_meter

_meter = get_meter("worker_service.jobs")
_job_duration = _meter.create_histogram(
    "worker.job.duration_ms", unit="ms", description="arq job execution wall time, keyed by job name",
)


def now_iso() -> str:
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
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def log_job_picked_up(logger: logging.Logger, ctx: dict, job_name: str, requested_at=None, **ids) -> datetime:
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

    finished_at = datetime.now(timezone.utc)
    duration_ms = (finished_at - picked_up_at).total_seconds() * 1000
    id_str = ", ".join(f"{k}={v}" for k, v in ids.items())
    logger.info(
        "%s: finished at %s (status=%s, duration=%.1fms%s%s)",
        job_name, finished_at.isoformat(timespec="milliseconds"), status, duration_ms,
        ", " if id_str else "", id_str,
    )
    # Every arq job (run_investigation, run_ingestion, update_chat_memory, refresh_dashboard,
    # reconcile_stuck_investigations) calls this on completion, so this one histogram covers
    # "worker execution duration" for the whole worker service without per-task instrumentation.
    _job_duration.record(duration_ms, {"job": job_name, "status": status})
