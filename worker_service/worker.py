import asyncio
import logging
import os
import time

from arq import cron
from arq.worker import func as arq_func

from worker_service import engine_bootstrap
from worker_service.tasks.dashboard_refresh import refresh_dashboard
from worker_service.tasks.ingestion import run_ingestion
from worker_service.tasks.investigation import run_investigation, update_chat_memory
from worker_service.tasks.reconciliation import reconcile_stuck_investigations

from analyzerEngine.ingestion.storage.local_store import LocalParquetStore
from analyzerEngine.sandbox.sandbox_manager import get_manager as get_sandbox_manager
from analyzerEngine.vectordb.chroma_store import ChromaVectorStore

from shared.db import close_client, ensure_indexes
from shared.logging_config import configure_logging
from shared.observability import get_meter, init_observability, start_prometheus_metrics_server
from shared.redis_client import close_redis, get_arq_redis_settings, get_redis

configure_logging("worker_service")
# Must run before on_startup constructs Mongo/Redis clients (ctx["storage"]/ctx["vector_store"]
# etc. below), so those get auto-instrumented - see shared/observability.py.
init_observability("worker_service")

logging.getLogger("autogen_core.events").setLevel(logging.WARNING)

# arq's default queue is a Redis sorted set at this key (see arq.constants.default_queue_name) -
# polled in the background below rather than read inline in the observable gauge callback, since
# OTel's async-instrument callbacks run synchronously on the SDK's export path and shouldn't do
# their own network I/O.
_ARQ_QUEUE_KEY = "arq:queue"
_queue_size_cache = {"value": 0}


def _queue_size_callback(options):
    from opentelemetry.metrics import Observation
    yield Observation(_queue_size_cache["value"], {"queue": _ARQ_QUEUE_KEY})


_meter = get_meter("worker_service")
_meter.create_observable_gauge(
    "worker.queue.size", callbacks=[_queue_size_callback],
    description="Pending arq job count (polled every ~15s)",
)


async def _poll_queue_size():
    redis = get_redis()
    while True:
        try:
            _queue_size_cache["value"] = await redis.zcard(_ARQ_QUEUE_KEY)
        except Exception:
            logging.getLogger("worker").exception("failed to poll arq queue size - leaving last known value")
        await asyncio.sleep(15)


async def on_startup(ctx):
    try:
        await ensure_indexes()
    except Exception:
        logging.getLogger("worker").exception(
            "on_startup: ensure_indexes failed - worker cannot start without a working Mongo "
            "connection, re-raising so the process exits with a clear traceback instead of "
            "hanging or crashing silently"
        )
        raise

    metrics_port = os.environ.get("PROMETHEUS_METRICS_PORT")
    if metrics_port:
        try:
            start_prometheus_metrics_server(int(metrics_port))
        except Exception:
            logging.getLogger("worker").exception(
                "failed to start Prometheus metrics server on port %s - continuing without it",
                metrics_port,
            )

    ctx["queue_size_poll_task"] = asyncio.create_task(_poll_queue_size())

    try:
        ctx["storage"] = LocalParquetStore(root_dir=engine_bootstrap.PARQUET_ROOT)
        ctx["vector_store"] = ChromaVectorStore()
        sandbox_manager = get_sandbox_manager(socket_root=engine_bootstrap.SANDBOX_SOCKET_ROOT)
        ctx["sandbox_manager"] = sandbox_manager
    except Exception:
        logging.getLogger("worker").exception(
            "on_startup: failed to construct storage/vector_store/sandbox_manager - re-raising "
            "so the crash reason is visible in logs instead of the container just exiting"
        )
        raise

    # Bring the pool up to min_size *now*, before this worker starts pulling jobs off the
    # queue - this is what actually eliminates first-request latency: previously each new
    # chat paid full container-create/start on its first run_python call (fire-and-forget
    # pre-warmed, but still awaited before the investigation could finish). With a shared
    # warm pool there's no per-chat sandbox to create - jobs just acquire whatever's idle.
    warm_start = time.perf_counter()
    try:
        await asyncio.to_thread(sandbox_manager.warm_pool)
        logging.getLogger("worker").info(
            "sandbox pool warmed to min_size=%d in %.1fms",
            sandbox_manager.min_size, (time.perf_counter() - warm_start) * 1000,
        )
    except Exception:
        logging.getLogger("worker").exception(
            "failed to warm the sandbox pool at startup - continuing without it; the pool "
            "will grow on-demand as executions come in"
        )

    logging.getLogger("worker").info("worker started, engine loaded from %s", engine_bootstrap.ENGINE_DIR)


async def on_shutdown(ctx):
    poll_task = ctx.get("queue_size_poll_task")
    if poll_task is not None:
        poll_task.cancel()
    sandbox_manager = ctx.get("sandbox_manager")
    if sandbox_manager is not None:
        await asyncio.to_thread(sandbox_manager.shutdown_all)
    await close_redis()
    await close_client()


class WorkerSettings:
    functions = [
        run_ingestion,
        run_investigation,
        refresh_dashboard,
        arq_func(update_chat_memory, max_tries=5),
    ]
    # Sweeps for investigations stuck at status="running" (worker died mid-run, or died after
    # sending the result but before finishing background bookkeeping) - see reconciliation.py's
    # module docstring. Every 5 minutes against a 10-minute stuck threshold there, so nothing sits
    # broken for more than ~15 minutes worst case.
    cron_jobs = [
        cron(reconcile_stuck_investigations, minute=set(range(0, 60, 5)), timeout=300),
    ]
    redis_settings = get_arq_redis_settings()
    on_startup = on_startup
    on_shutdown = on_shutdown
    job_timeout = 1800
    max_jobs = 4
    poll_delay = 0.05
