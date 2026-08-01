import asyncio
import logging
import os
import time

from arq.worker import func as arq_func

from worker_service import engine_bootstrap
from worker_service.tasks.chat_title import generate_chat_title
from worker_service.tasks.dashboard_refresh import refresh_dashboard
from worker_service.tasks.ingestion import run_ingestion
from worker_service.tasks.investigation import run_investigation, update_chat_memory

from analyzerEngine.ingestion.storage.local_store import LocalParquetStore
from analyzerEngine.sandbox.sandbox_manager import get_manager as get_sandbox_manager
from analyzerEngine.vectordb.chroma_store import ChromaVectorStore

from shared.db import close_client, ensure_indexes
from shared.logging_config import configure_logging
from shared.observability import start_prometheus_metrics_server
from shared.redis_client import close_redis, get_arq_redis_settings

configure_logging("worker_service")

logging.getLogger("autogen_core.events").setLevel(logging.WARNING)


async def on_startup(ctx):
    await ensure_indexes()

    metrics_port = os.environ.get("PROMETHEUS_METRICS_PORT")
    if metrics_port:
        try:
            start_prometheus_metrics_server(int(metrics_port))
        except Exception:
            logging.getLogger("worker").exception(
                "failed to start Prometheus metrics server on port %s - continuing without it",
                metrics_port,
            )

    ctx["storage"] = LocalParquetStore(root_dir=engine_bootstrap.PARQUET_ROOT)
    ctx["vector_store"] = ChromaVectorStore()

    sandbox_manager = get_sandbox_manager(socket_root=engine_bootstrap.SANDBOX_SOCKET_ROOT)
    ctx["sandbox_manager"] = sandbox_manager

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
        arq_func(generate_chat_title, max_tries=3),
    ]
    redis_settings = get_arq_redis_settings()
    on_startup = on_startup
    on_shutdown = on_shutdown
    job_timeout = 1800
    max_jobs = 4
    poll_delay = 0.05
