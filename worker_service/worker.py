"""arq worker entrypoint.

Run with (from the Server/ directory):

    arq worker_service.worker.WorkerSettings

This process is intentionally decoupled from api_service - it only talks to
Mongo and Redis, never receives HTTP requests, and keeps running/writing
progress regardless of whether any client is currently connected to an SSE
stream (see the "Refresh-safety" note in full_application_build_plan.md
Phase 5).
"""
import logging
import os

from worker_service import engine_bootstrap  # noqa: F401  (sys.path setup, see module docstring)
from worker_service.tasks.dashboard_refresh import refresh_dashboard
from worker_service.tasks.ingestion import run_ingestion
from worker_service.tasks.investigation import run_investigation

from analyzerEngine.ingestion.storage.local_store import LocalParquetStore
from analyzerEngine.vectordb.chroma_store import ChromaVectorStore

from shared.db import close_client, ensure_indexes
from shared.logging_config import configure_logging
from shared.observability import start_prometheus_metrics_server
from shared.redis_client import close_redis, get_arq_redis_settings

# Console + Loki (if LOKI_URL is set - see docker-compose.yml) instead of
# the old bare logging.basicConfig().
configure_logging("worker_service")

# autogen_core's own structured-tracing logger (distinct from our compact
# agents/logger.py output) dumps one full LLMCall record per model call -
# every tool's complete JSON schema plus the whole accumulated message
# history so far, repeated in full on every iteration. At INFO it inherits
# root's level from configure_logging() above (and would now also ship to
# Loki) and floods the terminal/log file with that, growing every call.
# Silence just this logger so our own "[tool
# call] ..." / "[tool result] ..." / assistant text lines (see
# analyzerEngine/agents/logger.py) stay the only per-step activity logged.
logging.getLogger("autogen_core.events").setLevel(logging.WARNING)


async def on_startup(ctx):
    await ensure_indexes()

    # arq has no HTTP server of its own, so Prometheus has nothing to scrape
    # here unless we open one ourselves. No-op if PROMETHEUS_METRICS_PORT
    # isn't set (e.g. running this worker bare, outside docker-compose).
    metrics_port = os.environ.get("PROMETHEUS_METRICS_PORT")
    if metrics_port:
        start_prometheus_metrics_server(int(metrics_port))

    # Built ONCE per worker process, not once per job - run_ingestion/run_investigation used to
    # construct a fresh LocalParquetStore + ChromaVectorStore on every single call. LocalParquetStore
    # itself is cheap (just stores a path), but ChromaVectorStore's __init__ opens a
    # chromadb.CloudClient(...) - a real network handshake to Chroma Cloud - so that cost was being
    # paid on every job instead of once at startup. `ctx` is arq's per-process dict, shared across
    # every job this worker runs (and passed as each job function's first argument) - see
    # https://arq-docs.helpmanual.io/#usage for on_startup/ctx. If ChromaVectorStore() fails here
    # (e.g. bad/missing CHROMA_* credentials), the worker refuses to start rather than having every
    # job fail individually once it gets deep into a run - fail fast, once, loudly.
    ctx["storage"] = LocalParquetStore(root_dir=engine_bootstrap.PARQUET_ROOT)
    ctx["vector_store"] = ChromaVectorStore()

    logging.getLogger("worker").info("worker started, engine loaded from %s", engine_bootstrap.ENGINE_DIR)


async def on_shutdown(ctx):
    await close_redis()
    await close_client()


class WorkerSettings:
    functions = [run_ingestion, run_investigation, refresh_dashboard]
    redis_settings = get_arq_redis_settings()
    on_startup = on_startup
    on_shutdown = on_shutdown
    # Investigations run a multi-agent tool-calling loop (up to 25
    # orchestrator iterations, each possibly delegating to a subagent with
    # its own loop) - the default 300s arq job timeout is too tight.
    # PDF ingestion needs headroom too: docling's CPU layout/table pipeline measured
    # 858.79s for a single dense SEC-filing PDF (see pdf_ingestor.py's _convert_cached -
    # before that fix, ingestion ran that conversion twice per file, which is what
    # actually blew the old 900s ceiling). One conversion alone can still land close to
    # 900s on a big enough document, so this leaves real margin instead of a near-exact
    # race. asyncio.to_thread work (docling conversion) isn't actually killable on
    # timeout - the awaiting coroutine gets cancelled but the OS thread runs on orphaned
    # - so a tighter timeout doesn't fail faster/cheaper, it just fails messier (see the
    # "No such file or directory" secondary error from the temp file being cleaned up out
    # from under the still-running thread). Better to just not hit it.
    job_timeout = 1800
    # Ingestion (esp. PDF/docling) and investigations are both
    # CPU/LLM-latency heavy, not memory-cheap - keep concurrency modest by
    # default; raise once you've checked memory headroom.
    max_jobs = 4
