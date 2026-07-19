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

from worker_service import engine_bootstrap  # noqa: F401  (sys.path setup, see module docstring)
from worker_service.tasks.ingestion import run_ingestion
from worker_service.tasks.investigation import run_investigation

from shared.db import close_client, ensure_indexes
from shared.redis_client import close_redis, get_arq_redis_settings

logging.basicConfig(level=logging.INFO)

# autogen_core's own structured-tracing logger (distinct from our compact
# agents/logger.py output) dumps one full LLMCall record per model call -
# every tool's complete JSON schema plus the whole accumulated message
# history so far, repeated in full on every iteration. At INFO it inherits
# root's level from basicConfig above and floods the terminal/log file with
# that, growing every call. Silence just this logger so our own "[tool
# call] ..." / "[tool result] ..." / assistant text lines (see
# analyzerEngine/agents/logger.py) stay the only per-step activity logged.
logging.getLogger("autogen_core.events").setLevel(logging.WARNING)


async def on_startup(ctx):
    await ensure_indexes()
    logging.getLogger("worker").info("worker started, engine loaded from %s", engine_bootstrap.ENGINE_DIR)


async def on_shutdown(ctx):
    await close_redis()
    await close_client()


class WorkerSettings:
    functions = [run_ingestion, run_investigation]
    redis_settings = get_arq_redis_settings()
    on_startup = on_startup
    on_shutdown = on_shutdown
    # Investigations run a multi-agent tool-calling loop (up to 25
    # orchestrator iterations, each possibly delegating to a subagent with
    # its own loop) - the default 300s arq job timeout is too tight.
    job_timeout = 900
    # Ingestion (esp. PDF/docling) and investigations are both
    # CPU/LLM-latency heavy, not memory-cheap - keep concurrency modest by
    # default; raise once you've checked memory headroom.
    max_jobs = 4
