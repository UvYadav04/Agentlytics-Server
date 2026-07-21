"""Shared observability helpers for api_service and worker_service:
Langfuse client access and Prometheus metrics-server bootstrap.

Langfuse: actual LLM-call tracing (every ChatCompletionClient.create() ->
one Langfuse generation) is wired up in
analyzerEngine/llm_provider/langfuse_wrapper.py, NOT here - that module
deliberately avoids importing from shared/ so analyzerEngine stays
importable as its own root (see engine_bootstrap.py's docstring), and reads
its own local config.get_settings() instead. It has its own copy of this
same lazy-singleton pattern.

get_langfuse_client() below is a separate accessor for anything in
api_service that wants to talk to Langfuse directly - e.g. the feedback
router pushing a user's thumbs up/down as a Langfuse score against the
trace/generation ID returned alongside a chat response. Not wired into
anything yet; use this if/when that's needed. Returns None if
LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY aren't configured, same
fail-open behavior as the analyzerEngine copy.

Prometheus: worker_service has no HTTP server of its own (arq), so
start_prometheus_metrics_server() opens a dedicated metrics-only HTTP
server via prometheus_client - see worker_service/worker.py's on_startup.
api_service doesn't need this; it uses prometheus-fastapi-instrumentator
to expose /metrics on its existing FastAPI port instead (see main.py).
"""
import logging

from shared.config import get_settings

logger = logging.getLogger("shared.observability")

_langfuse_client = None
_langfuse_checked = False


def get_langfuse_client():
    global _langfuse_client, _langfuse_checked
    if _langfuse_checked:
        return _langfuse_client
    _langfuse_checked = True

    settings = get_settings()
    public_key = settings.get("LANGFUSE_PUBLIC_KEY")
    secret_key = settings.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        logger.info("LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set - Langfuse tracing disabled")
        return None

    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning("langfuse is not installed - add it to requirements.txt")
        return None

    _langfuse_client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=settings.get("LANGFUSE_HOST", "http://langfuse-web:3000"),
    )
    return _langfuse_client


def start_prometheus_metrics_server(port: int) -> None:
    from prometheus_client import start_http_server

    start_http_server(port)
    logger.info("Prometheus metrics server listening on :%s/metrics", port)
