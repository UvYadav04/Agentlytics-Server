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
