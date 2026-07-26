"""Shared logging setup for api_service and worker_service.

Both services call configure_logging("<service_name>") once at startup
instead of logging.basicConfig(). It sets up:

  - a console handler (same as before - `docker compose logs` still works
    exactly as it did)
  - a Loki handler, IF LOKI_URL is set (docker-compose.yml sets it to
    http://loki:3100/loki/api/v1/push for both services). Every
    logger.info/warning/error/exception call anywhere in the app then also
    ships to Loki, tagged with service/environment/level labels, so you can
    filter by service in Grafana's Explore view instead of grepping
    container stdout.

If LOKI_URL isn't set (e.g. running api_service/worker_service bare on a
dev machine without the observability stack up), this quietly falls back to
console-only logging - Loki is additive, never required.
"""
import logging
import os

from shared.config import get_settings

_configured_services: set[str] = set()


def configure_logging(service_name: str, level: str | None = None) -> logging.Logger:
    """Configure the root logger once per process. Safe to call multiple times
    (e.g. once from main.py and again from a router module) - only the first
    call for a given service_name actually attaches handlers."""
    if service_name in _configured_services:
        return logging.getLogger()

    settings = get_settings()
    resolved_level = (level or settings.get("LOG_LEVEL") or os.environ.get("LOG_LEVEL") or "INFO").upper()

    root = logging.getLogger()
    root.setLevel(resolved_level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    loki_url = settings.get("LOKI_URL") or os.environ.get("LOKI_URL")
    if loki_url:
        _attach_loki_handler(root, loki_url, service_name, settings)
    else:
        logging.getLogger(service_name).info(
            "LOKI_URL not set - logging to console only, Loki shipping disabled"
        )

    _configured_services.add(service_name)
    return root


class _SafeLokiHandler:
    """Mixin that swallows any emit()-time failure instead of letting it propagate.

    Loki shipping is meant to be purely additive (see module docstring) - if Loki is down,
    unreachable (NameResolutionError when the observability compose overlay isn't up), or just
    slow, that must never take the console handler / the rest of the app down with it. The
    underlying logging_loki.LokiHandler.emit() can raise requests.exceptions.ConnectionError
    straight through; logging's own Handler.handleError() safety net only prints "--- Logging
    error ---" and normally doesn't re-raise, but that's an extra assumption not worth relying on
    (e.g. inside arq's worker loop, an exception raised while handling a log record during
    on_startup has in practice taken the whole process down). Catch here explicitly so the
    contract is "never raises," full stop, not "probably doesn't raise."
    """

    def emit(self, record):
        try:
            super().emit(record)
        except Exception:
            pass


def _attach_loki_handler(root: logging.Logger, loki_url: str, service_name: str, settings) -> None:
    try:
        import logging_loki
    except ImportError:
        logging.getLogger(service_name).warning(
            "LOKI_URL is set but python-logging-loki isn't installed - "
            "add it to requirements.txt (pip install python-logging-loki)"
        )
        return

    environment = settings.get("ENVIRONMENT") or os.environ.get("ENVIRONMENT") or "development"

    class SafeLokiHandler(_SafeLokiHandler, logging_loki.LokiHandler):
        pass

    loki_handler = SafeLokiHandler(
        url=loki_url,
        tags={"service": service_name, "environment": environment},
        version="1",
    )
    # INFO+ ships to Loki (per the "info before tools, errors always" plan) -
    # DEBUG stays console-only so Loki doesn't get flooded with chatter.
    loki_handler.setLevel(logging.INFO)
    loki_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(loki_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
