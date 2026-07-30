import logging
import os

from shared.config import get_settings

_configured_services: set[str] = set()


def configure_logging(service_name: str, level: str | None = None) -> logging.Logger:
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
    loki_handler.setLevel(logging.INFO)
    loki_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(loki_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
