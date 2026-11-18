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

    _configured_services.add(service_name)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
