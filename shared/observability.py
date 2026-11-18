"""Observability interface.

OpenTelemetry has been removed in favor of the custom internal monitoring system
(see monitoring/) and Prometheus/Loki. This module provides zero-overhead no-op
tracers/meters to preserve compatibility with existing callers without requiring
any OpenTelemetry packages, exporters, or background gRPC threads.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from shared.config import get_settings

logger = logging.getLogger("shared.observability")


class _NoOpSpan:
    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return False

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, exception: BaseException, *args: Any, **kwargs: Any) -> None:
        pass

    def add_event(self, name: str, *args: Any, **kwargs: Any) -> None:
        pass


class _NoOpTracer:
    def start_as_current_span(self, name: str, *args: Any, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def start_span(self, name: str, *args: Any, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()


class _NoOpMetric:
    def add(self, amount: float | int, attributes: Optional[dict[str, Any]] = None) -> None:
        pass

    def record(self, amount: float | int, attributes: Optional[dict[str, Any]] = None) -> None:
        pass


class _NoOpMeter:
    def create_counter(self, name: str, *args: Any, **kwargs: Any) -> _NoOpMetric:
        return _NoOpMetric()

    def create_histogram(self, name: str, *args: Any, **kwargs: Any) -> _NoOpMetric:
        return _NoOpMetric()

    def create_observable_gauge(self, name: str, *args: Any, **kwargs: Any) -> _NoOpMetric:
        return _NoOpMetric()


_noop_tracer = _NoOpTracer()
_noop_meter = _NoOpMeter()


def init_observability(service_name: str) -> None:
    """No-op: OpenTelemetry export is disabled in favor of internal monitoring."""
    pass


def instrument_fastapi(app: Any) -> None:
    """No-op: FastAPI OpenTelemetry middleware is removed."""
    pass


def get_tracer(name: str) -> _NoOpTracer:
    return _noop_tracer


def get_meter(name: str) -> _NoOpMeter:
    return _noop_meter


_langfuse_client = None


def get_langfuse_client() -> Any:
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    settings = get_settings()
    if not settings.get("LANGFUSE_PUBLIC_KEY") or not settings.get("LANGFUSE_SECRET_KEY"):
        return None
    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key=settings.get("LANGFUSE_PUBLIC_KEY"),
            secret_key=settings.get("LANGFUSE_SECRET_KEY"),
            host=settings.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except ImportError:
        return None
    except Exception:
        logger.exception("Failed to construct Langfuse client")
        return None
    return _langfuse_client


def start_prometheus_metrics_server(port: int) -> None:
    from prometheus_client import start_http_server
    start_http_server(port)
    logger.info("Prometheus metrics server listening on :%s/metrics", port)
