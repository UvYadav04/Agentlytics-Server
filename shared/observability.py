import logging
import os

from shared.config import get_settings

logger = logging.getLogger("shared.observability")

_initialized_services: set[str] = set()
_tracer_provider = None
_otel_enabled = False


def init_observability(service_name: str) -> None:
    if service_name in _initialized_services:
        return
    _initialized_services.add(service_name)

    settings = get_settings()
    otlp_endpoint = settings.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not otlp_endpoint:
        logger.info(
            "OTEL_EXPORTER_OTLP_ENDPOINT not set - OpenTelemetry export to Grafana Alloy is "
            "disabled (the app runs normally either way, just without traces/metrics/logs "
            "leaving the process)"
        )
        return

    try:
        _init_otel(service_name, otlp_endpoint, settings)
    except Exception:
        logger.exception(
            "OpenTelemetry initialization failed for service=%s - continuing without it "
            "(the app must keep running even if the observability backend is unavailable)",
            service_name,
        )


def _build_resource(service_name: str, settings):
    from opentelemetry.sdk.resources import Resource

    attributes = {"service.name": settings.get("OTEL_SERVICE_NAME") or service_name}
    raw_extra = settings.get("OTEL_RESOURCE_ATTRIBUTES") or os.environ.get("OTEL_RESOURCE_ATTRIBUTES")
    # Standard OTel format: comma-separated key=value pairs, e.g. "deployment.environment=prod".
    for pair in (raw_extra or "").split(","):
        if "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        key, value = key.strip(), value.strip()
        if key:
            attributes[key] = value
    return Resource.create(attributes)


_NOISY_REDIS_COMMANDS = {"ZRANGEBYSCORE", "ZADD", "ZREM", "ZCARD", "PING", "BRPOPLPUSH", "EVALSHA", "HSET", "HGET"}


def _is_noisy_span(span) -> bool:
    attrs = getattr(span, "attributes", None) or {}
    if attrs.get("db.system") == "redis":
        command = (span.name or "").split(" ")[0].upper()
        if command in _NOISY_REDIS_COMMANDS:
            return True
    return False


class _FilteringSpanProcessor:
    def __init__(self, inner):
        self._inner = inner

    def on_start(self, span, parent_context=None):
        self._inner.on_start(span, parent_context=parent_context)

    def on_end(self, span):
        if _is_noisy_span(span):
            return
        self._inner.on_end(span)

    def shutdown(self):
        self._inner.shutdown()

    def force_flush(self, timeout_millis=30000):
        return self._inner.force_flush(timeout_millis)


def _init_otel(service_name: str, otlp_endpoint: str, settings) -> None:
    global _tracer_provider, _otel_enabled

    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = _build_resource(service_name, settings)

    export_timeout_ms = int(settings.get("OTEL_EXPORT_TIMEOUT_MS", 15000) or 15000)

    tracer_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, timeout=export_timeout_ms // 1000)
    batch_processor = BatchSpanProcessor(
        span_exporter,
        export_timeout_millis=export_timeout_ms,
        schedule_delay_millis=5000,
    )
    tracer_provider.add_span_processor(_FilteringSpanProcessor(batch_processor))

    trace.set_tracer_provider(tracer_provider)
    _tracer_provider = tracer_provider

    langfuse_ready = False
    if settings.get("LANGFUSE_PUBLIC_KEY") and settings.get("LANGFUSE_SECRET_KEY"):
        langfuse_ready = get_langfuse_client() is not None
        if langfuse_ready:
            logger.info("Langfuse client constructed with the shared tracer_provider - LLM-shaped spans export to Langfuse Cloud")
    else:
        logger.info("LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set - Langfuse tracing disabled")

    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=otlp_endpoint, timeout=export_timeout_ms // 1000),
        export_timeout_millis=export_timeout_ms,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    from shared.logging_config import attach_otel_logging
    attach_otel_logging(service_name, resource, otlp_endpoint)

    _auto_instrument()
    _otel_enabled = True

    logger.info(
        "OpenTelemetry initialized: service=%s endpoint=%s langfuse=%s",
        service_name, otlp_endpoint, langfuse_ready,
    )


def _auto_instrument() -> None:
    instrumentors = [
        ("pymongo", "opentelemetry.instrumentation.pymongo", "PymongoInstrumentor"),
        ("redis", "opentelemetry.instrumentation.redis", "RedisInstrumentor"),
        ("httpx", "opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"),
        ("requests", "opentelemetry.instrumentation.requests", "RequestsInstrumentor"),
        ("botocore (S3)", "opentelemetry.instrumentation.botocore", "BotocoreInstrumentor"),
    ]
    for label, module_name, class_name in instrumentors:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)().instrument()
        except Exception:
            logger.exception("%s auto-instrumentation failed - continuing without it", label)

    try:
        from opentelemetry.instrumentation.logging import LoggingInstrumentor
        LoggingInstrumentor().instrument(set_logging_format=False)
    except Exception:
        logger.exception("logging auto-instrumentation (trace/span id injection) failed")


def instrument_fastapi(app) -> None:
    if not _otel_enabled:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app, tracer_provider=_tracer_provider)
    except Exception:
        logger.exception("FastAPI auto-instrumentation failed")


def get_tracer(name: str):
    from opentelemetry import trace
    return trace.get_tracer(name)


def get_meter(name: str):
    from opentelemetry import metrics
    return metrics.get_meter(name)


_langfuse_client = None


def get_langfuse_client():
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    settings = get_settings()
    if not settings.get("LANGFUSE_PUBLIC_KEY") or not settings.get("LANGFUSE_SECRET_KEY"):
        return None
    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse(tracer_provider=_tracer_provider)
    except ImportError:
        logger.warning("langfuse is not installed - add it to requirements.txt")
        return None
    except Exception:
        logger.exception("failed to construct Langfuse client")
        return None
    return _langfuse_client


def start_prometheus_metrics_server(port: int) -> None:
    from prometheus_client import start_http_server
    start_http_server(port)
    logger.info("Prometheus metrics server listening on :%s/metrics", port)