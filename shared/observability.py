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

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))

    langfuse_public_key = settings.get("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = settings.get("LANGFUSE_SECRET_KEY")
    if langfuse_public_key and langfuse_secret_key:
        try:
            from langfuse.opentelemetry import LangfuseSpanProcessor
            tracer_provider.add_span_processor(LangfuseSpanProcessor())
            logger.info("Langfuse span processor attached - LLM-shaped spans export to Langfuse Cloud")
        except ImportError:
            logger.warning("langfuse is not installed - Langfuse tracing disabled")
        except Exception:
            logger.exception("failed to attach LangfuseSpanProcessor - Langfuse tracing disabled")
    else:
        logger.info("LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set - Langfuse tracing disabled")

    trace.set_tracer_provider(tracer_provider)
    _tracer_provider = tracer_provider

    metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=otlp_endpoint))
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    from shared.logging_config import attach_otel_logging
    attach_otel_logging(service_name, resource, otlp_endpoint)

    _auto_instrument()
    _otel_enabled = True

    logger.info(
        "OpenTelemetry initialized: service=%s endpoint=%s langfuse=%s",
        service_name, otlp_endpoint, bool(langfuse_public_key and langfuse_secret_key),
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


def get_langfuse_client():
    settings = get_settings()
    if not settings.get("LANGFUSE_PUBLIC_KEY") or not settings.get("LANGFUSE_SECRET_KEY"):
        return None
    try:
        from langfuse import get_client
        return get_client()
    except ImportError:
        logger.warning("langfuse is not installed - add it to requirements.txt")
        return None
    except Exception:
        logger.exception("failed to get Langfuse client")
        return None


def start_prometheus_metrics_server(port: int) -> None:
    from prometheus_client import start_http_server
    start_http_server(port)
    logger.info("Prometheus metrics server listening on :%s/metrics", port)
