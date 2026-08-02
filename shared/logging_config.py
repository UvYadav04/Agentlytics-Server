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

    old_factory = logging.getLogRecordFactory()

    def _record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        if not hasattr(record, "otelTraceID"):
            record.otelTraceID = "-"
        if not hasattr(record, "otelSpanID"):
            record.otelSpanID = "-"
        return record

    logging.setLogRecordFactory(_record_factory)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] [trace_id=%(otelTraceID)s span_id=%(otelSpanID)s] %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    _configured_services.add(service_name)
    return root


def attach_otel_logging(service_name: str, resource, otlp_endpoint: str) -> None:

    try:
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(OTLPLogExporter(endpoint=otlp_endpoint))
        )
        set_logger_provider(logger_provider)

        otel_handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
        logging.getLogger().addHandler(otel_handler)
        logging.getLogger("shared.logging_config").info(
            "OTel log export attached: service=%s endpoint=%s", service_name, otlp_endpoint
        )
    except Exception:
        logging.getLogger("shared.logging_config").exception(
            "failed to attach OTel log export - continuing with console logging only"
        )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
