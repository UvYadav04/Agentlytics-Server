"""Base tracing decorator and context manager.

Provides automatic execution timing, parent-child nesting via contextvars,
error capture, and persistent span storage without impacting runtime safety.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
import traceback
from typing import Any, Callable, Optional

from monitoring.context import (
    generate_span_id,
    get_current_span_id,
    get_current_tags,
    get_current_trace_id,
    span_scope,
)
from monitoring.schema import SpanRecord, SpanStatus, StepType
from monitoring.storage import get_monitoring_store

logger = logging.getLogger("monitoring.wrappers.base")


class TraceSpanContext:
    """Sync & async context manager to trace an arbitrary block of code."""

    def __init__(
        self,
        name: str,
        component: str,
        step_type: str = StepType.CUSTOM.value,
        trace_id: Optional[str] = None,
        inputs: Optional[dict[str, Any]] = None,
        tags: Optional[dict[str, Any]] = None,
    ):
        self.name = name
        self.component = component
        self.step_type = step_type
        self.trace_id = trace_id or get_current_trace_id() or "untraced"
        self.parent_span_id = get_current_span_id()
        self.span_id = generate_span_id()
        self.inputs = inputs or {}
        self.tags = {**get_current_tags(), **(tags or {})}
        self.metrics: dict[str, Any] = {}
        self.outputs: dict[str, Any] = {}
        self.error: Optional[dict[str, Any]] = None
        self.status = SpanStatus.RUNNING.value
        self.ttft_ms: Optional[float] = None
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration_ms: float = 0.0
        self._scope_token = None

    def set_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def set_output(self, key: str, value: Any) -> None:
        self.outputs[key] = value

    def set_ttft(self, ttft_ms: float) -> None:
        self.ttft_ms = round(ttft_ms, 2)

    def __enter__(self) -> TraceSpanContext:
        self.start_time = time.time()
        self._scope = span_scope(self.span_id)
        self._scope.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)

        if exc_val is not None:
            self.status = SpanStatus.ERROR.value
            self.error = {
                "type": exc_type.__name__ if exc_type else "Exception",
                "message": str(exc_val),
                "traceback": traceback.format_exc(),
            }
        else:
            self.status = SpanStatus.SUCCESS.value

        try:
            self._scope.__exit__(exc_type, exc_val, exc_tb)
        except Exception:
            pass

        self._flush()

    async def __aenter__(self) -> TraceSpanContext:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)

    def _flush(self) -> None:
        try:
            record = SpanRecord(
                span_id=self.span_id,
                trace_id=self.trace_id,
                parent_span_id=self.parent_span_id,
                step_name=self.name,
                component=self.component,
                step_type=self.step_type,
                status=self.status,
                start_time=self.start_time,
                end_time=self.end_time,
                duration_ms=self.duration_ms,
                ttft_ms=self.ttft_ms,
                inputs=self.inputs,
                outputs=self.outputs,
                metrics=self.metrics,
                error=self.error,
                tags=self.tags,
            )
            get_monitoring_store().save_span(record)
        except Exception:
            logger.exception("Failed to flush span %s", self.span_id)


def trace_span(
    name: str,
    component: str,
    step_type: str = StepType.CUSTOM.value,
    inputs: Optional[dict[str, Any]] = None,
    tags: Optional[dict[str, Any]] = None,
) -> TraceSpanContext:
    """Create a context manager for tracking an execution span."""
    return TraceSpanContext(name=name, component=component, step_type=step_type, inputs=inputs, tags=tags)


def trace_step(
    name: Optional[str] = None,
    component: str = "app",
    step_type: str = StepType.CUSTOM.value,
    capture_args: bool = True,
    capture_result: bool = True,
):
    """Decorator to trace any sync or async function execution."""
    def decorator(fn: Callable):
        step_name = name or fn.__name__

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            inputs = {}
            if capture_args:
                try:
                    bound = inspect.signature(fn).bind_partial(*args, **kwargs)
                    inputs = {k: v for k, v in bound.arguments.items() if k not in ("self", "cls")}
                except Exception:
                    inputs = {"args_count": len(args), "kwargs_keys": list(kwargs.keys())}

            with trace_span(name=step_name, component=component, step_type=step_type, inputs=inputs) as ctx:
                result = await fn(*args, **kwargs)
                if capture_result:
                    ctx.set_output("result", result)
                return result

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            inputs = {}
            if capture_args:
                try:
                    bound = inspect.signature(fn).bind_partial(*args, **kwargs)
                    inputs = {k: v for k, v in bound.arguments.items() if k not in ("self", "cls")}
                except Exception:
                    inputs = {"args_count": len(args), "kwargs_keys": list(kwargs.keys())}

            with trace_span(name=step_name, component=component, step_type=step_type, inputs=inputs) as ctx:
                result = fn(*args, **kwargs)
                if capture_result:
                    ctx.set_output("result", result)
                return result

        return async_wrapper if inspect.iscoroutinefunction(fn) else sync_wrapper

    return decorator

