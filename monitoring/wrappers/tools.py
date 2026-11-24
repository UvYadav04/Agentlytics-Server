"""Tool execution tracing wrapper for agent tools and external functions."""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, Optional

from monitoring.sanitizer import safe_serialize
from monitoring.schema import StepType
from monitoring.wrappers.base import trace_span

logger = logging.getLogger("monitoring.wrappers.tools")


def _record_tool_metrics(span: Any, result: Any) -> None:
    from monitoring.schema import SpanStatus

    if isinstance(result, dict):
        if "execution_time_ms" in result:
            span.set_metric("db_execution_time_ms", result["execution_time_ms"])
        if "row_count" in result:
            span.set_metric("row_count", result["row_count"])
        if "success" in result:
            span.set_metric("success", result["success"])
            if result["success"] is False:
                span.status = SpanStatus.ERROR.value
                span.error = {
                    "type": "ToolExecutionError",
                    "message": str(result.get("error") or "Tool execution failed"),
                }
        elif result.get("error"):
            span.status = SpanStatus.ERROR.value
            span.error = {
                "type": "ToolExecutionError",
                "message": str(result.get("error")),
            }

        if "chart_id" in result:
            span.set_metric("chart_id", result["chart_id"])
        elif isinstance(result.get("specification"), dict) and "chart_id" in result["specification"]:
            span.set_metric("chart_id", result["specification"]["chart_id"])
        if "tables" in result and isinstance(result["tables"], list):
            span.set_metric("table_count", len(result["tables"]))
        if "exit_code" in result:
            span.set_metric("exit_code", result["exit_code"])
            if result["exit_code"] != 0:
                span.status = SpanStatus.ERROR.value
                span.error = {
                    "type": "SandboxExecutionError",
                    "message": str(result.get("error") or result.get("stderr") or f"Command failed with exit code {result['exit_code']}"),
                }
    elif hasattr(result, "error") and getattr(result, "error", None):
        span.status = SpanStatus.ERROR.value
        span.error = {
            "type": "ToolExecutionError",
            "message": str(getattr(result, "error")),
        }


def trace_tool(
    name: Optional[str] = None,
    component: str = "agent.tool",
    step_type: str = StepType.TOOL.value,
    tags: Optional[dict[str, Any]] = None,
):
    """Decorator to trace tool execution invoked by agents."""
    def decorator(fn: Callable):
        tool_name = name or fn.__name__

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            try:
                bound = inspect.signature(fn).bind_partial(*args, **kwargs)
                inputs = {k: safe_serialize(v) for k, v in bound.arguments.items() if k not in ("self", "cls")}
            except Exception:
                inputs = {"args_count": len(args), "kwargs_keys": list(kwargs.keys())}

            with trace_span(name=tool_name, component=component, step_type=step_type, inputs=inputs, tags=tags) as span:
                result = await fn(*args, **kwargs)
                _record_tool_metrics(span, result)
                span.set_output("output", safe_serialize(result))
                return result

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            try:
                bound = inspect.signature(fn).bind_partial(*args, **kwargs)
                inputs = {k: safe_serialize(v) for k, v in bound.arguments.items() if k not in ("self", "cls")}
            except Exception:
                inputs = {"args_count": len(args), "kwargs_keys": list(kwargs.keys())}

            with trace_span(name=tool_name, component=component, step_type=step_type, inputs=inputs, tags=tags) as span:
                result = fn(*args, **kwargs)
                _record_tool_metrics(span, result)
                span.set_output("output", safe_serialize(result))
                return result

        return async_wrapper if inspect.iscoroutinefunction(fn) else sync_wrapper

    return decorator

