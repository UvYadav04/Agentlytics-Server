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


def trace_tool(name: Optional[str] = None, component: str = "agent.tool"):
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

            with trace_span(name=tool_name, component=component, step_type=StepType.TOOL.value, inputs=inputs) as span:
                result = await fn(*args, **kwargs)
                span.set_output("output", safe_serialize(result))
                return result

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            try:
                bound = inspect.signature(fn).bind_partial(*args, **kwargs)
                inputs = {k: safe_serialize(v) for k, v in bound.arguments.items() if k not in ("self", "cls")}
            except Exception:
                inputs = {"args_count": len(args), "kwargs_keys": list(kwargs.keys())}

            with trace_span(name=tool_name, component=component, step_type=StepType.TOOL.value, inputs=inputs) as span:
                result = fn(*args, **kwargs)
                span.set_output("output", safe_serialize(result))
                return result

        return async_wrapper if inspect.iscoroutinefunction(fn) else sync_wrapper

    return decorator
