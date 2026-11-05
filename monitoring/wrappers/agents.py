"""Agent workflow tracing wrapper.

Captures agent task objectives, constraints, tool invocation sequences,
iterations count, and final answers produced.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, Optional

from monitoring.sanitizer import safe_serialize
from monitoring.schema import StepType
from monitoring.wrappers.base import trace_span

logger = logging.getLogger("monitoring.wrappers.agents")


def trace_agent(name: Optional[str] = None, component: str = "agent"):
    """Decorator to trace high-level agent execution cycles."""
    def decorator(fn: Callable):
        agent_name = name or fn.__name__

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            objective = kwargs.get("objective") or kwargs.get("task") or (args[1] if len(args) > 1 else "")
            constraints = kwargs.get("constraints") or {}

            inputs = {
                "objective": safe_serialize(objective),
                "constraints": safe_serialize(constraints),
            }

            with trace_span(name=agent_name, component=component, step_type=StepType.AGENT.value, inputs=inputs) as span:
                result = await fn(*args, **kwargs)
                span.set_output("result", safe_serialize(result))
                return result

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            objective = kwargs.get("objective") or kwargs.get("task") or (args[1] if len(args) > 1 else "")
            constraints = kwargs.get("constraints") or {}

            inputs = {
                "objective": safe_serialize(objective),
                "constraints": safe_serialize(constraints),
            }

            with trace_span(name=agent_name, component=component, step_type=StepType.AGENT.value, inputs=inputs) as span:
                result = fn(*args, **kwargs)
                span.set_output("result", safe_serialize(result))
                return result

        return async_wrapper if inspect.iscoroutinefunction(fn) else sync_wrapper

    return decorator
