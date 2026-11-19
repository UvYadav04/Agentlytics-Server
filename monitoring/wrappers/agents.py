"""Agent workflow tracing wrapper.

Captures agent task objectives, constraints, tool invocation sequences,
iterations count, and final answers produced.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, Optional

from monitoring.context import get_current_trace_id
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
            investigation_id = (
                kwargs.get("investigation_id")
                or kwargs.get("trace_id")
                or (constraints if isinstance(constraints, dict) else {}).get("investigation_id")
                or get_current_trace_id()
            )

            self_obj = args[0] if args and hasattr(args[0], "__class__") and type(args[0]).__name__ not in ("type", "str", "int", "dict") else None
            assigned_files = getattr(self_obj, "assigned_files", None)
            connections = getattr(self_obj, "available_connections", None) or getattr(self_obj, "database_connections", None)
            conn_id = kwargs.get("connection_id") or getattr(self_obj, "connection_id", None)
            direct_route = getattr(self_obj, "direct_route", False)

            inputs = {
                "objective": safe_serialize(objective),
                "constraints": safe_serialize(constraints),
            }
            if assigned_files:
                inputs["assigned_files"] = safe_serialize(assigned_files)
            if connections:
                inputs["database_connections"] = safe_serialize(connections)
            if conn_id:
                inputs["connection_id"] = conn_id
            if direct_route:
                inputs["direct_route"] = True

            tags = {"objective": str(objective)[:250]} if objective else {}

            with trace_span(
                name=agent_name,
                component=component,
                step_type=StepType.AGENT.value,
                trace_id=investigation_id,
                inputs=inputs,
                tags=tags,
            ) as span:
                result = await fn(*args, **kwargs)
                span.set_output("result", safe_serialize(result))
                return result

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            objective = kwargs.get("objective") or kwargs.get("task") or (args[1] if len(args) > 1 else "")
            constraints = kwargs.get("constraints") or {}
            investigation_id = (
                kwargs.get("investigation_id")
                or kwargs.get("trace_id")
                or (constraints if isinstance(constraints, dict) else {}).get("investigation_id")
                or get_current_trace_id()
            )

            self_obj = args[0] if args and hasattr(args[0], "__class__") and type(args[0]).__name__ not in ("type", "str", "int", "dict") else None
            assigned_files = getattr(self_obj, "assigned_files", None)
            connections = getattr(self_obj, "available_connections", None) or getattr(self_obj, "database_connections", None)
            conn_id = kwargs.get("connection_id") or getattr(self_obj, "connection_id", None)
            direct_route = getattr(self_obj, "direct_route", False)

            inputs = {
                "objective": safe_serialize(objective),
                "constraints": safe_serialize(constraints),
            }
            if assigned_files:
                inputs["assigned_files"] = safe_serialize(assigned_files)
            if connections:
                inputs["database_connections"] = safe_serialize(connections)
            if conn_id:
                inputs["connection_id"] = conn_id
            if direct_route:
                inputs["direct_route"] = True

            tags = {"objective": str(objective)[:250]} if objective else {}

            with trace_span(
                name=agent_name,
                component=component,
                step_type=StepType.AGENT.value,
                trace_id=investigation_id,
                inputs=inputs,
                tags=tags,
            ) as span:
                result = fn(*args, **kwargs)
                span.set_output("result", safe_serialize(result))
                return result

        return async_wrapper if inspect.iscoroutinefunction(fn) else sync_wrapper

    return decorator

