"""Sandbox execution tracing wrappers.

Captures Python code executed, tables injected, timeout settings,
stdout, stderr, exit code, chart artifacts generated, and execution latency.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, Optional

from monitoring.sanitizer import safe_serialize
from monitoring.schema import StepType
from monitoring.wrappers.base import trace_span

logger = logging.getLogger("monitoring.wrappers.sandbox")


def _get_arg(bound_args: dict[str, Any], candidate_names: list[str], default: Any = None) -> Any:
    for name in candidate_names:
        if name in bound_args:
            return bound_args[name]
    return default


def trace_sandbox(name: str = "sandbox_execute", component: str = "sandbox.docker"):
    """Decorator for sandbox code execution calls."""
    def decorator(fn: Callable):
        sig = inspect.signature(fn)

        def _extract_inputs(args, kwargs):
            try:
                bound = sig.bind_partial(*args, **kwargs).arguments
            except Exception:
                bound = {}
            code = _get_arg(bound, ["code", "script", "command"], "")
            tables = _get_arg(bound, ["tables", "data", "dfs"], {})
            workspace_id = _get_arg(bound, ["workspace_id", "workspace"], None)
            timeout = _get_arg(bound, ["timeout_seconds", "timeout"], None)

            return {
                "code": safe_serialize(code),
                "tables_count": len(tables) if isinstance(tables, dict) else 0,
                "table_names": list(tables.keys()) if isinstance(tables, dict) else [],
                "workspace_id": workspace_id,
                "timeout_seconds": timeout,
            }, workspace_id

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            inputs, workspace_id = _extract_inputs(args, kwargs)
            tags = {"workspace_id": workspace_id} if workspace_id else {}

            with trace_span(name=name, component=component, step_type=StepType.SANDBOX.value, inputs=inputs, tags=tags) as span:
                result = await fn(*args, **kwargs)
                _record_sandbox_result(span, result)
                return result

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            inputs, workspace_id = _extract_inputs(args, kwargs)
            tags = {"workspace_id": workspace_id} if workspace_id else {}

            with trace_span(name=name, component=component, step_type=StepType.SANDBOX.value, inputs=inputs, tags=tags) as span:
                result = fn(*args, **kwargs)
                _record_sandbox_result(span, result)
                return result

        return async_wrapper if inspect.iscoroutinefunction(fn) else sync_wrapper

    return decorator


def _record_sandbox_result(span: Any, result: Any) -> None:
    if not isinstance(result, dict):
        span.set_output("result", safe_serialize(result))
        return

    stdout = result.get("stdout") or result.get("output")
    stderr = result.get("stderr") or result.get("error")
    exit_code = result.get("exit_code") or result.get("status_code") or (0 if result.get("status") == "ok" else 1)
    artifacts = result.get("artifacts") or result.get("files") or []

    span.set_metric("exit_code", exit_code)
    span.set_metric("artifact_count", len(artifacts) if isinstance(artifacts, list) else 0)

    span.set_output("exit_code", exit_code)
    span.set_output("stdout", safe_serialize(stdout))
    span.set_output("stderr", safe_serialize(stderr))
    if artifacts:
        span.set_output("artifacts", safe_serialize(artifacts))
