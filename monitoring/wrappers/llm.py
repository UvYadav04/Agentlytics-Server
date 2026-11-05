"""LLM Tracing wrappers and AutoGen ChatCompletionClient interceptor.

Captures prompts, completions, tokens (prompt, completion, total),
TTFT (time-to-first-token for streaming), latency, and tool invocations.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import time
from typing import Any, Callable, Optional, Sequence

from monitoring.context import get_current_span_id, get_current_trace_id
from monitoring.sanitizer import safe_serialize
from monitoring.schema import StepType
from monitoring.wrappers.base import trace_span

logger = logging.getLogger("monitoring.wrappers.llm")


def _get_arg(bound_args: dict[str, Any], candidate_names: list[str], default: Any = None) -> Any:
    for name in candidate_names:
        if name in bound_args:
            return bound_args[name]
    return default


def _serialize_messages(messages: Any) -> list[dict[str, Any]]:
    """Converts diverse message objects (OpenAI, AutoGen, dicts) into standard dicts."""
    out = []
    if not messages:
        return out
    if isinstance(messages, str):
        return [{"role": "user", "content": safe_serialize(messages)}]
    for m in messages:
        if isinstance(m, dict):
            out.append(m)
        else:
            role = getattr(m, "type", None) or getattr(m, "role", None) or type(m).__name__
            content = getattr(m, "content", None)
            if not isinstance(content, (str, list, dict, type(None))):
                content = str(content)
            out.append({"role": role, "content": safe_serialize(content)})
    return out


def _serialize_tools(tools: Any) -> list[dict[str, Any]]:
    out = []
    for t in tools or []:
        try:
            schema = t.schema if hasattr(t, "schema") else t
            if isinstance(schema, dict):
                out.append({
                    "name": schema.get("name"),
                    "description": schema.get("description"),
                    "parameters": schema.get("parameters"),
                })
            else:
                out.append({"name": getattr(t, "name", str(t))})
        except Exception:
            out.append({"name": getattr(t, "name", str(t))})
    return out


def _extract_usage(result: Any) -> dict[str, int]:
    """Extract token counts from various SDK response objects (OpenAI, AutoGen, etc.)."""
    usage = getattr(result, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    prompt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0
    total = getattr(usage, "total_tokens", None) or (prompt + completion)
    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(total),
    }


def trace_llm(name: str = "llm_call", provider: Optional[str] = None, model: Optional[str] = None):
    """Decorator for standalone LLM calls (e.g. OpenAI / Groq SDK calls)."""
    def decorator(fn: Callable):
        sig = inspect.signature(fn)

        def _extract_inputs(args, kwargs):
            try:
                bound = sig.bind_partial(*args, **kwargs).arguments
            except Exception:
                bound = {}
            messages = _get_arg(bound, ["messages", "prompt", "input"], [])
            call_model = model or _get_arg(bound, ["model", "model_name"], "unknown")
            call_provider = provider or _get_arg(bound, ["provider", "provider_name"], "unknown")
            temperature = _get_arg(bound, ["temperature", "temp"], None)
            return {
                "messages": _serialize_messages(messages),
                "model": call_model,
                "provider": call_provider,
                "temperature": temperature,
            }, call_provider, call_model

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            inputs, call_provider, call_model = _extract_inputs(args, kwargs)
            tags = {"provider": call_provider, "model": call_model}

            with trace_span(name=name, component=f"llm.{call_provider}", step_type=StepType.LLM.value, inputs=inputs, tags=tags) as span:
                result = await fn(*args, **kwargs)
                usage = _extract_usage(result)
                span.set_metric("prompt_tokens", usage["prompt_tokens"])
                span.set_metric("completion_tokens", usage["completion_tokens"])
                span.set_metric("total_tokens", usage["total_tokens"])

                content = getattr(result, "content", None)
                if content is None and hasattr(result, "choices") and result.choices:
                    content = getattr(result.choices[0].message, "content", None)
                span.set_output("completion", safe_serialize(content or str(result)))
                return result

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            inputs, call_provider, call_model = _extract_inputs(args, kwargs)
            tags = {"provider": call_provider, "model": call_model}

            with trace_span(name=name, component=f"llm.{call_provider}", step_type=StepType.LLM.value, inputs=inputs, tags=tags) as span:
                result = fn(*args, **kwargs)
                usage = _extract_usage(result)
                span.set_metric("prompt_tokens", usage["prompt_tokens"])
                span.set_metric("completion_tokens", usage["completion_tokens"])
                span.set_metric("total_tokens", usage["total_tokens"])

                content = getattr(result, "content", None)
                if content is None and hasattr(result, "choices") and result.choices:
                    content = getattr(result.choices[0].message, "content", None)
                span.set_output("completion", safe_serialize(content or str(result)))
                return result

        return async_wrapper if inspect.iscoroutinefunction(fn) else sync_wrapper

    return decorator


class MonitoringTracedChatCompletionClient:
    """Wrapper around AutoGen ChatCompletionClient to trace all agent LLM calls."""

    def __init__(self, inner: Any, provider_name: str, model: Optional[str] = None):
        self._inner = inner
        self._provider_name = provider_name
        self._model = model

    async def create(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                      extra_create_args={}, cancellation_token=None):
        inputs = {
            "messages": _serialize_messages(messages),
            "tools": _serialize_tools(tools),
            "tool_choice": str(tool_choice),
            "model": self._model or self._provider_name,
            "provider": self._provider_name,
        }
        tags = {
            "provider": self._provider_name,
            "model": self._model or "default",
            "tool_count": len(tools or []),
        }

        with trace_span(
            name=f"{self._provider_name}.create",
            component=f"llm.{self._provider_name}",
            step_type=StepType.LLM.value,
            inputs=inputs,
            tags=tags,
        ) as span:
            result = await self._inner.create(
                messages, tools=tools, tool_choice=tool_choice, json_output=json_output,
                extra_create_args=extra_create_args, cancellation_token=cancellation_token,
            )

            usage = _extract_usage(result)
            span.set_metric("prompt_tokens", usage["prompt_tokens"])
            span.set_metric("completion_tokens", usage["completion_tokens"])
            span.set_metric("total_tokens", usage["total_tokens"])

            tool_calls = [
                c for c in (result.content if isinstance(result.content, list) else [])
                if hasattr(c, "name") or hasattr(c, "arguments")
            ]

            span.set_output("content", safe_serialize(result.content))
            span.set_output("finish_reason", getattr(result, "finish_reason", None))
            if tool_calls:
                span.set_output("tool_calls", safe_serialize(tool_calls))

            return result

    async def create_stream(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                             extra_create_args={}, cancellation_token=None):
        inputs = {
            "messages": _serialize_messages(messages),
            "tools": _serialize_tools(tools),
            "tool_choice": str(tool_choice),
            "model": self._model or self._provider_name,
            "provider": self._provider_name,
            "streaming": True,
        }
        tags = {
            "provider": self._provider_name,
            "model": self._model or "default",
            "tool_count": len(tools or []),
            "streaming": True,
        }

        with trace_span(
            name=f"{self._provider_name}.create_stream",
            component=f"llm.{self._provider_name}",
            step_type=StepType.LLM.value,
            inputs=inputs,
            tags=tags,
        ) as span:
            start_stream = time.perf_counter()
            first_chunk_received = False
            last_chunk = None
            chunk_count = 0

            async for chunk in self._inner.create_stream(
                messages, tools=tools, tool_choice=tool_choice, json_output=json_output,
                extra_create_args=extra_create_args, cancellation_token=cancellation_token,
            ):
                if not first_chunk_received:
                    first_chunk_received = True
                    ttft_ms = (time.perf_counter() - start_stream) * 1000
                    span.set_ttft(ttft_ms)

                chunk_count += 1
                last_chunk = chunk
                yield chunk

            usage = _extract_usage(last_chunk)
            span.set_metric("prompt_tokens", usage["prompt_tokens"])
            span.set_metric("completion_tokens", usage["completion_tokens"])
            span.set_metric("total_tokens", usage["total_tokens"])
            span.set_metric("chunk_count", chunk_count)
            span.set_output("content", safe_serialize(getattr(last_chunk, "content", last_chunk)))

    async def close(self) -> None:
        await self._inner.close()

    def actual_usage(self):
        return self._inner.actual_usage()

    def total_usage(self):
        return self._inner.total_usage()

    def count_tokens(self, messages: Sequence[Any], *, tools: Sequence[Any] = []) -> int:
        return self._inner.count_tokens(messages, tools=tools)

    def remaining_tokens(self, messages: Sequence[Any], *, tools: Sequence[Any] = []) -> int:
        return self._inner.remaining_tokens(messages, tools=tools)

    @property
    def capabilities(self):
        return self._inner.capabilities

    @property
    def model_info(self):
        return self._inner.model_info
