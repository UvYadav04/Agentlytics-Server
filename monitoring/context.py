"""Context propagation for monitoring using Python contextvars.

Supports nested spans, ambient trace IDs (e.g. investigation_id),
and metadata tags across async/sync execution boundaries.
"""

from __future__ import annotations

import contextvars
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Generator, Optional

_current_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "monitoring_current_trace_id", default=None
)
_current_span_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "monitoring_current_span_id", default=None
)
_current_trace_tags: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "monitoring_current_trace_tags", default={}
)


def generate_span_id() -> str:
    """Generate a unique step/span identifier."""
    return f"spn_{uuid.uuid4().hex[:12]}"


def generate_trace_id() -> str:
    """Generate a unique trace identifier."""
    return f"trc_{uuid.uuid4().hex[:16]}"


def get_current_trace_id() -> Optional[str]:
    """Retrieve the active trace ID (e.g. investigation_id)."""
    return _current_trace_id.get()


def get_current_span_id() -> Optional[str]:
    """Retrieve the active parent span ID."""
    return _current_span_id.get()


def get_current_tags() -> dict[str, Any]:
    """Retrieve active trace tags."""
    return dict(_current_trace_tags.get())


def set_trace_id(trace_id: str) -> contextvars.Token:
    """Set the active trace ID directly."""
    return _current_trace_id.set(trace_id)


def set_span_id(span_id: Optional[str]) -> contextvars.Token:
    """Set the active span ID directly."""
    return _current_span_id.set(span_id)


def reset_trace_id(token: contextvars.Token) -> None:
    _current_trace_id.reset(token)


def reset_span_id(token: contextvars.Token) -> None:
    _current_span_id.reset(token)


@contextmanager
def trace_scope(trace_id: str, tags: Optional[dict[str, Any]] = None) -> Generator[str, None, None]:
    """Sync context manager for a trace boundary (e.g. around an investigation)."""
    t_token = _current_trace_id.set(trace_id)
    merged_tags = {**_current_trace_tags.get(), **(tags or {})}
    tags_token = _current_trace_tags.set(merged_tags)
    s_token = _current_span_id.set(None)
    try:
        yield trace_id
    finally:
        _current_trace_id.reset(t_token)
        _current_trace_tags.reset(tags_token)
        _current_span_id.reset(s_token)


@asynccontextmanager
async def async_trace_scope(trace_id: str, tags: Optional[dict[str, Any]] = None) -> AsyncGenerator[str, None]:
    """Async context manager for a trace boundary."""
    t_token = _current_trace_id.set(trace_id)
    merged_tags = {**_current_trace_tags.get(), **(tags or {})}
    tags_token = _current_trace_tags.set(merged_tags)
    s_token = _current_span_id.set(None)
    try:
        yield trace_id
    finally:
        _current_trace_id.reset(t_token)
        _current_trace_tags.reset(tags_token)
        _current_span_id.reset(s_token)


@contextmanager
def span_scope(span_id: str) -> Generator[str, None, None]:
    """Make the given span_id the current parent for any nested operations."""
    token = _current_span_id.set(span_id)
    try:
        yield span_id
    finally:
        _current_span_id.reset(token)


@asynccontextmanager
async def async_span_scope(span_id: str) -> AsyncGenerator[str, None]:
    """Async variant to make span_id the active parent."""
    token = _current_span_id.set(span_id)
    try:
        yield span_id
    finally:
        _current_span_id.reset(token)

