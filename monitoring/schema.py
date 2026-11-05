"""Data structures and schemas for monitoring traces and spans."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class StepType(str, Enum):
    LLM = "llm"
    RETRIEVER = "retriever"
    RERANKER = "reranker"
    GENERATOR = "generator"
    SANDBOX = "sandbox"
    TOOL = "tool"
    AGENT = "agent"
    ROUTER = "router"
    CUSTOM = "custom"


class SpanStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    RUNNING = "running"


@dataclass
class SpanRecord:
    span_id: str
    trace_id: str
    step_name: str
    component: str
    step_type: str = StepType.CUSTOM.value
    parent_span_id: Optional[str] = None
    status: str = SpanStatus.SUCCESS.value
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    ttft_ms: Optional[float] = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: Optional[dict[str, Any]] = None
    tags: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpanRecord:
        # Filter keys to match dataclass fields
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class TraceSummary:
    trace_id: str
    root_step: Optional[str] = None
    status: str = SpanStatus.SUCCESS.value
    start_time: float = 0.0
    end_time: float = 0.0
    total_duration_ms: float = 0.0
    total_spans: int = 0
    error_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    tags: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

