"""Unified Monitoring and Observability system for AI pipelines.

Provides fine-grained, semantically aware tracing for LLMs, RAG (retrievers,
rerankers, generators), agents, tools, and code sandboxes.
"""

from monitoring.context import (
    async_span_scope,
    async_trace_scope,
    generate_span_id,
    generate_trace_id,
    get_current_span_id,
    get_current_tags,
    get_current_trace_id,
    set_span_id,
    set_trace_id,
    span_scope,
    trace_scope,
)
from monitoring.schema import SpanRecord, SpanStatus, StepType, TraceSummary
from monitoring.storage import MonitoringStore, get_monitoring_store
from monitoring.wrappers.agents import trace_agent
from monitoring.wrappers.base import TraceSpanContext, trace_span, trace_step
from monitoring.wrappers.llm import (
    MonitoringTracedChatCompletionClient,
    trace_llm,
)
from monitoring.wrappers.rag import (
    trace_generator,
    trace_reranker,
    trace_retriever,
)
from monitoring.wrappers.sandbox import trace_sandbox
from monitoring.wrappers.tools import trace_tool

__all__ = [
    # Context
    "trace_scope",
    "async_trace_scope",
    "span_scope",
    "async_span_scope",
    "get_current_trace_id",
    "get_current_span_id",
    "get_current_tags",
    "set_trace_id",
    "set_span_id",
    "generate_span_id",
    "generate_trace_id",
    # Schema
    "SpanRecord",
    "StepType",
    "SpanStatus",
    "TraceSummary",
    # Storage
    "MonitoringStore",
    "get_monitoring_store",
    # Wrappers
    "trace_step",
    "trace_span",
    "TraceSpanContext",
    "trace_llm",
    "MonitoringTracedChatCompletionClient",
    "trace_retriever",
    "trace_reranker",
    "trace_generator",
    "trace_sandbox",
    "trace_tool",
    "trace_agent",
]

