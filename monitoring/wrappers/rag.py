"""RAG Tracing wrappers for Retriever, Reranker, and Generator.

Captures input queries, retrieved chunks with scores, reranked orders,
context chunks used by generator, chunk counts, and latencies.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, Optional, Sequence

from monitoring.sanitizer import safe_serialize
from monitoring.schema import StepType
from monitoring.wrappers.base import trace_span

logger = logging.getLogger("monitoring.wrappers.rag")


def _format_chunk(chunk: Any, rank: Optional[int] = None) -> dict[str, Any]:
    """Extract standard summary from chunk objects (Chroma ChunkRecord, dict, etc.)."""
    chunk_id = getattr(chunk, "chunk_id", None) or getattr(chunk, "id", None)
    text = getattr(chunk, "text", None) or getattr(chunk, "page_content", None) or str(chunk)
    meta = getattr(chunk, "metadata", None) or {}
    score = getattr(chunk, "score", None) or getattr(chunk, "distance", None)

    summary = {
        "chunk_id": chunk_id,
        "text_preview": (text[:250] + "...") if len(text) > 250 else text,
        "file_id": meta.get("file_id") if isinstance(meta, dict) else None,
        "score": round(float(score), 4) if score is not None else None,
    }
    if rank is not None:
        summary["rank"] = rank
    return summary


def _get_arg(bound_args: dict[str, Any], candidate_names: list[str], default: Any = None) -> Any:
    for name in candidate_names:
        if name in bound_args:
            return bound_args[name]
    return default


def trace_retriever(name: str = "retriever_query", component: str = "rag.retriever"):
    """Decorator to trace vector database retrieval steps."""
    def decorator(fn: Callable):
        sig = inspect.signature(fn)

        def _extract_inputs(args, kwargs):
            try:
                bound = sig.bind_partial(*args, **kwargs).arguments
            except Exception:
                bound = {}
            query = _get_arg(bound, ["query_text", "query", "text", "q"], "")
            top_k = _get_arg(bound, ["top_k", "k", "n_results", "limit"], None)
            filters = _get_arg(bound, ["filters", "where", "filter"], None)
            return {
                "query": safe_serialize(query),
                "top_k": top_k,
                "filters": safe_serialize(filters),
            }, top_k

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            inputs, top_k = _extract_inputs(args, kwargs)
            with trace_span(name=name, component=component, step_type=StepType.RETRIEVER.value, inputs=inputs) as span:
                chunks = await fn(*args, **kwargs)
                chunk_list = list(chunks) if isinstance(chunks, (list, tuple)) else []
                span.set_metric("chunks_retrieved", len(chunk_list))
                if top_k is not None:
                    span.set_metric("top_k", top_k)
                span.set_output("chunk_count", len(chunk_list))
                span.set_output("chunks", [_format_chunk(c) for c in chunk_list[:25]])
                return chunks

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            inputs, top_k = _extract_inputs(args, kwargs)
            with trace_span(name=name, component=component, step_type=StepType.RETRIEVER.value, inputs=inputs) as span:
                chunks = fn(*args, **kwargs)
                chunk_list = list(chunks) if isinstance(chunks, (list, tuple)) else []
                span.set_metric("chunks_retrieved", len(chunk_list))
                if top_k is not None:
                    span.set_metric("top_k", top_k)
                span.set_output("chunk_count", len(chunk_list))
                span.set_output("chunks", [_format_chunk(c) for c in chunk_list[:25]])
                return chunks

        return async_wrapper if inspect.iscoroutinefunction(fn) else sync_wrapper

    return decorator


def trace_reranker(name: str = "rerank_chunks", component: str = "rag.reranker", model_name: Optional[str] = None):
    """Decorator to trace chunk reranking operations."""
    def decorator(fn: Callable):
        sig = inspect.signature(fn)

        def _extract_inputs(args, kwargs):
            try:
                bound = sig.bind_partial(*args, **kwargs).arguments
            except Exception:
                bound = {}
            query = _get_arg(bound, ["query", "query_text", "text"], "")
            chunks = _get_arg(bound, ["chunks", "docs", "documents"], [])
            top_k = _get_arg(bound, ["top_k", "k", "limit"], None)
            model = model_name or getattr(args[0], "model_name", "reranker") if args else "reranker"
            return {
                "query": safe_serialize(query),
                "chunks_in_count": len(chunks) if isinstance(chunks, (list, tuple)) else 0,
                "top_k": top_k,
                "model": model,
            }, chunks, model

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            inputs, chunks, model = _extract_inputs(args, kwargs)
            tags = {"model": model}
            with trace_span(name=name, component=component, step_type=StepType.RERANKER.value, inputs=inputs, tags=tags) as span:
                ranked = await fn(*args, **kwargs)
                ranked_list = list(ranked) if isinstance(ranked, (list, tuple)) else []
                span.set_metric("chunks_in", len(chunks) if isinstance(chunks, (list, tuple)) else 0)
                span.set_metric("chunks_out", len(ranked_list))
                span.set_output("ranked_count", len(ranked_list))
                span.set_output("ranked_chunks", [_format_chunk(c, rank=i + 1) for i, c in enumerate(ranked_list[:25])])
                return ranked

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            inputs, chunks, model = _extract_inputs(args, kwargs)
            tags = {"model": model}
            with trace_span(name=name, component=component, step_type=StepType.RERANKER.value, inputs=inputs, tags=tags) as span:
                ranked = fn(*args, **kwargs)
                ranked_list = list(ranked) if isinstance(ranked, (list, tuple)) else []
                span.set_metric("chunks_in", len(chunks) if isinstance(chunks, (list, tuple)) else 0)
                span.set_metric("chunks_out", len(ranked_list))
                span.set_output("ranked_count", len(ranked_list))
                span.set_output("ranked_chunks", [_format_chunk(c, rank=i + 1) for i, c in enumerate(ranked_list[:25])])
                return ranked

        return async_wrapper if inspect.iscoroutinefunction(fn) else sync_wrapper

    return decorator


def trace_generator(name: str = "rag_generate", component: str = "rag.generator"):
    """Decorator to trace RAG synthesis / generation calls."""
    def decorator(fn: Callable):
        sig = inspect.signature(fn)

        def _extract_inputs(args, kwargs):
            try:
                bound = sig.bind_partial(*args, **kwargs).arguments
            except Exception:
                bound = {}
            prompt = _get_arg(bound, ["prompt", "query", "text", "question"], "")
            context = _get_arg(bound, ["context", "chunks", "docs"], [])
            return {
                "prompt": safe_serialize(prompt),
                "context_chunks_count": len(context) if isinstance(context, (list, tuple)) else 0,
            }

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            inputs = _extract_inputs(args, kwargs)
            with trace_span(name=name, component=component, step_type=StepType.GENERATOR.value, inputs=inputs) as span:
                result = await fn(*args, **kwargs)
                span.set_output("answer", safe_serialize(result))
                return result

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            inputs = _extract_inputs(args, kwargs)
            with trace_span(name=name, component=component, step_type=StepType.GENERATOR.value, inputs=inputs) as span:
                result = fn(*args, **kwargs)
                span.set_output("answer", safe_serialize(result))
                return result

        return async_wrapper if inspect.iscoroutinefunction(fn) else sync_wrapper

    return decorator
