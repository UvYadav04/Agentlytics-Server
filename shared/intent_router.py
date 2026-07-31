from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

import numpy as np

from shared.config import get_settings
from shared.intent_classifier import CANDIDATE_LABELS, IntentResult, classify_intent
from shared.onnx_intent import embed as _onnx_embed

logger = logging.getLogger("intent_router")

_EXAMPLES_PATH_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intent_examples.json")

DEFAULT_SIMILARITY_THRESHOLD = 0.60
DEFAULT_MARGIN_THRESHOLD = 0.05


@dataclass
class RouteResult:
    query: str
    intent: str | None 
    method: str  
    top_intent: str | None  
    top_similarity: float
    second_intent: str | None
    second_similarity: float
    margin: float
    llm_result: IntentResult | None  

    latency_ms: float = 0.0
    error: str | None = None


class _ExampleIndex:
    __slots__ = ("texts", "matrix", "label_masks")

    def __init__(self, texts: list[str], matrix: np.ndarray, label_masks: dict[str, np.ndarray]):
        self.texts = texts  
        self.matrix = matrix  
        self.label_masks = label_masks


_index: _ExampleIndex | None = None


def _settings():
    return get_settings()


def _examples_path() -> str:
    configured = _settings().get("INTENT_EXAMPLES_PATH", "") or ""
    if not configured:
        return _EXAMPLES_PATH_DEFAULT
    if os.path.isabs(configured):
        return configured
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), configured)


def similarity_threshold() -> float:
    return float(_settings().get("INTENT_SIMILARITY_THRESHOLD", DEFAULT_SIMILARITY_THRESHOLD) or DEFAULT_SIMILARITY_THRESHOLD)


def margin_threshold() -> float:
    return float(_settings().get("INTENT_MARGIN_THRESHOLD", DEFAULT_MARGIN_THRESHOLD) or DEFAULT_MARGIN_THRESHOLD)


def _load_examples() -> dict[str, list[str]]:
    path = _examples_path()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    examples = {
        label: [str(p) for p in phrases if str(p).strip()]
        for label, phrases in data.items()
        if label in CANDIDATE_LABELS
    }
    missing = [label for label in CANDIDATE_LABELS if not examples.get(label)]
    if missing:
        raise ValueError(f"intent_examples config at {path!r} has no example phrases for: {missing}")
    return examples


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0 
    return matrix / norms


def _embed(texts: list[str]) -> np.ndarray:
    return _onnx_embed(texts)


def init(force: bool = False) -> bool:
   
    global _index
    if _index is not None and not force:
        return True

    try:
        examples = _load_examples()
        labels: list[str] = []
        texts: list[str] = []
        for label, phrases in examples.items():
            for phrase in phrases:
                labels.append(label)
                texts.append(phrase)

        start = time.perf_counter()
        matrix = _normalize_rows(_embed(texts))
        labels_arr = np.array(labels)
        label_masks = {label: (labels_arr == label) for label in examples}
        _index = _ExampleIndex(texts=texts, matrix=matrix, label_masks=label_masks)
        logger.info(
            "intent_router: embedded %d example phrase(s) across %d intent(s) in %.1fms (path=%s)",
            len(texts), len(examples), (time.perf_counter() - start) * 1000, _examples_path(),
        )
        return True
    except Exception:
        logger.exception("intent_router: failed to build example embedding index - embedding tier disabled")
        _index = None
        return False


def _per_intent_scores(similarities: np.ndarray) -> dict[str, float]:
    return {label: float(similarities[mask].max()) for label, mask in _index.label_masks.items()}


def _top_two(intent_scores: dict[str, float]) -> tuple[str, float, str, float]:
    ranked = sorted(intent_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_intent, top_score = ranked[0]
    second_intent, second_score = ranked[1] if len(ranked) > 1 else ranked[0]
    return top_intent, top_score, second_intent, second_score


def _is_confident(top_similarity: float, margin: float) -> bool:
    return top_similarity >= similarity_threshold() and margin >= margin_threshold()


def _llm_fallback(query: str, timeout: float, embedding_fields: dict) -> RouteResult:
    llm_result = classify_intent(query, timeout=timeout)
    return RouteResult(
        query=query,
        intent=llm_result.top_label,    
        method="llm" if llm_result.top_label else "none",
        llm_result=llm_result,
        error=llm_result.error,
        **embedding_fields,
    )


def route_query_intent(query: str, timeout: float = 30.0, llm_fallback: bool = True) -> RouteResult:
    start = time.perf_counter()
    no_embedding_fields = {
        "top_intent": None, "top_similarity": 0.0,
        "second_intent": None, "second_similarity": 0.0, "margin": 0.0,
    }

    def _unresolved(error: str | None) -> RouteResult:
        if llm_fallback:
            result = _llm_fallback(query, timeout, no_embedding_fields)
            if error:
                result.error = error
            return result
        return RouteResult(
            query=query, intent=None, method="none", llm_result=None, error=error,
            **no_embedding_fields,
        )

    if _index is None and not init():
        return _finish(_unresolved(None), start)

    try:
        query_vec = _normalize_rows(_embed([query]))[0]
        similarities = _index.matrix @ query_vec  # vectorized cosine similarity, both sides pre-normalized
        intent_scores = _per_intent_scores(similarities)
        top_intent, top_similarity, second_intent, second_similarity = _top_two(intent_scores)
        margin = top_similarity - second_similarity
    except Exception as exc:
        logger.warning("intent_router: embedding call failed (%s)", exc)
        return _finish(_unresolved(str(exc)), start)

    embedding_fields = {
        "top_intent": top_intent, "top_similarity": top_similarity,
        "second_intent": second_intent, "second_similarity": second_similarity, "margin": margin,
    }

    if _is_confident(top_similarity, margin):
        result = RouteResult(
            query=query, intent=top_intent, method="embedding", llm_result=None, error=None,
            **embedding_fields,
        )
        return _finish(result, start)

    if llm_fallback:
        result = _llm_fallback(query, timeout, embedding_fields)
    else:
        result = RouteResult(
            query=query, intent=top_intent, method="none", llm_result=None, error=None,
            **embedding_fields,
        )
    return _finish(result, start)


def route_query_intent_fast(query: str) -> RouteResult:
    return route_query_intent(query, llm_fallback=False)


def _finish(result: RouteResult, start: float) -> RouteResult:
    result.latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "intent_router: query=%r method=%s final_intent=%s top_intent=%s top_similarity=%.3f "
        "second_intent=%s second_similarity=%.3f margin=%.3f latency_ms=%.1f%s",
        result.query, result.method, result.intent, result.top_intent, result.top_similarity,
        result.second_intent, result.second_similarity, result.margin, result.latency_ms,
        f" llm_intent={result.llm_result.top_label} llm_confidence={result.llm_result.top_score:.3f}"
        if result.llm_result is not None else "",
    )
    return result
