"""Hybrid intent router: embedding-similarity first, LLM fallback only when unsure.

Replaces shared/intent_classifier.py's classify_intent as api_service/routers/chats.py's real
routing decision (see that module's docstring - still called synchronously, in the foreground,
before the arq job is enqueued, so the route is known before worker_service ever picks the job
up). The old "always call the LLM" classifier took ~5s on every single message, even for
"hi" or "show my sales" - obviously not ambiguous. This module fixes that:

    query -> embed -> cosine similarity against every example embedding -> confident? -> route
                                                                        -> not confident? -> LLM

Same four labels either way (see shared/intent_classifier.CANDIDATE_LABELS): greeting, tabular,
document, orchestrator.

Example phrases per label live in a JSON config file (shared/intent_examples.json by default,
see INTENT_EXAMPLES_PATH below), not in this module - routing can be improved just by adding more
representative phrases there, no code change and no retraining required.

Embeddings come from DeepInfra's OpenAI-compatible /embeddings endpoint - same provider/client
pattern shared/intent_classifier.py's LLM call already uses (the `openai` SDK pointed at
DEEPINFRA_BASE_URL), so this adds no new provider dependency. Every example embedding is
generated ONCE, at application startup (see init(), called from api_service/main.py's lifespan)
and L2-normalized up front; at runtime only the query itself gets embedded (one call), so cosine
similarity against every example collapses into a single normalized dot product
(_index.matrix @ query_vector) - vectorized, no per-example Python loop.

Confidence is judged per INTENT, not per individual example: several examples for the SAME
(correct) intent scoring similarly high is a sign of MORE confidence, not less, so similarities
are max-pooled within each intent before comparing the top intent against the runner-up intent
(see _per_intent_scores) - comparing the top-2 individual examples instead would incorrectly
punish an intent that just happens to have multiple close phrasings in its example list.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

import numpy as np
from openai import OpenAI

from shared.config import get_settings
from shared.intent_classifier import CANDIDATE_LABELS, IntentResult, classify_intent

logger = logging.getLogger("intent_router")

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

# Sub-1s budget end to end - much tighter than the LLM fallback's own 30s default, since the
# entire point of this tier is to be fast; a slow/stuck embeddings call should fail fast and let
# the caller fall back to the LLM rather than eat the latency budget this module exists to save.
DEFAULT_EMBEDDING_TIMEOUT = 5.0

_EXAMPLES_PATH_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intent_examples.json")

# Below this, the embedding tier is not trusted at all regardless of margin - a genuinely
# ambiguous/unfamiliar query should score low against every example, not just close to a tie.
DEFAULT_SIMILARITY_THRESHOLD = 0.60
# How much the winning intent must beat the runner-up intent by - a high top score with a near-
# tied runner-up (e.g. "analyze this" scoring close to both tabular and document) means the
# query is genuinely ambiguous between two intents even though it's clearly NOT a greeting.
DEFAULT_MARGIN_THRESHOLD = 0.05


@dataclass
class RouteResult:
    query: str
    intent: str | None  # final routing decision - "greeting"/"tabular"/"document"/"orchestrator", or None
    method: str  # "embedding" | "llm" | "none" (no embedding tier AND no usable LLM result)
    top_intent: str | None  # embedding tier's best-scoring intent, even when method == "llm"
    top_similarity: float
    second_intent: str | None
    second_similarity: float
    margin: float
    llm_result: IntentResult | None  # populated only when the LLM fallback actually ran
    # Both defaulted - every call site below constructs a RouteResult before the total elapsed
    # time is known, then hands it to _finish() to fill in latency_ms and log it; error defaults
    # to None for the (common) case where nothing went wrong.
    latency_ms: float = 0.0
    error: str | None = None


class _ExampleIndex:
    """Every example phrase's normalized embedding, held in memory for the life of the process -
    built once by init(), never regenerated per request (see module docstring)."""

    __slots__ = ("texts", "matrix", "label_masks")

    def __init__(self, texts: list[str], matrix: np.ndarray, label_masks: dict[str, np.ndarray]):
        self.texts = texts  # texts[i] is the example phrase behind matrix row i, for logging/debugging
        self.matrix = matrix  # shape (n_examples, dim), each row L2-normalized
        self.label_masks = label_masks  # label -> boolean mask selecting that label's rows in matrix


_index: _ExampleIndex | None = None
_client: OpenAI | None = None


def _settings():
    return get_settings()


def _embedding_model() -> str:
    return _settings().get("INTENT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL


def _embedding_timeout() -> float:
    return float(_settings().get("INTENT_EMBEDDING_TIMEOUT", DEFAULT_EMBEDDING_TIMEOUT) or DEFAULT_EMBEDDING_TIMEOUT)


def _examples_path() -> str:
    """Resolves INTENT_EXAMPLES_PATH if set, else _EXAMPLES_PATH_DEFAULT. A relative override
    (e.g. "intent_examples.json") is resolved against shared/ itself - i.e. right where this
    module lives - NOT the process's CWD, which is whatever directory the service happened to be
    started from (/app in the Docker images) and has no reason to contain this file. Without this,
    a bare relative filename in shared/.env breaks with FileNotFoundError at startup."""
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


def _get_client() -> OpenAI | None:
    """Shared across every call this process makes (not rebuilt per request) - None (not raised)
    when DEEPINFRA_API_KEY isn't configured, so callers can degrade to "embedding tier
    unavailable" the same never-crash way shared/intent_classifier.py already does."""
    global _client
    if _client is not None:
        return _client
    api_key = _settings().get("DEEPINFRA_API_KEY")
    if not api_key:
        return None
    _client = OpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL, timeout=_embedding_timeout())
    return _client


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
    norms[norms == 0] = 1.0  # guard against a degenerate all-zero embedding
    return matrix / norms


def _embed(client: OpenAI, model: str, texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(model=model, input=texts)
    # DeepInfra (like OpenAI) returns `data` in the same order as `input` and includes an
    # `index` field per item - sort by it explicitly rather than trusting list order, in case
    # that guarantee ever changes upstream.
    ordered = sorted(response.data, key=lambda item: item.index)
    return np.array([item.embedding for item in ordered], dtype=np.float32)


def init(force: bool = False) -> bool:
    """Build the in-memory example index once, at application startup (see api_service/main.py's
    lifespan). Safe to call more than once - a no-op unless force=True or nothing has been built
    yet. Returns True once the index is ready, False if it couldn't be built (no
    DEEPINFRA_API_KEY yet, a bad/missing examples config, or the embeddings call itself failing)
    - callers should treat False as "embedding tier unavailable this run", never as a reason to
    fail startup: classify() below falls straight through to the LLM classifier whenever
    `_index is None`, exactly like this feature never existed."""
    global _index
    if _index is not None and not force:
        return True

    client = _get_client()
    if client is None:
        logger.warning(
            "intent_router: DEEPINFRA_API_KEY not configured - embedding tier disabled, "
            "every query will use the LLM classifier"
        )
        return False

    try:
        examples = _load_examples()
        labels: list[str] = []
        texts: list[str] = []
        for label, phrases in examples.items():
            for phrase in phrases:
                labels.append(label)
                texts.append(phrase)

        model = _embedding_model()
        start = time.perf_counter()
        matrix = _normalize_rows(_embed(client, model, texts))
        labels_arr = np.array(labels)
        label_masks = {label: (labels_arr == label) for label in examples}
        _index = _ExampleIndex(texts=texts, matrix=matrix, label_masks=label_masks)
        logger.info(
            "intent_router: embedded %d example phrase(s) across %d intent(s) in %.1fms (model=%s, path=%s)",
            len(texts), len(examples), (time.perf_counter() - start) * 1000, model, _examples_path(),
        )
        return True
    except Exception:
        logger.exception("intent_router: failed to build example embedding index - embedding tier disabled")
        _index = None
        return False


def _per_intent_scores(similarities: np.ndarray) -> dict[str, float]:
    """Max-pool similarities within each intent (see module docstring for why max, not mean/top-
    individual-example) - cheap since there are only as many masks as CANDIDATE_LABELS (4
    today), each a vectorized boolean-index + max over up to a few hundred example rows."""
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


def route_query_intent(query: str, timeout: float = 30.0) -> RouteResult:
    """Hybrid routing decision for ONE query - embedding similarity first, LLM fallback only when
    the embedding result isn't confident (see _is_confident). Never raises: any internal failure
    (missing API key, a bad config file, a network error) degrades straight to the LLM
    classifier, same always-safe contract shared/intent_classifier.classify_intent already has.

    `timeout` is forwarded to the LLM fallback call only - the embedding call itself uses its own
    much shorter INTENT_EMBEDDING_TIMEOUT (see module docstring)."""
    start = time.perf_counter()
    no_embedding_fields = {
        "top_intent": None, "top_similarity": 0.0,
        "second_intent": None, "second_similarity": 0.0, "margin": 0.0,
    }

    if _index is None and not init():
        result = _llm_fallback(query, timeout, no_embedding_fields)
        return _finish(result, start)

    client = _get_client()
    try:
        query_vec = _normalize_rows(_embed(client, _embedding_model(), [query]))[0]
        similarities = _index.matrix @ query_vec  # vectorized cosine similarity, both sides pre-normalized
        intent_scores = _per_intent_scores(similarities)
        top_intent, top_similarity, second_intent, second_similarity = _top_two(intent_scores)
        margin = top_similarity - second_similarity
    except Exception as exc:
        logger.warning("intent_router: embedding call failed (%s) - falling back to the LLM classifier", exc)
        result = _llm_fallback(query, timeout, no_embedding_fields)
        result.error = str(exc)
        return _finish(result, start)

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

    result = _llm_fallback(query, timeout, embedding_fields)
    return _finish(result, start)


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
