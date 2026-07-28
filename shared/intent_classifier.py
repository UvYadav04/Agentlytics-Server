"""DeepInfra-based intent classifier - now a REAL routing decision, not just a shadow test.

Classifies a query into one of four categories: "greeting", "tabular", "document",
"orchestrator" (see _SYSTEM_PROMPT below for the full category definitions given to the model).
Calls DeepInfra's OpenAI-compatible chat completion API directly via the `openai` SDK pointed at
DeepInfra's base_url - same provider this codebase already uses elsewhere (see
analyzerEngine/llm_provider/providers/deepinfra_client.py), just called directly here instead of
through autogen's wrapper, since this module lives in shared/ (used by both api_service and
worker_service) and has no autogen dependency.

Qwen3-family models are hybrid reasoning models that wrap every response in a <think>...</think>
block unless told otherwise (see deepinfra_client.py's DISABLE_THINKING comment) - left on, that
reasoning text would blow past max_tokens before the model ever emits the JSON we actually want,
or leak into the JSON block itself. extra_body below turns it off the same way.

Called synchronously, in the foreground, from api_service/routers/chats.py's send_message -
deliberately, so the route is known BEFORE the arq job is enqueued (there's no cheap way to
decide the route after the fact without a second job round trip), and so its latency shows up
in that route's own timing logs instead of being hidden in a background task. send_message only
trusts "tabular"/"document"/"orchestrator" above INTENT_ROUTE_CONFIDENCE_THRESHOLD as a real
route - "greeting", low confidence, and any classifier error all fall back to route=None, which
worker_service/tasks/investigation.py treats as "run the full Orchestrator" (today's original,
always-safe behavior). Even a trusted "tabular"/"document" route gets a second, independent
safety check in investigation.py's _select_direct_route_files (file selection must be
unambiguous) before it's actually allowed to skip the Orchestrator.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from openai import OpenAI

from shared.config import get_settings
from shared.semantic_cache import cached_check, cached_store

logger = logging.getLogger("intent_classifier")

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

# shared/semantic_cache.py index name for this module's LLM fallback tier. A tight threshold on
# purpose - a false-positive cache hit here means misrouting a query to the wrong
# tabular/document/orchestrator handler, so it's worth erring towards more cache misses (i.e.
# more LLM calls) in exchange for fewer wrong ones. 1 day TTL: routing preferences don't need to
# be cached forever, and this keeps the index from growing unbounded across many distinct queries.
_CACHE_NAME = "intent_classifier"
_CACHE_DISTANCE_THRESHOLD = 0.05
_CACHE_TTL_SECONDS = 86400

# Qwen3-family models are hybrid reasoning models - disable the <think> block so the
# response is just the JSON we asked for (see module docstring).
_DISABLE_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}

CANDIDATE_LABELS = ["greeting", "tabular", "document", "orchestrator"]

_SYSTEM_PROMPT = """
You are an intent classifier for a data analysis platform.

Classify the user request into EXACTLY ONE category.

Categories:

1. greeting
- Greetings, thanks, farewell.
Examples:
"hi"
"hello"
"good morning"

2. tabular
Use this when answering the request requires computation over structured data such as CSV, Excel, SQL or Parquet.

3. document
Use this ONLY when the answer already exists inside uploaded documents and no computation is required.

4. orchestrator
Use when multiple capabilities are required or the query is vague/ambigous.

Rules:

If ANY mathematical computation, aggregation, chart, anomaly detection, statistics or analysis is required,
ALWAYS choose tabular.

If greetings intent, return a response along with a greeting message.

If unsure,
choose orchestrator.

Return ONLY JSON.

{
  "intent":"tabular",
  "reasoning":"",
  "response":"",
  "confidence":0.98
}
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class IntentResult:
    query: str
    top_label: str | None
    top_score: float
    labels: list[str]
    scores: list[float]
    latency_ms: float
    error: str | None = None
    cached: bool = False  # True when top_label/top_score came from shared/semantic_cache.py
    # instead of an actual DeepInfra call this time - purely informational (logging/debugging),
    # every other field means the same thing either way.


def _parse_reply(raw_text: str) -> tuple[str | None, float, str | None]:
    """The model is told to return ONLY JSON, but instruct models still sometimes wrap it
    in markdown fences or add a stray sentence - pull the first {...} block out rather than
    assuming the whole reply is clean JSON, then validate the intent is one we know."""
    match = _JSON_BLOCK_RE.search(raw_text or "")
    if not match:
        return None, 0.0, f"no JSON object found in reply: {raw_text!r}"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return None, 0.0, f"failed to parse JSON reply ({exc}): {raw_text!r}"

    intent = str(data.get("intent") or "").strip().lower()
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    if intent not in CANDIDATE_LABELS:
        return None, confidence, f"model returned an unknown intent: {intent!r}"
    return intent, confidence, None


def classify_intent(query: str, timeout: float = 30.0) -> IntentResult:
    """Never raises - any failure (missing key, network, bad/unparseable response) is
    captured on `.error` with top_label=None, same "never take down the caller" contract
    as shared/query_router.py's classify()."""
    settings = get_settings()
    model = settings.get("INTENT_CLASSIFIER_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    api_key = settings.get("DEEPINFRA_API_KEY")

    start = time.perf_counter()
    if not api_key:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning("intent_classifier: DEEPINFRA_API_KEY not set, skipping call")
        return IntentResult(
            query=query, top_label=None, top_score=0.0, labels=[], scores=[],
            latency_ms=latency_ms, error="DEEPINFRA_API_KEY not configured",
        )

    cache_enabled = (settings.get("INTENT_CACHE_ENABLED", "true") or "true").lower() != "false"
    if cache_enabled:
        cached_value = cached_check(_CACHE_NAME, query, distance_threshold=_CACHE_DISTANCE_THRESHOLD, ttl=_CACHE_TTL_SECONDS)
        if cached_value is not None:
            intent, _, confidence_str = cached_value.partition("|")
            try:
                confidence = float(confidence_str)
            except ValueError:
                confidence = 0.0
            latency_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "intent_classifier: query=%r top_label=%s top_score=%.3f latency_ms=%.1f model=%s cache=hit",
                query, intent, confidence, latency_ms, model,
            )
            return IntentResult(
                query=query, top_label=intent, top_score=confidence,
                labels=[intent], scores=[confidence], latency_ms=latency_ms, error=None, cached=True,
            )

    labels, scores, top_label, top_score, error = [], [], None, 0.0, None
    try:
        client = OpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL, timeout=timeout)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=200,
            temperature=0,
            response_format={"type": "json_object"},
            extra_body=_DISABLE_THINKING,
        )
        raw_text = response.choices[0].message.content or ""
        intent, confidence, parse_error = _parse_reply(raw_text)
        if intent is not None:
            labels, scores = [intent], [confidence]
            top_label, top_score = intent, confidence
            if cache_enabled:
                cached_store(
                    _CACHE_NAME, query, f"{intent}|{confidence}",
                    distance_threshold=_CACHE_DISTANCE_THRESHOLD, ttl=_CACHE_TTL_SECONDS,
                )
        error = parse_error
    except Exception as exc:  # noqa: BLE001 - test harness must never crash the request
        error = str(exc)

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "intent_classifier: query=%r top_label=%s top_score=%.3f latency_ms=%.1f model=%s error=%s",
        query, top_label, top_score, latency_ms, model, error,
    )
    return IntentResult(
        query=query, top_label=top_label, top_score=top_score,
        labels=labels, scores=scores, latency_ms=latency_ms, error=error, cached=False,
    )
