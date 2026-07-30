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

_CACHE_NAME = "intent_classifier"
_CACHE_DISTANCE_THRESHOLD = 0.05
_CACHE_TTL_SECONDS = 86400

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
