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

try:
    from monitoring.wrappers.base import trace_step
except Exception:
    def trace_step(*args, **kwargs):
        return lambda fn: fn

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

_CACHE_NAME = "intent_classifier"
_CACHE_DISTANCE_THRESHOLD = 0.05
_CACHE_TTL_SECONDS = 86400

_DISABLE_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}

CANDIDATE_LABELS = ["greeting", "tabular", "document", "orchestrator"]

_SYSTEM_PROMPT = """
You are an intent classifier for a data analysis platform.

Classify the user request into EXACTLY ONE category: "greeting", "tabular", "document", or "orchestrator".

Categories & Guidelines:

1. greeting
- Pleasantries, hellos, goodbyes, gratitude, or assistant identity questions.
- Rule: If greetings intent, provide a friendly greeting in the "response" field.
- Example: "Hi there, who are you and how can you help?"
  Output: {"intent": "greeting", "reasoning": "User is greeting and asking about bot capabilities.", "response": "Hello! I am your data analysis assistant. I can help analyze your tables, documents, and reports.", "confidence": 0.99}

2. tabular
- Queries requiring numerical computation, statistics (correlation, variance, distribution), aggregations, sorting, filtering, or charts over structured data (CSV, Excel, SQL, Parquet).
- Rule: If ANY mathematical computation, statistical correlation between variables, trend analysis, or chart creation over table columns is required, ALWAYS choose tabular.
- Example: "What is the correlation between marketing spend and customer acquisition?"
  Output: {"intent": "tabular", "reasoning": "Requires statistical correlation analysis across numerical table columns.", "response": "", "confidence": 0.98}

3. document
- Queries seeking information, summaries, or facts that exist inside unstructured text or PDF documents (e.g., policies, handbooks, contracts, agreements, compliance guidelines) where no math or table computation is required.
- Rule: Use when the answer is retrieved directly from document text.
- Example: "What is the company policy on remote work and parental leave?"
  Output: {"intent": "document", "reasoning": "Requires retrieving policy details from uploaded document text without numerical calculations.", "response": "", "confidence": 0.98}

4. orchestrator
- Use when:
  a) Cross-modal synthesis: Query requires combining or cross-referencing text from documents with numbers from spreadsheets/tables (e.g., explaining why sales dropped based on a memo, or checking financial metrics against an executive summary).
  b) Deliverable exports: Query explicitly asks to export, generate, build, or download a file, dashboard, spreadsheet, or report.
  c) The query is ambiguous, open-ended, or requires multi-step planning across multiple files.
- Rule: If unsure, or if both documents and tabular data must be connected, choose orchestrator.
- Example: "Correlate the executive summary with the financial spreadsheet metrics and export the results to a CSV."
  Output: {"intent": "orchestrator", "reasoning": "Requires cross-referencing document text with spreadsheet metrics and exporting a deliverable file.", "response": "", "confidence": 0.98}

Return ONLY valid JSON matching this schema:
{
  "intent": "tabular",
  "reasoning": "Brief explanation of the routing choice",
  "response": "",
  "confidence": 0.98
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
    cached: bool = False


def _parse_reply(raw_text: str) -> tuple[str | None, float, str | None]:
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


@trace_step(name="classify_intent", component="intent.classifier", step_type="router")
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
    except Exception as exc:  # noqa: BLE001
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
