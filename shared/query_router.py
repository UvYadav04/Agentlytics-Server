"""Shadow query router.

Classifies short/simple queries ("hi", "hello", "thanks"...) against a
canned intent list so we can evaluate whether a fast-path (bypassing the
arq queue entirely) would be safe to turn on - *before* actually wiring it
up. Nothing in this module changes request handling: it only produces a
`ClassificationResult` for the caller to log.

Tiers, cheapest first:
  1. exact  - normalized text is byte-for-byte a canned phrase
  2. fuzzy  - difflib similarity ratio against every canned phrase
              (stdlib only, no new dependency)
  3. hf     - optional sentence-embedding cosine similarity, only runs if
              `sentence-transformers` is installed *and*
              QUERY_ROUTER_HF_ENABLED=true. Lazily loaded on first use so
              importing this module never requires the dependency.

Context gating: a short message that follows a prior turn in the same
chat ("no", "make it red instead") can score high against a canned
phrase purely by being short - but it is very likely a reply/correction,
not small talk. So any match is marked `context_gated` (and
`would_shortcircuit=False`) whenever the chat already has prior
messages. The classification itself still runs and gets logged, so we
can measure how often this would matter.

See api_service/routers/chats.py for the (shadow-only) call site.
"""
from __future__ import annotations

import difflib
import logging
import re
import time
from dataclasses import dataclass

from shared.config import get_settings

logger = logging.getLogger("query_router")

# Short/ambiguous items ("ok", "yes", "no") are deliberately included here
# even though they're common follow-up replies too ("ok" after a
# clarifying question, "yes"/"no" answering one) - that's exactly what
# context-gating (above) exists for. A bare "yes"/"no"/"ok" only ever gets
# `would_shortcircuit=True` when it's the *first* message in the chat
# (nothing to be a follow-up to); with prior context it still classifies
# for logging, but `context_gated=True` blocks the shortcircuit. So the
# list can be broad - the gating check is what keeps it safe, not the
# list being narrow.
CANNED_INTENTS: dict[str, list[str]] = {
    "greeting": [
        "hi", "hello", "hey", "yo", "hiya", "sup", "howdy",
        "good morning", "good afternoon", "good evening",
    ],
    "farewell": ["bye", "goodbye", "see you", "cya", "later", "take care", "gtg"],
    "thanks": ["thanks", "thank you", "thanks a lot", "thx", "ty", "appreciate it"],
    "affirmation": [
        "ok", "okay", "sure", "alright", "sounds good", "got it",
        "cool", "great", "perfect", "awesome",
    ],
    "acknowledgment": ["yes", "yeah", "yep", "no", "nope", "nah"],
    "bot_identity": [
        "whats your name", "what is your name", "who are you", "what are you",
        "are you a bot", "are you human", "are you an ai",
        "what can you do", "how do you work", "who made you",
    ],
    "status_smalltalk": [
        "how are you", "how are you doing", "hows it going", "whats up",
        "how have you been", "hows your day",
    ],
    "filler": ["test", "testing", "are you there", "anyone there", "can you hear me"],
}

_FLAT_PHRASES: list[tuple[str, str]] = [
    (phrase, intent) for intent, phrases in CANNED_INTENTS.items() for phrase in phrases
]
_EXACT_LOOKUP: dict[str, str] = dict(_FLAT_PHRASES)

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = _PUNCT_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


@dataclass
class ClassificationResult:
    query: str
    normalized: str
    tier: str  # "exact" | "fuzzy" | "hf" | "hf_unavailable" | "none"
    intent: str | None
    score: float
    has_prior_context: bool
    context_gated: bool
    would_shortcircuit: bool
    latency_ms: float
    error: str | None = None


def _exact_match(normalized: str) -> tuple[str, float] | None:
    intent = _EXACT_LOOKUP.get(normalized)
    return (intent, 1.0) if intent else None


def _fuzzy_match(normalized: str, threshold: float) -> tuple[str, float] | None:
    if not normalized:
        return None
    best_intent, best_score = None, 0.0
    for phrase, intent in _FLAT_PHRASES:
        score = difflib.SequenceMatcher(None, normalized, phrase).ratio()
        if score > best_score:
            best_intent, best_score = intent, score
    if best_intent and best_score >= threshold:
        return (best_intent, best_score)
    return None


# --- Optional HF tier ---------------------------------------------------
# Lazily loaded: importing this module never requires sentence-transformers.
# Flip on with QUERY_ROUTER_HF_ENABLED=true once the dependency is installed
# (see shared/requirements.txt for the optional install note).

_hf_model = None
_hf_ready = False
_hf_load_failed = False
_canned_embeddings = None


def _load_hf_model() -> None:
    global _hf_model, _hf_ready, _hf_load_failed, _canned_embeddings
    if _hf_ready or _hf_load_failed:
        return
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model_name = get_settings().get("QUERY_ROUTER_HF_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        _hf_model = SentenceTransformer(model_name)
        phrases = [p for p, _ in _FLAT_PHRASES]
        _canned_embeddings = _hf_model.encode(phrases, convert_to_tensor=True)
        _hf_ready = True
        logger.info("query_router: HF model %s loaded (%d canned phrases)", model_name, len(phrases))
    except Exception as exc:  # noqa: BLE001 - any load failure just disables tier 2
        _hf_load_failed = True
        logger.warning("query_router: HF tier unavailable (%s)", exc)


def _hf_match(text: str, threshold: float) -> tuple[str, float] | None:
    from sentence_transformers import util  # only reached once load succeeded

    query_emb = _hf_model.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, _canned_embeddings)[0]
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])
    if best_score >= threshold:
        return (_FLAT_PHRASES[best_idx][1], best_score)
    return None


def classify(query: str, has_prior_context: bool) -> ClassificationResult:
    """Classify a query for shadow evaluation.

    Never raises: any internal failure is captured on `.error` and reported
    as `tier="none"`, so a bug here can never take down the caller (which
    today just logs the result - see chats.py).
    """
    start = time.perf_counter()
    settings = get_settings()
    fuzzy_threshold = float(settings.get("QUERY_ROUTER_FUZZY_THRESHOLD", 0.84) or 0.84)
    hf_threshold = float(settings.get("QUERY_ROUTER_HF_THRESHOLD", 0.80) or 0.80)
    hf_enabled = str(settings.get("QUERY_ROUTER_HF_ENABLED", "false") or "false").lower() == "true"
    max_words = int(settings.get("QUERY_ROUTER_MAX_WORDS", 6) or 6)

    tier, intent, score, error, normalized = "none", None, 0.0, None, ""
    try:
        normalized = normalize(query)

        # Long messages are obviously not small talk - skip the tiers
        # entirely rather than wasting fuzzy/HF compute on them.
        if normalized and len(normalized.split()) <= max_words:
            match = _exact_match(normalized)
            if match:
                tier, (intent, score) = "exact", match
            else:
                match = _fuzzy_match(normalized, fuzzy_threshold)
                if match:
                    tier, (intent, score) = "fuzzy", match
                elif hf_enabled:
                    _load_hf_model()
                    if _hf_ready:
                        match = _hf_match(normalized, hf_threshold)
                        if match:
                            tier, (intent, score) = "hf", match
                        else:
                            tier = "hf"  # ran, no confident match
                    else:
                        tier = "hf_unavailable"
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        tier, intent, score = "none", None, 0.0

    matched = intent is not None
    context_gated = has_prior_context and matched
    would_shortcircuit = matched and not context_gated

    latency_ms = (time.perf_counter() - start) * 1000
    return ClassificationResult(
        query=query,
        normalized=normalized,
        tier=tier,
        intent=intent,
        score=score,
        has_prior_context=has_prior_context,
        context_gated=context_gated,
        would_shortcircuit=would_shortcircuit,
        latency_ms=latency_ms,
        error=error,
    )
