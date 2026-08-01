"""Auto-titles a chat from its first message, the same way most chat products title a new
thread - see worker_service/tasks/chat_title.py for the arq job that calls generate_title() and
api_service/routers/chats.py's send_message for where that job gets enqueued (only when the
message that just landed was the chat's first).

Mirrors shared/light_response.py's DeepInfra call shape on purpose (same client construction,
same "thinking disabled" extra_body, same settings-driven model/api-key lookup) - this is the
same class of problem: a small, latency-sensitive LLM call that must never be allowed to break
the caller. Unlike light_response, a bad/missing model call here isn't fatal to anything - it
just means a slightly worse title - so generate_title() always returns *something* usable
(never None/empty), falling back to a deterministic heuristic built from the query itself.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from openai import OpenAI

from shared.config import get_settings

logger = logging.getLogger("chat_title")

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

_DISABLE_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}

MAX_TITLE_CHARS = 60
FALLBACK_WORD_COUNT = 8

_SYSTEM_PROMPT = """You title conversations for a data analysis app, the same way chat apps
title threads from the user's first message.

Reply with ONLY the title - 3 to 6 words, plain text, no quotes, no markdown, no emoji, no
trailing period. Capture the specific topic or task in the message, not a generic label like
"Data Analysis" or "User Question". If the message is a greeting or too vague to summarize,
still produce a short, sensible title (e.g. "Quick Hello", "General Question") rather than
refusing or explaining that you can't."""


@dataclass
class ChatTitleResult:
    query: str
    title: str  # always non-empty - see generate_title()
    model: str
    latency_ms: float
    error: str | None = None
    fallback: bool = False  # True if `title` came from the heuristic, not the model


def _clean(text: str) -> str:
    text = text.strip()
    # Strip one layer of surrounding quotes/backticks - models add these despite being told
    # not to often enough that it's worth handling rather than shipping "\"Sales Trends\"".
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        text = text[1:-1].strip()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(" .!:;-").strip()


def _cap_length(title: str) -> str:
    if len(title) <= MAX_TITLE_CHARS:
        return title
    # Truncate on a word boundary rather than mid-word.
    return title[:MAX_TITLE_CHARS].rsplit(" ", 1)[0].rstrip(" .!:;-")


def _heuristic_title(query: str) -> str:
    """Deterministic, dependency-free fallback so a chat is never left without SOME title, even
    if DEEPINFRA_API_KEY isn't configured or the model call fails/times out/returns junk."""
    words = query.strip().split()
    title = _cap_length(_clean(" ".join(words[:FALLBACK_WORD_COUNT])))
    if not title:
        return "New chat"
    return title[0].upper() + title[1:]


def generate_title(query: str, timeout: float = 15.0) -> ChatTitleResult:
    settings = get_settings()
    model = settings.get("CHAT_TITLE_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    api_key = settings.get("DEEPINFRA_API_KEY")

    start = time.perf_counter()
    if not api_key:
        logger.warning("chat_title: DEEPINFRA_API_KEY not set, using heuristic fallback")
        return ChatTitleResult(
            query=query, title=_heuristic_title(query), model="heuristic",
            latency_ms=(time.perf_counter() - start) * 1000,
            error="DEEPINFRA_API_KEY not configured", fallback=True,
        )

    title, error = None, None
    try:
        client = OpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL, timeout=timeout)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                # Titling only needs the opening of a long message - capped so an unusually
                # long first query doesn't blow up prompt size for zero benefit to the title.
                {"role": "user", "content": query[:2000]},
            ],
            max_tokens=30,
            temperature=0.3,
            extra_body=_DISABLE_THINKING,
        )
        raw = (response.choices[0].message.content or "").strip()
        title = _cap_length(_clean(raw))
        if not title:
            error = "empty/unusable title from model"
    except Exception as exc:  # noqa: BLE001 - falls back to the heuristic below either way
        error = str(exc)

    latency_ms = (time.perf_counter() - start) * 1000
    fallback = not title
    if fallback:
        title = _heuristic_title(query)

    logger.info(
        "chat_title: query=%r title=%r latency_ms=%.1f model=%s error=%s fallback=%s",
        query, title, latency_ms, model, error, fallback,
    )
    return ChatTitleResult(
        query=query, title=title, model=model, latency_ms=latency_ms, error=error, fallback=fallback,
    )
