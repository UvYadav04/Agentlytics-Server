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
    title: str
    model: str
    latency_ms: float
    error: str | None = None
    fallback: bool = False


def _clean(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        text = text[1:-1].strip()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(" .!:;-").strip()


def _cap_length(title: str) -> str:
    if len(title) <= MAX_TITLE_CHARS:
        return title
    return title[:MAX_TITLE_CHARS].rsplit(" ", 1)[0].rstrip(" .!:;-")


def _heuristic_title(query: str) -> str:
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
    except Exception as exc:  # noqa: BLE001
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
