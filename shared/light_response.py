from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from openai import OpenAI

from shared.config import get_settings

logger = logging.getLogger("light_response")

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

_DISABLE_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}

_GREETING_PROMPT = """You are the assistant for a data analysis platform. The user just sent a
greeting, thanks, farewell, or other small talk - not a data question.

Reply warmly and briefly (1-2 sentences). Do not invent or reference any analysis, files, or
data - none was asked for. If it fits naturally you can mention you can analyze CSV/Excel/PDF
files once they upload one, but don't force that into every reply."""

_NO_FILES_PROMPT = """You are the assistant for a data analysis platform. This workspace
currently has NO files uploaded - there is nothing for you to compute over or read from.

Answer the user's question directly if it's general knowledge you already know. If the question
clearly depends on data or documents they haven't uploaded yet, say so plainly and invite them
to upload a CSV, Excel, PDF, or similar file so you can look into it. Keep it concise - a short
paragraph, not a report. Never pretend to have analyzed data that doesn't exist."""

_PROMPTS = {"greeting": _GREETING_PROMPT, "no_files": _NO_FILES_PROMPT}


@dataclass
class LightReplyResult:
    query: str
    route: str
    content: str | None
    model: str
    latency_ms: float
    error: str | None = None


def generate_light_reply(query: str, route: str, timeout: float = 20.0) -> LightReplyResult:
    if route not in _PROMPTS:
        raise ValueError(f"generate_light_reply: unknown route {route!r} (expected 'greeting' or 'no_files')")

    settings = get_settings()
    model = settings.get("LIGHT_RESPONSE_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    api_key = settings.get("DEEPINFRA_API_KEY")

    start = time.perf_counter()
    if not api_key:
        logger.warning("light_response: DEEPINFRA_API_KEY not set, skipping call")
        return LightReplyResult(
            query=query, route=route, content=None, model=model,
            latency_ms=(time.perf_counter() - start) * 1000,
            error="DEEPINFRA_API_KEY not configured",
        )

    content, error = None, None
    try:
        client = OpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL, timeout=timeout)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _PROMPTS[route]},
                {"role": "user", "content": query},
            ],
            max_tokens=300,
            temperature=0.4,
            extra_body=_DISABLE_THINKING,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            error = "empty reply from model"
            content = None
    except Exception as exc:  # noqa: BLE001 - caller decides whether to retry/fall back
        error = str(exc)

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "light_response: route=%s query=%r content_len=%s latency_ms=%.1f model=%s error=%s",
        route, query, len(content) if content else 0, latency_ms, model, error,
    )
    return LightReplyResult(
        query=query, route=route, content=content, model=model, latency_ms=latency_ms, error=error,
    )
