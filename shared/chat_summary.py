"""Builds a chat's export summary by concatenating its assistant answers - no LLM call, on
purpose (see api_service/routers/chats.py's export_chat_summary and the design discussion this
came out of: the answers stored per turn are already short, distilled final_answers, not raw
tool-call transcripts, so gluing them together is already readable without paying for another
model call).
"""
from __future__ import annotations

MAX_ANSWER_CHARS = 1500


def generate_chat_summary(turns: list[tuple[str, str]]) -> str:
    """turns: ordered list of (question, final_answer) pairs - one per completed exchange in the
    chat. Turns with an empty/blank answer are skipped. Always returns a non-empty string."""
    parts = []
    for question, answer in turns:
        answer = (answer or "").strip()
        if not answer:
            continue
        if len(answer) > MAX_ANSWER_CHARS:
            answer = answer[:MAX_ANSWER_CHARS].rsplit(" ", 1)[0] + "..."
        question = (question or "").strip()
        parts.append(f"Q: {question}\nA: {answer}" if question else answer)
    return "\n\n".join(parts) or "This chat has no answers yet."
