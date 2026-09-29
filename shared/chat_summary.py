from __future__ import annotations

MAX_ANSWER_CHARS = 1500


def generate_chat_summary(turns: list[tuple[str, str]]) -> str:
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
