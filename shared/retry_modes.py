from typing import Literal

RetryMode = Literal["regenerate", "think_harder", "show_reasoning", "more_detail", "more_concise", "simpler"]

DEFAULT_RETRY_MODE: RetryMode = "regenerate"

ORCHESTRATOR_ONLY_MODES = frozenset({"think_harder", "show_reasoning"})

PREVIOUS_ANSWER_CHARS = 1500

RETRY_INSTRUCTIONS: dict[str, str] = {
    "regenerate": (
        "The user asked for a fresh answer to the same question. Work it out again independently "
        "rather than restating an earlier answer."
    ),
    "think_harder": (
        "The user asked you to think harder. Investigate more thoroughly than usual: check the "
        "relevant data from more than one angle, verify every number against the source, consider "
        "alternative explanations, and call out caveats or uncertainty explicitly."
    ),
    "show_reasoning": (
        "The user wants to see your reasoning. Answer the question, then explain step by step how "
        "you got there: which files, tables or columns you used, what calculations or filters you "
        "applied, and any assumptions you made."
    ),
    "more_detail": (
        "The user wants a longer, more detailed answer. Expand on the findings with supporting "
        "numbers, breakdowns and context, while staying focused on the question."
    ),
    "more_concise": (
        "The user wants a shorter answer. Lead with the direct answer and keep only the essential "
        "supporting facts - a few sentences or a short list."
    ),
    "simpler": (
        "The user wants a simpler explanation. Use plain, non-technical language, avoid jargon, "
        "and explain any necessary terms briefly."
    ),
}


def retry_constraints(mode: str | None, previous_answer: str | None) -> dict:
    if not mode:
        return {}
    constraints = {
        "regeneration": RETRY_INSTRUCTIONS.get(mode, RETRY_INSTRUCTIONS[DEFAULT_RETRY_MODE]),
    }
    if mode != DEFAULT_RETRY_MODE and previous_answer:
        trimmed = previous_answer.strip()
        if len(trimmed) > PREVIOUS_ANSWER_CHARS:
            trimmed = trimmed[:PREVIOUS_ANSWER_CHARS].rstrip() + "..."
        constraints["previous_answer_to_improve"] = trimmed
    return constraints
