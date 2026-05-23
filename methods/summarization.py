"""Baseline 2: Summarisation with budget-controlled verbatim recent turns."""

import os
from dotenv import load_dotenv
from utils.llm import llm_call
from utils.token_counter import count_tokens, turns_to_text, count_turns_tokens

load_dotenv()
_SUMMARIZE_MODEL = os.getenv("SUMMARIZE_MODEL", None)

SUMMARIZE_PROMPT = """\
Write an exhaustive, comprehensive record of the following conversation.
Include EVERY detail: all people mentioned with their ages, occupations, relationships, \
and preferences; all events, facts, numbers, dates, and locations; all decisions made. \
Be as thorough and detailed as possible. Do not omit anything.

Conversation:
{turns}

Comprehensive record:"""


def _summarise(turns_text: str, max_tokens: int = 4096) -> str:
    return llm_call(
        SUMMARIZE_PROMPT.format(turns=turns_text),
        system="You are a faithful dialogue summarisation assistant. Be exhaustive and detailed.",
        model=_SUMMARIZE_MODEL,
        max_tokens=max_tokens,
    )


def recursive_summarize(
    history: list,
    max_summary_tokens: int = 5000,
    summarize_every: int = 10,
) -> tuple[str, int]:
    """
    Summarisation baseline.

    Step 1 — Summarise the first half of the conversation with one LLM call
              (LLM writes freely, typically ~800–1500 tokens).
    Step 2 — Fill remaining budget (max_summary_tokens - summary_tokens) with
              verbatim recent turns, greedy from most-recent backward.

    max_summary_tokens is the total context budget (summary + verbatim combined).
    Larger budget → more verbatim turns → higher accuracy, lower compression.
    """
    if not history:
        return "", 0

    # Summarise first half; treat second half as verbatim candidates
    split = max(1, len(history) // 2)
    old_turns = history[:split]
    candidate_recent = history[split:]

    summary = _summarise(turns_to_text(old_turns))
    summary_tokens = count_tokens(summary)

    # Greedy fill: add turns from most-recent until budget exhausted
    remaining = max(0, max_summary_tokens - summary_tokens - 50)
    verbatim = []
    for turn in reversed(candidate_recent):
        t = turns_to_text([turn])
        t_tokens = count_tokens(t)
        if t_tokens <= remaining:
            verbatim.insert(0, t)
            remaining -= t_tokens
        else:
            break

    context_parts = [f"[Conversation Summary]\n{summary}"]
    if verbatim:
        context_parts.append("[Recent Turns]\n" + "\n".join(verbatim))
    context = "\n\n".join(context_parts)
    return context, count_tokens(context)


def run(sample: dict, max_summary_tokens: int = 5000, summarize_every: int = 10) -> tuple[str, int, int]:
    """Return (compressed_context, tokens_used, full_tokens)."""
    full_tokens = count_turns_tokens(sample["full_history"])
    context, tokens_used = recursive_summarize(
        sample["full_history"], max_summary_tokens, summarize_every
    )
    return context, tokens_used, full_tokens
