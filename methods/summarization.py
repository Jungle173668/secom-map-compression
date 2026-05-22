"""Baseline 2: True RecurSum — one running summary updated incrementally."""

import os
from dotenv import load_dotenv
from utils.llm import llm_call
from utils.token_counter import count_tokens, turns_to_text, count_turns_tokens

load_dotenv()
_SUMMARIZE_MODEL = os.getenv("SUMMARIZE_MODEL", None)

SUMMARIZE_PROMPT = """\
Summarise the following conversation. Your target length is approximately {max_tokens} tokens.

When the target is large (e.g. 5000+ tokens), you MUST write a thorough, exhaustive summary:
read the entire conversation carefully and include as much detail as possible — every person \
mentioned, their age, occupation, preferences, relationships, and specific events, facts, \
numbers, dates, locations, and decisions. A larger token budget means more detail, not repetition.

When the target is small (e.g. under 2000 tokens), be concise but still preserve the most \
important named facts.

Rules:
- Preserve all names, ages, numbers, dates, locations, and specific decisions.
- Do not add information not present in the original text.
- Write in flowing prose or structured bullet points — whichever fits the length better.

Conversation:
{turns}

Summary (approximately {max_tokens} tokens):"""


def _summarise(turns_text: str, max_tokens: int) -> str:
    prompt = SUMMARIZE_PROMPT.format(turns=turns_text, max_tokens=max_tokens)
    return llm_call(
        prompt,
        system="You are a faithful dialogue summarisation assistant.",
        model=_SUMMARIZE_MODEL,
    )


def recursive_summarize(
    history: list,
    max_summary_tokens: int = 2000,
    summarize_every: int = 10,
) -> tuple[str, int]:
    """
    Summarisation baseline: one LLM call to summarise all old turns, keep recent verbatim.

    Design:
      old turns  = history[:-summarize_every]  → one LLM call → summary
      recent     = history[-summarize_every:]   → kept verbatim

    Control:
      max_summary_tokens → target summary length → compression ratio.
      Larger value → more detail → lower compression → higher accuracy.

    This is the simplest offline instantiation of 'LLM rewrites old turns':
    one call, all old content compressed equally, recent turns preserved.
    """
    if not history:
        return "", 0

    if len(history) <= summarize_every:
        context = turns_to_text(history)
        return context, count_tokens(context)

    old_turns = history[:-summarize_every]
    recent_turns = history[-summarize_every:]

    summary = _summarise(turns_to_text(old_turns), max_summary_tokens)
    recent_text = turns_to_text(recent_turns)

    context = "[Conversation Summary]\n" + summary + "\n\n[Recent Turns]\n" + recent_text
    return context, count_tokens(context)


def run(sample: dict, max_summary_tokens: int = 2000, summarize_every: int = 10) -> tuple[str, int, int]:
    """Return (compressed_context, tokens_used, full_tokens)."""
    full_tokens = count_turns_tokens(sample["full_history"])
    context, tokens_used = recursive_summarize(
        sample["full_history"], max_summary_tokens, summarize_every
    )
    return context, tokens_used, full_tokens
