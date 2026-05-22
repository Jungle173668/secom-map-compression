"""Baseline 1: Sliding window — keep the last N turns verbatim."""

from utils.token_counter import count_tokens, turns_to_text, count_turns_tokens


def sliding_window(history: list, n_turns: int) -> tuple[str, int]:
    """Keep the last n_turns turns verbatim."""
    kept = history[-n_turns:] if n_turns < len(history) else history
    context = turns_to_text(kept)
    return context, count_tokens(context)


def run(sample: dict, n_turns: int = 150) -> tuple[str, int, int]:
    """Return (context, tokens_used, full_tokens)."""
    full_tokens = count_turns_tokens(sample["full_history"])
    context, tokens_used = sliding_window(sample["full_history"], n_turns)
    return context, tokens_used, full_tokens
