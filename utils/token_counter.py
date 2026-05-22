"""Token counting utilities (cl100k_base as a universal approximation)."""

import tiktoken

_enc = None


def _get_encoder():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_get_encoder().encode(text))


def turns_to_text(turns: list) -> str:
    """Convert a list of (user, assistant) pairs or dicts to a single string."""
    parts = []
    for turn in turns:
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            parts.append(f"User: {turn[0]}\nAssistant: {turn[1]}")
        elif isinstance(turn, dict):
            speaker = turn.get("speaker", "Speaker")
            text = turn.get("text", "")
            parts.append(f"{speaker}: {text}")
        else:
            parts.append(str(turn))
    return "\n".join(parts)


def count_turns_tokens(turns: list) -> int:
    return count_tokens(turns_to_text(turns))
