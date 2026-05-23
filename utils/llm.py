"""LLM abstraction layer supporting OpenAI and Anthropic."""

import os
import time
import json
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

MAX_RETRIES = 6
BASE_WAIT = 2.0  # seconds; doubles each retry


def llm_call(
    prompt: str,
    system: str = "You are a helpful assistant.",
    model: str = None,
    json_mode: bool = False,
    max_tokens: int = None,
) -> str:
    if LLM_PROVIDER == "openai":
        return _openai_call(prompt, system, model or OPENAI_MODEL, json_mode, max_tokens)
    elif LLM_PROVIDER == "anthropic":
        return _anthropic_call(prompt, system, model or ANTHROPIC_MODEL, json_mode, max_tokens)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. Set it to 'openai' or 'anthropic' in .env"
        )


def _openai_call(prompt: str, system: str, model: str, json_mode: bool, max_tokens: int = None) -> str:
    from openai import OpenAI, RateLimitError

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    wait = BASE_WAIT
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except RateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(wait)
            wait *= 2


def _anthropic_call(prompt: str, system: str, model: str, json_mode: bool, max_tokens: int = None) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    full_prompt = prompt
    if json_mode:
        full_prompt += "\n\nRespond with valid JSON only. No markdown, no explanation."

    wait = BASE_WAIT
    for attempt in range(MAX_RETRIES):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens or 2048,
                system=system,
                messages=[{"role": "user", "content": full_prompt}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(wait)
                wait *= 2
            else:
                raise
