"""Exp16 wrapper that adds thinking/reasoning controls only.

This runner reuses `run_dissociation_test.py` end-to-end and keeps the base
prompt/endorsement channel unchanged. It only adds optional controls for:
- `enable_thinking=False` in chat templating when supported (e.g., Gemma)
- OpenRouter/OpenAI-compatible `reasoning` payload fields
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import src.exp16.run_dissociation_test as base

_DISABLE_THINKING = False
_OPENROUTER_EXCLUDE_REASONING = False
_REASONING_EFFORT: str | None = None
FREEGEN_NATURAL_SUFFIX = "Answer the question in one short sentence using the full answer text, not option letters."


def _format_chat_prompt_text_conditioned(tokenizer, prompt_text: str, plain_prompt: bool) -> str:
    if plain_prompt:
        return prompt_text
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt_text}]
        # Gemma supports `enable_thinking=False`; other tokenizers may not.
        if _DISABLE_THINKING:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                pass
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt_text


def _request_openai_compatible_chat_completion_text_conditioned(
    *,
    api_base: str,
    api_key: str,
    api_provider: str,
    openrouter_http_referer: str,
    openrouter_app_title: str,
    model_name: str,
    prompt: str,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    max_tokens: int,
    timeout_seconds: float,
) -> str:
    url = f"{api_base.rstrip('/')}/chat/completions"
    body: dict[str, object] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if not base._is_gpt5_family(model_name):
        body["temperature"] = temperature
        body["top_p"] = top_p
        if presence_penalty != 0.0:
            body["presence_penalty"] = presence_penalty

    reasoning_payload: dict[str, object] = {}
    if _OPENROUTER_EXCLUDE_REASONING and api_provider == "openrouter":
        reasoning_payload["exclude"] = True
    if _REASONING_EFFORT is not None and _REASONING_EFFORT != "":
        reasoning_payload["effort"] = _REASONING_EFFORT
    if reasoning_payload:
        body["reasoning"] = reasoning_payload

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if api_provider == "openrouter":
        if openrouter_http_referer:
            headers["HTTP-Referer"] = openrouter_http_referer
        if openrouter_app_title:
            headers["X-Title"] = openrouter_app_title
    req = urllib.request.Request(
        url,
        method="POST",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise ValueError("No choices returned from OpenAI-compatible API")
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        raise ValueError("Malformed OpenAI-compatible API choice")
    message = choice0.get("message", {})
    if not isinstance(message, dict):
        raise ValueError("Malformed OpenAI-compatible API message")
    text = base._extract_openai_content_text(message.get("content", ""))
    if not text.strip():
        raise ValueError("Empty completion from OpenAI-compatible API")
    return text


def _consume_wrapper_args() -> None:
    global _DISABLE_THINKING, _OPENROUTER_EXCLUDE_REASONING, _REASONING_EFFORT
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="When tokenizer supports it (e.g., Gemma), set enable_thinking=False in chat templating.",
    )
    parser.add_argument(
        "--openrouter-exclude-reasoning",
        action="store_true",
        help="For API provider=openrouter, send reasoning.exclude=true.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        help="Optional reasoning effort hint for OpenAI/OpenRouter payloads (e.g., low, medium, high).",
    )
    args, remaining = parser.parse_known_args(sys.argv[1:])
    _DISABLE_THINKING = bool(args.disable_thinking)
    _OPENROUTER_EXCLUDE_REASONING = bool(args.openrouter_exclude_reasoning)
    _REASONING_EFFORT = args.reasoning_effort
    has_natural_suffix = any(
        arg == "--natural-suffix" or arg.startswith("--natural-suffix=")
        for arg in remaining
    )
    if not has_natural_suffix:
        remaining.extend(["--natural-suffix", FREEGEN_NATURAL_SUFFIX])
    sys.argv = [sys.argv[0], *remaining]


def main() -> None:
    _consume_wrapper_args()
    # Keep base prompt/endorsement channel unchanged; patch only thinking/reasoning controls.
    base._format_chat_prompt = _format_chat_prompt_text_conditioned
    base._request_openai_compatible_chat_completion = _request_openai_compatible_chat_completion_text_conditioned
    base.main()


if __name__ == "__main__":
    main()
