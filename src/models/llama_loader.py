"""
Helpers for loading causal-LM models (Llama, Qwen, etc.) via HF Transformers.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(dtype: Optional[Union[str, torch.dtype]]) -> Union[str, torch.dtype]:
    if dtype is None or dtype == "auto":
        return "auto"
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        if not hasattr(torch, dtype):
            raise ValueError(f"Unknown torch dtype string: {dtype}")
        return getattr(torch, dtype)
    raise ValueError(f"Unsupported dtype value: {dtype}")


def _from_pretrained_with_token(factory, model_name: str, token: Optional[str], **kwargs):
    """
    Call Transformers from_pretrained with the correct auth kwarg.

    Newer versions use `token`, older versions use `use_auth_token`.
    """
    max_retries = int(os.environ.get("HF_DOWNLOAD_RETRIES", "4"))
    base_sleep_s = float(os.environ.get("HF_DOWNLOAD_RETRY_SLEEP", "2.0"))

    def _is_transient_download_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        transient_tokens = (
            "readtimeout",
            "connecttimeout",
            "connectionerror",
            "temporarily unavailable",
            "timed out",
            "503",
            "504",
            "connection reset",
        )
        return any(tok in msg for tok in transient_tokens)

    def _invoke():
        if token:
            try:
                return factory.from_pretrained(model_name, token=token, **kwargs)
            except TypeError:
                # Fall back for older Transformers that use use_auth_token.
                return factory.from_pretrained(model_name, use_auth_token=token, **kwargs)
        return factory.from_pretrained(model_name, **kwargs)

    # Retry transient download failures.
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return _invoke()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries - 1 or not _is_transient_download_error(exc):
                raise
            sleep_s = base_sleep_s * (2 ** attempt)
            print(
                f"[load_model_and_tokenizer] transient download failure "
                f"(attempt {attempt + 1}/{max_retries}): {exc}. "
                f"Retrying in {sleep_s:.1f}s..."
            )
            time.sleep(sleep_s)

    # Fallback: should be unreachable because loop either returns or raises.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unexpected download retry state in _from_pretrained_with_token")


def _parse_env_bool(value: Optional[str]) -> Optional[bool]:
    """Parse common truthy/falsey env var values."""
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean env value: {value!r}")


def _infer_trust_remote_code(model_name: str) -> bool:
    """Enable remote code only for model families that commonly require it."""
    return "qwen" in model_name.lower()


def _resolve_token_and_trust_remote_code(
    model_name: str,
    use_auth_token: Optional[str],
    trust_remote_code: Optional[bool],
) -> tuple[Optional[str], bool]:
    token = use_auth_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    env_trust_remote_code = _parse_env_bool(os.environ.get("HF_TRUST_REMOTE_CODE"))
    if trust_remote_code is None:
        trust_remote_code = (
            env_trust_remote_code
            if env_trust_remote_code is not None
            else _infer_trust_remote_code(model_name)
        )

    return token, trust_remote_code


def load_tokenizer(
    model_name: str,
    use_auth_token: Optional[str] = None,
    revision: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
):
    """
    Load only the tokenizer with the same defaults used for model loading.
    """
    token, trust_remote_code = _resolve_token_and_trust_remote_code(
        model_name=model_name,
        use_auth_token=use_auth_token,
        trust_remote_code=trust_remote_code,
    )
    resolved_revision = revision or os.environ.get("HF_MODEL_REVISION")

    shared_kwargs: dict = {}
    if resolved_revision:
        shared_kwargs["revision"] = resolved_revision

    tokenizer = _from_pretrained_with_token(
        AutoTokenizer,
        model_name,
        token,
        trust_remote_code=trust_remote_code,
        **shared_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model_and_tokenizer(
    model_name: str,
    device: Optional[str] = None,
    dtype: Optional[Union[str, torch.dtype]] = "auto",
    use_auth_token: Optional[str] = None,
    revision: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
    prefer_flash_attention: bool = True,
):
    """
    Load a model + tokenizer pair with sensible defaults for this project.
    """
    # Prefer explicit token, otherwise fall back to common env vars set via Modal secret.
    token, trust_remote_code = _resolve_token_and_trust_remote_code(
        model_name=model_name,
        use_auth_token=use_auth_token,
        trust_remote_code=trust_remote_code,
    )
    resolved_revision = revision or os.environ.get("HF_MODEL_REVISION")

    shared_kwargs: dict = {}
    if resolved_revision:
        shared_kwargs["revision"] = resolved_revision

    tokenizer = load_tokenizer(
        model_name,
        use_auth_token=token,
        revision=resolved_revision,
        trust_remote_code=trust_remote_code,
    )

    resolved_dtype = _resolve_dtype(dtype)
    device_map = None
    target_device = device
    if device is None or device == "auto":
        device_map = "auto"
        target_device = None

    # Llama requires eager attention for consistent head tensor shapes.
    # Qwen uses its own attention implementation and may not accept this kwarg.
    model_kwargs: dict = dict(
        dtype=resolved_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **shared_kwargs,
    )
    model_name_lower = model_name.lower()
    if model_name_lower.startswith("meta-llama/") or "llama" in model_name_lower:
        model_kwargs["attn_implementation"] = "eager"
    elif prefer_flash_attention and ("qwen" in model_name_lower or "gpt-oss" in model_name_lower):
        model_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = _from_pretrained_with_token(
            AutoModelForCausalLM,
            model_name,
            token,
            **model_kwargs,
        )
    except Exception as exc:
        if model_kwargs.get("attn_implementation") != "flash_attention_2":
            raise
        retry_kwargs = dict(model_kwargs)
        retry_kwargs.pop("attn_implementation", None)
        print(
            "[load_model_and_tokenizer] flash_attention_2 load failed for "
            f"{model_name}: {exc}. Retrying without explicit attention override."
        )
        model = _from_pretrained_with_token(
            AutoModelForCausalLM,
            model_name,
            token,
            **retry_kwargs,
        )
    if target_device:
        model.to(torch.device(target_device))
    return model, tokenizer
