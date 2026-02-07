"""
Helpers for loading Llama-family models via Hugging Face Transformers.
"""

from __future__ import annotations

import os
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
    if token:
        try:
            return factory.from_pretrained(model_name, token=token, **kwargs)
        except TypeError:
            # Fall back for older Transformers
            return factory.from_pretrained(model_name, use_auth_token=token, **kwargs)
    return factory.from_pretrained(model_name, **kwargs)


def load_model_and_tokenizer(
    model_name: str,
    device: Optional[str] = None,
    dtype: Optional[Union[str, torch.dtype]] = "auto",
    use_auth_token: Optional[str] = None,
):
    """
    Load a model + tokenizer pair with sensible defaults for this project.
    """
    # Prefer explicit token, otherwise fall back to common env vars set via Modal secret.
    token = use_auth_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    tokenizer = _from_pretrained_with_token(AutoTokenizer, model_name, token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    resolved_dtype = _resolve_dtype(dtype)
    device_map = None
    target_device = device
    if device is None or device == "auto":
        device_map = "auto"
        target_device = None

    model = _from_pretrained_with_token(
        AutoModelForCausalLM,
        model_name,
        token,
        dtype=resolved_dtype,
        device_map=device_map,
        trust_remote_code=False,
        attn_implementation="eager",  # Required for consistent head tensor shapes
    )
    if target_device:
        model.to(torch.device(target_device))
    return model, tokenizer
