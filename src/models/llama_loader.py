"""
Helpers for loading Llama-family models via Hugging Face Transformers.
"""

from __future__ import annotations

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


def load_model_and_tokenizer(
    model_name: str,
    device: Optional[str] = None,
    dtype: Optional[Union[str, torch.dtype]] = "auto",
    use_auth_token: Optional[str] = None,
):
    """
    Load a model + tokenizer pair with sensible defaults for this project.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    resolved_dtype = _resolve_dtype(dtype)
    device_map = None
    target_device = device
    if device is None or device == "auto":
        device_map = "auto"
        target_device = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=resolved_dtype,
        device_map=device_map,
        trust_remote_code=False,
        use_auth_token=use_auth_token,
        attn_implementation="eager",  # Required for consistent head tensor shapes
    )
    if target_device:
        model.to(torch.device(target_device))
    return model, tokenizer


