"""
Instrumentation for caching and patching attention head contexts.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import os

import torch

from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    Cache,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)


@dataclass
class HeadKey:
    example_id: str
    layer_idx: int
    head_idx: int


class HeadCache:
    """
    Stores cached head contexts captured during neutral runs.
    """

    def __init__(self):
        self._store: Dict[Tuple[str, int, int], torch.Tensor] = {}

    def set(self, key: HeadKey, value: torch.Tensor) -> None:
        self._store[(key.example_id, key.layer_idx, key.head_idx)] = value.detach().cpu()

    def get(self, key: HeadKey) -> Optional[torch.Tensor]:
        tensor = self._store.get((key.example_id, key.layer_idx, key.head_idx))
        return tensor

    def clear_example(self, example_id: str) -> None:
        to_delete = [k for k in self._store if k[0] == example_id]
        for key in to_delete:
            del self._store[key]


class HeadPatchRegistry:
    """
    Global state machine describing whether we are caching or patching.
    """

    def __init__(self):
        self.mode: str = "off"  # off | cache | patch
        self.example_id: Optional[str] = None
        self.pos_idx: Optional[int] = None
        self.target_layer: Optional[int] = None
        self.target_head: Optional[int] = None
        self.cache = HeadCache()
        self.mean_ablation: Dict[int, Dict[int, torch.Tensor]] = {}
        self.ablation_pos_idx: Optional[int] = None
        # Debug logging path can be set via env HEAD_PATCH_DEBUG_LOG; defaults fall back to RESULTS_DIR/exp2 or results/exp2 locally.
        self.debug_log_path = self._resolve_debug_path()

    @staticmethod
    def _resolve_debug_path() -> Optional[Path]:
        # Priority: explicit env; RESULTS_DIR (Modal volume); local results.
        if os.environ.get("HEAD_PATCH_DEBUG_LOG"):
            return Path(os.environ["HEAD_PATCH_DEBUG_LOG"])
        if os.environ.get("RESULTS_DIR"):
            return Path(os.environ["RESULTS_DIR"]) / "exp2" / "patch_debug.log"
        return Path("results/exp2/patch_debug.log")

    def _append_debug(self, line: str) -> None:
        """Best-effort append to debug log; never raise."""
        path = self.debug_log_path
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(line + "\n")
        except Exception:
            # Swallow all errors to avoid interfering with model forward.
            pass

    def reset(self):
        self.mode = "off"
        self.example_id = None
        self.pos_idx = None
        self.target_layer = None
        self.target_head = None
        return self

    def activate_cache(self, example_id: str, pos_idx: int):
        self.mode = "cache"
        self.example_id = example_id
        self.pos_idx = pos_idx
        self.target_layer = None
        self.target_head = None
        return self

    def activate_patch(self, example_id: str, layer_idx: int, head_idx: int, pos_idx: int):
        self.mode = "patch"
        self.example_id = example_id
        self.target_layer = layer_idx
        self.target_head = head_idx
        self.pos_idx = pos_idx
        return self

    def set_mean_ablation(self, vectors: Dict[Tuple[int, int], torch.Tensor], pos_idx: int):
        self.mean_ablation = {}
        for (layer, head), tensor in vectors.items():
            self.mean_ablation.setdefault(layer, {})[head] = tensor.detach().cpu()
        self.ablation_pos_idx = pos_idx

    def clear_mean_ablation(self):
        self.mean_ablation = {}
        self.ablation_pos_idx = None

    def process(
        self,
        layer_idx: int,
        attn_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        attn_output: [batch, num_heads, seq_len, head_dim]
        """
        if self.mode in {"cache", "patch"} and self.example_id is not None and self.pos_idx is not None:
            pos = min(self.pos_idx, attn_output.shape[2] - 1)
            num_heads = attn_output.shape[1]
            if self.mode == "cache":
                for head_idx in range(num_heads):
                    key = HeadKey(self.example_id, layer_idx, head_idx)
                    self.cache.set(key, attn_output[0, head_idx, pos, :])
            elif self.mode == "patch" and layer_idx == self.target_layer:
                if self.target_head < num_heads:
                    key = HeadKey(self.example_id, layer_idx, self.target_head)
                    cached = self.cache.get(key)
                    if cached is not None:
                        # DEBUG: measure how different the cached vs original values are
                        original = attn_output[0, self.target_head, pos, :].clone()
                        cached_tensor = cached.to(attn_output.device)
                        diff_norm = (original - cached_tensor).norm().item()
                        orig_norm = original.norm().item()
                        ratio = diff_norm / orig_norm if orig_norm > 0 else float("nan")
                        # Log only a small subset to avoid huge files
                        if layer_idx <= 2 and self.target_head == 0:
                            self._append_debug(
                                f"PATCH L{layer_idx}H{self.target_head} pos={pos}: "
                                f"diff_norm={diff_norm:.6f}, orig_norm={orig_norm:.6f}, ratio={ratio:.6f}, "
                                f"cache_shape={tuple(cached_tensor.shape)}"
                            )
                        attn_output = attn_output.clone()
                        attn_output[:, self.target_head, pos, :] = cached_tensor

        ablation_layer = self.mean_ablation.get(layer_idx) if self.mean_ablation else None
        ablation_pos = self.ablation_pos_idx if self.ablation_pos_idx is not None else self.pos_idx
        if ablation_layer and ablation_pos is not None:
            pos = min(ablation_pos, attn_output.shape[2] - 1)
            num_heads = attn_output.shape[1]
            attn_output = attn_output.clone()
            for head_idx, vec in ablation_layer.items():
                if head_idx < num_heads:
                    attn_output[:, head_idx, pos, :] = vec.to(attn_output.device)

        return attn_output


def instrument_model_for_head_patching(model, registry: HeadPatchRegistry, force: bool = False) -> None:
    """
    Monkey-patch attention modules to intercept head contexts.
    """
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if getattr(attn, "_head_patch_wrapped", False) and not force:
            continue
        original_forward = attn.forward  # noqa: F841 - kept for potential future use

        def patched_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_values: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            _layer_idx=layer_idx,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            attention_interface = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

            attn_output = registry.process(_layer_idx, attn_output)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        attn.forward = patched_forward.__get__(attn, attn.__class__)  # bind method
        attn._head_patch_wrapped = True


@contextmanager
def cache_mode(registry: HeadPatchRegistry, example_id: str, pos_idx: int):
    registry.activate_cache(example_id, pos_idx)
    try:
        yield registry
    finally:
        registry.reset()


@contextmanager
def patch_mode(
    registry: HeadPatchRegistry,
    example_id: str,
    layer_idx: int,
    head_idx: int,
    pos_idx: int,
):
    registry.activate_patch(example_id, layer_idx, head_idx, pos_idx)
    try:
        yield registry
    finally:
        registry.reset()


@contextmanager
def mean_ablation_mode(
    registry: HeadPatchRegistry,
    head_vectors: Dict[Tuple[int, int], torch.Tensor],
    pos_idx: int,
):
    registry.set_mean_ablation(head_vectors, pos_idx)
    try:
        yield registry
    finally:
        registry.clear_mean_ablation()


def ensure_head_patching(model, registry: Optional[HeadPatchRegistry] = None, force: bool = False) -> HeadPatchRegistry:
    """
    Ensure the model has been instrumented and return the registry used.
    """
    registry = registry or HeadPatchRegistry()
    instrument_model_for_head_patching(model, registry, force=force)
    return registry


