"""
Instrumentation for caching and patching attention head contexts.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import os

import torch
from torch import nn

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

    def get_example(self, example_id: str) -> Dict[Tuple[int, int], torch.Tensor]:
        """Return {(layer_idx, head_idx): tensor} for a given example_id."""
        out: Dict[Tuple[int, int], torch.Tensor] = {}
        for (ex_id, layer, head), tensor in self._store.items():
            if ex_id == example_id:
                out[(layer, head)] = tensor
        return out

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
        # Multi-head value patching (e.g. patch selected heads to cached neutral values)
        self.value_patch: Dict[int, Dict[int, torch.Tensor]] = {}
        self.value_patch_pos_idx: Optional[int] = None
        self.value_patch_alpha: float = 1.0
        self.value_patch_mode: str = "replace"  # "replace" | "interp"
        # MLP ablation state
        self.mlp_ablation_layers: Set[int] = set()
        self.mlp_ablation_mode: str = "zero"  # "zero" | "mean"
        self.mlp_mean_cache: Dict[int, torch.Tensor] = {}
        # Debug logging path can be set via env HEAD_PATCH_DEBUG_LOG; defaults fall back to RESULTS_DIR/head_patching or results/head_patching locally.
        self.debug_log_path = self._resolve_debug_path()

    @staticmethod
    def _resolve_debug_path() -> Optional[Path]:
        """
        Opt-in debug logging. Set HEAD_PATCH_DEBUG_LOG explicitly to enable.
        Defaults to None to avoid creating local files (e.g., during builds).
        """
        if os.environ.get("HEAD_PATCH_DEBUG_LOG"):
            return Path(os.environ["HEAD_PATCH_DEBUG_LOG"])
        if os.environ.get("RESULTS_DIR"):
            return Path(os.environ["RESULTS_DIR"]) / "head_patching" / "patch_debug.log"
        # Default: no debug log
        return None

    def _append_debug(self, line: str) -> None:
        """Best-effort append to debug log; never raise."""
        path = self.debug_log_path
        if path is None:
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

    def set_value_patch(
        self,
        values: Dict[Tuple[int, int], torch.Tensor],
        pos_idx: int,
        *,
        alpha: float = 1.0,
        mode: str = "replace",
    ):
        """Patch selected head outputs at a given position.

        mode:
          - "replace": set head output to cached value
          - "interp": set head output to orig + alpha * (cached - orig)
        """
        self.value_patch = {}
        for (layer, head), tensor in values.items():
            self.value_patch.setdefault(layer, {})[head] = tensor.detach().cpu()
        self.value_patch_pos_idx = pos_idx
        self.value_patch_alpha = float(alpha)
        self.value_patch_mode = mode

    def clear_value_patch(self):
        self.value_patch = {}
        self.value_patch_pos_idx = None
        self.value_patch_alpha = 1.0
        self.value_patch_mode = "replace"

    def set_mlp_ablation(self, layers: List[int], mode: str = "zero", mean_cache: Optional[Dict[int, torch.Tensor]] = None):
        """Set which MLP layers to ablate and how."""
        self.mlp_ablation_layers = set(layers)
        self.mlp_ablation_mode = mode
        if mean_cache:
            self.mlp_mean_cache = {k: v.detach().cpu() for k, v in mean_cache.items()}

    def clear_mlp_ablation(self):
        self.mlp_ablation_layers = set()
        self.mlp_ablation_mode = "zero"
        self.mlp_mean_cache = {}

    def process_mlp(self, layer_idx: int, mlp_output: torch.Tensor) -> torch.Tensor:
        """Process MLP output, applying ablation if configured."""
        if layer_idx not in self.mlp_ablation_layers:
            return mlp_output

        if self.mlp_ablation_mode == "zero":
            return torch.zeros_like(mlp_output)
        elif self.mlp_ablation_mode == "mean" and layer_idx in self.mlp_mean_cache:
            mean_vec = self.mlp_mean_cache[layer_idx].to(mlp_output.device)
            # Broadcast mean across sequence
            return mean_vec.unsqueeze(0).unsqueeze(0).expand_as(mlp_output)
        return mlp_output

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

        value_patch_layer = self.value_patch.get(layer_idx) if self.value_patch else None
        if value_patch_layer and self.value_patch_pos_idx is not None:
            pos = min(self.value_patch_pos_idx, attn_output.shape[2] - 1)
            num_heads = attn_output.shape[1]
            attn_output = attn_output.clone()
            for head_idx, target in value_patch_layer.items():
                if head_idx >= num_heads:
                    continue
                target_tensor = target.to(attn_output.device)
                if self.value_patch_mode == "interp":
                    orig = attn_output[:, head_idx, pos, :].clone()
                    attn_output[:, head_idx, pos, :] = orig + self.value_patch_alpha * (target_tensor - orig)
                else:
                    attn_output[:, head_idx, pos, :] = target_tensor

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


@contextmanager
def head_value_patch_mode(
    registry: HeadPatchRegistry,
    head_values: Dict[Tuple[int, int], torch.Tensor],
    pos_idx: int,
    *,
    alpha: float = 1.0,
    mode: str = "replace",
):
    registry.set_value_patch(head_values, pos_idx, alpha=alpha, mode=mode)
    try:
        yield registry
    finally:
        registry.clear_value_patch()


def ensure_head_patching(model, registry: Optional[HeadPatchRegistry] = None, force: bool = False) -> HeadPatchRegistry:
    """
    Ensure the model has been instrumented and return the registry used.
    """
    registry = registry or HeadPatchRegistry()
    instrument_model_for_head_patching(model, registry, force=force)
    return registry


def instrument_model_for_mlp_ablation(model, registry: HeadPatchRegistry, force: bool = False) -> None:
    """
    Monkey-patch MLP modules to allow ablation.
    """
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if getattr(mlp, "_mlp_ablation_wrapped", False) and not force:
            continue

        original_forward = mlp.forward

        def patched_mlp_forward(
            self,
            x: torch.Tensor,
            _layer_idx=layer_idx,
            _original=original_forward,
        ):
            output = _original(x)
            return registry.process_mlp(_layer_idx, output)

        mlp.forward = patched_mlp_forward.__get__(mlp, mlp.__class__)
        mlp._mlp_ablation_wrapped = True


@contextmanager
def mlp_ablation_mode(
    registry: HeadPatchRegistry,
    layers: List[int],
    mode: str = "zero",
    mean_cache: Optional[Dict[int, torch.Tensor]] = None,
):
    """Context manager to ablate MLP layers."""
    registry.set_mlp_ablation(layers, mode, mean_cache)
    try:
        yield registry
    finally:
        registry.clear_mlp_ablation()


def ensure_full_instrumentation(
    model,
    registry: Optional[HeadPatchRegistry] = None,
    force: bool = False,
) -> HeadPatchRegistry:
    """
    Ensure both attention and MLP instrumentation are applied.
    """
    registry = registry or HeadPatchRegistry()
    instrument_model_for_head_patching(model, registry, force=force)
    instrument_model_for_mlp_ablation(model, registry, force=force)
    return registry


