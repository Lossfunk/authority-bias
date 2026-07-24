"""Optimized hook-based utilities for activation extraction and intervention.

Key improvements over output_hidden_states=True:
1. Selective layer capture - only hooks on layers you need.
   For a 64-layer model capturing 5 layers, this saves ~12x GPU memory.
2. Immediate CPU offload in hooks - frees GPU memory during forward pass.
3. Capping intervention - constrains activations along a direction.
4. Pin-memory support for async CPU transfer.
"""

from __future__ import annotations

import torch
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch import nn

from src.mechanism.hooks import get_transformer_layers, get_hidden_size


def _extract_hidden(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output:
        return output[0]
    raise ValueError("Hook output does not contain a hidden-state tensor")


def _replace_hidden(output, hidden):
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple) and output:
        return (hidden, *output[1:])
    raise ValueError("Hook output does not contain a hidden-state tensor")


class SelectiveLayerCapture:
    """Capture block outputs at specific layers using forward hooks.

    Usage:
        capture = SelectiveLayerCapture(model, target_layers=[22, 23, 24])
        with capture:
            outputs = model(**inputs, use_cache=False)
        acts = capture.extract_positions(positions_per_item, batch_size)
        capture.clear()
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: Optional[List[int]] = None,
        save_dtype: torch.dtype = torch.float16,
        pin_memory: bool = True,
    ):
        self.model = model
        self.layers = get_transformer_layers(model)
        self.n_layers = len(self.layers)
        self.target_layers = (
            target_layers if target_layers is not None
            else list(range(self.n_layers))
        )
        self.save_dtype = save_dtype
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self._captured: Dict[int, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHook] = []

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            hidden = _extract_hidden(output)
            t = hidden.detach().to(dtype=self.save_dtype, device="cpu", non_blocking=True)
            if self.pin_memory:
                t = t.pin_memory()
            self._captured[layer_idx] = t
        return hook

    def __enter__(self):
        for layer_idx in self.target_layers:
            h = self.layers[layer_idx].register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(h)
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def extract_positions(
        self,
        positions_per_item: List[Dict[str, Optional[int]]],
        batch_size: int,
        position_names: Sequence[str] = ("last_token", "instruction_last", "endorsement_last"),
    ) -> Dict[str, torch.Tensor]:
        """Extract activations at specific positions for each batch item.

        Returns:
            {position_name: Tensor(batch, n_target_layers, hidden_dim)}
        """
        h_dim = next(iter(self._captured.values())).shape[-1]
        n_target = len(self.target_layers)

        result: Dict[str, torch.Tensor] = {}
        for pos_name in position_names:
            buf = torch.zeros(batch_size, n_target, h_dim, dtype=self.save_dtype)
            for item_idx in range(batch_size):
                pos = positions_per_item[item_idx].get(pos_name)
                if pos is None:
                    continue
                for li, layer_idx in enumerate(self.target_layers):
                    buf[item_idx, li] = self._captured[layer_idx][item_idx, pos, :]
            result[pos_name] = buf
        return result

    def extract_all_positions_flat(
        self,
        positions_per_item: List[Dict[str, Optional[int]]],
        batch_size: int,
        position_names: Sequence[str] = ("last_token", "instruction_last", "endorsement_last"),
    ) -> Dict[str, torch.Tensor]:
        """Like extract_positions but returns (batch, n_all_layers, hidden_dim)
        where n_all_layers = n_model_layers + 1 (for backward compat with hidden_states indexing).

        Non-target layers are filled with zeros.
        """
        h_dim = next(iter(self._captured.values())).shape[-1]
        n_all = self.n_layers + 1  # +1 for embeddings layer at index 0

        result: Dict[str, torch.Tensor] = {}
        for pos_name in position_names:
            buf = torch.zeros(batch_size, n_all, h_dim, dtype=self.save_dtype)
            for item_idx in range(batch_size):
                pos = positions_per_item[item_idx].get(pos_name)
                if pos is None:
                    continue
                for layer_idx in self.target_layers:
                    hs_idx = layer_idx + 1  # hidden_states[0] = embeddings
                    buf[item_idx, hs_idx] = self._captured[layer_idx][item_idx, pos, :]
            result[pos_name] = buf
        return result

    def clear(self):
        self._captured.clear()


def collect_hidden_states_batch_hooked(
    model: nn.Module,
    device: torch.device,
    prepared_batch: Sequence[Dict],
    token_id_a: int,
    token_id_b: int,
    save_dtype: torch.dtype,
    target_layers: Optional[List[int]] = None,
    position_names: Sequence[str] = ("last_token", "instruction_last", "endorsement_last"),
) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, float], int]]:
    """Drop-in replacement for _collect_hidden_states_batch using hooks.

    When target_layers is provided, only those layers are captured via hooks.
    When target_layers is None, ALL layers are captured (equivalent to output_hidden_states=True
    but still using hooks for consistent behavior).
    """
    if not prepared_batch:
        return []

    input_ids_list = [row["input_ids"] for row in prepared_batch]
    attn_list = [row["attention_mask"] for row in prepared_batch]
    max_len = max(int(ids.shape[0]) for ids in input_ids_list)
    batch_size = len(prepared_batch)

    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    seq_lens = []
    for i, (ids, mask) in enumerate(zip(input_ids_list, attn_list)):
        seq_len = int(ids.shape[0])
        seq_lens.append(seq_len)
        input_ids[i, :seq_len] = ids
        attention_mask[i, :seq_len] = mask

    model_inputs = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }

    capture = SelectiveLayerCapture(
        model, target_layers=target_layers, save_dtype=save_dtype
    )
    with capture:
        with torch.inference_mode():
            outputs = model(**model_inputs, use_cache=False)

    batch_indices = torch.arange(batch_size, device=outputs.logits.device)
    last_indices = attention_mask.to(outputs.logits.device).sum(dim=1) - 1
    final_logits = outputs.logits[batch_indices, last_indices, :]

    positions_per_item = [row["positions"] for row in prepared_batch]
    views_batch = capture.extract_all_positions_flat(
        positions_per_item, batch_size, position_names
    )
    capture.clear()

    results = []
    for i, row in enumerate(prepared_batch):
        views: Dict[str, torch.Tensor] = {}
        for name in position_names:
            pos = row["positions"].get(name)
            if pos is None:
                continue
            views[name] = views_batch[name][i]

        logit_stats = {
            "logit_a": float(final_logits[i, token_id_a].item()),
            "logit_b": float(final_logits[i, token_id_b].item()),
        }
        results.append((views, logit_stats, seq_lens[i]))

    return results


class PositionPatcherBatched:
    """Apply additive patches at per-item positions within a batch.

    Supports variable patch positions per batch element.
    """

    def __init__(
        self,
        module: nn.Module,
        token_positions: List[int],
        vector: torch.Tensor,
    ):
        self.module = module
        self.token_positions = token_positions
        self.vector = vector
        self.handle = None

    def __enter__(self):
        positions = self.token_positions
        vec = self.vector

        def hook(_module, _inputs, output):
            hidden = _extract_hidden(output)
            patched = hidden.clone()
            for i, pos in enumerate(positions):
                if pos >= 0:
                    patched[i, pos, :] = patched[i, pos, :] + vec
            return _replace_hidden(output, patched)

        self.handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
        return False


class ActivationCapper:
    """Cap activations along a direction at a threshold (one-sided or two-sided).

    Following the Assistant Axis methodology: prevents drift beyond a normal range
    along a specified direction.
    """

    def __init__(
        self,
        module: nn.Module,
        direction: torch.Tensor,
        upper_threshold: Optional[float] = None,
        lower_threshold: Optional[float] = None,
        positions: str = "all",
    ):
        self.module = module
        self.direction = direction / (direction.norm() + 1e-8)
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.positions = positions
        self.handle = None

    def __enter__(self):
        v = self.direction
        upper = self.upper_threshold
        lower = self.lower_threshold
        pos_mode = self.positions

        def hook(_module, _inputs, output):
            hidden = _extract_hidden(output)
            d = v.to(hidden.device, dtype=hidden.dtype)

            if pos_mode == "all":
                proj = torch.einsum("bld,d->bl", hidden, d)
                correction = torch.zeros_like(proj)
                if upper is not None:
                    correction += (proj - upper).clamp(min=0.0)
                if lower is not None:
                    correction += (proj - lower).clamp(max=0.0)
                if correction.abs().sum() > 0:
                    patched = hidden - torch.einsum("bl,d->bld", correction, d)
                    return _replace_hidden(output, patched)
            else:
                proj = torch.einsum("bd,d->b", hidden[:, -1, :], d)
                correction = torch.zeros_like(proj)
                if upper is not None:
                    correction += (proj - upper).clamp(min=0.0)
                if lower is not None:
                    correction += (proj - lower).clamp(max=0.0)
                if correction.abs().sum() > 0:
                    patched = hidden.clone()
                    patched[:, -1, :] -= torch.einsum("b,d->bd", correction, d)
                    return _replace_hidden(output, patched)
            return output

        self.handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
        return False


def setup_h100_optimizations():
    """Apply H100/Hopper-specific optimizations."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def try_compile_model(model: nn.Module, backend: str = "inductor") -> nn.Module:
    """Attempt torch.compile; fall back gracefully if unavailable."""
    try:
        compiled = torch.compile(model, backend=backend, mode="reduce-overhead")
        print(f"[hooks_v2] torch.compile succeeded (backend={backend})")
        return compiled
    except Exception as exc:
        print(f"[hooks_v2] torch.compile failed ({exc}), using eager mode")
        return model
