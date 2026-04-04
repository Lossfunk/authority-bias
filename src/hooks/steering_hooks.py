"""
Residual stream capture and steering hooks.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from src.mechanism.hooks import get_transformer_layers


@dataclass
class CaptureRecord:
    layer: int
    position: int
    tensor: torch.Tensor


class ResidualHookManager:
    """
    Utility to register / remove hooks for residual capture + injection.
    """

    def __init__(self, model):
        self.model = model
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def clear(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _get_layer(self, layer_idx: int):
        return get_transformer_layers(self.model)[layer_idx]

    def register_capture(
        self,
        layer_idx: int,
        pos_idx: int,
        storage: Dict[int, CaptureRecord],
    ) -> None:
        """
        Capture the residual stream entering the specified layer.
        """

        def _hook(module, inputs):
            hidden_states = inputs[0]
            pos = min(pos_idx, hidden_states.shape[1] - 1)
            storage[layer_idx] = CaptureRecord(layer_idx, pos, hidden_states[:, pos, :].detach().cpu())

        handle = self._get_layer(layer_idx).register_forward_pre_hook(_hook, with_kwargs=False)
        self._handles.append(handle)

    def register_injection(
        self,
        layer_idx: int,
        pos_idx: int,
        vector: torch.Tensor,
        scale: float = 1.0,
    ) -> None:
        """
        Inject a steering vector at the specified layer / position.
        """

        def _hook(module, inputs):
            hidden_states = inputs[0]
            pos = min(pos_idx, hidden_states.shape[1] - 1)
            vec = vector.to(hidden_states.device)
            hidden_states = hidden_states.clone()
            hidden_states[:, pos, :] += scale * vec
            new_inputs = (hidden_states, *inputs[1:])
            return new_inputs

        handle = self._get_layer(layer_idx).register_forward_pre_hook(_hook, with_kwargs=False)
        self._handles.append(handle)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.clear()


@contextmanager
def capture_residual_stream(
    model,
    layer_idx: int,
    pos_idx: int,
) -> Dict[int, CaptureRecord]:
    """
    Context manager to capture residual activations for a single layer.
    """
    storage: Dict[int, CaptureRecord] = {}
    with ResidualHookManager(model) as manager:
        manager.register_capture(layer_idx, pos_idx, storage)
        yield storage


@contextmanager
def inject_steering_vector(
    model,
    layer_idx: int,
    pos_idx: int,
    vector: torch.Tensor,
    scale: float = 1.0,
):
    """
    Context manager to temporarily inject a steering vector.
    """
    with ResidualHookManager(model) as manager:
        manager.register_injection(layer_idx, pos_idx, vector, scale)
        yield

