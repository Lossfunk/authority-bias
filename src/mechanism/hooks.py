from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch import nn


COMPONENT_ORDER: Tuple[str, ...] = ("block_output", "attention_output", "mlp_output")


@dataclass(frozen=True)
class InterventionSpec:
    layer_idx: int
    component: str
    position_label: str
    intervention_type: str
    source_tensor: Optional[torch.Tensor] = None
    alpha: float = 1.0
    mean_tensor: Optional[torch.Tensor] = None


def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model: could not locate transformer layers")


def get_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    for attr in ("hidden_size", "n_embd", "d_model"):
        if config is not None and hasattr(config, attr):
            return int(getattr(config, attr))
    raise ValueError("Could not infer model hidden size")


def get_component_modules(model: nn.Module) -> Dict[Tuple[int, str], nn.Module]:
    layers = get_transformer_layers(model)
    mapping: Dict[Tuple[int, str], nn.Module] = {}
    for layer_idx, layer in enumerate(layers):
        mapping[(layer_idx, "block_output")] = layer
        if not hasattr(layer, "self_attn"):
            raise ValueError(f"Layer {layer_idx} missing self_attn module")
        if not hasattr(layer, "mlp"):
            raise ValueError(f"Layer {layer_idx} missing mlp module")
        mapping[(layer_idx, "attention_output")] = layer.self_attn
        mapping[(layer_idx, "mlp_output")] = layer.mlp
    return mapping


def _extract_hidden(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output:
        first = output[0]
        if torch.is_tensor(first):
            return first
    raise ValueError("Hook output does not contain a hidden-state tensor")


def _replace_hidden(output, hidden):
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple) and output:
        return (hidden, *output[1:])
    raise ValueError("Hook output does not contain a hidden-state tensor")


class ModelHookRunner:
    def __init__(self, model: nn.Module):
        self.model = model
        self.layers = get_transformer_layers(model)
        self.component_modules = get_component_modules(model)
        self.hidden_size = get_hidden_size(model)

    def _make_position_indices(self, token_positions: Mapping[str, Sequence[int]]) -> Dict[str, List[int]]:
        return {label: [int(idx) for idx in indices] for label, indices in token_positions.items()}

    def collect(
        self,
        *,
        model_inputs: Mapping[str, torch.Tensor],
        token_positions: Mapping[str, Sequence[int]],
        components: Sequence[str] = COMPONENT_ORDER,
    ) -> torch.Tensor:
        positions = self._make_position_indices(token_positions)
        position_labels = list(positions)
        component_to_index = {component: idx for idx, component in enumerate(components)}
        cache: MutableMapping[Tuple[int, str], torch.Tensor] = {}

        def make_collect_hook(layer_idx: int, component: str) -> Callable:
            def hook(_module, _inputs, output):
                hidden = _extract_hidden(output)
                cache[(layer_idx, component)] = hidden.detach().cpu()
                return output
            return hook

        with ExitStack() as stack:
            for (layer_idx, component), module in self.component_modules.items():
                if component not in component_to_index:
                    continue
                stack.enter_context(module.register_forward_hook(make_collect_hook(layer_idx, component)))
            with torch.inference_mode():
                _ = self.model(**model_inputs, use_cache=False)

        n_layers = len(self.layers)
        tensor = torch.zeros(
            n_layers,
            len(components),
            len(position_labels),
            self.hidden_size,
            dtype=torch.float32,
        )
        for layer_idx in range(n_layers):
            for component in components:
                hidden = cache.get((layer_idx, component))
                if hidden is None:
                    continue
                hidden = hidden[0]
                comp_idx = component_to_index[component]
                for pos_idx, label in enumerate(position_labels):
                    indices = [min(max(index, 0), hidden.shape[0] - 1) for index in positions[label]]
                    slice_tensor = hidden[indices]
                    if slice_tensor.ndim == 1:
                        pooled = slice_tensor
                    else:
                        pooled = slice_tensor.mean(dim=0)
                    tensor[layer_idx, comp_idx, pos_idx] = pooled.to(torch.float32)
        return tensor

    def intervene(
        self,
        *,
        model_inputs: Mapping[str, torch.Tensor],
        token_positions: Mapping[str, Sequence[int]],
        specs: Sequence[InterventionSpec],
    ):
        positions = self._make_position_indices(token_positions)
        by_target: Dict[Tuple[int, str], List[InterventionSpec]] = {}
        for spec in specs:
            by_target.setdefault((spec.layer_idx, spec.component), []).append(spec)

        def make_patch_hook(layer_idx: int, component: str) -> Callable:
            def hook(_module, _inputs, output):
                hidden = _extract_hidden(output)
                updated = hidden.clone()
                for spec in by_target.get((layer_idx, component), []):
                    if spec.position_label not in positions:
                        continue
                    idxs = [min(max(index, 0), hidden.shape[1] - 1) for index in positions[spec.position_label]]
                    if spec.intervention_type == "zero":
                        replacement = torch.zeros(hidden.shape[-1], dtype=hidden.dtype, device=hidden.device)
                    elif spec.intervention_type == "mean_replace":
                        if spec.mean_tensor is None:
                            raise ValueError("mean_replace requires mean_tensor")
                        replacement = spec.mean_tensor.to(hidden.device, dtype=hidden.dtype)
                    elif spec.intervention_type == "replace":
                        if spec.source_tensor is None:
                            raise ValueError("replace requires source_tensor")
                        replacement = spec.source_tensor.to(hidden.device, dtype=hidden.dtype)
                    elif spec.intervention_type == "interpolate":
                        if spec.source_tensor is None:
                            raise ValueError("interpolate requires source_tensor")
                        source = spec.source_tensor.to(hidden.device, dtype=hidden.dtype)
                        for idx in idxs:
                            updated[:, idx, :] = hidden[:, idx, :] + spec.alpha * (source - hidden[:, idx, :])
                        continue
                    elif spec.intervention_type == "subtract":
                        if spec.source_tensor is None:
                            raise ValueError("subtract requires source_tensor")
                        replacement = None
                        source = spec.source_tensor.to(hidden.device, dtype=hidden.dtype)
                        for idx in idxs:
                            updated[:, idx, :] = hidden[:, idx, :] - spec.alpha * source
                        continue
                    else:
                        raise ValueError(f"Unknown intervention type: {spec.intervention_type}")
                    for idx in idxs:
                        updated[:, idx, :] = replacement
                return _replace_hidden(output, updated)
            return hook

        handles = []
        for (layer_idx, component), module in self.component_modules.items():
            if (layer_idx, component) in by_target:
                handles.append(module.register_forward_hook(make_patch_hook(layer_idx, component)))
        try:
            with torch.inference_mode():
                return self.model(**model_inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
