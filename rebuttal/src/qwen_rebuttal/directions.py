from __future__ import annotations

import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .hooks import MultiResidualHooks, ResidualHook
from .io import atomic_write_json, sha256_file
from .model import QwenAdapter
from .types import PromptExample


@dataclass(frozen=True)
class LayerDirections:
    layer: int
    source: torch.Tensor
    user: torch.Tensor
    shared: torch.Tensor
    identity: torch.Tensor
    source_identity_coefficient: float
    user_identity_coefficient: float
    geometry: dict[str, float]


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if denominator <= 0:
        return math.nan
    return float(torch.dot(left, right) / denominator)


def decompose_directions(
    layer: int,
    source_centroid: torch.Tensor,
    user_centroid: torch.Tensor,
    neutral_centroid: torch.Tensor,
) -> LayerDirections:
    source = source_centroid - neutral_centroid
    user = user_centroid - neutral_centroid
    shared = (source + user) / 2
    raw_identity = source - user
    shared_norm_sq = torch.dot(shared, shared)
    if shared_norm_sq <= 0:
        raise RuntimeError(f"Shared direction at layer {layer} has zero norm")
    identity = raw_identity - torch.dot(raw_identity, shared) / shared_norm_sq * shared
    identity_norm = torch.linalg.vector_norm(identity)
    if identity_norm <= 0:
        raise RuntimeError(f"Identity residual at layer {layer} has zero norm")
    identity_unit = identity / identity_norm
    source_coefficient = float(torch.dot(source, identity_unit))
    user_coefficient = float(torch.dot(user, identity_unit))
    raw_norm = torch.linalg.vector_norm(raw_identity)
    geometry = {
        "source_user_cosine": _cosine(source, user),
        "source_norm": float(torch.linalg.vector_norm(source)),
        "user_norm": float(torch.linalg.vector_norm(user)),
        "shared_norm": float(torch.linalg.vector_norm(shared)),
        "raw_identity_norm": float(raw_norm),
        "identity_residual_norm": float(identity_norm),
        "identity_residual_fraction": float(identity_norm / raw_norm) if raw_norm > 0 else math.nan,
        "source_identity_coefficient": source_coefficient,
        "user_identity_coefficient": user_coefficient,
    }
    return LayerDirections(
        layer=layer,
        source=source,
        user=user,
        shared=shared,
        identity=identity_unit,
        source_identity_coefficient=source_coefficient,
        user_identity_coefficient=user_coefficient,
        geometry=geometry,
    )


def collect_endpoint_activations(
    adapter: QwenAdapter,
    prompts: list[PromptExample],
    layers: tuple[int, ...],
    batch_size: int,
) -> dict[int, torch.Tensor]:
    collected: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    tokenizer = adapter.tokenizer
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        if any(prompt.cue_token_end is None for prompt in batch):
            raise RuntimeError("All direction-fitting prompts need verified cue endpoints")
        encoded = tokenizer(
            [prompt.rendered_text for prompt in batch],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(adapter.device)
        attention_mask = encoded["attention_mask"].to(adapter.device)
        padded_length = input_ids.shape[1]
        positions = [
            padded_length - int(mask.sum()) + int(prompt.cue_token_end)
            for prompt, mask in zip(batch, attention_mask, strict=True)
        ]
        hooks = [
            ResidualHook(
                module=adapter.layers[layer],
                layer=layer,
                mode="collect",
                phase="prefill",
                token_positions=positions,
            )
            for layer in layers
        ]
        with torch.inference_mode(), MultiResidualHooks(hooks):
            adapter.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
        for layer, hook in zip(layers, hooks, strict=True):
            if len(hook.collected) != len(batch):
                raise RuntimeError(f"Layer {layer} collection returned the wrong batch size")
            collected[layer].extend(hook.collected)
    return {layer: torch.stack(rows) for layer, rows in collected.items()}


def fit_layer_directions(
    activations: dict[str, dict[int, torch.Tensor]], layers: tuple[int, ...]
) -> dict[int, LayerDirections]:
    required = {"source", "user", "neutral"}
    if set(activations) != required:
        raise ValueError(f"Expected activation groups {required}, found {set(activations)}")
    return {
        layer: decompose_directions(
            layer,
            activations["source"][layer].mean(dim=0),
            activations["user"][layer].mean(dim=0),
            activations["neutral"][layer].mean(dim=0),
        )
        for layer in layers
    }


def shuffled_identity_controls(
    source_rows: torch.Tensor,
    user_rows: torch.Tensor,
    shared: torch.Tensor,
    seeds: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    pooled = torch.cat([source_rows, user_rows], dim=0)
    group_size = source_rows.shape[0]
    controls: dict[int, torch.Tensor] = {}
    shared_norm_sq = torch.dot(shared, shared)
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(pooled.shape[0], generator=generator)
        raw = pooled[order[:group_size]].mean(0) - pooled[order[group_size:]].mean(0)
        residual = raw - torch.dot(raw, shared) / shared_norm_sq * shared
        norm = torch.linalg.vector_norm(residual)
        if norm <= 0:
            raise RuntimeError(f"Degenerate shuffled control for seed {seed}")
        controls[seed] = residual / norm
    return controls


def matched_random_controls(
    reference: torch.Tensor, seeds: tuple[int, ...]
) -> dict[int, torch.Tensor]:
    controls: dict[int, torch.Tensor] = {}
    norm = torch.linalg.vector_norm(reference)
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        random = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
        controls[seed] = random / torch.linalg.vector_norm(random) * norm
    return controls


def save_directions(
    path: Path,
    fitted: dict[int, LayerDirections],
    shuffled: dict[int, dict[int, torch.Tensor]],
    random: dict[int, dict[int, torch.Tensor]],
) -> str:
    payload = {
        "layers": {
            layer: {
                "source": value.source,
                "user": value.user,
                "shared": value.shared,
                "identity": value.identity,
                "source_identity_coefficient": value.source_identity_coefficient,
                "user_identity_coefficient": value.user_identity_coefficient,
            }
            for layer, value in fitted.items()
        },
        "shuffled": shuffled,
        "random": random,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    atomic_write_json(
        path.with_suffix(".geometry.json"),
        {str(layer): asdict(value)["geometry"] for layer, value in fitted.items()},
    )
    return sha256_file(path)
