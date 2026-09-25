"""The submitted paper's own mitigation and causal-split vectors, rebuilt here.

Reviewer 1474 asks whether the CAA baseline in the submitted Table 6 was given a
fair search. Answering that means running optimized CAA against *the same vector
the submission actually mitigates with*, on the same items and the same readout.
The submission's mitigation vector is not the one the rest of this pipeline fits,
so it is rebuilt here from the submission's own arithmetic.

Two references, two different vectors
-------------------------------------
Both of the paper's constructions average a cue's correct- and wrong-endorsement
activations and subtract a reference. They differ only in the reference, and that
difference is what separates its two experiments:

  v_auth  = mean(source_C1, source_W1) - neutral-cue        (Table 6 mitigation)
  v_src   = mean(source_C1, source_W1) - no cue at all      (Table 4 causal split)
  v_usr   = mean(user_C1,   user_W1)   - no cue at all      (Table 4 causal split)

`v_auth` is referenced against a note whose speaker is unspecified, so the
endorsement is present in both terms and cancels: it isolates *who is speaking*.
`v_src` is referenced against a prompt with no note at all, so it carries the
endorsement as well as the speaker. Removing the first is a mitigation; removing
the second is a causal test of cue-family separability. Reporting one against
CAA while calling it the other is the confusion this module exists to prevent.

Source of the arithmetic, all in the submission tree:
  * `src/exp16/run_steering_test.py::_direction_from_store` -> `shared_within_label`
  * `src/exp20/run_source_user_authority_split.py::_fit_directions`
  * `src/exp20/common.py::derive_deconfound_payloads` (the residualization)

The assistant axis is *not* refitted. It is loaded from the submission's own
saved tensor so that `v_auth_resid` is the vector the submitted table was
produced with, rather than an approximation of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .directions import collect_final_activations, collect_span_activations
from .model import QwenAdapter
from .types import PromptExample


@dataclass(frozen=True)
class PaperVectors:
    """Unit-normalized copies of the submission's vectors at one layer."""

    layer: int
    authority: torch.Tensor
    assistant: torch.Tensor
    authority_resid: torch.Tensor
    source: torch.Tensor
    user: torch.Tensor
    geometry: dict[str, float]


def _unit(vector: torch.Tensor, name: str) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or norm <= 0:
        raise RuntimeError(f"Paper vector {name} has zero or non-finite norm")
    return vector / norm


def _residualize(vector: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Remove the component of `vector` along `axis`, as `exp19.common.residualize`."""
    unit_axis = _unit(axis, "assistant_axis")
    return vector - torch.dot(vector, unit_axis) * unit_axis


def load_assistant_axis(
    path: Path, *, layer: int, hidden_size: int
) -> tuple[torch.Tensor, int | None]:
    """Read the submission's assistant axis, matching `exp19.common.load_axis_vector`.

    The submission stores this tensor in three shapes across model families, and
    its loader accepts all three. The shapes are reproduced rather than
    normalised away, because a run that silently picked a different layer than
    the submitted table used would be indistinguishable from a correct one.

    Returns the unit axis together with the layer the stored tensor declares,
    which is not always the layer it is applied at.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    declared_layer: int | None = None
    if torch.is_tensor(payload):
        axis = payload.float()
        if axis.ndim == 1:
            vector = axis
        elif axis.ndim == 2:
            if not 0 <= layer < axis.shape[0]:
                raise RuntimeError(
                    f"Layer {layer} is outside the axis stack {tuple(axis.shape)} at {path}"
                )
            vector = axis[layer]
            declared_layer = layer
        else:
            raise RuntimeError(f"Unsupported assistant-axis rank {axis.ndim} at {path}")
    elif isinstance(payload, dict):
        declared = payload.get("layer")
        declared_layer = int(declared) if isinstance(declared, int) else None
        for key in ("axis_layer", "vector", "axis"):
            value = payload.get(key)
            if torch.is_tensor(value):
                vector = value.float()
                if vector.ndim == 2:
                    vector = vector[layer]
                    declared_layer = layer
                break
        else:
            raise RuntimeError(f"No assistant-axis tensor in {path}: keys={sorted(payload)}")
    else:
        raise RuntimeError(f"Unsupported assistant-axis payload {type(payload).__name__}")

    if vector.ndim != 1 or vector.shape[0] != hidden_size:
        raise RuntimeError(
            f"Assistant axis at {path} has shape {tuple(vector.shape)}, "
            f"expected ({hidden_size},)"
        )
    return _unit(vector, "assistant_axis"), declared_layer


def _centroid(activations: dict[int, torch.Tensor], layer: int) -> torch.Tensor:
    return activations[layer].mean(dim=0).float()


def build_paper_vectors(
    *,
    adapter: QwenAdapter,
    layer: int,
    source_prompts: list[PromptExample],
    user_prompts: list[PromptExample],
    neutral_prompts: list[PromptExample],
    no_cue_prompts: list[PromptExample],
    assistant_axis_path: Path,
    batch_size: int,
) -> PaperVectors:
    """Fit the paper's vector set at one layer from already-tokenized prompts.

    Cued conditions are collected across the whole cue span, which is the
    submission's `endorsement_span` fit position and also where every
    intervention below is applied. The no-cue condition has no span, so it is
    collected at its final token, which is the submission's own fallback.
    """
    layers = (layer,)
    span = {
        "source": collect_span_activations(adapter, source_prompts, layers, batch_size),
        "user": collect_span_activations(adapter, user_prompts, layers, batch_size),
        "neutral": collect_span_activations(adapter, neutral_prompts, layers, batch_size),
    }
    no_cue = collect_final_activations(adapter, no_cue_prompts, layers, batch_size)

    source_centroid = _centroid(span["source"], layer)
    user_centroid = _centroid(span["user"], layer)
    neutral_centroid = _centroid(span["neutral"], layer)
    no_cue_centroid = _centroid(no_cue, layer)

    authority = _unit(source_centroid - neutral_centroid, "authority")
    source = _unit(source_centroid - no_cue_centroid, "source")
    user = _unit(user_centroid - no_cue_centroid, "user")

    hidden_size = int(source_centroid.shape[0])
    assistant, declared_axis_layer = load_assistant_axis(
        assistant_axis_path, layer=layer, hidden_size=hidden_size
    )
    authority_resid = _unit(_residualize(authority, assistant), "authority_resid")

    geometry = {
        "layer": float(layer),
        "cos_authority_assistant": float(torch.dot(authority, assistant)),
        "cos_authority_residualized": float(torch.dot(authority, authority_resid)),
        "cos_source_user": float(torch.dot(source, user)),
        "cos_authority_source": float(torch.dot(authority, source)),
        "norm_authority_raw": float(
            torch.linalg.vector_norm(source_centroid - neutral_centroid)
        ),
        "norm_source_raw": float(torch.linalg.vector_norm(source_centroid - no_cue_centroid)),
        "norm_user_raw": float(torch.linalg.vector_norm(user_centroid - no_cue_centroid)),
        # The submission stores one assistant axis per family and applies it at
        # every sweep layer, so for some families the axis was fitted at a
        # different layer than the one it is applied at. That is reproduced, not
        # corrected, and recorded here so the run can be read honestly.
        "assistant_axis_declared_layer": (
            float(declared_axis_layer) if declared_axis_layer is not None else float("nan")
        ),
        "n_source": float(len(source_prompts)),
        "n_user": float(len(user_prompts)),
        "n_neutral": float(len(neutral_prompts)),
        "n_no_cue": float(len(no_cue_prompts)),
    }
    return PaperVectors(
        layer=layer,
        authority=authority,
        assistant=assistant,
        authority_resid=authority_resid,
        source=source,
        user=user,
        geometry=geometry,
    )


def save_paper_vectors(path: Path, vectors: PaperVectors) -> dict[str, Any]:
    payload = {
        "layer": vectors.layer,
        "authority": vectors.authority,
        "assistant": vectors.assistant,
        "authority_resid": vectors.authority_resid,
        "source": vectors.source,
        "user": vectors.user,
    }
    torch.save(payload, path)
    return payload


def load_paper_vectors(path: Path, geometry_path: Path) -> PaperVectors:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return PaperVectors(
        layer=int(payload["layer"]),
        authority=payload["authority"],
        assistant=payload["assistant"],
        authority_resid=payload["authority_resid"],
        source=payload["source"],
        user=payload["user"],
        geometry=json.loads(geometry_path.read_text()),
    )
