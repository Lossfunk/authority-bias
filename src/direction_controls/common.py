from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    direction_path: Path
    axis_path: Path | None
    masks_path: Path
    extraction_dir: Path
    trivia_dataset_path: Path = Path("data/exp7_mc_dataset.jsonl")
    piqa_dataset_path: Path = Path("data/piqa_mc_validation.jsonl")
    trivia_reference_summary_path: Path | None = None
    piqa_reference_summary_path: Path | None = None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize(vec: torch.Tensor) -> torch.Tensor:
    vec = vec.float()
    n = float(vec.norm().item())
    if n == 0.0:
        return vec
    return vec / n


def residualize(base: torch.Tensor, nuisance: torch.Tensor) -> torch.Tensor:
    base = base.float()
    nuisance = normalize(nuisance)
    coeff = torch.dot(base, nuisance)
    return base - coeff * nuisance


def load_direction_payload(path: Path, key: str = "vector") -> Tuple[torch.Tensor, int, str, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        if torch.is_tensor(payload):
            return normalize(payload.float()), 0, "unknown", {"source": str(path)}
        raise ValueError(f"Unsupported direction payload type at {path}: {type(payload).__name__}")
    if key in payload and torch.is_tensor(payload[key]):
        vec = payload[key].float()
    elif "vector" in payload and torch.is_tensor(payload["vector"]):
        vec = payload["vector"].float()
    else:
        raise ValueError(f"No tensor key '{key}' or 'vector' in {path}")
    layer = int(payload.get("layer", 0))
    position = str(payload.get("position", "unknown"))
    meta = {k: v for k, v in payload.items() if not torch.is_tensor(v)}
    return normalize(vec), layer, position, meta


def load_axis_vector(path: Path, layer: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        if "axis_layer" in payload and torch.is_tensor(payload["axis_layer"]):
            return normalize(payload["axis_layer"].float())
        if "axis" in payload and torch.is_tensor(payload["axis"]):
            axis = payload["axis"].float()
            if axis.ndim == 1:
                return normalize(axis)
            return normalize(axis[layer])
        if "vector" in payload and torch.is_tensor(payload["vector"]):
            return normalize(payload["vector"].float())
        if "vector_by_layer" in payload and isinstance(payload["vector_by_layer"], dict):
            mapping = payload["vector_by_layer"]
            if layer in mapping and torch.is_tensor(mapping[layer]):
                return normalize(mapping[layer].float())
            if str(layer) in mapping and torch.is_tensor(mapping[str(layer)]):
                return normalize(mapping[str(layer)].float())
        raise ValueError(f"Unsupported dict axis payload in {path}: keys={list(payload.keys())}")
    if torch.is_tensor(payload):
        axis = payload.float()
        if axis.ndim == 1:
            return normalize(axis)
        if layer < 0 or layer >= axis.shape[0]:
            raise ValueError(f"Layer {layer} out of range for axis {tuple(axis.shape)} @ {path}")
        return normalize(axis[layer])
    raise ValueError(f"Unsupported axis payload type at {path}: {type(payload).__name__}")


def save_direction_payload(
    *,
    path: Path,
    vector: torch.Tensor,
    layer: int,
    position: str,
    direction_name: str,
    extra: Dict[str, Any] | None = None,
) -> None:
    ensure_parent(path)
    payload: Dict[str, Any] = {
        "vector": normalize(vector).cpu(),
        "layer": int(layer),
        "position": str(position),
        "direction_name": direction_name,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def run_module(module: str, args: Sequence[str], *, dry_run: bool = False, env: Dict[str, str] | None = None) -> int:
    cmd = [sys.executable, "-m", module, *args]
    print("$", " ".join(cmd))
    if dry_run:
        return 0
    run_env = None
    if env is not None:
        run_env = os.environ.copy()
        run_env.update(env)
    proc = subprocess.run(cmd, check=False, env=run_env)
    return int(proc.returncode)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_int_csv(raw: str) -> List[int]:
    return [int(x) for x in parse_csv(raw)]


def parse_float_csv(raw: str) -> List[float]:
    return [float(x) for x in parse_csv(raw)]


def default_model_specs(repo_root: Path) -> Dict[str, ModelSpec]:
    return {
        "gpt_oss": ModelSpec(
            name="gpt_oss",
            model_id="openai/gpt-oss-20b",
            direction_path=repo_root / "results/authority/gpt-oss/mechanism/gpt_oss_compliance_analysis/primary_direction.pt",
            axis_path=repo_root / "external/assistant-axis/hardened/assistant_axis.pt",
            masks_path=repo_root / "results/authority/gpt-oss/mechanism/gpt_oss_authority_activations/label_masks.json",
            extraction_dir=repo_root / "results/authority/gpt-oss/mechanism/gpt_oss_authority_activations",
            trivia_reference_summary_path=repo_root / "results/authority/gpt-oss/mechanism/forward_patch_test_256/steering_summary.json",
            piqa_reference_summary_path=repo_root / "results/authority/gpt-oss/mechanism/piqa_forward_patch_200_freegen/piqa_summary.json",
        ),
        "gemma4": ModelSpec(
            name="gemma4",
            model_id="google/gemma-4-26B-A4B-it",
            direction_path=repo_root / "results/authority/gemma4/mechanism/gemma4_compliance_analysis/primary_direction.pt",
            axis_path=repo_root / "results/authority/gemma4/mechanism/assistant_axis_hardened/assistant_axis.pt",
            masks_path=repo_root / "results/authority/gemma4/mechanism/gemma4_authority_activations_no_thinking/label_masks.json",
            extraction_dir=repo_root / "results/authority/gemma4/mechanism/gemma4_authority_activations_no_thinking",
            trivia_reference_summary_path=repo_root / "results/authority/gemma4/mechanism/forward_patch_test_256_no_thinking/steering_summary.json",
            piqa_reference_summary_path=repo_root / "results/authority/gemma4/mechanism/piqa_forward_patch_200_freegen_no_thinking/piqa_summary.json",
        ),
        "olmo2": ModelSpec(
            name="olmo2",
            model_id="allenai/OLMo-2-0325-32B-Instruct",
            direction_path=repo_root / "results/authority/olmo2/mechanism/olmo2_compliance_analysis/primary_direction.pt",
            axis_path=repo_root / "results/authority/olmo2/mechanism/assistant_axis_hardened/assistant_axis.pt",
            masks_path=repo_root / "results/authority/olmo2/mechanism/olmo2_authority_activations/label_masks.json",
            extraction_dir=repo_root / "results/authority/olmo2/mechanism/olmo2_authority_activations",
            trivia_reference_summary_path=repo_root / "results/authority/olmo2/mechanism/forward_patch_test_256/steering_summary.json",
            piqa_reference_summary_path=repo_root / "results/authority/olmo2/mechanism/piqa_forward_patch_200_freegen/piqa_summary.json",
        ),
        "olmo31": ModelSpec(
            name="olmo31",
            model_id="allenai/OLMo-3.1-32B-Instruct",
            direction_path=repo_root / "results/authority/olmo31/mechanism/compliance_analysis_all_prior_wrong_shared_h100/primary_direction.pt",
            axis_path=repo_root / "results/authority/olmo31/mechanism/assistant_axis_hardened/assistant_axis.pt",
            masks_path=repo_root / "results/authority/olmo31/mechanism/authority_activations_all_prior_wrong_shared_h100/label_masks.json",
            extraction_dir=repo_root / "results/authority/olmo31/mechanism/authority_activations_all_prior_wrong_shared_h100",
            trivia_reference_summary_path=repo_root / "results/authority/authority_steering/olmo31_steering_interpolate_mean_l15_l18_l22_apw/steering_summary.json",
            piqa_reference_summary_path=None,
        ),
        "qwen35": ModelSpec(
            name="qwen35",
            model_id="Qwen/Qwen3.5-27B",
            direction_path=repo_root / "results/authority/qwen35/mechanism/qwen35_compliance_analysis_shared_h100/primary_direction.pt",
            axis_path=repo_root / "results/authority/qwen35/mechanism/assistant_axis_hardened/assistant_axis.pt",
            masks_path=repo_root / "results/authority/qwen35/mechanism/qwen35_authority_activations_shared_h100/label_masks.json",
            extraction_dir=repo_root / "results/authority/qwen35/mechanism/qwen35_authority_activations_shared_h100",
            trivia_reference_summary_path=repo_root / "results/authority/qwen35/mechanism/forward_patch_expanded_512_w1_minus_c1/steering_summary.json",
            piqa_reference_summary_path=repo_root / "results/authority/qwen35/mechanism/piqa_forward_patch_200_freegen/piqa_summary.json",
        ),
    }
