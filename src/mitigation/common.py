from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import torch

from src.authority_steering.run_steering_test import (
    _direction_from_store,
    _load_extraction_store,
    _load_masks,
    _load_uids_from_masks,
)
from src.direction_controls.common import (
    ModelSpec,
    default_model_specs,
    load_axis_vector,
    load_direction_payload,
    normalize,
    residualize,
    run_module,
    write_json,
)


TARGET_POSITION = "endorsement_span"
FULL_ANSWER_SUFFIX = "Answer the question in one short sentence using the full answer text, not option letters."
DEFAULT_ALPHAS = "0,0.25,0.5,0.75,1"


@dataclass(frozen=True)
class Exp20Sweep:
    layers: tuple[int, ...]
    trivia_batch_size: int
    piqa_batch_size: int
    piqa_max_batch_tokens: int


MODEL_SWEEPS: Dict[str, Exp20Sweep] = {
    "qwen35": Exp20Sweep((5,), 128, 128, 131072),
    "gpt_oss": Exp20Sweep((16, 18, 20), 256, 256, 262144),
    "gemma4": Exp20Sweep((18, 20, 22), 128, 128, 131072),
    "olmo2": Exp20Sweep((10, 16, 22), 96, 96, 98304),
    "olmo31": Exp20Sweep((15, 18, 22), 96, 96, 98304),
}


def parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def csv(values: Iterable[Any]) -> str:
    return ",".join(str(v) for v in values)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def direction_for_layer(direction_obj: Any, layer: int) -> torch.Tensor:
    if isinstance(direction_obj, dict):
        if layer in direction_obj:
            value = direction_obj[layer]
        elif str(layer) in direction_obj:
            value = direction_obj[str(layer)]
        else:
            raise KeyError(f"Layer {layer} missing from direction mapping")
        return normalize(value.float() if torch.is_tensor(value) else torch.tensor(value, dtype=torch.float32))
    if torch.is_tensor(direction_obj):
        return normalize(direction_obj.float())
    raise TypeError(f"Unsupported direction object: {type(direction_obj).__name__}")


def write_layerwise_payload(path: Path, vectors: Dict[int, torch.Tensor], *, name: str, position: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vector_by_layer": {str(k): normalize(v).cpu() for k, v in vectors.items()},
            "layer": int(next(iter(vectors))),
            "position": position,
            "direction_name": name,
        },
        path,
    )


def derive_deconfound_payloads(repo_root: Path, spec: ModelSpec, sweep: Exp20Sweep, out_dir: Path) -> dict[str, Path]:
    masks = _load_masks(spec.masks_path)
    uids = _load_uids_from_masks(masks, "w1")
    store = _load_extraction_store(spec.extraction_dir)
    _, _, source_position, _ = load_direction_payload(spec.direction_path, "vector")

    authority: dict[int, torch.Tensor] = {}
    assistant: dict[int, torch.Tensor] = {}
    residualized_map: dict[int, torch.Tensor] = {}
    norm_rows: list[dict[str, Any]] = []
    for layer in sweep.layers:
        dirs = _direction_from_store(store, uids=uids, position=source_position, layer_value=layer)
        a = normalize(dirs["shared_within_label"])
        ax = load_axis_vector(spec.axis_path, layer) if spec.axis_path else torch.zeros_like(a)
        r = normalize(residualize(a, ax))
        authority[layer] = a
        assistant[layer] = ax
        residualized_map[layer] = r
        norm_rows.append(
            {
                "layer": layer,
                "source_position": source_position,
                "authority_norm": float(a.norm()),
                "assistant_norm": float(ax.norm()),
                "residualized_norm": float(r.norm()),
                "cos_authority_assistant": float(torch.dot(a, normalize(ax))) if float(ax.norm()) else None,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "authority": out_dir / "authority.pt",
        "assistant": out_dir / "assistant.pt",
        "residualized": out_dir / "residualized.pt",
    }
    write_layerwise_payload(paths["authority"], authority, name="authority_layerwise", position=source_position)
    write_layerwise_payload(paths["assistant"], assistant, name="assistant_axis_layerwise", position=source_position)
    write_layerwise_payload(paths["residualized"], residualized_map, name="authority_residualized_layerwise", position=source_position)
    write_json(out_dir / "norms.json", {"rows": norm_rows})
    return paths


def trivia_args(
    *,
    spec: ModelSpec,
    direction_path: Path,
    layers: Sequence[int],
    alphas: str,
    output_dir: Path,
    batch_size: int,
    max_batch_tokens: int,
    mode: str,
    position: str = TARGET_POSITION,
    max_items: int = 512,
    no_compile: bool = False,
    conditions: str = "W1_note",
    uids_subset: str = "w1",
) -> list[str]:
    args = [
        "--model", spec.model_id,
        "--direction-path", str(direction_path),
        "--direction-key", "vector_by_layer",
        "--dataset-path", str(spec.trivia_dataset_path),
        "--masks-path", str(spec.masks_path),
        "--uids-subset", uids_subset,
        "--conditions", conditions,
        "--target-layers", csv(layers),
        "--single-layer-sweep",
        "--position-modes", position,
        "--components", "block_output",
        "--intervention-mode", mode,
        "--norm-scaling", "none",
        "--alphas", alphas,
        "--max-new-tokens", "256",
        "--batch-size", str(batch_size),
        "--max-batch-tokens", str(max_batch_tokens),
        "--answer-suffix", FULL_ANSWER_SUFFIX,
        "--output-dir", str(output_dir),
        "--max-items", str(max_items),
    ]
    if no_compile:
        args.append("--no-compile")
    return args


def piqa_args(
    *,
    spec: ModelSpec,
    direction_path: Path,
    layers: Sequence[int],
    alphas: str,
    output_dir: Path,
    batch_size: int,
    max_batch_tokens: int,
    mode: str,
    position: str = TARGET_POSITION,
    max_samples: int = 200,
    no_compile: bool = False,
    conditions: str = "W1_note",
) -> list[str]:
    args = [
        "--model", spec.model_id,
        "--mc-dataset-path", str(spec.piqa_dataset_path),
        "--masks-path", str(spec.masks_path),
        "--direction-path", str(direction_path),
        "--direction-key", "vector_by_layer",
        "--conditions", conditions,
        "--target-layers", csv(layers),
        "--single-layer-sweep",
        "--components", "block_output",
        "--position-mode", position,
        "--additive-modes", mode,
        "--norm-scaling", "none",
        "--direction-kinds", "shared_within_label",
        "--additive-alphas", alphas,
        "--sparse-topk-values", "0",
        "--patch-modes", "",
        "--extraction-dir", str(output_dir / "__no_extract__"),
        "--output-dir", str(output_dir),
        "--max-new-tokens", "256",
        "--batch-size", str(batch_size),
        "--max-batch-tokens", str(max_batch_tokens),
        "--answer-suffix", FULL_ANSWER_SUFFIX,
        "--max-samples", str(max_samples),
    ]
    if no_compile:
        args.append("--no-compile")
    return args


def specs_and_sweeps(repo_root: Path, models: str) -> list[tuple[str, ModelSpec, Exp20Sweep]]:
    specs = default_model_specs(repo_root)
    out = []
    for name in parse_csv(models):
        if name not in specs:
            raise KeyError(f"Unknown model {name}; choices={sorted(specs)}")
        if name not in MODEL_SWEEPS:
            raise KeyError(f"No exp20 sweep registered for {name}")
        out.append((name, specs[name], MODEL_SWEEPS[name]))
    return out
