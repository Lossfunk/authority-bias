from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from src.exp16.run_steering_test import _direction_from_store, _load_extraction_store, _load_masks, _load_uids_from_masks
from src.exp19.common import (
    ModelSpec,
    default_model_specs,
    load_axis_vector,
    load_direction_payload,
    normalize,
    residualize,
    run_module,
    write_json,
)


TRIVIA_SUFFIX = "Answer the question in one short sentence using the full answer text, not option letters."
PIQA_SUFFIX = TRIVIA_SUFFIX
TARGET_POSITION = "endorsement_span"


@dataclass(frozen=True)
class SweepSpec:
    layers: tuple[int, ...]
    trivia_batch_size: int
    piqa_batch_size: int
    piqa_max_batch_tokens: int


MODEL_SWEEPS: Dict[str, SweepSpec] = {
    "gpt_oss": SweepSpec(layers=(16, 18, 20), trivia_batch_size=256, piqa_batch_size=256, piqa_max_batch_tokens=262144),
    # gemma4's extraction store on the remote boxes only contains layers [15, 18, 20, 22].
    # Original forward-patch winners were 20/22/24 but L24 has no extraction activations
    # to derive per-layer directions from. The layer-position AUROC sweep ranks the
    # available layers as 22 > 18 > 20 > 15, so we sweep {18, 20, 22}, dropping L15 (weak)
    # and keeping L22 in place of the unavailable L24.
    "gemma4": SweepSpec(layers=(18, 20, 22), trivia_batch_size=128, piqa_batch_size=128, piqa_max_batch_tokens=131072),
    "olmo2": SweepSpec(layers=(10, 16, 22), trivia_batch_size=96, piqa_batch_size=96, piqa_max_batch_tokens=98304),
    "olmo31": SweepSpec(layers=(15, 18, 22), trivia_batch_size=96, piqa_batch_size=96, piqa_max_batch_tokens=98304),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the principled endorsement_span/layer correction on one GPU.")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--models", type=str, default="gpt_oss,gemma4,olmo2")
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("neurips-results/exp19/principled_correction_w1"),
    )
    p.add_argument("--alphas", type=str, default="0.5,1,2,4")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--run-trivia", action="store_true")
    p.add_argument("--run-piqa", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--force", action="store_true", help="Re-run stages even if their summary files exist")
    p.add_argument("--checkpoint-path", type=Path, default=None)
    return p.parse_args()


def _csv(values: Sequence[int | float | str]) -> str:
    return ",".join(str(v) for v in values)


def _parse_models(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _alpha_list(raw: str, *, include_zero: bool) -> str:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return ",".join((["0"] if include_zero else []) + vals)


def _derive_layerwise_payloads(repo_root: Path, spec: ModelSpec, sweep: SweepSpec, out_dir: Path) -> Dict[str, Path]:
    masks = _load_masks(spec.masks_path)
    direction_uids = _load_uids_from_masks(masks, "w1")
    store = _load_extraction_store(spec.extraction_dir)
    _, _, source_position, _ = load_direction_payload(spec.direction_path, "vector")

    authority_by_layer: Dict[str, torch.Tensor] = {}
    assistant_by_layer: Dict[str, torch.Tensor] = {}
    residualized_by_layer: Dict[str, torch.Tensor] = {}
    for layer in sweep.layers:
        dirs = _direction_from_store(store, uids=direction_uids, position=source_position, layer_value=layer)
        authority = normalize(dirs["shared_within_label"])
        assistant = load_axis_vector(spec.axis_path, layer)
        residualized_vec = normalize(residualize(authority, assistant))
        authority_by_layer[str(layer)] = authority.cpu()
        assistant_by_layer[str(layer)] = assistant.cpu()
        residualized_by_layer[str(layer)] = residualized_vec.cpu()

    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "authority": {
            "vector_by_layer": authority_by_layer,
            "layer": int(sweep.layers[0]),
            "position": source_position,
            "direction_name": "authority_layerwise",
        },
        "assistant": {
            "vector_by_layer": assistant_by_layer,
            "layer": int(sweep.layers[0]),
            "position": source_position,
            "direction_name": "assistant_axis_layerwise",
        },
        "residualized": {
            "vector_by_layer": residualized_by_layer,
            "layer": int(sweep.layers[0]),
            "position": source_position,
            "direction_name": "authority_residualized_layerwise",
        },
    }
    paths: Dict[str, Path] = {}
    for name, payload in payloads.items():
        path = out_dir / f"{name}.pt"
        torch.save(payload, path)
        paths[name] = path
    return paths


def _trivia_args(
    *,
    spec: ModelSpec,
    direction_path: Path,
    layers: Sequence[int],
    alphas: str,
    output_dir: Path,
    batch_size: int,
    no_compile: bool,
) -> list[str]:
    args = [
        "--model",
        spec.model_id,
        "--direction-path",
        str(direction_path),
        "--direction-key",
        "vector_by_layer",
        "--dataset-path",
        str(spec.trivia_dataset_path),
        "--masks-path",
        str(spec.masks_path),
        "--uids-subset",
        "w1",
        "--conditions",
        "W1_note",
        "--target-layers",
        _csv(layers),
        "--single-layer-sweep",
        "--position-modes",
        TARGET_POSITION,
        "--components",
        "block_output",
        "--intervention-mode",
        "project_out_direction",
        "--norm-scaling",
        "none",
        "--alphas",
        alphas,
        "--max-new-tokens",
        "256",
        "--batch-size",
        str(batch_size),
        "--answer-suffix",
        TRIVIA_SUFFIX,
        "--output-dir",
        str(output_dir),
        "--max-items",
        "512",
    ]
    if no_compile:
        args.append("--no-compile")
    return args


def _piqa_args(
    *,
    spec: ModelSpec,
    direction_path: Path,
    layers: Sequence[int],
    alphas: str,
    output_dir: Path,
    batch_size: int,
    max_batch_tokens: int,
    no_compile: bool,
) -> list[str]:
    args = [
        "--model",
        spec.model_id,
        "--mc-dataset-path",
        str(spec.piqa_dataset_path),
        "--masks-path",
        str(spec.masks_path),
        "--direction-path",
        str(direction_path),
        "--direction-key",
        "vector_by_layer",
        "--conditions",
        "W1_note",
        "--target-layers",
        _csv(layers),
        "--single-layer-sweep",
        "--components",
        "block_output",
        "--position-mode",
        TARGET_POSITION,
        "--additive-modes",
        "project_out_direction",
        "--norm-scaling",
        "none",
        "--direction-kinds",
        "shared_within_label",
        "--additive-alphas",
        alphas,
        "--sparse-topk-values",
        "0",
        "--patch-modes",
        "",
        "--extraction-dir",
        str(output_dir / "__no_extract__"),
        "--output-dir",
        str(output_dir),
        "--max-new-tokens",
        "256",
        "--batch-size",
        str(batch_size),
        "--max-batch-tokens",
        str(max_batch_tokens),
        "--answer-suffix",
        PIQA_SUFFIX,
        "--max-samples",
        "200",
    ]
    if no_compile:
        args.append("--no-compile")
    return args


def main() -> None:
    args = parse_args()
    run_trivia = args.run_trivia or (not args.run_trivia and not args.run_piqa)
    run_piqa = args.run_piqa or (not args.run_trivia and not args.run_piqa)
    specs = default_model_specs(args.repo_root)
    manifest_path = args.checkpoint_path or (args.output_root / "manifest.json")
    manifest: Dict[str, Any] = {"models": {}, "execute": bool(args.execute)}

    for model_name in _parse_models(args.models):
        spec = specs[model_name]
        sweep = MODEL_SWEEPS[model_name]
        model_root = args.output_root / model_name
        directions_dir = model_root / "directions"
        direction_paths = _derive_layerwise_payloads(args.repo_root, spec, sweep, directions_dir)
        model_manifest: Dict[str, Any] = {
            "layers": list(sweep.layers),
            "target_position": TARGET_POSITION,
            "source_position": load_direction_payload(spec.direction_path, "vector")[2],
            "directions": {k: str(v) for k, v in direction_paths.items()},
            "runs": [],
        }
        env = {"USE_HUB_KERNELS": "NO"}

        for variant in ("authority", "assistant", "residualized"):
            alpha_str = _alpha_list(args.alphas, include_zero=(variant == "authority"))
            if run_trivia:
                out_dir = model_root / f"trivia_{variant}"
                trivia_summary = out_dir / "steering_summary.json"
                if args.execute and not args.force and trivia_summary.exists():
                    print(f"[skip] trivia/{variant} already has summary -> {trivia_summary}")
                    model_manifest["runs"].append(
                        {"task": "trivia", "variant": variant, "args": None, "returncode": 0, "skipped": True}
                    )
                else:
                    cmd_args = _trivia_args(
                        spec=spec,
                        direction_path=direction_paths[variant],
                        layers=sweep.layers,
                        alphas=alpha_str,
                        output_dir=out_dir,
                        batch_size=sweep.trivia_batch_size,
                        no_compile=bool(args.no_compile),
                    )
                    rc = run_module("src.exp16.run_steering_test", cmd_args, dry_run=not args.execute, env=env)
                    model_manifest["runs"].append(
                        {"task": "trivia", "variant": variant, "args": cmd_args, "returncode": rc}
                    )
                    if args.execute and rc != 0:
                        raise SystemExit(f"Run failed: {model_name}/trivia/{variant}")
            if run_piqa:
                out_dir = model_root / f"piqa_{variant}"
                piqa_summary = out_dir / "piqa_summary.json"
                if args.execute and not args.force and piqa_summary.exists():
                    print(f"[skip] piqa/{variant} already has summary -> {piqa_summary}")
                    model_manifest["runs"].append(
                        {"task": "piqa", "variant": variant, "args": None, "returncode": 0, "skipped": True}
                    )
                else:
                    cmd_args = _piqa_args(
                        spec=spec,
                        direction_path=direction_paths[variant],
                        layers=sweep.layers,
                        alphas=alpha_str,
                        output_dir=out_dir,
                        batch_size=sweep.piqa_batch_size,
                        max_batch_tokens=sweep.piqa_max_batch_tokens,
                        no_compile=bool(args.no_compile),
                    )
                    rc = run_module("src.exp16.run_piqa_interventions", cmd_args, dry_run=not args.execute, env=env)
                    model_manifest["runs"].append(
                        {"task": "piqa", "variant": variant, "args": cmd_args, "returncode": rc}
                    )
                    if args.execute and rc != 0:
                        raise SystemExit(f"Run failed: {model_name}/piqa/{variant}")

        manifest["models"][model_name] = model_manifest
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
