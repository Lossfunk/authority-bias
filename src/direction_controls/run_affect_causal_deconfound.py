from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from src.authority_steering.va_subspace import load_basis_by_layer, normalize_torch, project_out_torch
from src.direction_controls.common import (
    default_model_specs,
    load_direction_payload,
    parse_csv,
    run_module,
    save_direction_payload,
    write_json,
)


MODEL_TO_VA_PLANE = {
    "qwen35": Path("results/authority/qwen35/mechanism/lexical_va_plane/va_plane.pt"),
    "gpt_oss": Path("results/authority/gpt-oss/mechanism/lexical_va_plane/va_plane.pt"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Causal affect deconfound for authority compliance. Runs three controls: "
            "(1) authority project-out positive control, (2) direct VA-plane project-out, "
            "(3) authority vector with VA-plane component removed."
        )
    )
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--models", type=str, default="qwen35")
    p.add_argument(
        "--variants",
        type=str,
        default="authority,affect_subspace,authority_affect_component,authority_affect_residualized",
    )
    p.add_argument("--output-root", type=Path, default=Path("results/authority/direction_controls/affect_causal_deconfound"))
    p.add_argument("--conditions", type=str, default="N0_note,W1_note,C1_note")
    p.add_argument("--alphas", type=str, default="0,1", help="Alphas for the authority positive control.")
    p.add_argument("--control-alpha", type=float, default=1.0, help="Alpha for non-baseline control variants.")
    p.add_argument("--max-items", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batch-tokens", type=int, default=65536)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--position-mode", type=str, default="endorsement_span")
    p.add_argument("--component", type=str, default="block_output")
    p.add_argument("--uids-subset", choices=("w1", "c1", "all"), default="w1")
    p.add_argument("--answer-suffix", type=str, default="Answer the question in one short sentence using the full answer text, not option letters.")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    return p.parse_args()


def _make_affect_directions(
    *,
    model_name: str,
    direction_path: Path,
    va_plane_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    authority_vec, layer, position, meta = load_direction_payload(direction_path, key="vector")
    basis_by_layer = load_basis_by_layer(va_plane_path)
    if layer not in basis_by_layer:
        raise ValueError(f"VA plane {va_plane_path} has no basis for layer {layer}; available={sorted(basis_by_layer)}")
    basis = basis_by_layer[layer].float()
    authority_resid = project_out_torch(authority_vec, basis)
    authority_affect_component = authority_vec.float() - authority_resid.float()

    out = output_dir / model_name / "directions"
    out.mkdir(parents=True, exist_ok=True)
    save_direction_payload(
        path=out / "authority.pt",
        vector=authority_vec,
        layer=layer,
        position=position,
        direction_name="authority",
        extra={"source_direction_path": str(direction_path), "source_meta": meta},
    )
    save_direction_payload(
        path=out / "authority_affect_residualized.pt",
        vector=authority_resid,
        layer=layer,
        position=position,
        direction_name="authority_affect_residualized",
        extra={"source_direction_path": str(direction_path), "va_plane_path": str(va_plane_path), "source_meta": meta},
    )
    save_direction_payload(
        path=out / "authority_affect_component.pt",
        vector=authority_affect_component,
        layer=layer,
        position=position,
        direction_name="authority_affect_component",
        extra={"source_direction_path": str(direction_path), "va_plane_path": str(va_plane_path), "source_meta": meta},
    )
    authority_norm = float(authority_vec.float().norm().item())
    resid_norm = float(authority_resid.float().norm().item())
    affect_norm = float(authority_affect_component.float().norm().item())
    geometry = {
        "model": model_name,
        "layer": int(layer),
        "position": position,
        "direction_path": str(direction_path),
        "va_plane_path": str(va_plane_path),
        "authority_norm": authority_norm,
        "authority_affect_residual_norm": resid_norm,
        "authority_affect_component_norm": affect_norm,
        "authority_affect_component_norm_fraction": affect_norm / authority_norm if authority_norm else 0.0,
        "basis_rows": int(basis.shape[0]),
        "basis_dim": int(basis.shape[1]),
    }
    write_json(output_dir / model_name / "affect_geometry.json", geometry)
    return geometry


def _run_steering(
    *,
    model_id: str,
    direction_path: Path,
    masks_path: Path,
    dataset_path: Path,
    va_plane_path: Path | None,
    output_dir: Path,
    layer: int,
    args: argparse.Namespace,
    intervention_mode: str,
    alphas: str,
) -> int:
    cmd = [
        "--model", model_id,
        "--direction-path", str(direction_path),
        "--direction-key", "vector",
        "--dataset-path", str(dataset_path),
        "--masks-path", str(masks_path),
        "--uids-subset", args.uids_subset,
        "--conditions", args.conditions,
        "--target-layers", str(layer),
        "--position-modes", args.position_mode,
        "--components", args.component,
        "--intervention-mode", intervention_mode,
        "--norm-scaling", "none",
        "--alphas", alphas,
        "--max-new-tokens", str(args.max_new_tokens),
        "--batch-size", str(args.batch_size),
        "--max-batch-tokens", str(args.max_batch_tokens),
        "--answer-suffix", args.answer_suffix,
        "--output-dir", str(output_dir),
        "--max-items", str(args.max_items),
    ]
    if va_plane_path is not None:
        cmd += ["--project-out-subspace-path", str(va_plane_path)]
    if args.no_compile:
        cmd.append("--no-compile")
    return run_module("src.authority_steering.run_steering_test", cmd, dry_run=not args.execute)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    specs = default_model_specs(repo_root)
    variants = parse_csv(args.variants)
    models = parse_csv(args.models)
    manifest: Dict[str, Any] = {
        "experiment": "affect_causal_deconfound",
        "conditions": parse_csv(args.conditions),
        "max_items": args.max_items,
        "position_mode": args.position_mode,
        "component": args.component,
        "models": {},
    }

    for model_name in models:
        if model_name not in specs:
            raise ValueError(f"Unknown model {model_name}; known={sorted(specs)}")
        if model_name not in MODEL_TO_VA_PLANE:
            raise ValueError(f"No default VA-plane path for {model_name}; add it to MODEL_TO_VA_PLANE")
        spec = specs[model_name]
        va_plane_path = repo_root / MODEL_TO_VA_PLANE[model_name]
        if not va_plane_path.exists():
            raise FileNotFoundError(f"Missing VA plane for {model_name}: {va_plane_path}")
        geometry = _make_affect_directions(
            model_name=model_name,
            direction_path=spec.direction_path,
            va_plane_path=va_plane_path,
            output_dir=args.output_root,
        )
        layer = int(geometry["layer"])
        model_out = args.output_root / model_name

        variant_to_call = {
            "authority": {
                "direction": model_out / "directions" / "authority.pt",
                "mode": "project_out_direction",
                "va": None,
                "alphas": args.alphas,
            },
            "affect_subspace": {
                "direction": model_out / "directions" / "authority.pt",
                "mode": "project_out_subspace",
                "va": va_plane_path,
                "alphas": f"{args.control_alpha:g}",
            },
            "authority_affect_component": {
                "direction": model_out / "directions" / "authority_affect_component.pt",
                "mode": "project_out_direction",
                "va": None,
                "alphas": f"{args.control_alpha:g}",
            },
            "authority_affect_residualized": {
                "direction": model_out / "directions" / "authority_affect_residualized.pt",
                "mode": "project_out_direction",
                "va": None,
                "alphas": f"{args.control_alpha:g}",
            },
        }

        manifest["models"][model_name] = {"geometry": geometry, "variants": {}}
        for variant in variants:
            if variant not in variant_to_call:
                raise ValueError(f"Unknown variant {variant}; known={sorted(variant_to_call)}")
            output_dir = model_out / f"trivia_{variant}"
            summary_path = output_dir / "steering_summary.json"
            if summary_path.exists() and not args.overwrite:
                print(f"[skip] {model_name}/{variant}: {summary_path} exists")
                manifest["models"][model_name]["variants"][variant] = {"summary_path": str(summary_path), "skipped": True}
                continue
            call = variant_to_call[variant]
            rc = _run_steering(
                model_id=spec.model_id,
                direction_path=call["direction"],
                masks_path=spec.masks_path,
                dataset_path=spec.trivia_dataset_path,
                va_plane_path=call["va"],
                output_dir=output_dir,
                layer=layer,
                args=args,
                intervention_mode=call["mode"],
                alphas=call["alphas"],
            )
            if rc != 0:
                raise SystemExit(f"{model_name}/{variant} failed with rc={rc}")
            manifest["models"][model_name]["variants"][variant] = {
                "summary_path": str(summary_path),
                "rows_path": str(output_dir / "steering_rows.jsonl"),
                "meta_path": str(output_dir / "steering_meta.json"),
                "intervention_mode": call["mode"],
                "alphas": call["alphas"],
            }

    write_json(args.output_root / "manifest.json", manifest)
    print(f"[done] wrote {args.output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
