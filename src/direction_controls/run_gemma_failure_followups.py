from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from joblib import Parallel, delayed

from src.direction_controls.common import parse_csv, run_module, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Gemma failure-analysis follow-up experiments (E1-E7).")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--output-root", type=Path, default=Path("results/authority/direction_controls/gemma_failure_followups"))
    p.add_argument("--steps", type=str, default="e1,e2,e3,e4,e5,e6,e7")
    p.add_argument("--layers-main", type=str, default="15,18,20,22")
    p.add_argument("--layers-shallow", type=str, default="2,5,8,10")
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--checkpoint-path", type=Path, default=None)
    p.add_argument("--piqa-alphas", type=str, default="0,0.3,0.5,0.7,1.0")
    p.add_argument("--piqa-conditions", type=str, default="N0_note")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batch-tokens", type=int, default=16384)
    p.add_argument("--piqa-max-new-tokens", type=int, default=256)
    p.add_argument("--n-jobs-directions", type=int, default=1)
    p.add_argument("--gpu-devices", type=str, default="")
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Pass --no-compile to downstream GPU modules (avoids CUDA graph capture failures).",
    )
    return p.parse_args()


def _gemma_paths(root: Path) -> Dict[str, Path]:
    return {
        "model_id": Path("google/gemma-4-26B-A4B-it"),
        "direction": root / "results/authority/gemma4/mechanism/gemma4_compliance_analysis/primary_direction.pt",
        "masks": root / "results/authority/gemma4/mechanism/gemma4_authority_activations_no_thinking/label_masks.json",
        "extract": root / "results/authority/gemma4/mechanism/gemma4_authority_activations_no_thinking",
        "trivia_dataset": root / "data/exp7_mc_dataset.jsonl",
        "piqa_dataset": root / "data/piqa_mc_validation.jsonl",
    }


def _run_piqa_for_direction(
    *,
    model_id: str,
    masks_path: Path,
    direction_path: Path,
    layer: int,
    out_dir: Path,
    alphas: str,
    conditions: str,
    max_items: int,
    execute: bool,
    batch_size: int,
    max_batch_tokens: int,
    max_new_tokens: int,
    run_env: Dict[str, str] | None,
    no_compile: bool = False,
) -> int:
    cmd = [
        "--model", model_id,
        "--mc-dataset-path", "data/piqa_mc_validation.jsonl",
        "--masks-path", str(masks_path),
        "--direction-path", str(direction_path),
        "--direction-key", "vector",
        "--conditions", conditions,
        "--target-layers", str(layer),
        "--components", "block_output",
        "--position-mode", "endorsement_span",
        "--additive-modes", "add",
        "--direction-kinds", "shared_within_label",
        "--additive-alphas", alphas,
        "--sparse-topk-values", "0",
        "--patch-modes", "",
        "--extraction-dir", str(out_dir / "__no_extract__"),
        "--output-dir", str(out_dir),
        "--batch-size", str(batch_size),
        "--max-batch-tokens", str(max_batch_tokens),
        "--max-new-tokens", str(max_new_tokens),
    ]
    if max_items > 0:
        cmd += ["--max-samples", str(max_items)]
    if no_compile:
        cmd += ["--no-compile"]
    return run_module("src.authority_steering.run_piqa_interventions", cmd, dry_run=not execute, env=run_env)


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    paths = _gemma_paths(root)
    model_id = str(paths["model_id"])
    selected = set(parse_csv(args.steps))
    gpu_pool = parse_csv(args.gpu_devices)
    default_env = {"CUDA_VISIBLE_DEVICES": gpu_pool[0]} if gpu_pool else None

    checkpoint = args.checkpoint_path or (args.output_root / "gemma_followups_manifest.json")
    if args.resume and checkpoint.exists():
        try:
            manifest: Dict[str, object] = json.loads(checkpoint.read_text(encoding="utf-8"))
            manifest["execute"] = bool(args.execute)
        except Exception:
            manifest = {"execute": bool(args.execute), "steps": {}}
    else:
        manifest = {"execute": bool(args.execute), "steps": {}}
    steps = manifest.setdefault("steps", {})

    def _save() -> None:
        write_json(checkpoint, manifest)

    def _skip(step: str, expected: Path) -> bool:
        if args.overwrite or not args.resume or not expected.exists():
            return False
        rec = steps.get(step)
        if isinstance(rec, dict):
            return int(rec.get("returncode", 1)) == 0
        if isinstance(rec, list):
            return all(int(x.get("returncode", 1)) == 0 for x in rec if isinstance(x, dict))
        return False

    if "e1" in selected:
        out = args.output_root / "e1_shallow_transfer"
        if _skip("e1", out / "piqa_summary.json"):
            print("[resume] skip e1")
        else:
            cmd = [
                "--model", model_id, "--mc-dataset-path", str(paths["piqa_dataset"]),
                "--masks-path", str(paths["masks"]),
                "--direction-path", str(paths["direction"]), "--direction-key", "vector",
                "--conditions", args.piqa_conditions,
                "--target-layers", args.layers_shallow, "--single-layer-sweep",
                "--components", "block_output", "--position-mode", "endorsement_span",
                "--additive-modes", "add", "--direction-kinds", "shared_within_label",
                "--additive-alphas", args.piqa_alphas, "--sparse-topk-values", "0",
                "--patch-modes", "", "--extraction-dir", str(out / "__no_extract__"),
                "--output-dir", str(out), "--batch-size", str(args.batch_size),
                "--max-batch-tokens", str(args.max_batch_tokens), "--max-new-tokens", str(args.piqa_max_new_tokens),
            ]
            if args.max_items > 0:
                cmd += ["--max-samples", str(args.max_items)]
            if args.no_compile:
                cmd += ["--no-compile"]
            rc = run_module("src.authority_steering.run_piqa_interventions", cmd, dry_run=not args.execute, env=default_env)
            steps["e1"] = {"module": "src.authority_steering.run_piqa_interventions", "args": cmd, "returncode": rc}
            _save()

    if "e2" in selected:
        rows: List[Dict[str, object]] = []
        for task, dataset in [("trivia", paths["trivia_dataset"]), ("piqa", paths["piqa_dataset"])]:
            out = args.output_root / "e2_margin" / f"{task}_margins.json"
            if args.resume and not args.overwrite and out.exists():
                rows.append({"task": task, "skipped": True, "returncode": 0})
                continue
            cmd = ["--model", model_id, "--dataset-path", str(dataset), "--task-name", task, "--condition-code", "N0_note", "--output-path", str(out)]
            if args.max_items > 0:
                cmd += ["--max-items", str(args.max_items)]
            if args.no_compile:
                cmd += ["--no-compile"]
            rc = run_module("src.direction_controls.compute_prompt_margins", cmd, dry_run=not args.execute, env=default_env)
            rows.append({"task": task, "module": "src.direction_controls.compute_prompt_margins", "args": cmd, "returncode": rc})
            steps["e2"] = rows
            _save()
        steps["e2"] = rows
        _save()

    def _run_layer_sweep(step_key: str, fit_module: str, fit_cmd: List[str], glob_pat: str, out_root: Path) -> None:
        if _skip(step_key, out_root / "directions" / ("snmf_direction_summary.json" if step_key == "e5" else "layerwise_direction_summary.json")):
            print(f"[resume] skip {step_key}")
            return
        rc = run_module(fit_module, fit_cmd, dry_run=not args.execute, env=default_env)
        runs: List[Dict[str, object]] = [{"stage": "fit", "module": fit_module, "args": fit_cmd, "returncode": rc}]
        steps[step_key] = runs
        _save()
        direction_files = sorted((out_root / "directions").glob(glob_pat))

        def _run_one(i: int, p: Path) -> Dict[str, object]:
            layer = int(p.stem.split("_")[1])
            expected = out_root / f"piqa_layer_{layer}" / "piqa_summary.json"
            if args.resume and not args.overwrite and expected.exists():
                return {"stage": f"piqa_layer_{layer}", "direction_file": str(p), "returncode": 0, "skipped": True}
            env = {"CUDA_VISIBLE_DEVICES": gpu_pool[i % len(gpu_pool)]} if gpu_pool else None
            rc2 = _run_piqa_for_direction(
                model_id=model_id,
                masks_path=paths["masks"],
                direction_path=p,
                layer=layer,
                out_dir=out_root / f"piqa_layer_{layer}",
                alphas=args.piqa_alphas,
                conditions=args.piqa_conditions,
                max_items=args.max_items,
                execute=args.execute,
                batch_size=args.batch_size,
                max_batch_tokens=args.max_batch_tokens,
                max_new_tokens=args.piqa_max_new_tokens,
                run_env=env,
                no_compile=bool(args.no_compile),
            )
            return {"stage": f"piqa_layer_{layer}", "direction_file": str(p), "returncode": rc2}

        if args.n_jobs_directions == 1:
            for i, p in enumerate(direction_files):
                runs.append(_run_one(i, p))
                steps[step_key] = runs
                _save()
        else:
            runs.extend(Parallel(n_jobs=args.n_jobs_directions, prefer="threads")(delayed(_run_one)(i, p) for i, p in enumerate(direction_files)))
            steps[step_key] = runs
            _save()

    if "e3" in selected:
        out = args.output_root / "e3_layerwise_refit"
        cmd = [
            "--extraction-dir", str(paths["extract"]), "--masks-path", str(paths["masks"]),
            "--layers", args.layers_main, "--position", "endorsement_mean", "--uids-subset", "w1",
            "--direction-kind", "w1_minus_c1", "--output-dir", str(out / "directions"), "--n-jobs", "4",
        ]
        _run_layer_sweep("e3", "src.direction_controls.fit_layerwise_directions", cmd, "layer_*_w1_minus_c1_raw.pt", out)

    if "e4" in selected:
        out = args.output_root / "e4_style_residualized_refit"
        cmd = [
            "--extraction-dir", str(paths["extract"]), "--masks-path", str(paths["masks"]),
            "--layers", args.layers_main, "--position", "endorsement_mean", "--uids-subset", "w1",
            "--direction-kind", "w1_minus_c1", "--residualize-style",
            "--output-dir", str(out / "directions"), "--n-jobs", "4",
        ]
        _run_layer_sweep("e4", "src.direction_controls.fit_layerwise_directions", cmd, "layer_*_w1_minus_c1_style_residualized.pt", out)

    if "e5" in selected:
        out = args.output_root / "e5_snmf"
        cmd = [
            "--extraction-dir", str(paths["extract"]), "--masks-path", str(paths["masks"]),
            "--layers", args.layers_main, "--position", "endorsement_mean", "--uids-subset", "w1",
            "--style", "authoritative_verified", "--components", "16",
            "--output-dir", str(out / "directions"), "--n-jobs", "4",
        ]
        _run_layer_sweep("e5", "src.direction_controls.discover_snmf_directions", cmd, "layer_*_snmf_feature_direction.pt", out)

    if "e6" in selected:
        out = args.output_root / "e6_router_drift/router_drift.json"
        if _skip("e6", out):
            print("[resume] skip e6")
        else:
            cmd = [
                "--model", model_id, "--dataset-path", str(paths["trivia_dataset"]),
                "--direction-path", str(paths["direction"]), "--mode", "drift",
                "--patch-alpha", "0.5", "--max-items", str(args.max_items if args.max_items > 0 else 64),
                "--output-path", str(out),
            ]
            if args.no_compile:
                cmd += ["--no-compile"]
            rc = run_module("src.direction_controls.analyze_router_drift", cmd, dry_run=not args.execute, env=default_env)
            steps["e6"] = {"module": "src.direction_controls.analyze_router_drift", "args": cmd, "returncode": rc}
            _save()

    if "e7" in selected:
        out = args.output_root / "e7_router_bias/router_bias.json"
        if _skip("e7", out):
            print("[resume] skip e7")
        else:
            cmd = [
                "--model", model_id, "--dataset-path", str(paths["trivia_dataset"]), "--mode", "router_bias",
                "--router-bias-alpha", "1.0", "--max-items", str(args.max_items if args.max_items > 0 else 64),
                "--output-path", str(out),
            ]
            if args.no_compile:
                cmd += ["--no-compile"]
            rc = run_module("src.direction_controls.analyze_router_drift", cmd, dry_run=not args.execute, env=default_env)
            steps["e7"] = {"module": "src.direction_controls.analyze_router_drift", "args": cmd, "returncode": rc}
            _save()

    write_json(checkpoint, manifest)
    print(f"[done] wrote {checkpoint}")


if __name__ == "__main__":
    main()
