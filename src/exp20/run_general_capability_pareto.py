from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.exp19.common import write_json
from src.exp20.common import derive_deconfound_payloads, parse_csv, specs_and_sweeps
from src.exp20.run_mitigation_pareto import _variant_path


def run_module_uv(module: str, args: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> int:
    cmd = ["uv", "run", "python", "-m", module, *args]
    print("$", " ".join(cmd))
    if dry_run:
        return 0
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    proc = subprocess.run(cmd, check=False, env=run_env)
    return int(proc.returncode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run general capability preservation evals (MMLU-Pro/GSM8K).")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--models", default="qwen35")
    p.add_argument("--output-root", type=Path, default=Path("results/authority/exp20/general_capability"))
    p.add_argument("--mitigation-root", type=Path, default=Path("results/authority/exp20/mitigation_pareto"))
    p.add_argument("--caa-root", type=Path, default=Path("results/authority/exp20/caa_sycophancy"))
    p.add_argument("--tasks", default="mmlu_pro,gsm8k")
    p.add_argument("--variants", default="residualized,assistant,caa")
    p.add_argument("--residualized-alphas", default="0,0.75,1")
    p.add_argument("--control-alphas", default="0,1")
    p.add_argument("--layers", default=None, help="Override model sweep layers, e.g. 5")
    p.add_argument("--mmlu-pro-samples", type=int, default=300)
    p.add_argument("--gsm8k-samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-batch-tokens", type=int, default=65536)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--position-mode", default="answer_position")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    return p.parse_args()


def _alphas_for_variant(args: argparse.Namespace, variant: str) -> str:
    return args.residualized_alphas if variant == "residualized" else args.control_alphas


def _samples_for_task(args: argparse.Namespace, task: str) -> int:
    if task == "mmlu_pro":
        return args.mmlu_pro_samples
    if task == "gsm8k":
        return args.gsm8k_samples
    raise ValueError(task)


def _summary_path(out_dir: Path, task: str) -> Path:
    return out_dir / f"{task}_summary.json"


def _collect_summary(model_dir: Path, tasks: list[str], variants: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"tasks": {}}
    for task in tasks:
        out["tasks"][task] = []
        for variant in variants:
            path = model_dir / f"{task}_{variant}" / f"{task}_summary.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            for row in payload.get("summary", []):
                out["tasks"][task].append(
                    {
                        "variant": variant,
                        "alpha": row["alpha"],
                        "accuracy": row["accuracy"],
                        "parsed": row["parsed"],
                        "total": row["total"],
                        "parse_rate": row["parse_rate"],
                        "config_id": row["config_id"],
                    }
                )
    return out


def main() -> None:
    args = parse_args()
    manifest: dict[str, Any] = {
        "models": {},
        "tasks": parse_csv(args.tasks),
        "variants": parse_csv(args.variants),
        "max_new_tokens": args.max_new_tokens,
        "position_mode": args.position_mode,
        "notes": {
            "mmlu_pro": "TIGER-Lab/MMLU-Pro test subset; answer parsed as option letter.",
            "gsm8k": "openai/gsm8k test subset; answer parsed as final numeric answer.",
            "success": "Capability should remain within ~1-3 pp of alpha=0 baseline.",
        },
    }
    env = {"USE_HUB_KERNELS": "NO"}

    for model_name, spec, sweep in specs_and_sweeps(args.repo_root, args.models):
        model_dir = args.output_root / model_name
        mitigation_model_dir = args.mitigation_root / model_name
        direction_dir = mitigation_model_dir / "directions"
        local_dirs = derive_deconfound_payloads(args.repo_root, spec, sweep, direction_dir)
        layers = args.layers if args.layers is not None else ",".join(str(x) for x in sweep.layers)

        model_manifest: dict[str, Any] = {"runs": [], "layers": layers}
        for variant in parse_csv(args.variants):
            direction_path, mode = _variant_path(mitigation_model_dir, args.caa_root, model_name, variant, local_dirs)
            if variant == "caa" and not direction_path.exists():
                print(f"[skip] {model_name}/{variant}: missing CAA direction {direction_path}")
                model_manifest["runs"].append({"variant": variant, "skipped": True, "reason": "missing_caa_direction"})
                continue
            for task in parse_csv(args.tasks):
                out_dir = model_dir / f"{task}_{variant}"
                summary = _summary_path(out_dir, task)
                if args.execute and summary.exists() and not args.force:
                    print(f"[skip] existing {summary}")
                    model_manifest["runs"].append({"task": task, "variant": variant, "skipped": True, "reason": "summary_exists"})
                    continue
                cmd = [
                    "--repo-root", str(args.repo_root),
                    "--model-name", model_name,
                    "--direction-path", str(direction_path),
                    "--direction-key", "vector_by_layer",
                    "--task", task,
                    "--output-dir", str(out_dir),
                    "--layers", layers,
                    "--alphas", _alphas_for_variant(args, variant),
                    "--mode", mode,
                    "--position-mode", args.position_mode,
                    "--max-samples", str(_samples_for_task(args, task)),
                    "--batch-size", str(args.batch_size),
                    "--max-batch-tokens", str(args.max_batch_tokens),
                    "--max-new-tokens", str(args.max_new_tokens),
                ]
                if args.force:
                    cmd.append("--force")
                if args.no_compile:
                    cmd.append("--no-compile")
                rc = run_module_uv("src.exp20.run_capability_eval", cmd, dry_run=not args.execute, env=env)
                model_manifest["runs"].append({"task": task, "variant": variant, "mode": mode, "returncode": rc})
                if args.execute and rc:
                    raise SystemExit(f"Failed {model_name}/{task}/{variant}")

        summary_payload = _collect_summary(model_dir, parse_csv(args.tasks), parse_csv(args.variants))
        summary_payload.update({"model": model_name, "layers": layers})
        write_json(model_dir / "general_capability_summary.json", summary_payload)
        manifest["models"][model_name] = model_manifest
        write_json(args.output_root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
