from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from src.exp19.common import run_module, write_json
from src.exp20.common import (
    DEFAULT_ALPHAS,
    derive_deconfound_payloads,
    parse_csv,
    piqa_args,
    specs_and_sweeps,
    trivia_args,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Exp20 mitigation Pareto sweeps.")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--models", default="qwen35,gpt_oss,olmo2,olmo31")
    p.add_argument("--output-root", type=Path, default=Path("neurips-results/exp20/mitigation_pareto"))
    p.add_argument("--alphas", default=DEFAULT_ALPHAS)
    p.add_argument("--variants", default="residualized,assistant,caa")
    p.add_argument("--caa-root", type=Path, default=Path("neurips-results/exp20/caa_sycophancy"))
    p.add_argument("--execute", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--run-trivia", action="store_true")
    p.add_argument("--run-piqa", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--max-trivia-items", type=int, default=512)
    p.add_argument("--max-piqa-samples", type=int, default=200)
    p.add_argument("--conditions", default="W1_note", help="CSV of condition codes (e.g. 'N0_note' for capability eval).")
    p.add_argument("--output-suffix", default="", help="Suffix appended to per-variant output dirs (e.g. '_capability').")
    p.add_argument("--uids-subset", default="w1", choices=("w1", "c1", "all"))
    return p.parse_args()


def _variant_path(model_dir: Path, caa_root: Path, model_name: str, variant: str, local_dirs: Dict[str, Path]) -> tuple[Path, str]:
    if variant in {"authority", "assistant", "residualized"}:
        return local_dirs[variant], "project_out_direction"
    if variant == "caa":
        path = caa_root / model_name / "directions" / "sycophancy.pt"
        return path, "subtract"
    raise ValueError(f"Unknown variant: {variant}")


def main() -> None:
    args = parse_args()
    run_trivia = args.run_trivia or (not args.run_trivia and not args.run_piqa)
    run_piqa = args.run_piqa or (not args.run_trivia and not args.run_piqa)
    manifest: Dict[str, Any] = {
        "execute": bool(args.execute),
        "alphas": args.alphas,
        "models": {},
        "notes": {
            "pareto_axis": "Compare methods at matched W1 reduction, not matched alpha.",
            "caa": "CAA vector is sycophancy-positive; suppression uses subtract mode.",
        },
    }
    env = {"USE_HUB_KERNELS": "NO"}

    for model_name, spec, sweep in specs_and_sweeps(args.repo_root, args.models):
        model_dir = args.output_root / model_name
        direction_dir = model_dir / "directions"
        local_dirs = derive_deconfound_payloads(args.repo_root, spec, sweep, direction_dir)
        model_manifest: Dict[str, Any] = {"layers": list(sweep.layers), "runs": []}

        for variant in parse_csv(args.variants):
            direction_path, mode = _variant_path(model_dir, args.caa_root, model_name, variant, local_dirs)
            if variant == "caa" and not direction_path.exists():
                print(f"[skip] {model_name}/{variant}: missing CAA direction {direction_path}")
                model_manifest["runs"].append({"variant": variant, "skipped": True, "reason": "missing_caa_direction"})
                continue
            suffix = args.output_suffix
            if run_trivia:
                out_dir = model_dir / f"trivia_{variant}{suffix}"
                summary = out_dir / "steering_summary.json"
                if args.execute and summary.exists() and not args.force:
                    print(f"[skip] existing {summary}")
                else:
                    cmd = trivia_args(
                        spec=spec,
                        direction_path=direction_path,
                        layers=sweep.layers,
                        alphas=args.alphas,
                        output_dir=out_dir,
                        batch_size=sweep.trivia_batch_size,
                        max_batch_tokens=sweep.piqa_max_batch_tokens,
                        mode=mode,
                        max_items=args.max_trivia_items,
                        no_compile=args.no_compile,
                        conditions=args.conditions,
                        uids_subset=args.uids_subset,
                    )
                    rc = run_module("src.exp16.run_steering_test", cmd, dry_run=not args.execute, env=env)
                    model_manifest["runs"].append({"task": "trivia", "variant": variant, "mode": mode, "conditions": args.conditions, "returncode": rc})
                    if args.execute and rc:
                        raise SystemExit(f"Failed {model_name}/trivia/{variant}")
            if run_piqa:
                out_dir = model_dir / f"piqa_{variant}{suffix}"
                summary = out_dir / "piqa_summary.json"
                if args.execute and summary.exists() and not args.force:
                    print(f"[skip] existing {summary}")
                else:
                    cmd = piqa_args(
                        spec=spec,
                        direction_path=direction_path,
                        layers=sweep.layers,
                        alphas=args.alphas,
                        output_dir=out_dir,
                        batch_size=sweep.piqa_batch_size,
                        max_batch_tokens=sweep.piqa_max_batch_tokens,
                        mode=mode,
                        max_samples=args.max_piqa_samples,
                        no_compile=args.no_compile,
                        conditions=args.conditions,
                    )
                    rc = run_module("src.exp16.run_piqa_interventions", cmd, dry_run=not args.execute, env=env)
                    model_manifest["runs"].append({"task": "piqa", "variant": variant, "mode": mode, "conditions": args.conditions, "returncode": rc})
                    if args.execute and rc:
                        raise SystemExit(f"Failed {model_name}/piqa/{variant}")
        manifest["models"][model_name] = model_manifest
        manifest_name = f"manifest{args.output_suffix}.json" if args.output_suffix else "manifest.json"
        write_json(args.output_root / manifest_name, manifest)


if __name__ == "__main__":
    main()
