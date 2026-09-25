from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run 3 targeted safety follow-ups (multilayer, decode-time, W1-source) via Python."
    )
    p.add_argument("--prompts-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-samples", type=int, default=0, help="0 = use all prompts in the JSONL.")
    return p.parse_args()


def _run(cmd: List[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    common = [
        py,
        "-m",
        "src.authority_steering.run_harmbench_interventions",
        "--model",
        args.model,
        "--prompts-jsonl",
        str(args.prompts_jsonl),
        "--batch-size",
        str(args.batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--additive-modes",
        "",
        "--direction-kinds",
        "",
        "--additive-alphas",
        "0",
        "--sparse-topk-values",
        "0",
        "--max-samples",
        str(args.max_samples),
    ]

    # 1) Multi-layer band patching
    _run(
        common
        + [
            "--target-layers",
            "16,18,20",
            "--patch-modes",
            "interpolate_mean,replace_mean",
            "--patch-alphas",
            "0.5,0.7,1.0",
            "--output-dir",
            str(args.output_dir / "multilayer"),
        ]
    )

    # 2) Decode-time patching
    _run(
        common
        + [
            "--target-layers",
            "16,18,20",
            "--single-layer-sweep",
            "--apply-phase",
            "decode",
            "--patch-modes",
            "interpolate_mean",
            "--patch-alphas",
            "0.5,0.7,1.0",
            "--output-dir",
            str(args.output_dir / "decode"),
        ]
    )

    # 3) W1 source-state patching
    _run(
        common
        + [
            "--target-layers",
            "16,18,20",
            "--single-layer-sweep",
            "--patch-modes",
            "interpolate_mean,replace_mean",
            "--patch-alphas",
            "0.7,1.0",
            "--patch-source-condition",
            "W1_note",
            "--output-dir",
            str(args.output_dir / "w1_source"),
        ]
    )

    print(f"\nDone. Outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
