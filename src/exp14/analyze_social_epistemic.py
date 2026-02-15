"""Exp14D: Social vs epistemic factorial decomposition (2x2x2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exp14 social vs epistemic factorial effects")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("new-phase-results/exp14"),
        help="Directory containing Exp14 *_results.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("new-phase-results/exp14d"),
        help="Directory for analysis outputs",
    )
    parser.add_argument("--high-tag", type=str, default="expert")
    parser.add_argument("--low-tag", type=str, default="someone_online")
    parser.add_argument("--low-evidence", type=str, default="bare")
    parser.add_argument("--high-evidence", type=str, default="reason2")
    parser.add_argument("--instruction-key", type=str, default="I0")
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prior-tag", type=str, default="expert")
    parser.add_argument("--prior-instruction-key", type=str, default="0")
    parser.add_argument("--prior-evidence-level", type=str, default="bare")
    return parser.parse_args()


def _normalize_tag(tag: str) -> str:
    return tag.lower().replace(" ", "_").replace("-", "_")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("rb") as f:
        lines = f.readlines()
    for idx, raw in enumerate(lines):
        text = raw.decode("utf-8", errors="strict").strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            if idx == len(lines) - 1:
                break
            raise ValueError(f"Invalid JSON at {path}:{idx + 1}") from exc
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _bootstrap_mean_ci(values: Sequence[float], *, n_boot: int, seed: int) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = arr.size
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, n)]
        boot[i] = float(np.mean(sample))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _summ_mean(values: Sequence[float], *, n_boot: int, seed: int) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    lo, hi = _bootstrap_mean_ci(values, n_boot=n_boot, seed=seed)
    return {"mean": float(np.mean(arr)), "ci95_low": lo, "ci95_high": hi}


def _slice_records(
    records: Sequence[Dict[str, Any]],
    *,
    prior_code: str,
) -> Dict[str, List[Dict[str, Any]]]:
    out = {"all": list(records), "prior_correct": [], "prior_wrong": []}
    for row in records:
        fc = row.get("fc_correct", {})
        if not isinstance(fc, dict) or prior_code not in fc:
            continue
        p = float(fc[prior_code])
        if p > 0.5:
            out["prior_correct"].append(row)
        elif p < 0.5:
            out["prior_wrong"].append(row)
    return out


def _extract_shift(
    row: Dict[str, Any],
    *,
    tag: str,
    evidence: str,
    instruction_key: str,
    direction: str,
) -> float | None:
    metrics = row.get("evidence_metrics", {})
    if not isinstance(metrics, dict):
        return None
    tag_payload = metrics.get(tag, {})
    if not isinstance(tag_payload, dict):
        return None
    level_payload = tag_payload.get(evidence, {})
    if not isinstance(level_payload, dict):
        return None
    instr_payload = level_payload.get(instruction_key, {})
    if not isinstance(instr_payload, dict):
        return None

    if direction == "wrong":
        key = "logit_shift_wrong"
    elif direction == "correct":
        key = "logit_shift_correct"
    else:
        raise ValueError(f"Unknown direction: {direction}")

    if key not in instr_payload:
        return None
    return float(instr_payload[key])


def _decompose(
    rows: Sequence[Dict[str, Any]],
    *,
    high_tag: str,
    low_tag: str,
    low_evidence: str,
    high_evidence: str,
    instruction_key: str,
    direction: str,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    cells = {
        "high_tag_low_evidence": [],
        "high_tag_high_evidence": [],
        "low_tag_low_evidence": [],
        "low_tag_high_evidence": [],
    }

    for row in rows:
        a = _extract_shift(
            row,
            tag=high_tag,
            evidence=low_evidence,
            instruction_key=instruction_key,
            direction=direction,
        )
        b = _extract_shift(
            row,
            tag=high_tag,
            evidence=high_evidence,
            instruction_key=instruction_key,
            direction=direction,
        )
        c = _extract_shift(
            row,
            tag=low_tag,
            evidence=low_evidence,
            instruction_key=instruction_key,
            direction=direction,
        )
        d = _extract_shift(
            row,
            tag=low_tag,
            evidence=high_evidence,
            instruction_key=instruction_key,
            direction=direction,
        )
        if any(v is None for v in (a, b, c, d)):
            continue

        cells["high_tag_low_evidence"].append(float(a))
        cells["high_tag_high_evidence"].append(float(b))
        cells["low_tag_low_evidence"].append(float(c))
        cells["low_tag_high_evidence"].append(float(d))

    n_items = len(cells["high_tag_low_evidence"])
    if n_items == 0:
        return {
            "n_items": 0,
            "cells": {},
            "effects": {},
        }

    a = np.array(cells["high_tag_low_evidence"], dtype=float)
    b = np.array(cells["high_tag_high_evidence"], dtype=float)
    c = np.array(cells["low_tag_low_evidence"], dtype=float)
    d = np.array(cells["low_tag_high_evidence"], dtype=float)

    authority = ((a + b) / 2.0) - ((c + d) / 2.0)
    evidence = ((b + d) / 2.0) - ((a + c) / 2.0)
    interaction = (b - a) - (d - c)

    return {
        "n_items": n_items,
        "cells": {
            key: _summ_mean(values, n_boot=n_boot, seed=seed + i)
            for i, (key, values) in enumerate(cells.items())
        },
        "effects": {
            "authority_main": _summ_mean(authority.tolist(), n_boot=n_boot, seed=seed + 10),
            "evidence_main": _summ_mean(evidence.tolist(), n_boot=n_boot, seed=seed + 11),
            "interaction": _summ_mean(interaction.tolist(), n_boot=n_boot, seed=seed + 12),
        },
    }


def _analyze_file(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    rows = _load_jsonl(path)
    if not rows:
        raise ValueError(f"No rows in {path}")

    model = str(rows[0].get("model", "unknown"))
    high_tag = _normalize_tag(args.high_tag)
    low_tag = _normalize_tag(args.low_tag)
    prior_tag = _normalize_tag(args.prior_tag)
    prior_code = f"N{args.prior_instruction_key}_{prior_tag}_{args.prior_evidence_level}"

    slices = _slice_records(rows, prior_code=prior_code)

    out_slices: Dict[str, Any] = {}
    for slice_name, slice_rows in slices.items():
        out_slices[slice_name] = {
            "wrong": _decompose(
                slice_rows,
                high_tag=high_tag,
                low_tag=low_tag,
                low_evidence=args.low_evidence,
                high_evidence=args.high_evidence,
                instruction_key=args.instruction_key,
                direction="wrong",
                n_boot=args.n_boot,
                seed=args.seed + int(abs(hash((slice_name, "wrong"))) % 100000),
            ),
            "correct": _decompose(
                slice_rows,
                high_tag=high_tag,
                low_tag=low_tag,
                low_evidence=args.low_evidence,
                high_evidence=args.high_evidence,
                instruction_key=args.instruction_key,
                direction="correct",
                n_boot=args.n_boot,
                seed=args.seed + int(abs(hash((slice_name, "correct"))) % 100000),
            ),
        }

    return {
        "experiment": "exp14_social_epistemic",
        "model": model,
        "source": str(path),
        "design": {
            "high_tag": high_tag,
            "low_tag": low_tag,
            "low_evidence": args.low_evidence,
            "high_evidence": args.high_evidence,
            "instruction_key": args.instruction_key,
            "prior_code": prior_code,
        },
        "slice_counts": {name: len(rows_) for name, rows_ in slices.items()},
        "slices": out_slices,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = sorted(args.results_dir.glob("*_results.jsonl"))
    if not results:
        raise FileNotFoundError(f"No *_results.jsonl files found in {args.results_dir}")

    analyses = []
    for path in results:
        print(f"Analyzing {path} ...")
        try:
            analysis = _analyze_file(path, args)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue
        analyses.append(analysis)
        out_path = args.output_dir / f"{path.stem.replace('_results', '')}_social_epistemic.json"
        out_path.write_text(json.dumps(analysis, indent=2))

    if not analyses:
        raise RuntimeError("No analyses generated.")

    (args.output_dir / "summary.json").write_text(json.dumps({"models": analyses}, indent=2))
    print(f"Wrote Exp14D analysis outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
