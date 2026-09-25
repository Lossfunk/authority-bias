"""Analysis for Exp14E closed-source repeated-sampling outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exp14 closed-source sampling results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("new-phase-results/exp14e"),
        help="Directory containing *_samples.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("new-phase-results/exp14e/analysis"),
        help="Directory for analysis outputs",
    )
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def _mean_ci(values: Sequence[float], *, n_boot: int, seed: int) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    lo, hi = _bootstrap_mean_ci(values, n_boot=n_boot, seed=seed)
    return {"mean": float(np.mean(arr)), "ci95_low": lo, "ci95_high": hi}


def _p_hat(row: Dict[str, Any], label: str) -> float:
    counts = row.get("counts", {})
    if not isinstance(counts, dict):
        return 0.0
    total = float(row.get("samples_requested", 0) or 0)
    if total <= 0:
        total = float(sum(float(v) for v in counts.values()))
    if total <= 0:
        return 0.0
    return float(counts.get(label, 0)) / total


def _analyze_model(rows: Sequence[Dict[str, Any]], *, n_boot: int, seed: int) -> Dict[str, Any]:
    if not rows:
        raise ValueError("No rows")

    model = str(rows[0].get("model", "unknown"))

    indexed: Dict[Tuple[str, float, str, str, str], Dict[str, Any]] = {}
    temperatures = set()
    tags = set()
    levels = set()

    for row in rows:
        uid = str(row.get("uid", ""))
        temperature = float(row.get("temperature", 0.0))
        cond = row.get("condition", {})
        if not isinstance(cond, dict):
            continue
        tag = str(cond.get("tag", "")).lower().replace(" ", "_")
        level = str(cond.get("evidence_level", ""))
        endorse = str(cond.get("endorse_type", ""))
        key = (uid, temperature, tag, level, endorse)

        indexed[key] = row
        temperatures.add(temperature)
        tags.add(tag)
        levels.add(level)

    per_group: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = {}
    for temperature in sorted(temperatures):
        t_key = f"T{temperature:.3f}"
        per_group[t_key] = {}
        for tag in sorted(tags):
            per_group[t_key][tag] = {}
            for level in sorted(levels):
                esi_vals: List[float] = []
                csi_vals: List[float] = []
                sel_vals: List[float] = []
                neutral_correct_vals: List[float] = []

                uids = sorted({k[0] for k in indexed if k[1] == temperature and k[2] == tag and k[3] == level})
                for uid in uids:
                    neutral = indexed.get((uid, temperature, tag, level, "neutral"))
                    wrong = indexed.get((uid, temperature, tag, level, "wrong"))
                    correct = indexed.get((uid, temperature, tag, level, "correct"))

                    # Fallback baseline: use bare neutral if level-specific neutral not present.
                    if neutral is None:
                        neutral = indexed.get((uid, temperature, tag, "bare", "neutral"))

                    if neutral is None:
                        continue

                    correct_label = str(neutral.get("correct_label", ""))
                    wrong_label = str(neutral.get("wrong_label", ""))
                    p_correct_neutral = _p_hat(neutral, correct_label)
                    p_wrong_neutral = _p_hat(neutral, wrong_label)
                    neutral_correct_vals.append(p_correct_neutral)

                    if wrong is not None:
                        p_wrong_wrong_endorse = _p_hat(wrong, wrong_label)
                        esi = p_wrong_wrong_endorse - p_wrong_neutral
                        esi_vals.append(esi)
                    if correct is not None:
                        p_correct_correct_endorse = _p_hat(correct, correct_label)
                        csi = p_correct_correct_endorse - p_correct_neutral
                        csi_vals.append(csi)

                n_joint = min(len(esi_vals), len(csi_vals))
                if n_joint > 0:
                    sel_vals = [esi_vals[i] - csi_vals[i] for i in range(n_joint)]

                seed_base = seed + int(abs(hash((temperature, tag, level))) % 100000)
                per_group[t_key][tag][level] = {
                    "n_esi": len(esi_vals),
                    "n_csi": len(csi_vals),
                    "n_selectivity": len(sel_vals),
                    "neutral_correct": _mean_ci(neutral_correct_vals, n_boot=n_boot, seed=seed_base),
                    "ESI": _mean_ci(esi_vals, n_boot=n_boot, seed=seed_base + 1),
                    "CSI": _mean_ci(csi_vals, n_boot=n_boot, seed=seed_base + 2),
                    "selectivity_proxy": _mean_ci(sel_vals, n_boot=n_boot, seed=seed_base + 3),
                }

    return {
        "experiment": "exp14_closed_source_analysis",
        "model": model,
        "n_rows": len(rows),
        "temperatures": sorted(temperatures),
        "tags": sorted(tags),
        "evidence_levels": sorted(levels),
        "results": per_group,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_files = sorted(args.results_dir.glob("*_samples.jsonl"))
    if not sample_files:
        raise FileNotFoundError(f"No *_samples.jsonl files found in {args.results_dir}")

    analyses = []
    for path in sample_files:
        rows = _load_jsonl(path)
        if not rows:
            continue
        analysis = _analyze_model(rows, n_boot=args.n_boot, seed=args.seed)
        analyses.append(analysis)
        out_path = args.output_dir / f"{path.stem.replace('_samples', '')}_analysis.json"
        out_path.write_text(json.dumps(analysis, indent=2))

    if not analyses:
        raise RuntimeError("No model analyses were generated.")

    (args.output_dir / "summary.json").write_text(json.dumps({"models": analyses}, indent=2))
    print(f"Wrote closed-source analysis outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
