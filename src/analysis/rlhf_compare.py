"""RLHF Base vs Instruct comparison utilities.

Usage:
    python -m src.analysis.rlhf_compare \
      --base llama-results/llama31_base \
      --instruct llama-results/llama31_instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def gini(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and float(v) >= 0]
    if not vals:
        return 0.0
    vals.sort()
    total = sum(vals)
    if total <= 0:
        return 0.0
    n = len(vals)
    numerator = 0.0
    for i, x in enumerate(vals, start=1):
        numerator += (2 * i - n - 1) * x
    return numerator / (n * total)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _steering_stats(exp3_summary: Dict[str, Any]) -> Dict[str, Any]:
    e_clean = exp3_summary.get("E_clean_values") or []
    if not e_clean:
        return {}
    return {
        "E_clean_mean": mean(e_clean),
        "E_clean_median": median(e_clean),
        "E_clean_median_abs": median([abs(x) for x in e_clean]),
        "E_clean_n": len(e_clean),
    }


def _vector_grid_best(exp1_selection: Dict[str, Any]) -> Dict[str, Any]:
    grid = exp1_selection.get("grid") or []
    if not grid:
        return {}
    best = max(grid, key=lambda row: float(row.get("mean_effect", 0.0)))
    return {
        "best_layer": best.get("layer"),
        "best_alpha": best.get("alpha"),
        "best_mean_effect": best.get("mean_effect"),
        "best_std_effect": best.get("std_effect"),
        "best_n": best.get("n"),
    }


def load_run(run_dir: Path) -> Dict[str, Any]:
    exp0_path = run_dir / "exp0" / "baseline_metrics.json"
    exp1_path = run_dir / "exp1" / "vector_selection.json"
    exp2_scores_path = run_dir / "exp2" / "head_scores.jsonl"
    exp2_heads_path = run_dir / "exp2" / "syc_heads.json"
    exp3_summary_path = run_dir / "exp3" / "mediation_summary.json"

    missing = [
        str(p)
        for p in [exp0_path, exp1_path, exp2_scores_path, exp2_heads_path, exp3_summary_path]
        if not p.exists()
    ]
    if missing:
        raise SystemExit(
            "Missing required files under run_dir="
            f"{run_dir}:\n" + "\n".join(f"- {m}" for m in missing)
        )

    exp0 = _read_json(exp0_path)
    exp1 = _read_json(exp1_path)
    head_scores = _read_jsonl(exp2_scores_path)
    syc_heads_raw = _read_json(exp2_heads_path)
    exp3 = _read_json(exp3_summary_path)

    syc_head_set: set[Tuple[int, int]] = {
        (int(h["layer"]), int(h["head"])) for h in syc_heads_raw
    }
    import_abs_delta_sum = [abs(float(r.get("delta_sum", 0.0))) for r in head_scores]
    import_abs_delta_mean = [abs(float(r.get("delta_mean", 0.0))) for r in head_scores]

    return {
        "run_dir": str(run_dir),
        "model_name": exp0.get("model_name"),
        "exp0": {
            "num_examples": exp0.get("num_examples"),
            "syc_rate": exp0.get("syc_rate"),
            "D_syc_mean": exp0.get("D_syc_mean"),
            "D_syc_std": exp0.get("D_syc_std"),
        },
        "exp1": {
            "margin": exp1.get("margin"),
            "num_syc_examples": exp1.get("num_syc_examples"),
            "num_truth_examples": exp1.get("num_truth_examples"),
            "L_star": exp1.get("L_star"),
            "alpha_star": exp1.get("alpha_star"),
            **_vector_grid_best(exp1),
        },
        "exp2": {
            "syc_head_count": len(syc_head_set),
            "syc_unique_layers": len({l for (l, _) in syc_head_set}),
            "gini_abs_delta_sum": gini(import_abs_delta_sum),
            "gini_abs_delta_mean": gini(import_abs_delta_mean),
        },
        "exp3": {
            "num_examples": exp3.get("num_examples"),
            "num_examples_valid": exp3.get("num_examples_valid"),
            "head_set_used": exp3.get("head_set_used"),
            "num_heads_tested": exp3.get("num_heads_tested"),
            "num_unique_layers": exp3.get("num_unique_layers"),
            "mean_RAS_syc": exp3.get("mean_RAS_syc"),
            "mean_RAS_rand": exp3.get("mean_RAS_rand"),
            "mean_diff": exp3.get("mean_diff"),
            "cohens_d": exp3.get("cohens_d"),
            "p_value": exp3.get("p_value"),
            "threshold": exp3.get("threshold"),
            "threshold_mode": exp3.get("threshold_mode"),
            **_steering_stats(exp3),
        },
        "_syc_head_set": sorted(list(syc_head_set)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Base vs Instruct experiment runs")
    parser.add_argument("--base", type=Path, required=True, help="Run dir for base model")
    parser.add_argument("--instruct", type=Path, required=True, help="Run dir for instruct model")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <parent>/rlhf_comparison.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_run(args.base)
    instruct = load_run(args.instruct)

    base_heads = set(tuple(x) for x in base["_syc_head_set"])
    instruct_heads = set(tuple(x) for x in instruct["_syc_head_set"])

    out_path = args.out
    if out_path is None:
        parent = args.base.parent
        out_path = parent / "rlhf_comparison.json"

    comparison = {
        "base": {k: v for k, v in base.items() if not k.startswith("_")},
        "instruct": {k: v for k, v in instruct.items() if not k.startswith("_")},
        "comparison": {
            "syc_head_jaccard": jaccard(base_heads, instruct_heads),
            "syc_head_overlap": len(base_heads & instruct_heads),
            "syc_head_union": len(base_heads | instruct_heads),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(comparison, f, indent=2)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
