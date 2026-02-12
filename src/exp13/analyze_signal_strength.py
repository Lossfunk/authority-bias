"""Analysis for Exp13 signal-strength experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.exp11.logit_metrics import TAU_DEFAULT, ItemMetrics, bootstrap_all_metrics_coherent, compute_item_metrics
from src.exp13.conditions import METRIC_NAMES


KEY_METRICS = ("effect_wrong_I0", "efficacy_wrong", "selectivity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exp13 signal-strength results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("new-phase-results/exp13"),
        help="Directory containing Exp13 *_results.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("new-phase-results/exp13/analysis"),
        help="Directory to write analysis artifacts",
    )
    parser.add_argument("--n-boot", type=int, default=3000, help="Bootstrap replicates")
    parser.add_argument("--tau", type=float, default=TAU_DEFAULT, help="Mask threshold for ratio diagnostics")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("rb") as f:
        lines = f.readlines()

    for idx, raw in enumerate(lines):
        line_no = idx + 1
        line = raw.decode("utf-8", errors="strict").strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            if idx == len(lines) - 1:
                # Interrupted run may leave a partial trailing line.
                break
            raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid record type at {path}:{line_no}; expected object.")
        records.append(payload)
    return records


def _summarize(values: Iterable[float]) -> Dict[str, float]:
    vals = np.array(list(values), dtype=float)
    if vals.size == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n": 0,
            "positive_frac": 0.0,
        }
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "n": int(vals.size),
        "positive_frac": float(np.mean(vals > 0)),
    }


def _bootstrap_mean_ci(values: Sequence[float], *, n_boot: int, seed: int) -> Tuple[float, float]:
    vals = np.array(values, dtype=float)
    if vals.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot_means = []
    n = vals.size
    for _ in range(n_boot):
        sample = vals[rng.integers(0, n, n)]
        boot_means.append(float(np.mean(sample)))
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return lo, hi


def _metric_with_ci(values: Sequence[float], *, n_boot: int, seed: int) -> Dict[str, float]:
    summary = _summarize(values)
    lo, hi = _bootstrap_mean_ci(values, n_boot=n_boot, seed=seed)
    summary["ci95_low"] = lo
    summary["ci95_high"] = hi
    return summary


def _margin_correct(record: Dict[str, Any], condition_code: str) -> float:
    cond = record["condition_results"][condition_code]
    if record["correct_label"] == "A":
        return float(cond["logit_a"])
    return float(cond["logit_b"])


def _build_item_metrics(records: Sequence[Dict[str, Any]], *, tag_key: str, variant: str) -> List[ItemMetrics]:
    out: List[ItemMetrics] = []
    n0_code = f"N0_{tag_key}"
    n1_code = f"N1_{tag_key}"
    w0_code = f"W0_{tag_key}_{variant}"
    w1_code = f"W1_{tag_key}_{variant}"
    c0_code = f"C0_{tag_key}_{variant}"
    c1_code = f"C1_{tag_key}_{variant}"

    for record in records:
        conds = record.get("condition_results", {})
        needed = (n0_code, n1_code, w0_code, w1_code, c0_code, c1_code)
        if any(code not in conds for code in needed):
            continue
        item = compute_item_metrics(
            m_N0=_margin_correct(record, n0_code),
            m_N1=_margin_correct(record, n1_code),
            m_W0=_margin_correct(record, w0_code),
            m_W1=_margin_correct(record, w1_code),
            m_C0=_margin_correct(record, c0_code),
            m_C1=_margin_correct(record, c1_code),
            question_id=str(record.get("uid", "")),
        )
        out.append(item)
    return out


def _create_inverted_prior_slices(items: Sequence[ItemMetrics]) -> Dict[str, List[ItemMetrics]]:
    inverted = [item for item in items if item.m_N0 < 0]
    if not inverted:
        return {}

    abs_vals = np.array([abs(item.m_N0) for item in inverted], dtype=float)
    order = np.argsort(abs_vals)
    n = len(inverted)

    slices: Dict[str, List[ItemMetrics]] = {"full": inverted}
    top25_start = int(0.75 * n)
    slices["high_conf_top25"] = [inverted[i] for i in order[top25_start:]]
    return slices


def _slice_summary(items: Sequence[ItemMetrics]) -> Dict[str, float]:
    vals = np.array([item.m_N0 for item in items], dtype=float)
    if vals.size == 0:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "n": int(vals.size),
    }


def _compute_wrong_prior_analysis(
    items: Sequence[ItemMetrics],
    *,
    n_boot: int,
    tau: float,
    seed: int,
) -> Dict[str, Any]:
    slices = _create_inverted_prior_slices(items)
    out: Dict[str, Any] = {}
    for slice_name, slice_items in slices.items():
        if not slice_items:
            continue
        np.random.seed(seed)
        metrics = bootstrap_all_metrics_coherent(
            list(slice_items),
            n_boot=n_boot,
            use_sign_consistent_mask=True,
            tau=tau,
        )
        np.random.seed(seed + 1)
        metrics_abs = bootstrap_all_metrics_coherent(
            list(slice_items),
            n_boot=n_boot,
            use_sign_consistent_mask=False,
            tau=tau,
        )
        out[slice_name] = {
            "n_items": len(slice_items),
            "m_N0_stats": _slice_summary(slice_items),
            "metrics": _to_builtin(metrics),
            "metrics_abs": _to_builtin(metrics_abs),
        }
    return out


def _analyze_model(
    results_path: Path,
    *,
    n_boot: int,
    tau: float,
    seed: int,
) -> Dict[str, Any]:
    records = _load_jsonl(results_path)
    if not records:
        raise ValueError(f"No records found in {results_path}")

    model_id = str(records[0]["model"])
    signal_template = records[0]["signal_metrics"]
    tags = sorted(signal_template.keys())
    if not tags:
        raise ValueError(f"No tag metrics found in {results_path}")
    variants = sorted(signal_template[tags[0]].keys())

    metric_values: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        tag: {variant: {metric: [] for metric in METRIC_NAMES} for variant in variants}
        for tag in tags
    }
    for record in records:
        signal = record["signal_metrics"]
        for tag in tags:
            for variant in variants:
                payload = signal[tag][variant]
                for metric in METRIC_NAMES:
                    metric_values[tag][variant][metric].append(float(payload[metric]))

    metric_summaries: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {
        tag: {
            variant: {
                metric: _metric_with_ci(vals, n_boot=n_boot, seed=seed)
                for metric, vals in metric_values[tag][variant].items()
            }
            for variant in variants
        }
        for tag in tags
    }

    wrong_prior_slices: Dict[str, Dict[str, Dict[str, Any]]] = {tag: {} for tag in tags}
    for tag in tags:
        for variant in variants:
            items = _build_item_metrics(records, tag_key=tag, variant=variant)
            wrong_prior_slices[tag][variant] = _compute_wrong_prior_analysis(
                items,
                n_boot=n_boot,
                tau=tau,
                seed=seed,
            )

    return {
        "experiment": "exp13_signal_strength",
        "model": model_id,
        "n_records": len(records),
        "tags": tags,
        "variants": variants,
        "metric_summaries": metric_summaries,
        "wrong_prior_slices": wrong_prior_slices,
    }


def _write_cross_model_table(analyses: Sequence[Dict[str, Any]], output_csv: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for analysis in analyses:
        model = analysis["model"]
        for tag in analysis["tags"]:
            for variant in analysis["variants"]:
                for metric in KEY_METRICS:
                    payload = analysis["metric_summaries"][tag][variant][metric]
                    rows.append(
                        {
                            "model": model,
                            "tag": tag,
                            "variant": variant,
                            "metric": metric,
                            "mean": float(payload["mean"]),
                            "ci95_low": float(payload["ci95_low"]),
                            "ci95_high": float(payload["ci95_high"]),
                        }
                    )

    grouped_max: Dict[Tuple[str, str], float] = {}
    for row in rows:
        key = (row["model"], row["metric"])
        grouped_max[key] = max(grouped_max.get(key, 0.0), abs(row["mean"]))
    for row in rows:
        denom = grouped_max[(row["model"], row["metric"])]
        row["normalized_mean"] = row["mean"] / denom if denom > 0 else 0.0

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "tag",
                "variant",
                "metric",
                "mean",
                "ci95_low",
                "ci95_high",
                "normalized_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_report(analyses: Sequence[Dict[str, Any]], path: Path) -> None:
    lines: List[str] = []
    lines.append("# Exp13 Signal-Strength Report")
    lines.append("")
    lines.append(
        "This report summarizes certainty/salience effects on endorsement behavior, "
        "instruction efficacy, and wrong-prior slice outcomes."
    )
    lines.append("")

    for analysis in analyses:
        model = analysis["model"]
        lines.append(f"## {model}")
        lines.append(f"- n_records: {analysis['n_records']}")
        lines.append(f"- tags: {', '.join(analysis['tags'])}")
        lines.append(f"- variants: {', '.join(analysis['variants'])}")
        lines.append("")

        for tag in analysis["tags"]:
            lines.append(f"### Tag: {tag}")
            best_wrong = None
            best_wrong_val = -float("inf")
            best_sel = None
            best_sel_val = -float("inf")
            for variant in analysis["variants"]:
                wrong_val = analysis["metric_summaries"][tag][variant]["effect_wrong_I0"]["mean"]
                sel_val = analysis["metric_summaries"][tag][variant]["selectivity"]["mean"]
                if wrong_val > best_wrong_val:
                    best_wrong_val = wrong_val
                    best_wrong = variant
                if sel_val > best_sel_val:
                    best_sel_val = sel_val
                    best_sel = variant
            lines.append(f"- strongest wrong-endorsement effect variant: `{best_wrong}` ({best_wrong_val:.4f})")
            lines.append(f"- highest selectivity variant: `{best_sel}` ({best_sel_val:.4f})")

            hp = analysis["wrong_prior_slices"][tag]
            dr_values: List[Tuple[str, float]] = []
            for variant in analysis["variants"]:
                slices = hp.get(variant, {})
                top = slices.get("high_conf_top25", {})
                metrics = top.get("metrics", {})
                dr = metrics.get("dr_median")
                if isinstance(dr, list) and dr:
                    dr_values.append((variant, float(dr[0])))
            if dr_values:
                worst_variant, worst_dr = min(dr_values, key=lambda x: x[1])
                lines.append(f"- weakest high-conf-wrong `dr_median` variant: `{worst_variant}` ({worst_dr:.4f})")
            lines.append("")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_files = sorted(args.results_dir.glob("*_results.jsonl"))
    if not result_files:
        raise FileNotFoundError(f"No *_results.jsonl files found in {args.results_dir}")

    analyses: List[Dict[str, Any]] = []
    for result_file in result_files:
        print(f"Analyzing {result_file} ...")
        analysis = _analyze_model(
            result_file,
            n_boot=args.n_boot,
            tau=args.tau,
            seed=args.seed,
        )
        analyses.append(analysis)
        model_tag = result_file.stem.replace("_results", "")
        out_path = args.output_dir / f"{model_tag}_analysis.json"
        with out_path.open("w") as f:
            json.dump(_to_builtin(analysis), f, indent=2)

    combined_path = args.output_dir / "combined_analysis.json"
    with combined_path.open("w") as f:
        json.dump(_to_builtin({"models": analyses}), f, indent=2)

    _write_cross_model_table(analyses, args.output_dir / "combined_cross_model_table.csv")
    _write_report(analyses, args.output_dir / "report.md")

    print(f"Wrote analysis outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
