"""Analysis for Exp14A evidence-quality results.

Outputs:
- per-model monotonicity summaries (Kendall tau, Page trend, monotonic fraction)
- per-slice curves across evidence levels (wrong/correct endorsements)
- asymmetry diagnostics and prior-wrong key checks
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats

from src.exp14.conditions import DEFAULT_EVIDENCE_LEVELS, normalize_tag


DEFAULT_SLICES = (
    "all",
    "prior_correct",
    "prior_wrong",
    "prior_wrong_high_conf_top25",
    "prior_wrong_top10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exp14 evidence-quality outputs")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("new-phase-results/llama-3.1-8b-results/exp14"),
        help="Directory containing Exp14 *_results.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("new-phase-results/llama-3.1-8b-results/exp14/analysis"),
        help="Directory to write analysis artifacts",
    )
    parser.add_argument("--n-boot", type=int, default=3000, help="Bootstrap replicates")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prior-tag",
        type=str,
        default="expert",
        help="Tag used to define prior slices (default: expert)",
    )
    parser.add_argument(
        "--prior-instruction-key",
        type=str,
        default="0",
        help="Instruction key used for prior slices (default: 0)",
    )
    parser.add_argument(
        "--prior-evidence-level",
        type=str,
        default="bare",
        help="Evidence level used for prior slices (default: bare)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default="",
        help="Optional comma-separated tag subset to analyze",
    )
    parser.add_argument(
        "--evidence-order",
        type=str,
        default=",".join(DEFAULT_EVIDENCE_LEVELS),
        help="Comma-separated evidence level order",
    )
    return parser.parse_args()


def _parse_csv(raw: str) -> List[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


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
        line = raw.decode("utf-8", errors="strict").strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            if idx == len(lines) - 1:
                break
            raise ValueError(f"Invalid JSON at {path}:{idx + 1}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid row type at {path}:{idx + 1}; expected object")
        records.append(payload)
    return records


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


def _bootstrap_vector_ci(matrix: np.ndarray, *, n_boot: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap CI for per-column means of matrix [n_items, n_levels]."""
    if matrix.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    rng = np.random.default_rng(seed)
    n_items, n_levels = matrix.shape
    boot = np.empty((n_boot, n_levels), dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n_items, n_items)
        sample = matrix[idx, :]
        boot[i, :] = np.mean(sample, axis=0)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    return lo, hi


def _monotonic_non_decreasing(values: Sequence[float], tol: float = 1e-9) -> bool:
    return all(values[i] <= values[i + 1] + tol for i in range(len(values) - 1))


def _discover_axes(records: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    if not records:
        raise ValueError("No records to analyze")
    metrics = records[0].get("evidence_metrics", {})
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Records missing evidence_metrics payload")

    tag_keys = sorted(metrics.keys())
    first_tag = tag_keys[0]
    level_payload = metrics[first_tag]
    if not isinstance(level_payload, dict) or not level_payload:
        raise ValueError("Invalid evidence_metrics[tag] payload")
    evidence_levels = sorted(level_payload.keys(), key=lambda x: (x != "bare", x))

    first_level = evidence_levels[0]
    instr_payload = level_payload[first_level]
    if not isinstance(instr_payload, dict) or not instr_payload:
        raise ValueError("Invalid evidence_metrics[tag][level] payload")
    instruction_keys = sorted(instr_payload.keys())
    return tag_keys, evidence_levels, instruction_keys


def _get_prior_margin(record: Dict[str, Any], prior_code: str) -> float | None:
    logit_correct = record.get("logit_correct", {})
    if isinstance(logit_correct, dict) and prior_code in logit_correct:
        return float(logit_correct[prior_code])

    conds = record.get("condition_results", {})
    if not isinstance(conds, dict) or prior_code not in conds:
        return None

    cond = conds[prior_code]
    if not isinstance(cond, dict):
        return None
    correct_label = record.get("correct_label")
    if correct_label == "A" and "logit_a" in cond:
        return float(cond["logit_a"])
    if correct_label == "B" and "logit_b" in cond:
        return float(cond["logit_b"])
    return None


def _build_slices(
    records: Sequence[Dict[str, Any]],
    *,
    prior_code: str,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {name: [] for name in DEFAULT_SLICES}
    out["all"] = list(records)

    prior_wrong_rows: List[Tuple[float, Dict[str, Any]]] = []
    for record in records:
        fc_correct = record.get("fc_correct", {})
        if not isinstance(fc_correct, dict) or prior_code not in fc_correct:
            continue
        p = float(fc_correct[prior_code])
        margin = _get_prior_margin(record, prior_code)
        if margin is None:
            continue

        if p > 0.5:
            out["prior_correct"].append(record)
        elif p < 0.5:
            out["prior_wrong"].append(record)
            prior_wrong_rows.append((abs(margin), record))

    if prior_wrong_rows:
        prior_wrong_rows.sort(key=lambda x: x[0])
        n = len(prior_wrong_rows)
        top25_start = int(np.floor(0.75 * n))
        top10_start = int(np.floor(0.90 * n))
        out["prior_wrong_high_conf_top25"] = [row for _, row in prior_wrong_rows[top25_start:]]
        out["prior_wrong_top10"] = [row for _, row in prior_wrong_rows[top10_start:]]

    return out


def _collect_matrix(
    records: Sequence[Dict[str, Any]],
    *,
    tag_key: str,
    instruction_key: str,
    evidence_levels: Sequence[str],
    metric_name: str,
) -> Tuple[np.ndarray, List[str]]:
    rows: List[List[float]] = []
    uids: List[str] = []
    for record in records:
        metrics = record.get("evidence_metrics", {})
        if not isinstance(metrics, dict):
            continue
        tag_payload = metrics.get(tag_key, {})
        if not isinstance(tag_payload, dict):
            continue

        values: List[float] = []
        ok = True
        for level in evidence_levels:
            level_payload = tag_payload.get(level, {})
            if not isinstance(level_payload, dict):
                ok = False
                break
            instr_payload = level_payload.get(instruction_key, {})
            if not isinstance(instr_payload, dict) or metric_name not in instr_payload:
                ok = False
                break
            values.append(float(instr_payload[metric_name]))
        if not ok:
            continue

        rows.append(values)
        uids.append(str(record.get("uid", "")))

    if not rows:
        return np.zeros((0, len(evidence_levels)), dtype=float), []
    return np.array(rows, dtype=float), uids


def _summarize_trend(
    matrix: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    if matrix.size == 0:
        return {
            "n_items": 0,
            "mean_by_level": [],
            "fraction_monotonic_non_decreasing": 0.0,
            "kendall_tau_mean": 0.0,
            "kendall_tau_ci95": [0.0, 0.0],
            "page_trend_test": {
                "statistic": 0.0,
                "pvalue": 1.0,
                "method": "unavailable",
            },
        }

    means = np.mean(matrix, axis=0)
    lo, hi = _bootstrap_vector_ci(matrix, n_boot=n_boot, seed=seed)
    mean_by_level = [
        {
            "mean": float(means[i]),
            "ci95_low": float(lo[i]),
            "ci95_high": float(hi[i]),
        }
        for i in range(matrix.shape[1])
    ]

    mono_flags = [_monotonic_non_decreasing(row.tolist()) for row in matrix]
    frac_mono = float(np.mean(mono_flags)) if mono_flags else 0.0

    x = np.arange(matrix.shape[1], dtype=float)
    taus: List[float] = []
    for row in matrix:
        tau, _ = stats.kendalltau(x, row)
        if not np.isfinite(tau):
            tau = 0.0
        taus.append(float(tau))

    tau_mean = float(np.mean(taus)) if taus else 0.0
    tau_ci = _bootstrap_mean_ci(taus, n_boot=n_boot, seed=seed + 11)

    page_result = {
        "statistic": 0.0,
        "pvalue": 1.0,
        "method": "unavailable",
    }
    if matrix.shape[0] >= 3 and matrix.shape[1] >= 2:
        try:
            page = stats.page_trend_test(matrix)
            page_result = {
                "statistic": float(page.statistic),
                "pvalue": float(page.pvalue),
                "method": str(page.method),
            }
        except Exception:
            pass

    return {
        "n_items": int(matrix.shape[0]),
        "mean_by_level": mean_by_level,
        "fraction_monotonic_non_decreasing": frac_mono,
        "kendall_tau_mean": tau_mean,
        "kendall_tau_ci95": [float(tau_ci[0]), float(tau_ci[1])],
        "page_trend_test": page_result,
    }


def _slice_key_diagnostic(
    records: Sequence[Dict[str, Any]],
    *,
    tag_key: str,
    instruction_key: str,
    low_level: str,
    high_level: str,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    metrics = []
    for record in records:
        payload = record.get("evidence_metrics", {})
        if not isinstance(payload, dict):
            continue
        tag_payload = payload.get(tag_key, {})
        if not isinstance(tag_payload, dict):
            continue

        try:
            low = float(tag_payload[low_level][instruction_key]["logit_shift_correct"])
            high = float(tag_payload[high_level][instruction_key]["logit_shift_correct"])
        except Exception:
            continue
        metrics.append(high - low)

    if not metrics:
        return {
            "n_items": 0,
            "mean_delta": 0.0,
            "ci95": [0.0, 0.0],
            "positive_frac": 0.0,
        }

    ci = _bootstrap_mean_ci(metrics, n_boot=n_boot, seed=seed)
    return {
        "n_items": len(metrics),
        "mean_delta": float(np.mean(metrics)),
        "ci95": [float(ci[0]), float(ci[1])],
        "positive_frac": float(np.mean(np.array(metrics) > 0)),
    }


def _analyze_model(
    result_path: Path,
    *,
    n_boot: int,
    seed: int,
    prior_tag: str,
    prior_instruction_key: str,
    prior_evidence_level: str,
    requested_tags: Sequence[str],
    evidence_order: Sequence[str],
) -> Dict[str, Any]:
    records = _load_jsonl(result_path)
    if not records:
        raise ValueError(f"No records found in {result_path}")

    model_id = str(records[0].get("model", "unknown"))
    discovered_tags, discovered_levels, instruction_keys = _discover_axes(records)

    if requested_tags:
        tag_keys = [tag for tag in requested_tags if tag in discovered_tags]
        if not tag_keys:
            raise ValueError(f"Requested tags {requested_tags} not found in {result_path}")
    else:
        tag_keys = discovered_tags

    level_order = [level for level in evidence_order if level in discovered_levels]
    for level in discovered_levels:
        if level not in level_order:
            level_order.append(level)

    prior_tag_key = normalize_tag(prior_tag)
    if prior_tag_key not in discovered_tags:
        prior_tag_key = tag_keys[0]
    prior_code = f"N{prior_instruction_key}_{prior_tag_key}_{prior_evidence_level}"

    slices = _build_slices(records, prior_code=prior_code)

    monotonicity: Dict[str, Dict[str, Dict[str, Any]]] = {}
    asymmetry: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for slice_name, slice_records in slices.items():
        monotonicity[slice_name] = {}
        asymmetry[slice_name] = {}
        for tag_key in tag_keys:
            monotonicity[slice_name][tag_key] = {}
            asymmetry[slice_name][tag_key] = {}
            for instr_key in instruction_keys:
                wrong_matrix, _ = _collect_matrix(
                    slice_records,
                    tag_key=tag_key,
                    instruction_key=instr_key,
                    evidence_levels=level_order,
                    metric_name="logit_shift_wrong",
                )
                correct_matrix, _ = _collect_matrix(
                    slice_records,
                    tag_key=tag_key,
                    instruction_key=instr_key,
                    evidence_levels=level_order,
                    metric_name="logit_shift_correct",
                )
                asym_matrix, _ = _collect_matrix(
                    slice_records,
                    tag_key=tag_key,
                    instruction_key=instr_key,
                    evidence_levels=level_order,
                    metric_name="logit_asymmetry",
                )

                base_seed = seed + int(abs(hash((slice_name, tag_key, instr_key))) % 100000)
                monotonicity[slice_name][tag_key][instr_key] = {
                    "wrong": _summarize_trend(wrong_matrix, n_boot=n_boot, seed=base_seed + 1),
                    "correct": _summarize_trend(correct_matrix, n_boot=n_boot, seed=base_seed + 2),
                }
                asymmetry[slice_name][tag_key][instr_key] = _summarize_trend(
                    asym_matrix,
                    n_boot=n_boot,
                    seed=base_seed + 3,
                )

    key_diagnostic: Dict[str, Dict[str, Any]] = {}
    prior_wrong_records = slices.get("prior_wrong", [])
    for tag_key in tag_keys:
        key_diagnostic[tag_key] = {}
        for instr_key in instruction_keys:
            key_diagnostic[tag_key][instr_key] = _slice_key_diagnostic(
                prior_wrong_records,
                tag_key=tag_key,
                instruction_key=instr_key,
                low_level=level_order[0],
                high_level=level_order[-1],
                n_boot=n_boot,
                seed=seed + int(abs(hash((tag_key, instr_key, "diag"))) % 100000),
            )

    slice_counts = {name: len(rows) for name, rows in slices.items()}
    return {
        "experiment": "exp14_evidence_quality",
        "model": model_id,
        "n_records": len(records),
        "result_path": str(result_path),
        "tags": tag_keys,
        "instruction_keys": instruction_keys,
        "evidence_levels": level_order,
        "prior_definition": {
            "prior_code": prior_code,
            "prior_tag": prior_tag_key,
            "prior_instruction_key": prior_instruction_key,
            "prior_evidence_level": prior_evidence_level,
        },
        "slice_counts": slice_counts,
        "monotonicity": monotonicity,
        "asymmetry": asymmetry,
        "key_diagnostic_prior_wrong_e3_minus_e0_correct_shift": key_diagnostic,
    }


def _write_monotonicity_table(analyses: Sequence[Dict[str, Any]], output_csv: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for analysis in analyses:
        model = analysis["model"]
        for slice_name, slice_payload in analysis["monotonicity"].items():
            for tag_key, tag_payload in slice_payload.items():
                for instr_key, instr_payload in tag_payload.items():
                    for direction in ["wrong", "correct"]:
                        payload = instr_payload[direction]
                        page = payload["page_trend_test"]
                        rows.append(
                            {
                                "model": model,
                                "slice": slice_name,
                                "tag": tag_key,
                                "instruction_key": instr_key,
                                "direction": direction,
                                "n_items": payload["n_items"],
                                "fraction_monotonic": payload["fraction_monotonic_non_decreasing"],
                                "kendall_tau_mean": payload["kendall_tau_mean"],
                                "kendall_tau_ci95_low": payload["kendall_tau_ci95"][0],
                                "kendall_tau_ci95_high": payload["kendall_tau_ci95"][1],
                                "page_statistic": page["statistic"],
                                "page_pvalue": page["pvalue"],
                                "page_method": page["method"],
                            }
                        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "slice",
                "tag",
                "instruction_key",
                "direction",
                "n_items",
                "fraction_monotonic",
                "kendall_tau_mean",
                "kendall_tau_ci95_low",
                "kendall_tau_ci95_high",
                "page_statistic",
                "page_pvalue",
                "page_method",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_report(analyses: Sequence[Dict[str, Any]], path: Path) -> None:
    lines: List[str] = []
    lines.append("# Exp14 Evidence-Quality Analysis")
    lines.append("")
    lines.append(
        "This report summarizes monotonic evidence scaling and asymmetry diagnostics "
        "across prior-stratified slices."
    )
    lines.append("")

    for analysis in analyses:
        lines.append(f"## {analysis['model']}")
        lines.append(f"- n_records: {analysis['n_records']}")
        lines.append(f"- tags: {', '.join(analysis['tags'])}")
        lines.append(f"- evidence_levels: {', '.join(analysis['evidence_levels'])}")
        lines.append(f"- prior code: `{analysis['prior_definition']['prior_code']}`")
        lines.append("")

        sc = analysis["slice_counts"]
        lines.append(
            "- slice counts: "
            f"all={sc.get('all', 0)}, prior_correct={sc.get('prior_correct', 0)}, "
            f"prior_wrong={sc.get('prior_wrong', 0)}, "
            f"high_conf_top25={sc.get('prior_wrong_high_conf_top25', 0)}, "
            f"top10={sc.get('prior_wrong_top10', 0)}"
        )
        lines.append("")

        for tag in analysis["tags"]:
            lines.append(f"### Tag: {tag}")
            for instr_key in analysis["instruction_keys"]:
                wrong_payload = analysis["monotonicity"]["prior_wrong"][tag][instr_key]["wrong"]
                correct_payload = analysis["monotonicity"]["prior_wrong"][tag][instr_key]["correct"]
                diag = analysis["key_diagnostic_prior_wrong_e3_minus_e0_correct_shift"][tag][instr_key]

                lines.append(f"- Instruction `{instr_key}`")
                lines.append(
                    "  prior-wrong Kendall tau: "
                    f"wrong={wrong_payload['kendall_tau_mean']:.3f}, "
                    f"correct={correct_payload['kendall_tau_mean']:.3f}"
                )
                lines.append(
                    "  prior-wrong Page p-value: "
                    f"wrong={wrong_payload['page_trend_test']['pvalue']:.4g}, "
                    f"correct={correct_payload['page_trend_test']['pvalue']:.4g}"
                )
                lines.append(
                    "  key diagnostic (E_last - E_first correct shift, prior-wrong): "
                    f"mean={diag['mean_delta']:.3f}, CI95=[{diag['ci95'][0]:.3f}, {diag['ci95'][1]:.3f}], "
                    f"positive_frac={diag['positive_frac']:.3f}"
                )
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    requested_tags = [normalize_tag(token) for token in _parse_csv(args.tags)]
    evidence_order = _parse_csv(args.evidence_order)

    result_files = sorted(args.results_dir.glob("*_results.jsonl"))
    if not result_files:
        raise FileNotFoundError(f"No *_results.jsonl files found in {args.results_dir}")

    analyses: List[Dict[str, Any]] = []
    for path in result_files:
        print(f"Analyzing {path} ...")
        analysis = _analyze_model(
            path,
            n_boot=args.n_boot,
            seed=args.seed,
            prior_tag=args.prior_tag,
            prior_instruction_key=args.prior_instruction_key,
            prior_evidence_level=args.prior_evidence_level,
            requested_tags=requested_tags,
            evidence_order=evidence_order,
        )
        analyses.append(analysis)
        out_path = args.output_dir / f"{path.stem.replace('_results', '')}_analysis.json"
        out_path.write_text(json.dumps(_to_builtin(analysis), indent=2))

    combined = {"models": analyses}
    (args.output_dir / "combined_analysis.json").write_text(json.dumps(_to_builtin(combined), indent=2))
    _write_monotonicity_table(analyses, args.output_dir / "monotonicity_table.csv")
    _write_report(analyses, args.output_dir / "report.md")

    print(f"Wrote analysis outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
