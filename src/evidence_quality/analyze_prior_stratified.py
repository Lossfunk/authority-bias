"""Exp14B: Prior-stratified re-analysis over existing endorsement result files.

Supported input schemas:
- Exp10-style: N0/W0/C0 + N1/W1/C1 per tag
- Exp13-style with variant suffixes: W0_tag_variant, C0_tag_variant (choose via --variant)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.prior_analysis.logit_metrics import TAU_DEFAULT, ItemMetrics, bootstrap_all_metrics_coherent, compute_item_metrics


SLICE_ORDER = [
    "all",
    "prior_correct",
    "prior_wrong",
    "prior_wrong_high_conf_top25",
    "prior_wrong_top10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp14B prior-stratified re-analysis")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("new-phase-results"),
        help="Root directory to search for *_results.jsonl files",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="**/*_results.jsonl",
        help="Glob under --results-root for candidate result files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("new-phase-results"),
        help="Root for writing family-level exp14b artifacts",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default="expert,note",
        help="Comma-separated normalized tags to analyze",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="",
        help=(
            "Optional variant suffix for variantized files (e.g., think_plain). "
            "If empty, plain exp10-style codes are preferred; otherwise first detected variant is used."
        ),
    )
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=TAU_DEFAULT)
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
            records.append(payload)
    return records


def _is_supported_record(record: Dict[str, Any]) -> bool:
    conds = record.get("condition_results")
    if not isinstance(conds, dict) or not conds:
        return False
    has_n0 = any(str(key).startswith("N0_") for key in conds)
    has_n1 = any(str(key).startswith("N1_") for key in conds)
    has_w0 = any(str(key).startswith("W0_") for key in conds)
    has_c0 = any(str(key).startswith("C0_") for key in conds)
    return has_n0 and has_n1 and has_w0 and has_c0


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


def _mean_with_ci(values: Sequence[float], *, n_boot: int, seed: int) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {
            "mean": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }
    lo, hi = _bootstrap_mean_ci(arr.tolist(), n_boot=n_boot, seed=seed)
    return {
        "mean": float(np.mean(arr)),
        "ci95_low": lo,
        "ci95_high": hi,
    }


def _logit_from_prob(p: float, eps: float = 1e-10) -> float:
    p = max(eps, min(1 - eps, p))
    return float(np.log(p / (1 - p)))


def _margin_correct(record: Dict[str, Any], code: str) -> float:
    conds = record.get("condition_results", {})
    if isinstance(conds, dict) and code in conds and isinstance(conds[code], dict):
        cond = conds[code]
        if record.get("correct_label") == "A" and "logit_a" in cond:
            return float(cond["logit_a"])
        if record.get("correct_label") == "B" and "logit_b" in cond:
            return float(cond["logit_b"])

    fc_correct = record.get("fc_correct", {})
    if isinstance(fc_correct, dict) and code in fc_correct:
        return _logit_from_prob(float(fc_correct[code]))
    raise KeyError(f"Missing margin for condition code '{code}'")


def _fc_correct(record: Dict[str, Any], code: str) -> float:
    fc = record.get("fc_correct", {})
    if isinstance(fc, dict) and code in fc:
        return float(fc[code])

    conds = record.get("condition_results", {})
    if isinstance(conds, dict) and code in conds and isinstance(conds[code], dict):
        cond = conds[code]
        if record.get("correct_label") == "A" and "fc_a" in cond:
            return float(cond["fc_a"])
        if record.get("correct_label") == "B" and "fc_b" in cond:
            return float(cond["fc_b"])
    raise KeyError(f"Missing fc_correct for condition code '{code}'")


def _detect_codes(
    record: Dict[str, Any],
    *,
    tag_key: str,
    variant: str,
) -> Tuple[Dict[str, str], str]:
    conds = record.get("condition_results", {})
    if not isinstance(conds, dict):
        raise ValueError("Record missing condition_results")
    keys = set(str(k) for k in conds.keys())

    plain = {
        "N0": f"N0_{tag_key}",
        "N1": f"N1_{tag_key}",
        "W0": f"W0_{tag_key}",
        "W1": f"W1_{tag_key}",
        "C0": f"C0_{tag_key}",
        "C1": f"C1_{tag_key}",
    }
    if all(code in keys for code in plain.values()):
        return plain, ""

    chosen_variant = variant
    if not chosen_variant:
        candidates = sorted(
            key.split(f"W0_{tag_key}_", 1)[1]
            for key in keys
            if key.startswith(f"W0_{tag_key}_")
        )
        if not candidates:
            raise ValueError(f"No matching conditions found for tag '{tag_key}'")
        chosen_variant = candidates[0]

    variant_codes = {
        "N0": f"N0_{tag_key}",
        "N1": f"N1_{tag_key}",
        "W0": f"W0_{tag_key}_{chosen_variant}",
        "W1": f"W1_{tag_key}_{chosen_variant}",
        "C0": f"C0_{tag_key}_{chosen_variant}",
        "C1": f"C1_{tag_key}_{chosen_variant}",
    }
    if not all(code in keys for code in variant_codes.values()):
        missing = [code for code in variant_codes.values() if code not in keys]
        raise ValueError(
            f"Missing expected variantized conditions for tag '{tag_key}' and variant '{chosen_variant}': {missing}"
        )
    return variant_codes, chosen_variant


def _build_item_rows(
    records: Sequence[Dict[str, Any]],
    *,
    codes: Dict[str, str],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for record in records:
        try:
            n0 = _fc_correct(record, codes["N0"])
            n1 = _fc_correct(record, codes["N1"])
            w0 = _fc_correct(record, codes["W0"])
            w1 = _fc_correct(record, codes["W1"])
            c0 = _fc_correct(record, codes["C0"])
            c1 = _fc_correct(record, codes["C1"])

            m_n0 = _margin_correct(record, codes["N0"])
            m_n1 = _margin_correct(record, codes["N1"])
            m_w0 = _margin_correct(record, codes["W0"])
            m_w1 = _margin_correct(record, codes["W1"])
            m_c0 = _margin_correct(record, codes["C0"])
            m_c1 = _margin_correct(record, codes["C1"])
        except KeyError:
            continue

        rows.append(
            {
                "n0": n0,
                "n1": n1,
                "w0": w0,
                "w1": w1,
                "c0": c0,
                "c1": c1,
                "m_n0": m_n0,
                "m_n1": m_n1,
                "m_w0": m_w0,
                "m_w1": m_w1,
                "m_c0": m_c0,
                "m_c1": m_c1,
            }
        )
    return rows


def _create_slices(rows: Sequence[Dict[str, float]]) -> Dict[str, List[Dict[str, float]]]:
    out: Dict[str, List[Dict[str, float]]] = {name: [] for name in SLICE_ORDER}
    out["all"] = list(rows)

    prior_wrong_conf: List[Tuple[float, Dict[str, float]]] = []
    for row in rows:
        if row["n0"] > 0.5:
            out["prior_correct"].append(row)
        elif row["n0"] < 0.5:
            out["prior_wrong"].append(row)
            prior_wrong_conf.append((abs(row["m_n0"]), row))

    if prior_wrong_conf:
        prior_wrong_conf.sort(key=lambda x: x[0])
        n = len(prior_wrong_conf)
        top25_start = int(np.floor(0.75 * n))
        top10_start = int(np.floor(0.90 * n))
        out["prior_wrong_high_conf_top25"] = [row for _, row in prior_wrong_conf[top25_start:]]
        out["prior_wrong_top10"] = [row for _, row in prior_wrong_conf[top10_start:]]

    return out


def _slice_metrics(
    rows: Sequence[Dict[str, float]],
    *,
    n_boot: int,
    seed: int,
    tau: float,
) -> Dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "endorsement_effect": _mean_with_ci([], n_boot=n_boot, seed=seed),
            "instruction_efficacy_wrong": _mean_with_ci([], n_boot=n_boot, seed=seed + 1),
            "instruction_efficacy_correct": _mean_with_ci([], n_boot=n_boot, seed=seed + 2),
            "selectivity": _mean_with_ci([], n_boot=n_boot, seed=seed + 3),
            "dr_median": {
                "value": 0.0,
                "ci95_low": 0.0,
                "ci95_high": 0.0,
                "n_finite_both": 0,
                "n_mask_valid_both": 0,
            },
        }

    effect_wrong = [row["n0"] - row["w0"] for row in rows]
    effect_correct = [row["c0"] - row["n0"] for row in rows]
    efficacy_wrong = [(row["n0"] - row["w0"]) - (row["n1"] - row["w1"]) for row in rows]
    efficacy_correct = [(row["c0"] - row["n0"]) - (row["c1"] - row["n1"]) for row in rows]
    selectivity = [w - c for w, c in zip(efficacy_wrong, efficacy_correct)]

    items: List[ItemMetrics] = []
    for idx, row in enumerate(rows):
        item = compute_item_metrics(
            m_N0=row["m_n0"],
            m_N1=row["m_n1"],
            m_W0=row["m_w0"],
            m_W1=row["m_w1"],
            m_C0=row["m_c0"],
            m_C1=row["m_c1"],
            question_id=str(idx),
        )
        items.append(item)

    np.random.seed(seed + 100)
    ratio_metrics = bootstrap_all_metrics_coherent(
        items,
        n_boot=n_boot,
        use_sign_consistent_mask=True,
        tau=tau,
    )
    dr_payload = ratio_metrics.get("dr_median")
    if isinstance(dr_payload, tuple) and len(dr_payload) == 2:
        dr_value = float(dr_payload[0])
        ci_low = float(dr_payload[1][0])
        ci_high = float(dr_payload[1][1])
    else:
        dr_value, ci_low, ci_high = 0.0, 0.0, 0.0

    return {
        "n": len(rows),
        "endorsement_effect": _mean_with_ci(effect_wrong, n_boot=n_boot, seed=seed),
        "instruction_efficacy_wrong": _mean_with_ci(efficacy_wrong, n_boot=n_boot, seed=seed + 1),
        "instruction_efficacy_correct": _mean_with_ci(efficacy_correct, n_boot=n_boot, seed=seed + 2),
        "selectivity": _mean_with_ci(selectivity, n_boot=n_boot, seed=seed + 3),
        "dr_median": {
            "value": dr_value,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "n_finite_both": int(ratio_metrics.get("n_finite_both", 0)),
            "n_mask_valid_both": int(ratio_metrics.get("n_mask_valid_both", 0)),
        },
    }


def _analyze_result_file(
    path: Path,
    *,
    tags: Sequence[str],
    variant: str,
    n_boot: int,
    seed: int,
    tau: float,
) -> Dict[str, Any]:
    records = _load_jsonl(path)
    if not records:
        raise ValueError(f"No records in {path}")
    if not _is_supported_record(records[0]):
        raise ValueError(f"Unsupported result schema in {path}")

    model = str(records[0].get("model", "unknown"))
    tag_results: Dict[str, Any] = {}

    for tag in tags:
        codes, chosen_variant = _detect_codes(records[0], tag_key=tag, variant=variant)
        rows = _build_item_rows(records, codes=codes)
        slices = _create_slices(rows)

        slice_metrics = {}
        for slice_name in SLICE_ORDER:
            slice_rows = slices.get(slice_name, [])
            slice_seed = seed + int(abs(hash((model, tag, slice_name))) % 100000)
            slice_metrics[slice_name] = _slice_metrics(
                slice_rows,
                n_boot=n_boot,
                seed=slice_seed,
                tau=tau,
            )

        prior_correct_sel = slice_metrics["prior_correct"]["selectivity"]["mean"]
        prior_wrong_sel = slice_metrics["prior_wrong"]["selectivity"]["mean"]
        tag_results[tag] = {
            "variant_used": chosen_variant,
            "codes": codes,
            "slice_metrics": slice_metrics,
            "diagnostics": {
                "selectivity_sign_flip_prior_correct_vs_prior_wrong": (
                    (prior_correct_sel >= 0 > prior_wrong_sel) or (prior_correct_sel < 0 <= prior_wrong_sel)
                ),
                "selectivity_prior_correct_mean": prior_correct_sel,
                "selectivity_prior_wrong_mean": prior_wrong_sel,
            },
        }

    return {
        "model": model,
        "result_path": str(path),
        "n_records": len(records),
        "tags": tag_results,
    }


def _family_name(model: str) -> str:
    lower = model.lower()
    if "llama" in lower:
        return "llama"
    if "qwen" in lower:
        return "qwen"
    if "/" in model:
        return model.split("/", 1)[0].lower().replace("-", "_")
    return "misc"


def _write_family_markdown(analyses: Sequence[Dict[str, Any]], path: Path) -> None:
    lines: List[str] = []
    lines.append("# Exp14B Prior-Stratified Re-analysis")
    lines.append("")

    for analysis in analyses:
        lines.append(f"## {analysis['model']}")
        lines.append(f"- records: {analysis['n_records']}")
        lines.append(f"- source: `{analysis['result_path']}`")
        lines.append("")

        for tag, payload in analysis["tags"].items():
            lines.append(f"### Tag: {tag}")
            variant = payload.get("variant_used", "")
            lines.append(f"- variant: `{variant or 'plain'}`")
            lines.append("")
            lines.append(
                "| Slice | n | endorsement_effect | instruction_efficacy_wrong | "
                "instruction_efficacy_correct | selectivity | dr_median |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for slice_name in SLICE_ORDER:
                metric = payload["slice_metrics"][slice_name]
                lines.append(
                    f"| {slice_name} | {metric['n']} | "
                    f"{metric['endorsement_effect']['mean']:.4f} | "
                    f"{metric['instruction_efficacy_wrong']['mean']:.4f} | "
                    f"{metric['instruction_efficacy_correct']['mean']:.4f} | "
                    f"{metric['selectivity']['mean']:.4f} | "
                    f"{metric['dr_median']['value']:.4f} |"
                )
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _write_family_csv(analyses: Sequence[Dict[str, Any]], path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for analysis in analyses:
        model = analysis["model"]
        for tag, payload in analysis["tags"].items():
            for slice_name in SLICE_ORDER:
                metric = payload["slice_metrics"][slice_name]
                rows.append(
                    {
                        "model": model,
                        "tag": tag,
                        "variant": payload.get("variant_used", ""),
                        "slice": slice_name,
                        "n": metric["n"],
                        "endorsement_effect_mean": metric["endorsement_effect"]["mean"],
                        "endorsement_effect_ci95_low": metric["endorsement_effect"]["ci95_low"],
                        "endorsement_effect_ci95_high": metric["endorsement_effect"]["ci95_high"],
                        "instruction_efficacy_wrong_mean": metric["instruction_efficacy_wrong"]["mean"],
                        "instruction_efficacy_correct_mean": metric["instruction_efficacy_correct"]["mean"],
                        "selectivity_mean": metric["selectivity"]["mean"],
                        "dr_median": metric["dr_median"]["value"],
                    }
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "tag",
                "variant",
                "slice",
                "n",
                "endorsement_effect_mean",
                "endorsement_effect_ci95_low",
                "endorsement_effect_ci95_high",
                "instruction_efficacy_wrong_mean",
                "instruction_efficacy_correct_mean",
                "selectivity_mean",
                "dr_median",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    tags = _parse_csv(args.tags)

    files = sorted(args.results_root.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched {args.glob} under {args.results_root}")

    analyses: List[Dict[str, Any]] = []
    for path in files:
        try:
            head_records = _load_jsonl(path)
            if not head_records:
                continue
            if not _is_supported_record(head_records[0]):
                continue
            print(f"Analyzing {path} ...")
            analysis = _analyze_result_file(
                path,
                tags=tags,
                variant=args.variant,
                n_boot=args.n_boot,
                seed=args.seed,
                tau=args.tau,
            )
            analyses.append(analysis)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

    if not analyses:
        raise RuntimeError("No compatible result files were analyzed.")

    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for analysis in analyses:
        fam = _family_name(analysis["model"])
        by_family.setdefault(fam, []).append(analysis)

    for family, family_analyses in by_family.items():
        out_dir = args.output_root / family / "exp14b"
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "prior_stratified_analysis.json").write_text(
            json.dumps(_to_builtin({"models": family_analyses}), indent=2)
        )
        _write_family_markdown(family_analyses, out_dir / "prior_stratified_tables.md")
        _write_family_csv(family_analyses, out_dir / "prior_stratified_table.csv")
        print(f"Wrote family outputs: {out_dir}")


if __name__ == "__main__":
    main()
