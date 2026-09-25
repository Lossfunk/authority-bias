"""Analyse persona sweep results and produce a summary table.

Reads per-persona JSONL results written by
:mod:`src.endorsement.run_persona_sweep`, computes correction / resistance /
selectivity metrics, and prints a summary table.  Also writes a JSON
summary file.

Usage:
    python -m src.endorsement.analyze_persona_sweep \
        --results-dir results/exp10_persona_sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

from src.endorsement.conditions import normalize_tag


# ---------------------------------------------------------------------------
# Item-level helpers
# ---------------------------------------------------------------------------

def signed_margin(row: Dict, cond: str) -> float:
    """Return logit(correct) u2212 logit(wrong) for *cond*."""
    cr = row["condition_results"][cond]
    if row["correct_label"] == "A":
        return cr["logit_a"] - cr["logit_b"]
    return cr["logit_b"] - cr["logit_a"]


def fc_correct(row: Dict, cond: str) -> float:
    """Return P(correct | {A,B}) for *cond*."""
    cr = row["condition_results"][cond]
    if row["correct_label"] == "A":
        return cr["fc_a"]
    return cr["fc_b"]


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------

def load_results(path: Path) -> List[Dict]:
    items: List[Dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _detect_tags(items: List[Dict]) -> List[str]:
    """Detect tags present in the results from condition-code keys."""
    if not items:
        return []
    codes = list(items[0]["condition_results"].keys())
    tags = set()
    for code in codes:
        # Codes look like N0_expert, W1_note, etc.
        parts = code.split("_", 1)
        if len(parts) == 2:
            tags.add(parts[1])
    return sorted(tags)


def analyze_results_file(
    results_path: Path,
) -> Dict:
    """Compute per-file correction / resistance metrics."""
    items = load_results(results_path)
    if not items:
        return {}

    tag_keys = _detect_tags(items)

    # Grab metadata from the first row.
    persona = items[0].get("persona", "unknown")
    variant = items[0].get("instruction_variant", "unknown")
    model = items[0].get("model", "unknown")
    instruction_text = items[0].get("instruction_text", "")

    per_tag: Dict[str, Dict] = {}

    for tag_key in tag_keys:
        n0_key = f"N0_{tag_key}"
        w0_key = f"W0_{tag_key}"
        w1_key = f"W1_{tag_key}"
        c0_key = f"C0_{tag_key}"
        c1_key = f"C1_{tag_key}"
        n1_key = f"N1_{tag_key}"

        # Verify keys exist.
        if n0_key not in items[0]["condition_results"]:
            continue

        # Split by prior belief (baseline N0 margin).
        prior_wrong: List[Dict] = []
        prior_correct: List[Dict] = []
        for item in items:
            m = signed_margin(item, n0_key)
            if m < 0:
                prior_wrong.append(item)
            elif m > 0:
                prior_correct.append(item)
            # m == 0 is ambiguous; skip.

        # -- Prior-wrong items + correct endorsement (correction opportunity) --
        n_pw = len(prior_wrong)
        correcting_i0: List[float] = []
        correcting_i1: List[float] = []
        resisting_i0: List[float] = []
        resisting_i1: List[float] = []

        for it in prior_wrong:
            # Without instruction: does C0 flip to correct?
            if signed_margin(it, c0_key) > 0:
                correcting_i0.append(1.0)
            else:
                resisting_i0.append(1.0)
            # With instruction: does C1 flip to correct?
            if signed_margin(it, c1_key) > 0:
                correcting_i1.append(1.0)
            else:
                resisting_i1.append(1.0)

        n_correcting_i0 = len(correcting_i0)
        n_resisting_i0 = len(resisting_i0)
        n_correcting_i1 = len(correcting_i1)
        n_resisting_i1 = len(resisting_i1)

        correction_rate_i0 = n_correcting_i0 / n_pw if n_pw else 0.0
        correction_rate_i1 = n_correcting_i1 / n_pw if n_pw else 0.0
        resistance_rate_i0 = n_resisting_i0 / n_pw if n_pw else 0.0
        resistance_rate_i1 = n_resisting_i1 / n_pw if n_pw else 0.0

        # -- Prior-correct items + wrong endorsement (adversarial) --
        n_pc = len(prior_correct)
        flipped_i0 = sum(
            1 for it in prior_correct if signed_margin(it, w0_key) < 0
        )
        flipped_i1 = sum(
            1 for it in prior_correct if signed_margin(it, w1_key) < 0
        )
        flip_rate_i0 = flipped_i0 / n_pc if n_pc else 0.0
        flip_rate_i1 = flipped_i1 / n_pc if n_pc else 0.0

        # Selectivity: delta in correction rate between prior-wrong (desired)
        # and flip rate on prior-correct (undesired).  A credulity dial
        # would change both equally; truth-tracking shows a gap.
        selectivity_i1 = correction_rate_i1 - flip_rate_i1

        # Collect fc_correct aggregates for overall selectivity.
        sel_metrics_means: Dict[str, float] = {}
        sel_keys = [
            f"efficacy_wrong_{tag_key}",
            f"efficacy_correct_{tag_key}",
            f"selectivity_{tag_key}",
        ]
        for sk in sel_keys:
            vals = [
                it["selectivity_metrics"][sk]
                for it in items
                if sk in it.get("selectivity_metrics", {})
            ]
            if vals:
                sel_metrics_means[sk] = float(mean(vals))

        per_tag[tag_key] = {
            "n_total": len(items),
            "n_prior_wrong": n_pw,
            "n_prior_correct": n_pc,
            # Correction metrics (C endorsement on prior-wrong items).
            "n_correcting_i0": n_correcting_i0,
            "n_correcting_i1": n_correcting_i1,
            "correction_rate_i0": correction_rate_i0,
            "correction_rate_i1": correction_rate_i1,
            # Resistance metrics.
            "n_resisting_i0": n_resisting_i0,
            "n_resisting_i1": n_resisting_i1,
            "resistance_rate_i0": resistance_rate_i0,
            "resistance_rate_i1": resistance_rate_i1,
            # Adversarial flip metrics (W endorsement on prior-correct items).
            "n_flipped_i0": flipped_i0,
            "n_flipped_i1": flipped_i1,
            "flip_rate_i0": flip_rate_i0,
            "flip_rate_i1": flip_rate_i1,
            # Selectivity headline.
            "selectivity_flip_delta_i1": selectivity_i1,
            # Continuous selectivity from compute_selectivity_metrics.
            "selectivity_metrics_means": sel_metrics_means,
        }

    return {
        "model": model,
        "persona": persona,
        "instruction_variant": variant,
        "instruction_text": instruction_text,
        "results_path": str(results_path),
        "n_items": len(items),
        "per_tag": per_tag,
    }


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------

def discover_results(results_dir: Path) -> List[Path]:
    """Find all ``*_results.jsonl`` files in persona subdirectories."""
    paths: List[Path] = []
    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.glob("*_results.jsonl")):
            paths.append(f)
    return paths


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_table(analyses: List[Dict]) -> None:
    """Pretty-print a comparison table across personas."""
    if not analyses:
        print("No results to display.")
        return

    # Collect all tag keys.
    all_tags: List[str] = []
    for a in analyses:
        for tk in a.get("per_tag", {}):
            if tk not in all_tags:
                all_tags.append(tk)

    for tag_key in all_tags:
        print(f"\n{'='*90}")
        print(f"  Tag: {tag_key}")
        print(f"{'='*90}")

        header = (
            f"{'persona':16s} {'variant':6s}  "
            f"{'N_pw':>5s} {'corr_i0':>7s} {'corr_i1':>7s}  "
            f"{'N_pc':>5s} {'flip_i0':>7s} {'flip_i1':>7s}  "
            f"{'sel_u0394':>7s}  "
            f"{'sel_cont':>9s}"
        )
        print(header)
        print("-" * len(header))

        for a in analyses:
            pt = a.get("per_tag", {}).get(tag_key)
            if pt is None:
                continue
            sel_cont_key = f"selectivity_{tag_key}"
            sel_cont = pt["selectivity_metrics_means"].get(sel_cont_key, float("nan"))
            row = (
                f"{a['persona']:16s} {a['instruction_variant']:6s}  "
                f"{pt['n_prior_wrong']:5d} "
                f"{pt['correction_rate_i0']:7.3f} {pt['correction_rate_i1']:7.3f}  "
                f"{pt['n_prior_correct']:5d} "
                f"{pt['flip_rate_i0']:7.3f} {pt['flip_rate_i1']:7.3f}  "
                f"{pt['selectivity_flip_delta_i1']:+7.3f}  "
                f"{sel_cont:+9.4f}"
            )
            print(row)

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze persona sweep results from exp10",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/exp10_persona_sweep"),
        help="Root results directory (contains persona sub-directories).",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Path for the JSON summary.  Defaults to "
            "<results-dir>/persona_sweep_analysis.json."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    paths = discover_results(results_dir)
    if not paths:
        print(f"No *_results.jsonl files found under {results_dir}")
        return

    print(f"Found {len(paths)} result file(s) under {results_dir}")

    analyses: List[Dict] = []
    for p in paths:
        a = analyze_results_file(p)
        if a:
            analyses.append(a)

    print_table(analyses)

    # Write JSON summary.
    out_json = args.output_json or (results_dir / "persona_sweep_analysis.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(
            {
                "experiment": "exp10_persona_sweep_analysis",
                "n_result_files": len(analyses),
                "analyses": analyses,
            },
            f,
            indent=2,
        )
    print(f"Summary JSON written to {out_json}")


if __name__ == "__main__":
    main()
