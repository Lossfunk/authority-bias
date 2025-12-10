"""
Aggregate per-head deltas and select sycophancy / random head sets.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate head patching scores")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/exp2_path_patching.yaml"),
        help="Config file (used for result paths + coverage fraction)",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_raw_rows(raw_path: Path, pos_label: str | None = None) -> List[Dict]:
    rows: List[Dict] = []
    with raw_path.open("r") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if pos_label is None or row.get("pos_label") == pos_label:
                    rows.append(row)
    return rows


def aggregate_head_scores(rows: List[Dict]) -> List[Dict]:
    """Aggregate raw per-example deltas into per-head statistics."""
    agg = defaultdict(lambda: {"deltas": [], "pos_count": 0})
    for row in rows:
        key = (row["layer"], row["head"])
        agg[key]["deltas"].append(row["delta"])
        if row["delta"] > 0:
            agg[key]["pos_count"] += 1

    head_scores = []
    for (layer, head), info in agg.items():
        deltas = info["deltas"]
        n = len(deltas)
        delta_sum = sum(deltas)
        delta_mean = delta_sum / n if n > 0 else 0.0

        # Compute standard deviation
        if n > 1:
            variance = sum((d - delta_mean) ** 2 for d in deltas) / (n - 1)
            delta_std = math.sqrt(variance)
        else:
            delta_std = 0.0

        # Standard error and t-statistic
        delta_stderr = delta_std / math.sqrt(n) if n > 0 else 0.0
        t_stat = delta_mean / delta_stderr if delta_stderr > 0 else 0.0

        head_scores.append(
            {
                "layer": layer,
                "head": head,
                "delta_sum": delta_sum,
                "delta_mean": delta_mean,
                "delta_std": delta_std,
                "delta_stderr": delta_stderr,
                "t_stat": t_stat,
                "count": n,
                "pos_fraction": info["pos_count"] / n if n > 0 else 0.0,
            }
        )
    return head_scores


def get_rank_value(hs: Dict, rank_metric: str) -> float:
    """Get the value to rank by based on the metric name."""
    if rank_metric == "delta_sum":
        return hs["delta_sum"]
    elif rank_metric == "delta_mean":
        return hs["delta_mean"]
    elif rank_metric == "abs_delta_sum":
        return abs(hs["delta_sum"])
    elif rank_metric == "abs_delta_mean":
        return abs(hs["delta_mean"])
    elif rank_metric == "t_stat":
        return abs(hs["t_stat"])
    else:
        return hs["delta_sum"]


def select_positive_coverage(
    head_scores: List[Dict],
    coverage_fraction: float,
    bypass_positive_filter: bool = False,
) -> List[Dict]:
    """Select heads that cover coverage_fraction of positive delta mass."""
    if bypass_positive_filter:
        candidates = head_scores[:]
    else:
        candidates = [hs for hs in head_scores if hs["delta_sum"] > 0]

    candidates.sort(key=lambda x: x["delta_sum"], reverse=True)
    total = sum(hs["delta_sum"] for hs in candidates if hs["delta_sum"] > 0)
    if total <= 0:
        return []

    threshold = coverage_fraction * total
    selected: List[Dict] = []
    running = 0.0
    for hs in candidates:
        selected.append({
            "layer": hs["layer"],
            "head": hs["head"],
            "delta_sum": hs["delta_sum"],
            "delta_mean": hs["delta_mean"],
        })
        running += max(hs["delta_sum"], 0)
        if running >= threshold:
            break
    return selected


def select_top_k_per_layer(
    head_scores: List[Dict],
    top_k: int,
    rank_metric: str,
) -> List[Dict]:
    """Select top K heads per layer by the specified metric."""
    by_layer: Dict[int, List[Dict]] = defaultdict(list)
    for hs in head_scores:
        by_layer[hs["layer"]].append(hs)

    selected: List[Dict] = []
    for layer in sorted(by_layer.keys()):
        layer_heads = by_layer[layer]
        layer_heads.sort(key=lambda x: get_rank_value(x, rank_metric), reverse=True)
        for hs in layer_heads[:top_k]:
            selected.append({
                "layer": hs["layer"],
                "head": hs["head"],
                "delta_sum": hs["delta_sum"],
                "delta_mean": hs["delta_mean"],
                "rank_value": get_rank_value(hs, rank_metric),
            })
    return selected


def select_global_top_k_abs(
    head_scores: List[Dict],
    top_k: int,
    rank_metric: str,
) -> List[Dict]:
    """Select top K heads globally by absolute effect."""
    sorted_heads = sorted(
        head_scores,
        key=lambda x: get_rank_value(x, rank_metric),
        reverse=True,
    )
    selected: List[Dict] = []
    for hs in sorted_heads[:top_k]:
        selected.append({
            "layer": hs["layer"],
            "head": hs["head"],
            "delta_sum": hs["delta_sum"],
            "delta_mean": hs["delta_mean"],
            "rank_value": get_rank_value(hs, rank_metric),
        })
    return selected


def select_abs_top_k_per_layer(
    head_scores: List[Dict],
    top_k: int,
) -> List[Dict]:
    """Select top-K per layer by absolute delta_sum."""
    by_layer: Dict[int, List[Dict]] = defaultdict(list)
    for hs in head_scores:
        by_layer[hs["layer"]].append(hs)
    selected: List[Dict] = []
    for layer in sorted(by_layer.keys()):
        layer_heads = sorted(by_layer[layer], key=lambda x: abs(x["delta_sum"]), reverse=True)
        for hs in layer_heads[:top_k]:
            selected.append(
                {
                    "layer": hs["layer"],
                    "head": hs["head"],
                    "delta_sum": hs["delta_sum"],
                    "delta_mean": hs["delta_mean"],
                }
            )
    return selected


def generate_random_controls(
    selected: List[Dict],
    num_heads: int,
    rng: random.Random,
) -> List[Dict]:
    """Generate random control heads (one per selected head's layer)."""
    if not selected:
        return []

    random_heads: List[Dict] = []
    for hs in selected:
        layer = hs["layer"]
        excluded = {h["head"] for h in selected if h["layer"] == layer}
        candidates = [h for h in range(num_heads) if h not in excluded]
        if not candidates:
            continue
        random_heads.append({"layer": layer, "head": rng.choice(candidates)})
    return random_heads


def compute_layer_bounds(selected: List[Dict]) -> Dict:
    """Compute min/max layer from selected heads."""
    if not selected:
        return {}
    return {
        "L_min": min(h["layer"] for h in selected),
        "L_max": max(h["layer"] for h in selected),
    }


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_cfg = cfg["output"]
    sel_cfg = cfg["head_selection"]

    pos_label_filter = sel_cfg.get("pos_label")  # None means use all positions
    rows = load_raw_rows(Path(output_cfg["raw_results"]), pos_label=pos_label_filter)
    if not rows:
        raise RuntimeError("No rows found in raw results")

    head_scores = aggregate_head_scores(rows)
    save_jsonl(Path(output_cfg["head_scores"]), head_scores)

    rng = random.Random(cfg.get("seed", 42))
    num_heads = rows[0]["num_heads"]

    # Selection mode dispatch
    mode = sel_cfg.get("selection_mode", "positive_coverage")
    rank_metric = sel_cfg.get("rank_metric", "delta_sum")
    bypass_positive = sel_cfg.get("bypass_positive_filter", False)

    if mode == "positive_coverage":
        coverage = sel_cfg["coverage_fraction"]
        syc_heads = select_positive_coverage(head_scores, coverage, bypass_positive)
        # Fallback if empty
        if not syc_heads:
            fallback_k = sel_cfg.get("fallback_top_k", 20)
            syc_heads = select_global_top_k_abs(head_scores, fallback_k, "abs_delta_sum")
    elif mode == "top_k_per_layer":
        top_k = sel_cfg.get("top_k_per_layer", 3)
        syc_heads = select_top_k_per_layer(head_scores, top_k, rank_metric)
    elif mode == "abs_top_k_per_layer":
        top_k = sel_cfg.get("top_k_per_layer", 2)
        syc_heads = select_abs_top_k_per_layer(head_scores, top_k)
    elif mode == "global_top_k_abs":
        top_k = sel_cfg.get("top_k", 20)
        syc_heads = select_global_top_k_abs(head_scores, top_k, rank_metric)
    else:
        raise ValueError(f"Unknown selection_mode: {mode}")

    rand_heads = generate_random_controls(syc_heads, num_heads, rng)
    bounds = compute_layer_bounds(syc_heads)

    # Save main outputs
    out_dir = Path(output_cfg["syc_heads"]).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(output_cfg["syc_heads"]).open("w") as f:
        json.dump(syc_heads, f, indent=2)
    with Path(output_cfg["rand_heads"]).open("w") as f:
        json.dump(rand_heads, f, indent=2)

    # Generate exploratory set (cross-layer coverage)
    exploratory_heads = []
    if sel_cfg.get("emit_exploratory_set", False):
        exp_top_k = sel_cfg.get("exploratory_top_k_per_layer", 2)
        exploratory_heads = select_top_k_per_layer(head_scores, exp_top_k, "abs_delta_sum")
        exp_path = out_dir / "exploratory_heads.json"
        with exp_path.open("w") as f:
            json.dump(exploratory_heads, f, indent=2)

    # Summary
    summary = {
        "num_rows": len(rows),
        "num_head_scores": len(head_scores),
        "selection_mode": mode,
        "rank_metric": rank_metric,
        "syc_head_count": len(syc_heads),
        "rand_head_count": len(rand_heads),
        "exploratory_head_count": len(exploratory_heads),
        "unique_layers_in_syc": len(set(h["layer"] for h in syc_heads)),
        "unique_layers_in_exploratory": len(set(h["layer"] for h in exploratory_heads)),
    }
    summary.update(bounds)
    with Path(output_cfg["summary"]).open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {len(syc_heads)} syc heads ({summary['unique_layers_in_syc']} layers)")
    print(f"Saved {len(exploratory_heads)} exploratory heads ({summary['unique_layers_in_exploratory']} layers)")


if __name__ == "__main__":
    main()
def select_abs_top_k_per_layer(
    head_scores: List[Dict],
    top_k: int,
) -> List[Dict]:
    """Select top-K per layer by absolute delta_sum."""
    by_layer: Dict[int, List[Dict]] = defaultdict(list)
    for hs in head_scores:
        by_layer[hs["layer"]].append(hs)
    selected: List[Dict] = []
    for layer in sorted(by_layer.keys()):
        layer_heads = sorted(by_layer[layer], key=lambda x: abs(x["delta_sum"]), reverse=True)
        for hs in layer_heads[:top_k]:
            selected.append(
                {
                    "layer": hs["layer"],
                    "head": hs["head"],
                    "delta_sum": hs["delta_sum"],
                    "delta_mean": hs["delta_mean"],
                }
            )
    return selected
