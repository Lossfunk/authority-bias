"""
Aggregate per-head deltas and select sycophancy / random head sets.
"""

from __future__ import annotations

import argparse
import json
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


def load_raw_rows(raw_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with raw_path.open("r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def aggregate_head_scores(rows: List[Dict]) -> List[Dict]:
    agg = defaultdict(lambda: {"delta_sum": 0.0, "count": 0})
    for row in rows:
        key = (row["layer"], row["head"])
        agg[key]["delta_sum"] += row["delta"]
        agg[key]["count"] += 1
    head_scores = []
    for (layer, head), info in agg.items():
        head_scores.append(
            {
                "layer": layer,
                "head": head,
                "delta_sum": info["delta_sum"],
                "delta_mean": info["delta_sum"] / max(info["count"], 1),
                "count": info["count"],
            }
        )
    return head_scores


def select_heads(
    head_scores: List[Dict],
    coverage_fraction: float,
    rng: random.Random,
    num_heads: int,
) -> Tuple[List[Dict], List[Dict], Dict]:
    positive_heads = [hs for hs in head_scores if hs["delta_sum"] > 0]
    positive_heads.sort(key=lambda x: x["delta_sum"], reverse=True)
    total = sum(hs["delta_sum"] for hs in positive_heads)
    threshold = coverage_fraction * total
    selected: List[Dict] = []
    running = 0.0
    for hs in positive_heads:
        selected.append({"layer": hs["layer"], "head": hs["head"], "delta_sum": hs["delta_sum"]})
        running += hs["delta_sum"]
        if running >= threshold:
            break
    if not selected:
        return [], [], {}

    random_heads: List[Dict] = []
    for hs in selected:
        layer = hs["layer"]
        excluded = {h["head"] for h in selected if h["layer"] == layer}
        candidates = [h for h in range(num_heads) if h not in excluded]
        if not candidates:
            continue
        random_heads.append({"layer": layer, "head": rng.choice(candidates)})

    layer_bounds = {
        "L_min": min(h["layer"] for h in selected),
        "L_max": max(h["layer"] for h in selected),
    }
    return selected, random_heads, layer_bounds


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_cfg = cfg["output"]
    rows = load_raw_rows(Path(output_cfg["raw_results"]))
    if not rows:
        raise RuntimeError("No rows found in raw results")
    head_scores = aggregate_head_scores(rows)
    save_jsonl(Path(output_cfg["head_scores"]), head_scores)

    coverage = cfg["head_selection"]["coverage_fraction"]
    rng = random.Random(cfg.get("seed", 42))
    num_heads = rows[0]["num_heads"]
    syc_heads, rand_heads, bounds = select_heads(head_scores, coverage, rng, num_heads)

    (Path(output_cfg["syc_heads"]).parent).mkdir(parents=True, exist_ok=True)
    with Path(output_cfg["syc_heads"]).open("w") as f:
        json.dump(syc_heads, f, indent=2)
    with Path(output_cfg["rand_heads"]).open("w") as f:
        json.dump(rand_heads, f, indent=2)

    summary = {
        "num_rows": len(rows),
        "num_head_scores": len(head_scores),
        "coverage_fraction": coverage,
        "syc_head_count": len(syc_heads),
        "rand_head_count": len(rand_heads),
    }
    summary.update(bounds)
    with Path(output_cfg["summary"]).open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()


