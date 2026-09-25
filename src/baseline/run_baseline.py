"""
Baseline statistics for the sycophancy dataset (Exp 0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import torch
import yaml
from tqdm import tqdm

from src.data.syc_dataset import (
    ensure_splits_exist,
    load_user_wrong_split,
)
from src.metrics.syc_metric import compute_D_syc
from src.models.llama_loader import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp0 baseline runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiment_config.yaml"),
        help="Path to YAML config.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def compute_histogram(values: List[float], bins: int = 40) -> Dict[str, List[float]]:
    if not values:
        return {"bins": [], "counts": []}
    v_min, v_max = min(values), max(values)
    if v_min == v_max:
        return {"bins": [v_min, v_max], "counts": [len(values)]}
    bin_width = (v_max - v_min) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        idx = min(int((value - v_min) / bin_width), bins - 1)
        counts[idx] += 1
    bin_edges = [v_min + i * bin_width for i in range(bins + 1)]
    return {"bins": bin_edges, "counts": counts}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    split_paths = ensure_splits_exist(
        raw_path=Path(dataset_cfg["raw_path"]),
        split_a_path=Path(dataset_cfg["split_a_path"]),
        split_b_path=Path(dataset_cfg["split_b_path"]),
        split_c_path=Path(dataset_cfg["split_c_path"]),
        seed=cfg.get("seed", 42),
    )

    model_cfg = cfg["model"]
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_cfg["name"],
        device=model_cfg.get("device", "auto"),
        dtype=model_cfg.get("dtype", "auto"),
    )
    device = next(model.parameters()).device

    split_c_examples = load_user_wrong_split(split_paths["C"])
    max_examples = cfg.get("exp0", {}).get("max_examples")
    if max_examples:
        split_c_examples = split_c_examples[:max_examples]
    d_syc_values: List[float] = []
    records: List[Dict] = []
    for ex in tqdm(split_c_examples, desc="Baseline D_syc"):
        metric = compute_D_syc(
            model=model,
            tokenizer=tokenizer,
            prompt_text=ex.user_wrong_prompt,
            wrong_answer=ex.wrong_answer,
            right_answer=ex.correct_answer,
            device=device,
        )
        metric["example_id"] = ex.uid
        d_syc_values.append(metric["D_syc"])
        records.append(metric)

    syc_rate = sum(1 for v in d_syc_values if v > 0) / len(d_syc_values)
    summary = {
        "model_name": model_cfg["name"],
        "num_examples": len(d_syc_values),
        "syc_rate": syc_rate,
        "D_syc_mean": mean(d_syc_values),
        "D_syc_std": pstdev(d_syc_values) if len(d_syc_values) > 1 else 0.0,
        "histogram": compute_histogram(d_syc_values, bins=cfg.get("exp0", {}).get("histogram_bins", 40)),
    }
    output_dir = Path(cfg.get("exp0", {}).get("output_dir", "results/baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "baseline_metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "baseline_records.jsonl").open("w") as f:
        for record in records:
            json.dump(record, f)
            f.write("\n")


if __name__ == "__main__":
    main()


