"""
Experiment 1: Contrastive Activation Addition (CAA) vector extraction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from tqdm import tqdm

from src.data.syc_dataset import (
    ensure_splits_exist,
    load_user_wrong_split,
)
from src.metrics.syc_metric import compute_D_syc
from src.models.llama_loader import load_model_and_tokenizer
from src.hooks.steering_hooks import CaptureRecord, ResidualHookManager, inject_steering_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp1 CAA vector runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/exp1_caa.yaml"),
        help="Config file",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def tokenize_prompt(tokenizer, prompt: str, device: torch.device):
    enc = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def determine_margin(
    values: List[float],
    min_per_class: int,
    start_percentile: float = 0.8,
    step: float = 0.05,
) -> float:
    """
    Choose a symmetric |D_syc| margin that yields at least `min_per_class`
    examples on each side. Start from a high percentile of |D_syc| and relax
    until both classes have enough samples.
    """
    assert 0 < start_percentile <= 1.0
    assert 0 < step < 1.0
    abs_values = sorted(abs(v) for v in values)
    if not abs_values:
        raise RuntimeError("No D_syc values provided for margin selection.")

    def percentile(p: float) -> float:
        idx = max(0, min(len(abs_values) - 1, int(p * (len(abs_values) - 1))))
        return abs_values[idx]

    p = start_percentile
    while p >= 0:
        m = percentile(p)
        pos = sum(1 for v in values if v >= m)
        neg = sum(1 for v in values if v <= -m)
        if pos >= min_per_class and neg >= min_per_class:
            return m
        p -= step
    raise RuntimeError("Unable to find margin that satisfies min_per_class for both labels.")


def collect_activations(
    model,
    tokenizer,
    records: List[Dict],
    layers: List[int],
    device: torch.device,
) -> Dict[int, List[torch.Tensor]]:
    activations = {layer: [] for layer in layers}
    for rec in tqdm(records, desc="Collecting activations"):
        prompt = rec["example"].user_wrong_prompt
        inputs = tokenize_prompt(tokenizer, prompt, device)
        pos_idx = inputs["input_ids"].shape[1] - 1
        storage: Dict[int, CaptureRecord] = {}
        with ResidualHookManager(model) as manager:
            for layer in layers:
                manager.register_capture(layer, pos_idx, storage)
            with torch.no_grad():
                model(**inputs)
        for layer in layers:
            activations[layer].append(storage[layer].tensor.squeeze(0))
    return activations


def compute_layer_vectors(
    syc_acts: Dict[int, List[torch.Tensor]],
    truth_acts: Dict[int, List[torch.Tensor]],
) -> Dict[int, torch.Tensor]:
    vectors = {}
    for layer in syc_acts:
        mu_syc = torch.stack(syc_acts[layer]).mean(dim=0)
        mu_truth = torch.stack(truth_acts[layer]).mean(dim=0)
        vectors[layer] = (mu_truth - mu_syc)
    return vectors


def select_best_vector(
    model,
    tokenizer,
    records: List[Dict],
    vectors: Dict[int, torch.Tensor],
    device: torch.device,
    alpha_grid: List[float],
    dev_limit: int,
) -> Tuple[int, float, List[Dict]]:
    dev_records = records[: min(dev_limit, len(records))]
    grid_results: List[Dict] = []
    for layer, vector in vectors.items():
        vector = vector.to(device)
        for alpha in alpha_grid:
            effects = []
            for rec in dev_records:
                prompt = rec["example"].user_wrong_prompt
                base = rec["metric"]["D_syc"]
                pos_idx = rec["pos_idx"]
                with inject_steering_vector(model, layer, pos_idx, vector, alpha):
                    steered = compute_D_syc(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=prompt,
                        wrong_answer=rec["example"].wrong_answer,
                        right_answer=rec["example"].correct_answer,
                        device=device,
                    )
                effects.append(base - steered["D_syc"])
            if effects:
                effects_tensor = torch.tensor(effects, dtype=torch.float32, device=device)
                mean_effect = float(effects_tensor.mean().item())
                std_effect = float(effects_tensor.std(unbiased=False).item())
                n_effect = len(effects)
            else:
                mean_effect, std_effect, n_effect = 0.0, 0.0, 0
            grid_results.append(
                {
                    "layer": layer,
                    "alpha": alpha,
                    "mean_effect": mean_effect,
                    "std_effect": std_effect,
                    "n": n_effect,
                }
            )
    best = max(grid_results, key=lambda x: x["mean_effect"])
    return best["layer"], best["alpha"], grid_results


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    ensure_splits_exist(
        raw_path=Path(dataset_cfg["raw_path"]),
        split_a_path=Path(dataset_cfg["split_a_path"]),
        split_b_path=Path(dataset_cfg["split_b_path"]),
        split_c_path=Path(dataset_cfg["split_c_path"]),
        seed=seed,
    )
    examples = load_user_wrong_split(Path(dataset_cfg["split_a_path"]))

    model_cfg = cfg["model"]
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_cfg["name"],
        device=model_cfg.get("device", "auto"),
        dtype=model_cfg.get("dtype", "auto"),
    )
    device = next(model.parameters()).device

    records = []
    d_values = []
    for ex in tqdm(examples, desc="Labeling Split A"):
        prompt = ex.user_wrong_prompt
        inputs = tokenize_prompt(tokenizer, prompt, device)
        pos_idx = inputs["input_ids"].shape[1] - 1
        metric = compute_D_syc(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            wrong_answer=ex.wrong_answer,
            right_answer=ex.correct_answer,
            device=device,
        )
        records.append({"example": ex, "metric": metric, "pos_idx": pos_idx})
        d_values.append(metric["D_syc"])

    margin = determine_margin(
        d_values,
        min_per_class=cfg["exp1"]["min_per_class"],
        start_percentile=cfg["exp1"].get("margin_start_percentile", 0.8),
        step=cfg["exp1"].get("margin_step", 0.05),
    )
    syc_records = [rec for rec in records if rec["metric"]["D_syc"] >= margin]
    truth_records = [rec for rec in records if rec["metric"]["D_syc"] <= -margin]
    if not syc_records or not truth_records:
        raise RuntimeError("Failed to obtain labeled examples for both classes.")

    # Optional subsampling to rebalance classes and stabilize means.
    max_per_class = cfg["exp1"].get("max_per_class")
    rebalance = cfg["exp1"].get("subsample_majority", True)
    if rebalance or max_per_class:
        rng = torch.Generator().manual_seed(seed)
        syc_n = len(syc_records)
        truth_n = len(truth_records)
        target = min(syc_n, truth_n)
        if max_per_class:
            target = min(target, max_per_class)

        def downsample(records: List[Dict], target_size: int) -> List[Dict]:
            if len(records) <= target_size:
                return records
            idx = torch.randperm(len(records), generator=rng)[:target_size].tolist()
            return [records[i] for i in idx]

        if rebalance:
            syc_records = downsample(syc_records, target)
            truth_records = downsample(truth_records, target)
        else:
            if max_per_class:
                syc_records = downsample(syc_records, max_per_class)
                truth_records = downsample(truth_records, max_per_class)

    head_summary_path = Path(cfg["exp1"]["head_summary_path"])
    layer_limit = cfg["exp1"].get("layer_limit_override")
    if head_summary_path.exists():
        with head_summary_path.open("r") as f:
            summary = json.load(f)
        layer_limit = summary.get("L_min")
    total_layers = model.config.num_hidden_layers
    if layer_limit is None:
        layer_limit = total_layers
    candidate_layers = [l for l in range(min(layer_limit, total_layers))] or list(range(total_layers))

    syc_acts = collect_activations(model, tokenizer, syc_records, candidate_layers, device)
    truth_acts = collect_activations(model, tokenizer, truth_records, candidate_layers, device)
    layer_vectors = compute_layer_vectors(syc_acts, truth_acts)

    L_star, alpha_star, grid = select_best_vector(
        model=model,
        tokenizer=tokenizer,
        records=syc_records + truth_records,
        vectors=layer_vectors,
        device=device,
        alpha_grid=cfg["exp1"]["alpha_grid"],
        dev_limit=cfg["exp1"]["dev_subset_size"],
    )

    output_dir = Path(cfg["exp1"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {layer: vec.cpu() for layer, vec in layer_vectors.items()},
        output_dir / "layer_vectors.pt",
    )
    selection = {
        "margin": margin,
        "num_syc_examples": len(syc_records),
        "num_truth_examples": len(truth_records),
        "candidate_layers": candidate_layers,
        "L_star": L_star,
        "alpha_star": alpha_star,
        "grid": grid,
    }
    with (output_dir / "vector_selection.json").open("w") as f:
        json.dump(selection, f, indent=2)


if __name__ == "__main__":
    main()
