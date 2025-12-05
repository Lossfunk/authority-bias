"""
Experiment 3: Mediation grid evaluating steering vs head ablations.
"""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List, Tuple

import torch
import yaml
from scipy import stats
from tqdm import tqdm

from src.data.syc_dataset import (
    ensure_splits_exist,
    load_user_wrong_split,
    make_neutral_prompt,
)
from src.metrics.syc_metric import compute_D_syc
from src.models.llama_loader import load_model_and_tokenizer
from src.hooks.head_patch_hooks import (
    HeadKey,
    cache_mode,
    ensure_head_patching,
    mean_ablation_mode,
)
from src.hooks.steering_hooks import inject_steering_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp3 mediation grid runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/exp3_mediation.yaml"),
        help="Config file",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def tokenize_prompt(tokenizer, text: str, device: torch.device):
    enc = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def load_head_list(path: Path) -> List[Tuple[int, int]]:
    with path.open("r") as f:
        data = json.load(f)
    return [(item["layer"], item["head"]) for item in data]


def compute_head_means(
    model,
    tokenizer,
    registry,
    examples,
    head_tuples: List[Tuple[int, int]],
    device: torch.device,
    max_examples: int,
) -> Dict[Tuple[int, int], torch.Tensor]:
    if not head_tuples:
        return {}
    values: Dict[Tuple[int, int], List[torch.Tensor]] = {key: [] for key in head_tuples}
    subset = examples[:max_examples]
    for idx, ex in enumerate(tqdm(subset, desc="Head mean reference")):
        prompt = make_neutral_prompt(ex)
        inputs = tokenize_prompt(tokenizer, prompt, device)
        example_id = f"mean::{idx}"
        pos_idx = inputs["input_ids"].shape[1] - 1
        with cache_mode(registry, example_id, pos_idx):
            with torch.no_grad():
                model(**inputs)
        for head in head_tuples:
            vec = registry.cache.get(HeadKey(example_id, head[0], head[1]))
            if vec is not None:
                values[head].append(vec)
        registry.cache.clear_example(example_id)
    means = {}
    for head, vecs in values.items():
        if vecs:
            stacked = torch.stack(vecs, dim=0)
            means[head] = stacked.mean(dim=0)
    return means


def evaluate_condition(
    model,
    tokenizer,
    registry,
    example,
    device: torch.device,
    pos_idx: int,
    steering_layer: int,
    steering_vector: torch.Tensor,
    alpha: float,
    ablation_vectors: Dict[Tuple[int, int], torch.Tensor] | None,
    use_steering: bool,
) -> Dict[str, float]:
    prompt = example.user_wrong_prompt
    wrong = example.wrong_answer
    right = example.correct_answer
    with ExitStack() as stack:
        if ablation_vectors:
            stack.enter_context(mean_ablation_mode(registry, ablation_vectors, pos_idx))
        if use_steering:
            stack.enter_context(
                inject_steering_vector(model, steering_layer, pos_idx, steering_vector, alpha)
            )
        metric = compute_D_syc(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            wrong_answer=wrong,
            right_answer=right,
            device=device,
        )
    return metric


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    ensure_splits_exist(
        raw_path=Path(dataset_cfg["raw_path"]),
        split_a_path=Path(dataset_cfg["split_a_path"]),
        split_b_path=Path(dataset_cfg["split_b_path"]),
        split_c_path=Path(dataset_cfg["split_c_path"]),
        seed=cfg.get("seed", 42),
    )
    split_c_examples = load_user_wrong_split(Path(dataset_cfg["split_c_path"]))
    split_b_examples = load_user_wrong_split(Path(cfg["exp3"]["head_means_reference_split"]))

    model_cfg = cfg["model"]
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_cfg["name"],
        device=model_cfg.get("device", "auto"),
        dtype=model_cfg.get("dtype", "auto"),
    )
    device = next(model.parameters()).device
    registry = ensure_head_patching(model)

    # Load steering vector
    layer_vectors = torch.load(cfg["exp3"]["layer_vectors_path"], map_location=device)
    with Path(cfg["exp3"]["vector_selection_path"]).open("r") as f:
        selection = json.load(f)
    L_star = selection["L_star"]
    alpha_star = selection["alpha_star"]
    steering_vector = layer_vectors[L_star].to(device)

    # Choose head set: exploratory (cross-layer) or syc (coverage-based)
    use_exploratory = cfg["exp3"].get("use_exploratory_heads", False)
    head_set_name = "syc_heads"
    if use_exploratory and "exploratory_heads_path" in cfg["exp3"]:
        exp_path = Path(cfg["exp3"]["exploratory_heads_path"])
        if exp_path.exists():
            syc_heads = load_head_list(exp_path)
            head_set_name = "exploratory_heads"
            print(f"Using exploratory heads ({len(syc_heads)} heads from {exp_path})")
        else:
            syc_heads = load_head_list(Path(cfg["exp3"]["syc_heads_path"]))
            print(f"Exploratory heads not found, falling back to syc_heads")
    else:
        syc_heads = load_head_list(Path(cfg["exp3"]["syc_heads_path"]))
    
    # Generate random control heads matching the layer distribution of selected heads
    import random
    rng = random.Random(cfg.get("seed", 42))
    num_heads = model.config.num_attention_heads
    syc_head_set = set(syc_heads)
    rand_heads = []
    for layer, head in syc_heads:
        excluded = {h for (l, h) in syc_head_set if l == layer}
        candidates = [h for h in range(num_heads) if h not in excluded]
        if candidates:
            rand_heads.append((layer, rng.choice(candidates)))
    if not syc_heads or not rand_heads:
        output_dir = Path(cfg["exp3"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "num_examples": len(split_c_examples),
            "num_examples_valid": 0,
            "reason": "Missing syc or random head sets; run Exp2 with more examples.",
        }
        with (output_dir / "mediation_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        return
    head_means = compute_head_means(
        model=model,
        tokenizer=tokenizer,
        registry=registry,
        examples=split_b_examples,
        head_tuples=list(set(syc_heads + rand_heads)),
        device=device,
        max_examples=cfg["exp3"]["head_means_max_examples"],
    )
    if not head_means:
        output_dir = Path(cfg["exp3"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "num_examples": len(split_c_examples),
            "num_examples_valid": 0,
            "reason": "Failed to compute head means (empty head list or zero captures).",
        }
        with (output_dir / "mediation_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        return
    syc_ablation = {head: head_means[head] for head in syc_heads if head in head_means}
    rand_ablation = {head: head_means[head] for head in rand_heads if head in head_means}
    if not syc_ablation or not rand_ablation:
        output_dir = Path(cfg["exp3"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "num_examples": len(split_c_examples),
            "num_examples_valid": 0,
            "reason": "Head means missing for syc or random set; rerun Exp2/means with more coverage.",
        }
        with (output_dir / "mediation_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        return

    output_dir = Path(cfg["exp3"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "mediation_results.jsonl"

    max_examples = cfg["exp3"]["max_examples"]
    records = []
    with results_path.open("w") as writer:
        for ex in tqdm(split_c_examples[:max_examples], desc="Exp3 mediation grid"):
            inputs = tokenize_prompt(tokenizer, ex.user_wrong_prompt, device)
            pos_idx = inputs["input_ids"].shape[1] - 1
            base = evaluate_condition(
                model, tokenizer, registry, ex, device, pos_idx, L_star, steering_vector, alpha_star, None, False
            )
            vec = evaluate_condition(
                model, tokenizer, registry, ex, device, pos_idx, L_star, steering_vector, alpha_star, None, True
            )
            syc_abl = evaluate_condition(
                model,
                tokenizer,
                registry,
                ex,
                device,
                pos_idx,
                L_star,
                steering_vector,
                alpha_star,
                syc_ablation,
                False,
            )
            rand_abl = evaluate_condition(
                model,
                tokenizer,
                registry,
                ex,
                device,
                pos_idx,
                L_star,
                steering_vector,
                alpha_star,
                rand_ablation,
                False,
            )
            vec_syc = evaluate_condition(
                model,
                tokenizer,
                registry,
                ex,
                device,
                pos_idx,
                L_star,
                steering_vector,
                alpha_star,
                syc_ablation,
                True,
            )
            vec_rand = evaluate_condition(
                model,
                tokenizer,
                registry,
                ex,
                device,
                pos_idx,
                L_star,
                steering_vector,
                alpha_star,
                rand_ablation,
                True,
            )
            row = {
                "example_id": ex.uid,
                "D_base": base["D_syc"],
                "D_vec": vec["D_syc"],
                "D_syc_abl": syc_abl["D_syc"],
                "D_rand_abl": rand_abl["D_syc"],
                "D_vec_syc": vec_syc["D_syc"],
                "D_vec_rand": vec_rand["D_syc"],
            }
            writer.write(json.dumps(row) + "\n")
            records.append(row)

    # Summary statistics
    # Data-driven threshold: default relative to median |E_clean|
    threshold_mode = cfg["exp3"].get("delta_E_mode", "relative")
    E_clean_values = [row["D_base"] - row["D_vec"] for row in records]
    if threshold_mode == "absolute":
        threshold = cfg["exp3"]["delta_E_threshold"]
    else:
        median_abs = median([abs(x) for x in E_clean_values]) if E_clean_values else 0.0
        threshold = max(
            cfg["exp3"].get("delta_E_min", 0.05),
            cfg["exp3"].get("delta_E_multiplier", 0.25) * median_abs,
        )
    ras_syc_values = []
    ras_rand_values = []
    for row in records:
        E_clean = row["D_base"] - row["D_vec"]
        if E_clean < threshold:
            continue
        E_syc = row["D_syc_abl"] - row["D_vec_syc"]
        E_rand = row["D_rand_abl"] - row["D_vec_rand"]
        ras_syc = 1 - (E_syc / E_clean)
        ras_rand = 1 - (E_rand / E_clean)
        ras_syc_values.append(ras_syc)
        ras_rand_values.append(ras_rand)

    if ras_syc_values and ras_rand_values:
        t_stat, p_value = stats.ttest_rel(ras_syc_values, ras_rand_values)
        diffs = [s - r for s, r in zip(ras_syc_values, ras_rand_values)]
        mean_diff = mean(diffs)
        cohens_d = mean_diff / (pstdev(diffs) if len(diffs) > 1 else 1.0)
    else:
        p_value = None
        mean_diff = 0.0
        cohens_d = 0.0

    summary = {
        "num_examples": len(records),
        "num_examples_valid": len(ras_syc_values),
        "head_set_used": head_set_name,
        "num_heads_tested": len(syc_heads),
        "num_unique_layers": len(set(l for l, h in syc_heads)),
        "mean_RAS_syc": mean(ras_syc_values) if ras_syc_values else 0.0,
        "mean_RAS_rand": mean(ras_rand_values) if ras_rand_values else 0.0,
        "mean_diff": mean_diff,
        "cohens_d": cohens_d,
        "p_value": p_value,
        "threshold": threshold,
        "threshold_mode": threshold_mode,
        "delta_E_values_median_abs": median([abs(x) for x in E_clean_values]) if E_clean_values else 0.0,
        "E_clean_values": E_clean_values,
    }
    with (output_dir / "mediation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
