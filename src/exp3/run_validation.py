"""
Validation experiments to address methodological concerns:

1. Circuit Verification: Verify the syc circuit by showing ablation reduces baseline sycophancy
2. Sign-aware Mediation: Re-run mediation with only positive-delta heads
3. Position-split Mediation: Test prompt_last and first_answer heads separately

Usage:
    conda activate persona-vectors
    uv run python -m src.exp3.run_validation --config config/exp3_mediation.yaml
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List, Tuple, Optional

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
    parser = argparse.ArgumentParser(description="Validation experiments for mediation")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/exp3_mediation.yaml"),
        help="Config file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: llama-results/validation)",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def tokenize_prompt(tokenizer, text: str, device: torch.device):
    enc = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def load_head_list(path: Path) -> List[Dict]:
    """Load heads with their delta values."""
    with path.open("r") as f:
        return json.load(f)


def load_head_scores(path: Path) -> List[Dict]:
    """Load aggregated head scores from JSONL."""
    scores = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                scores.append(json.loads(line))
    return scores


def load_raw_results_by_position(path: Path) -> Dict[str, List[Dict]]:
    """Load raw patching results grouped by position label."""
    by_pos: Dict[str, List[Dict]] = defaultdict(list)
    with path.open("r") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                pos = row.get("pos_label", "unknown")
                by_pos[pos].append(row)
    return by_pos


def aggregate_by_head(rows: List[Dict]) -> Dict[Tuple[int, int], Dict]:
    """Aggregate deltas by (layer, head)."""
    agg = defaultdict(lambda: {"deltas": [], "count": 0})
    for row in rows:
        key = (row["layer"], row["head"])
        agg[key]["deltas"].append(row["delta"])
        agg[key]["count"] += 1
    
    result = {}
    for (layer, head), info in agg.items():
        deltas = info["deltas"]
        delta_sum = sum(deltas)
        delta_mean = delta_sum / len(deltas) if deltas else 0.0
        result[(layer, head)] = {
            "layer": layer,
            "head": head,
            "delta_sum": delta_sum,
            "delta_mean": delta_mean,
            "count": len(deltas),
        }
    return result


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
    for idx, ex in enumerate(tqdm(subset, desc="Computing head means")):
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
    ablation_vectors: Optional[Dict[Tuple[int, int], torch.Tensor]],
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


def run_circuit_verification(
    model,
    tokenizer,
    registry,
    examples,
    head_means: Dict[Tuple[int, int], torch.Tensor],
    device: torch.device,
    max_examples: int,
    output_path: Path,
):
    """
    Experiment 1: Circuit Verification
    
    Compare baseline D_syc vs D_syc with syc heads ablated (no steering).
    If ablation reduces sycophancy, the circuit is verified.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Circuit Verification")
    print("="*60)
    
    results = []
    for ex in tqdm(examples[:max_examples], desc="Circuit verification"):
        inputs = tokenize_prompt(tokenizer, ex.user_wrong_prompt, device)
        pos_idx = inputs["input_ids"].shape[1] - 1
        
        # Baseline (no ablation, no steering)
        base_metric = compute_D_syc(
            model=model,
            tokenizer=tokenizer,
            prompt_text=ex.user_wrong_prompt,
            wrong_answer=ex.wrong_answer,
            right_answer=ex.correct_answer,
            device=device,
        )
        
        # With syc head ablation (no steering)
        with mean_ablation_mode(registry, head_means, pos_idx):
            abl_metric = compute_D_syc(
                model=model,
                tokenizer=tokenizer,
                prompt_text=ex.user_wrong_prompt,
                wrong_answer=ex.wrong_answer,
                right_answer=ex.correct_answer,
                device=device,
            )
        
        results.append({
            "example_id": ex.uid,
            "D_base": base_metric["D_syc"],
            "D_ablated": abl_metric["D_syc"],
            "reduction": base_metric["D_syc"] - abl_metric["D_syc"],
        })
    
    # Compute summary statistics
    D_base_values = [r["D_base"] for r in results]
    D_abl_values = [r["D_ablated"] for r in results]
    reductions = [r["reduction"] for r in results]
    
    syc_rate_base = sum(1 for d in D_base_values if d > 0) / len(D_base_values)
    syc_rate_abl = sum(1 for d in D_abl_values if d > 0) / len(D_abl_values)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(D_base_values, D_abl_values)
    
    summary = {
        "num_examples": len(results),
        "num_heads_ablated": len(head_means),
        "D_base_mean": mean(D_base_values),
        "D_ablated_mean": mean(D_abl_values),
        "mean_reduction": mean(reductions),
        "syc_rate_base": syc_rate_base,
        "syc_rate_ablated": syc_rate_abl,
        "syc_rate_reduction": syc_rate_base - syc_rate_abl,
        "t_stat": t_stat,
        "p_value": p_value,
        "conclusion": "VERIFIED" if (mean(reductions) > 0 and p_value < 0.05) else "NOT VERIFIED",
    }
    
    # Save results
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "circuit_verification_results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with (output_path / "circuit_verification_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults:")
    print(f"  Baseline D_syc mean: {summary['D_base_mean']:.4f}")
    print(f"  Ablated D_syc mean:  {summary['D_ablated_mean']:.4f}")
    print(f"  Mean reduction:      {summary['mean_reduction']:.4f}")
    print(f"  Syc rate base:       {summary['syc_rate_base']:.2%}")
    print(f"  Syc rate ablated:    {summary['syc_rate_ablated']:.2%}")
    print(f"  p-value:             {summary['p_value']:.4f}")
    print(f"  Conclusion:          {summary['conclusion']}")
    
    return summary


def run_sign_aware_mediation(
    model,
    tokenizer,
    registry,
    examples,
    head_scores: List[Dict],
    split_b_examples,
    steering_layer: int,
    steering_vector: torch.Tensor,
    alpha: float,
    device: torch.device,
    max_examples: int,
    head_means_max: int,
    output_path: Path,
    seed: int = 42,
):
    """
    Experiment 2: Sign-aware Mediation
    
    Re-run mediation with only positive-delta heads (heads that PROMOTE sycophancy).
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Sign-aware Mediation")
    print("="*60)
    
    # Filter to positive-delta heads only
    positive_heads = [hs for hs in head_scores if hs["delta_sum"] > 0]
    negative_heads = [hs for hs in head_scores if hs["delta_sum"] < 0]
    
    print(f"Total heads: {len(head_scores)}")
    print(f"Positive-delta heads (promote syc): {len(positive_heads)}")
    print(f"Negative-delta heads (suppress syc): {len(negative_heads)}")
    
    # Select top positive heads (by delta_sum)
    positive_heads_sorted = sorted(positive_heads, key=lambda x: x["delta_sum"], reverse=True)
    selected_heads = positive_heads_sorted[:64]  # Match original count
    
    syc_head_tuples = [(h["layer"], h["head"]) for h in selected_heads]
    
    # Generate random controls
    import random
    rng = random.Random(seed)
    num_model_heads = model.config.num_attention_heads
    syc_head_set = set(syc_head_tuples)
    rand_head_tuples = []
    for layer, head in syc_head_tuples:
        excluded = {h for (l, h) in syc_head_set if l == layer}
        candidates = [h for h in range(num_model_heads) if h not in excluded]
        if candidates:
            rand_head_tuples.append((layer, rng.choice(candidates)))
    
    # Compute head means
    all_heads = list(set(syc_head_tuples + rand_head_tuples))
    head_means = compute_head_means(
        model, tokenizer, registry, split_b_examples,
        all_heads, device, head_means_max
    )
    
    syc_ablation = {h: head_means[h] for h in syc_head_tuples if h in head_means}
    rand_ablation = {h: head_means[h] for h in rand_head_tuples if h in head_means}
    
    # Run mediation grid
    records = []
    for ex in tqdm(examples[:max_examples], desc="Sign-aware mediation"):
        inputs = tokenize_prompt(tokenizer, ex.user_wrong_prompt, device)
        pos_idx = inputs["input_ids"].shape[1] - 1
        
        base = evaluate_condition(model, tokenizer, registry, ex, device, pos_idx,
                                  steering_layer, steering_vector, alpha, None, False)
        vec = evaluate_condition(model, tokenizer, registry, ex, device, pos_idx,
                                 steering_layer, steering_vector, alpha, None, True)
        syc_abl = evaluate_condition(model, tokenizer, registry, ex, device, pos_idx,
                                     steering_layer, steering_vector, alpha, syc_ablation, False)
        rand_abl = evaluate_condition(model, tokenizer, registry, ex, device, pos_idx,
                                      steering_layer, steering_vector, alpha, rand_ablation, False)
        vec_syc = evaluate_condition(model, tokenizer, registry, ex, device, pos_idx,
                                     steering_layer, steering_vector, alpha, syc_ablation, True)
        vec_rand = evaluate_condition(model, tokenizer, registry, ex, device, pos_idx,
                                      steering_layer, steering_vector, alpha, rand_ablation, True)
        
        records.append({
            "example_id": ex.uid,
            "D_base": base["D_syc"],
            "D_vec": vec["D_syc"],
            "D_syc_abl": syc_abl["D_syc"],
            "D_rand_abl": rand_abl["D_syc"],
            "D_vec_syc": vec_syc["D_syc"],
            "D_vec_rand": vec_rand["D_syc"],
        })
    
    # Compute RAS statistics
    E_clean_values = [r["D_base"] - r["D_vec"] for r in records]
    median_abs = median([abs(x) for x in E_clean_values]) if E_clean_values else 0.0
    threshold = max(0.05, 0.25 * median_abs)
    
    ras_syc_values = []
    ras_rand_values = []
    for row in records:
        E_clean = row["D_base"] - row["D_vec"]
        if abs(E_clean) < threshold:
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
        t_stat, p_value, mean_diff, cohens_d = 0.0, 1.0, 0.0, 0.0
    
    summary = {
        "experiment": "sign_aware_mediation",
        "num_examples": len(records),
        "num_examples_valid": len(ras_syc_values),
        "num_positive_heads_selected": len(selected_heads),
        "num_negative_heads_excluded": len(negative_heads),
        "mean_RAS_syc": mean(ras_syc_values) if ras_syc_values else 0.0,
        "mean_RAS_rand": mean(ras_rand_values) if ras_rand_values else 0.0,
        "mean_diff": mean_diff,
        "cohens_d": cohens_d,
        "p_value": p_value,
        "threshold": threshold,
        "conclusion": "MEDIATION" if (mean_diff > 0.1 and p_value < 0.05) else "BYPASS",
        "selected_heads": selected_heads[:10],  # Top 10 for reference
    }
    
    # Save results
    with (output_path / "sign_aware_mediation_results.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with (output_path / "sign_aware_mediation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults:")
    print(f"  Valid examples:     {summary['num_examples_valid']}")
    print(f"  Mean RAS (syc):     {summary['mean_RAS_syc']:.4f}")
    print(f"  Mean RAS (rand):    {summary['mean_RAS_rand']:.4f}")
    print(f"  Mean diff:          {summary['mean_diff']:.4f}")
    print(f"  Cohen's d:          {summary['cohens_d']:.4f}")
    print(f"  p-value:            {summary['p_value']:.4f}")
    print(f"  Conclusion:         {summary['conclusion']}")
    
    return summary


def run_position_split_mediation(
    model,
    tokenizer,
    registry,
    examples,
    raw_results_path: Path,
    split_b_examples,
    steering_layer: int,
    steering_vector: torch.Tensor,
    alpha: float,
    device: torch.device,
    max_examples: int,
    head_means_max: int,
    output_path: Path,
    seed: int = 42,
):
    """
    Experiment 3: Position-split Mediation
    
    Run mediation separately for prompt_last and first_answer heads.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Position-split Mediation")
    print("="*60)
    
    # Load raw results by position
    by_position = load_raw_results_by_position(raw_results_path)
    print(f"Positions found: {list(by_position.keys())}")
    
    results_by_pos = {}
    
    for pos_label in ["prompt_last", "first_answer"]:
        if pos_label not in by_position:
            print(f"  Skipping {pos_label} (no data)")
            continue
        
        print(f"\n--- Processing {pos_label} ---")
        rows = by_position[pos_label]
        head_agg = aggregate_by_head(rows)
        
        # Select top positive-delta heads for this position
        positive_heads = [h for h in head_agg.values() if h["delta_sum"] > 0]
        positive_heads_sorted = sorted(positive_heads, key=lambda x: x["delta_sum"], reverse=True)
        selected_heads = positive_heads_sorted[:32]  # Top 32 per position
        
        print(f"  Total heads: {len(head_agg)}")
        print(f"  Positive heads: {len(positive_heads)}")
        print(f"  Selected: {len(selected_heads)}")
        
        if not selected_heads:
            print(f"  No positive heads found, skipping")
            continue
        
        syc_head_tuples = [(h["layer"], h["head"]) for h in selected_heads]
        
        # Generate random controls
        import random
        rng = random.Random(seed)
        num_model_heads = model.config.num_attention_heads
        syc_head_set = set(syc_head_tuples)
        rand_head_tuples = []
        for layer, head in syc_head_tuples:
            excluded = {h for (l, h) in syc_head_set if l == layer}
            candidates = [h for h in range(num_model_heads) if h not in excluded]
            if candidates:
                rand_head_tuples.append((layer, rng.choice(candidates)))
        
        # Compute head means
        all_heads = list(set(syc_head_tuples + rand_head_tuples))
        head_means = compute_head_means(
            model, tokenizer, registry, split_b_examples,
            all_heads, device, head_means_max
        )
        
        syc_ablation = {h: head_means[h] for h in syc_head_tuples if h in head_means}
        rand_ablation = {h: head_means[h] for h in rand_head_tuples if h in head_means}
        
        # Run mediation
        records = []
        for ex in tqdm(examples[:max_examples], desc=f"Mediation ({pos_label})"):
            inputs = tokenize_prompt(tokenizer, ex.user_wrong_prompt, device)
            pos_idx = inputs["input_ids"].shape[1] - 1
            
            base = evaluate_condition(model, tokenizer, registry, ex, device, pos_idx,
                                      steering_layer, steering_vector, alpha, None, False)
            vec = evaluate_condition(model, tokenizer, registry, ex, device, pos_idx,
                                     steering_layer, steering_vector, alpha, None, True)
            vec_syc = evaluate_condition(model, tokenizer, registry, ex, device, pos_idx,
                                         steering_layer, steering_vector, alpha, syc_ablation, True)
            vec_rand = evaluate_condition(model, tokenizer, registry, ex, device, pos_idx,
                                          steering_layer, steering_vector, alpha, rand_ablation, True)
            syc_abl = evaluate_condition(model, tokenizer, registry, ex, device, pos_idx,
                                         steering_layer, steering_vector, alpha, syc_ablation, False)
            rand_abl = evaluate_condition(model, tokenizer, registry, ex, device, pos_idx,
                                          steering_layer, steering_vector, alpha, rand_ablation, False)
            
            records.append({
                "example_id": ex.uid,
                "D_base": base["D_syc"],
                "D_vec": vec["D_syc"],
                "D_syc_abl": syc_abl["D_syc"],
                "D_rand_abl": rand_abl["D_syc"],
                "D_vec_syc": vec_syc["D_syc"],
                "D_vec_rand": vec_rand["D_syc"],
            })
        
        # Compute RAS
        E_clean_values = [r["D_base"] - r["D_vec"] for r in records]
        median_abs = median([abs(x) for x in E_clean_values]) if E_clean_values else 0.0
        threshold = max(0.05, 0.25 * median_abs)
        
        ras_syc_values = []
        ras_rand_values = []
        for row in records:
            E_clean = row["D_base"] - row["D_vec"]
            if abs(E_clean) < threshold:
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
            t_stat, p_value, mean_diff, cohens_d = 0.0, 1.0, 0.0, 0.0
        
        summary = {
            "position": pos_label,
            "num_examples_valid": len(ras_syc_values),
            "num_heads_selected": len(selected_heads),
            "mean_RAS_syc": mean(ras_syc_values) if ras_syc_values else 0.0,
            "mean_RAS_rand": mean(ras_rand_values) if ras_rand_values else 0.0,
            "mean_diff": mean_diff,
            "cohens_d": cohens_d,
            "p_value": p_value,
            "top_heads": [(h["layer"], h["head"], h["delta_sum"]) for h in selected_heads[:5]],
        }
        
        results_by_pos[pos_label] = summary
        
        # Save per-position results
        with (output_path / f"position_{pos_label}_results.jsonl").open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        
        print(f"  Valid examples: {summary['num_examples_valid']}")
        print(f"  Mean RAS (syc): {summary['mean_RAS_syc']:.4f}")
        print(f"  Mean RAS (rand): {summary['mean_RAS_rand']:.4f}")
        print(f"  Mean diff: {summary['mean_diff']:.4f}")
        print(f"  p-value: {summary['p_value']:.4f}")
    
    # Save combined summary
    with (output_path / "position_split_summary.json").open("w") as f:
        json.dump(results_by_pos, f, indent=2)
    
    return results_by_pos


def main():
    args = parse_args()
    cfg = load_config(args.config)

    exp3_cfg = cfg.get("exp3", {})
    default_output_dir: Path | None = None
    if exp3_cfg.get("validation_output_dir"):
        default_output_dir = Path(exp3_cfg["validation_output_dir"])
    elif exp3_cfg.get("output_dir"):
        default_output_dir = Path(exp3_cfg["output_dir"]).parent / "validation"

    output_dir = args.output_dir or default_output_dir or Path("llama-results/validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("VALIDATION EXPERIMENTS")
    print("="*60)
    
    # Setup
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
    print(f"\nLoading model: {model_cfg['name']}")
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_cfg["name"],
        device=model_cfg.get("device", "auto"),
        dtype=model_cfg.get("dtype", "auto"),
    )
    device = next(model.parameters()).device
    registry = ensure_head_patching(model, force=True)
    
    # Load steering vector
    layer_vectors = torch.load(cfg["exp3"]["layer_vectors_path"], map_location=device)
    with Path(cfg["exp3"]["vector_selection_path"]).open("r") as f:
        selection = json.load(f)
    L_star = selection["L_star"]
    alpha_star = selection["alpha_star"]
    steering_vector = layer_vectors[L_star].to(device)
    print(f"Using steering vector: L={L_star}, α={alpha_star}")
    
    # Load existing head data
    syc_heads_path = Path(cfg["exp3"]["syc_heads_path"])
    head_scores_path = Path(exp3_cfg.get("head_scores_path", syc_heads_path.parent / "head_scores.jsonl"))
    raw_results_path = Path(exp3_cfg.get("raw_results_path", syc_heads_path.parent / "head_results_raw.jsonl"))
    
    syc_heads = load_head_list(syc_heads_path)
    head_scores = load_head_scores(head_scores_path)
    
    max_examples = cfg["exp3"]["max_examples"]
    head_means_max = cfg["exp3"]["head_means_max_examples"]
    
    all_summaries = {}
    
    # ==== EXPERIMENT 1: Circuit Verification ====
    print("\n" + "="*60)
    print("Computing head means for circuit verification...")
    syc_head_tuples = [(h["layer"], h["head"]) for h in syc_heads]
    syc_head_means = compute_head_means(
        model, tokenizer, registry, split_b_examples,
        syc_head_tuples, device, head_means_max
    )
    
    circuit_summary = run_circuit_verification(
        model=model,
        tokenizer=tokenizer,
        registry=registry,
        examples=split_c_examples,
        head_means=syc_head_means,
        device=device,
        max_examples=max_examples,
        output_path=output_dir,
    )
    all_summaries["circuit_verification"] = circuit_summary
    
    # ==== EXPERIMENT 2: Sign-aware Mediation ====
    sign_aware_summary = run_sign_aware_mediation(
        model=model,
        tokenizer=tokenizer,
        registry=registry,
        examples=split_c_examples,
        head_scores=head_scores,
        split_b_examples=split_b_examples,
        steering_layer=L_star,
        steering_vector=steering_vector,
        alpha=alpha_star,
        device=device,
        max_examples=max_examples,
        head_means_max=head_means_max,
        output_path=output_dir,
        seed=cfg.get("seed", 42),
    )
    all_summaries["sign_aware_mediation"] = sign_aware_summary
    
    # ==== EXPERIMENT 3: Position-split Mediation ====
    position_summary = run_position_split_mediation(
        model=model,
        tokenizer=tokenizer,
        registry=registry,
        examples=split_c_examples,
        raw_results_path=raw_results_path,
        split_b_examples=split_b_examples,
        steering_layer=L_star,
        steering_vector=steering_vector,
        alpha=alpha_star,
        device=device,
        max_examples=max_examples,
        head_means_max=head_means_max,
        output_path=output_dir,
        seed=cfg.get("seed", 42),
    )
    all_summaries["position_split"] = position_summary
    
    # ==== FINAL SUMMARY ====
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    with (output_dir / "validation_summary.json").open("w") as f:
        json.dump(all_summaries, f, indent=2)
    
    print(f"\n1. Circuit Verification: {circuit_summary['conclusion']}")
    print(f"   Syc rate reduction: {circuit_summary['syc_rate_reduction']:.2%}")
    
    print(f"\n2. Sign-aware Mediation: {sign_aware_summary['conclusion']}")
    print(f"   RAS diff: {sign_aware_summary['mean_diff']:.4f}")
    
    print(f"\n3. Position-split Mediation:")
    for pos, data in position_summary.items():
        conclusion = "MEDIATION" if data['mean_diff'] > 0.1 else "BYPASS"
        print(f"   {pos}: RAS_diff={data['mean_diff']:.4f} ({conclusion})")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
