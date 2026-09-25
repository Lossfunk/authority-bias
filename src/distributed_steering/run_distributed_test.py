"""
Experiment 4: Test the Distributed Hypothesis.

Tests whether steering vectors operate through attention heads or MLP/residual pathways.

Conditions:
1. Baseline: Steering with no ablation
2. All attention heads ablated: Does steering still work?
3. MLP layers ablated: Does steering break?

If attention ablation ≈ no effect → steering bypasses attention
If MLP ablation kills steering → steering operates via MLP
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from scipy import stats
from tqdm import tqdm

from src.data.syc_dataset import load_user_wrong_split
from src.hooks.head_patch_hooks import (
    HeadPatchRegistry,
    ensure_full_instrumentation,
    mean_ablation_mode,
    mlp_ablation_mode,
)
from src.hooks.steering_hooks import inject_steering_vector
from src.metrics.syc_metric import compute_D_syc
from src.models.llama_loader import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp4: Distributed Hypothesis Test")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/distributed_steering.yaml"),
        help="Config file",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def tokenize_prompt(tokenizer, prompt: str, device: torch.device):
    enc = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def compute_all_head_ablation_vectors(
    model,
    num_layers: int,
    num_heads: int,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Create zero vectors for all attention heads (zero ablation)."""
    head_dim = model.config.hidden_size // num_heads
    vectors = {}
    for layer in range(num_layers):
        for head in range(num_heads):
            vectors[(layer, head)] = torch.zeros(head_dim)
    return vectors


def measure_steering_effect(
    model,
    tokenizer,
    records: List[Dict],
    steering_vector: torch.Tensor,
    steering_layer: int,
    steering_alpha: float,
    device: torch.device,
    registry: HeadPatchRegistry,
    ablation_mode: str = "none",
    ablation_layers: List[int] = None,
) -> Dict:
    """
    Measure steering effect under different ablation conditions.

    ablation_mode: "none" | "all_attn" | "mlp"
    """
    effects = []
    base_d_syc_values = []
    steered_d_syc_values = []

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    for rec in tqdm(records, desc=f"Measuring ({ablation_mode})", leave=False):
        prompt = rec["prompt"]
        wrong_answer = rec["wrong_answer"]
        right_answer = rec["right_answer"]

        inputs = tokenize_prompt(tokenizer, prompt, device)
        pos_idx = inputs["input_ids"].shape[1] - 1

        # Compute baseline D_syc (no steering, with ablation if specified)
        if ablation_mode == "all_attn":
            head_vectors = compute_all_head_ablation_vectors(model, num_layers, num_heads)
            with mean_ablation_mode(registry, head_vectors, pos_idx):
                base_metric = compute_D_syc(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt,
                    wrong_answer=wrong_answer,
                    right_answer=right_answer,
                    device=device,
                )
        elif ablation_mode == "mlp" and ablation_layers:
            with mlp_ablation_mode(registry, ablation_layers, mode="zero"):
                base_metric = compute_D_syc(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt,
                    wrong_answer=wrong_answer,
                    right_answer=right_answer,
                    device=device,
                )
        else:
            base_metric = compute_D_syc(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt,
                wrong_answer=wrong_answer,
                right_answer=right_answer,
                device=device,
            )

        # Compute steered D_syc (with steering, with ablation if specified)
        if ablation_mode == "all_attn":
            head_vectors = compute_all_head_ablation_vectors(model, num_layers, num_heads)
            with mean_ablation_mode(registry, head_vectors, pos_idx):
                with inject_steering_vector(model, steering_layer, pos_idx, steering_vector, steering_alpha):
                    steered_metric = compute_D_syc(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=prompt,
                        wrong_answer=wrong_answer,
                        right_answer=right_answer,
                        device=device,
                    )
        elif ablation_mode == "mlp" and ablation_layers:
            with mlp_ablation_mode(registry, ablation_layers, mode="zero"):
                with inject_steering_vector(model, steering_layer, pos_idx, steering_vector, steering_alpha):
                    steered_metric = compute_D_syc(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=prompt,
                        wrong_answer=wrong_answer,
                        right_answer=right_answer,
                        device=device,
                    )
        else:
            with inject_steering_vector(model, steering_layer, pos_idx, steering_vector, steering_alpha):
                steered_metric = compute_D_syc(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt,
                    wrong_answer=wrong_answer,
                    right_answer=right_answer,
                    device=device,
                )

        base_d = base_metric["D_syc"]
        steered_d = steered_metric["D_syc"]
        effect = base_d - steered_d  # Positive = reduced sycophancy

        effects.append(effect)
        base_d_syc_values.append(base_d)
        steered_d_syc_values.append(steered_d)

    effects_tensor = torch.tensor(effects)
    return {
        "ablation_mode": ablation_mode,
        "ablation_layers": ablation_layers,
        "n": len(effects),
        "mean_effect": float(effects_tensor.mean()),
        "std_effect": float(effects_tensor.std()),
        "mean_base_d_syc": float(torch.tensor(base_d_syc_values).mean()),
        "mean_steered_d_syc": float(torch.tensor(steered_d_syc_values).mean()),
        "effects": [float(e) for e in effects],
    }


def compute_statistical_tests(results: Dict) -> Dict:
    """Compute statistical tests for a single alpha's results."""
    baseline_effects = results["baseline"]["effects"]
    attn_effects = results["all_attn_ablated"]["effects"]
    mlp_effects = results["mlp_ablated"]["effects"]

    # Paired t-tests
    t_stat_attn, p_val_attn = stats.ttest_rel(baseline_effects, attn_effects)
    t_stat_mlp, p_val_mlp = stats.ttest_rel(baseline_effects, mlp_effects)
    t_stat_attn_mlp, p_val_attn_mlp = stats.ttest_rel(attn_effects, mlp_effects)

    # One-sample t-tests
    t_stat_baseline, p_val_baseline = stats.ttest_1samp(baseline_effects, 0)
    t_stat_attn_zero, p_val_attn_zero = stats.ttest_1samp(attn_effects, 0)
    t_stat_mlp_zero, p_val_mlp_zero = stats.ttest_1samp(mlp_effects, 0)

    # Cohen's d
    def cohens_d_paired(x, y):
        diff = [a - b for a, b in zip(x, y)]
        mean_diff = sum(diff) / len(diff)
        std_diff = (sum((d - mean_diff)**2 for d in diff) / (len(diff)-1))**0.5
        return mean_diff / std_diff if std_diff > 0 else 0.0

    d_attn = cohens_d_paired(baseline_effects, attn_effects)
    d_mlp = cohens_d_paired(baseline_effects, mlp_effects)

    return {
        "one_sample": {
            "baseline": {"t": float(t_stat_baseline), "p": float(p_val_baseline)},
            "attn_ablated": {"t": float(t_stat_attn_zero), "p": float(p_val_attn_zero)},
            "mlp_ablated": {"t": float(t_stat_mlp_zero), "p": float(p_val_mlp_zero)},
        },
        "paired": {
            "baseline_vs_attn": {"t": float(t_stat_attn), "p": float(p_val_attn), "cohens_d": float(d_attn)},
            "baseline_vs_mlp": {"t": float(t_stat_mlp), "p": float(p_val_mlp), "cohens_d": float(d_mlp)},
            "attn_vs_mlp": {"t": float(t_stat_attn_mlp), "p": float(p_val_attn_mlp)},
        },
    }


def run_single_alpha(
    model,
    tokenizer,
    records: List[Dict],
    steering_vector: torch.Tensor,
    steering_layer: int,
    steering_alpha: float,
    device: torch.device,
    registry: HeadPatchRegistry,
    mlp_layers: List[int],
) -> Dict:
    """Run all conditions for a single alpha value."""
    results = {}

    # Condition 1: Baseline (no ablation)
    print(f"\n  [1/3] Baseline (no ablation)...")
    results["baseline"] = measure_steering_effect(
        model=model,
        tokenizer=tokenizer,
        records=records,
        steering_vector=steering_vector,
        steering_layer=steering_layer,
        steering_alpha=steering_alpha,
        device=device,
        registry=registry,
        ablation_mode="none",
    )
    print(f"    Effect: {results['baseline']['mean_effect']:.4f} ± {results['baseline']['std_effect']:.4f}")

    # Condition 2: All attention heads ablated
    print(f"  [2/3] All attention heads ablated...")
    results["all_attn_ablated"] = measure_steering_effect(
        model=model,
        tokenizer=tokenizer,
        records=records,
        steering_vector=steering_vector,
        steering_layer=steering_layer,
        steering_alpha=steering_alpha,
        device=device,
        registry=registry,
        ablation_mode="all_attn",
    )
    print(f"    Effect: {results['all_attn_ablated']['mean_effect']:.4f} ± {results['all_attn_ablated']['std_effect']:.4f}")

    # Condition 3: MLP layers ablated
    print(f"  [3/3] MLP layers {mlp_layers} ablated...")
    results["mlp_ablated"] = measure_steering_effect(
        model=model,
        tokenizer=tokenizer,
        records=records,
        steering_vector=steering_vector,
        steering_layer=steering_layer,
        steering_alpha=steering_alpha,
        device=device,
        registry=registry,
        ablation_mode="mlp",
        ablation_layers=mlp_layers,
    )
    print(f"    Effect: {results['mlp_ablated']['mean_effect']:.4f} ± {results['mlp_ablated']['std_effect']:.4f}")

    return results


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    # Load model
    model_cfg = cfg["model"]
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_cfg["name"],
        device=model_cfg.get("device", "auto"),
        dtype=model_cfg.get("dtype", "auto"),
    )
    device = next(model.parameters()).device

    # Instrument model for ablation
    registry = ensure_full_instrumentation(model)

    # Load steering vectors
    vector_cfg = cfg["steering_vector"]
    vectors_path = Path(vector_cfg["vectors_path"])
    layer_vectors = torch.load(vectors_path, map_location="cpu", weights_only=True)

    # Get layer values (support both single layer and sweep)
    if "layer_sweep" in vector_cfg:
        layer_values = vector_cfg["layer_sweep"]
    else:
        layer_values = [vector_cfg["layer"]]

    # Get alpha values (support both single alpha and sweep)
    if "alpha_sweep" in vector_cfg:
        alpha_values = vector_cfg["alpha_sweep"]
    else:
        alpha_values = [vector_cfg["alpha"]]

    print(f"Loaded steering vectors from: {vectors_path}")
    print(f"Layer sweep: {layer_values}")
    print(f"Alpha sweep: {alpha_values}")
    print(f"Total configurations to test: {len(layer_values) * len(alpha_values)}")

    # Load test data
    dataset_cfg = cfg["dataset"]
    examples = load_user_wrong_split(Path(dataset_cfg["test_path"]))

    # Limit examples if specified
    max_examples = cfg["exp4"].get("max_examples", len(examples))
    examples = examples[:max_examples]

    records = []
    for ex in examples:
        records.append({
            "prompt": ex.user_wrong_prompt,
            "wrong_answer": ex.wrong_answer,
            "right_answer": ex.correct_answer,
        })

    print(f"Testing on {len(records)} examples")

    mlp_layers = cfg["exp4"].get("mlp_ablation_layers", [8, 16, 24])
    output_dir = Path(cfg["exp4"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    config_idx = 0
    total_configs = len(layer_values) * len(alpha_values)

    # Run for each layer and alpha combination
    for steering_layer in layer_values:
        steering_vector = layer_vectors[steering_layer].to(device)

        for steering_alpha in alpha_values:
            config_idx += 1
            config_key = f"L{steering_layer}_a{steering_alpha}"

            print(f"\n{'='*60}")
            print(f"Layer={steering_layer}, Alpha={steering_alpha} ({config_idx}/{total_configs})")
            print("="*60)

            results = run_single_alpha(
                model=model,
                tokenizer=tokenizer,
                records=records,
                steering_vector=steering_vector,
                steering_layer=steering_layer,
                steering_alpha=steering_alpha,
                device=device,
                registry=registry,
                mlp_layers=mlp_layers,
            )

            # Compute stats
            baseline_effect = results["baseline"]["mean_effect"]
            attn_effect = results["all_attn_ablated"]["mean_effect"]
            mlp_effect = results["mlp_ablated"]["mean_effect"]

            attn_retention = attn_effect / baseline_effect if baseline_effect != 0 else float("nan")
            mlp_retention = mlp_effect / baseline_effect if baseline_effect != 0 else float("nan")

            statistical_tests = compute_statistical_tests(results)

            # Print summary
            print(f"\n  Summary for L={steering_layer}, α={steering_alpha}:")
            print(f"    Baseline effect:     {baseline_effect:+.4f} (p={statistical_tests['one_sample']['baseline']['p']:.4f})")
            print(f"    Attn ablated effect: {attn_effect:+.4f} (p={statistical_tests['one_sample']['attn_ablated']['p']:.4f})")
            print(f"    MLP ablated effect:  {mlp_effect:+.4f} (p={statistical_tests['one_sample']['mlp_ablated']['p']:.4f})")
            print(f"    Baseline vs Attn: p={statistical_tests['paired']['baseline_vs_attn']['p']:.4f}")

            all_results[config_key] = {
                "layer": steering_layer,
                "alpha": steering_alpha,
                "results": {
                    "baseline": {k: v for k, v in results["baseline"].items() if k != "effects"},
                    "all_attn_ablated": {k: v for k, v in results["all_attn_ablated"].items() if k != "effects"},
                    "mlp_ablated": {k: v for k, v in results["mlp_ablated"].items() if k != "effects"},
                },
                "retention": {
                    "attn_ablation": attn_retention,
                    "mlp_ablation": mlp_retention,
                },
                "statistical_tests": statistical_tests,
                "full_effects": results,
            }

    # Final summary across all configurations
    print("\n" + "="*60)
    print("LAYER x ALPHA SWEEP SUMMARY")
    print("="*60)
    print(f"\n{'Layer':<6} {'Alpha':<8} {'Baseline':<12} {'p(≠0)':<10} {'Attn Abl':<12} {'p(vs base)':<12}")
    print("-" * 70)

    best_config = None
    best_p = 1.0
    best_effect = 0.0

    for config_key, data in all_results.items():
        layer = data["layer"]
        alpha = data["alpha"]
        baseline = data["results"]["baseline"]["mean_effect"]
        p_baseline = data["statistical_tests"]["one_sample"]["baseline"]["p"]
        attn = data["results"]["all_attn_ablated"]["mean_effect"]
        p_vs_attn = data["statistical_tests"]["paired"]["baseline_vs_attn"]["p"]

        sig_baseline = "*" if p_baseline < 0.05 else ""
        sig_attn = "*" if p_vs_attn < 0.05 else ""

        print(f"{layer:<6} {alpha:<8} {baseline:+.4f}{sig_baseline:<4} {p_baseline:<10.4f} {attn:+.4f}{sig_attn:<4} {p_vs_attn:<12.4f}")

        # Track best config (prioritize significant positive effect)
        if baseline > 0 and p_baseline < best_p:
            best_p = p_baseline
            best_config = config_key
            best_effect = baseline

    if best_config:
        print(f"\nBest config: {best_config} (effect={best_effect:+.4f}, p={best_p:.4f})")
    else:
        print(f"\nNo configuration showed significant positive effect")

    # Save combined results
    sweep_summary = {
        "layer_sweep": layer_values,
        "alpha_sweep": alpha_values,
        "num_examples": len(records),
        "mlp_ablation_layers": mlp_layers,
        "best_config": best_config,
        "best_effect": best_effect,
        "best_p_value": best_p,
        "per_config_results": {k: {kk: vv for kk, vv in v.items() if kk != "full_effects"} for k, v in all_results.items()},
    }

    with (output_dir / "layer_alpha_sweep_results.json").open("w") as f:
        json.dump(sweep_summary, f, indent=2)

    # Save full results
    full_results = {k: v["full_effects"] for k, v in all_results.items()}
    with (output_dir / "layer_alpha_sweep_full.json").open("w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
