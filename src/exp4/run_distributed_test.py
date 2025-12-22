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
        default=Path("config/exp4_distributed.yaml"),
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

    # Load steering vector
    vector_cfg = cfg["steering_vector"]
    vectors_path = Path(vector_cfg["vectors_path"])
    layer_vectors = torch.load(vectors_path, map_location="cpu", weights_only=True)
    steering_layer = vector_cfg["layer"]
    steering_alpha = vector_cfg["alpha"]
    steering_vector = layer_vectors[steering_layer].to(device)

    print(f"Loaded steering vector: Layer {steering_layer}, α={steering_alpha}")

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

    results = {}

    # Condition 1: Baseline (no ablation)
    print("\n[1/3] Baseline (no ablation)...")
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
    print(f"  Effect: {results['baseline']['mean_effect']:.4f} ± {results['baseline']['std_effect']:.4f}")

    # Condition 2: All attention heads ablated
    print("\n[2/3] All attention heads ablated...")
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
    print(f"  Effect: {results['all_attn_ablated']['mean_effect']:.4f} ± {results['all_attn_ablated']['std_effect']:.4f}")

    # Condition 3: MLP layers ablated
    mlp_layers = cfg["exp4"].get("mlp_ablation_layers", [8, 16, 24])
    print(f"\n[3/3] MLP layers {mlp_layers} ablated...")
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
    print(f"  Effect: {results['mlp_ablated']['mean_effect']:.4f} ± {results['mlp_ablated']['std_effect']:.4f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    baseline_effect = results["baseline"]["mean_effect"]
    attn_effect = results["all_attn_ablated"]["mean_effect"]
    mlp_effect = results["mlp_ablated"]["mean_effect"]

    attn_retention = attn_effect / baseline_effect if baseline_effect != 0 else float("nan")
    mlp_retention = mlp_effect / baseline_effect if baseline_effect != 0 else float("nan")

    print(f"Baseline steering effect:     {baseline_effect:.4f}")
    print(f"With all attn ablated:        {attn_effect:.4f} ({attn_retention:.1%} retained)")
    print(f"With MLP layers ablated:      {mlp_effect:.4f} ({mlp_retention:.1%} retained)")

    print("\nINTERPRETATION:")
    if attn_retention > 0.7:
        print("  → Steering BYPASSES attention heads (effect retained with attn ablation)")
    else:
        print("  → Steering USES attention heads (effect reduced with attn ablation)")

    if mlp_retention < 0.3:
        print("  → Steering REQUIRES MLP layers (effect lost with MLP ablation)")
    else:
        print("  → Steering does NOT require these MLP layers")

    # Save results
    output_dir = Path(cfg["exp4"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "steering_layer": steering_layer,
        "steering_alpha": steering_alpha,
        "num_examples": len(records),
        "mlp_ablation_layers": mlp_layers,
        "results": {
            "baseline": {k: v for k, v in results["baseline"].items() if k != "effects"},
            "all_attn_ablated": {k: v for k, v in results["all_attn_ablated"].items() if k != "effects"},
            "mlp_ablated": {k: v for k, v in results["mlp_ablated"].items() if k != "effects"},
        },
        "retention": {
            "attn_ablation": attn_retention,
            "mlp_ablation": mlp_retention,
        },
    }

    with (output_dir / "distributed_test_results.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Save full results with per-example effects
    with (output_dir / "distributed_test_full.json").open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
