"""
Experiment 5: Contrastive Causal Mediation (CCM) for Sycophancy Steering

CCM finds which attention heads CAUSALLY mediate sycophancy by measuring
the indirect effect of each head on the output.

Algorithm:
1. For each example, create contrastive pair: (user_wrong_prompt, neutral_prompt)
2. For each attention head (layer, head):
   - Run model on user_wrong_prompt, cache head output at position p
   - Run model on neutral_prompt, but replace head output with cached value
   - Measure change in D_syc (indirect effect)
3. Rank heads by average indirect effect
4. Select top-K heads for steering
5. Apply head-specific steering vectors to selected heads
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import yaml
from tqdm import tqdm

from src.data.syc_dataset import load_user_wrong_split, make_neutral_prompt, SycophancyExample
from src.hooks.head_patch_hooks import (
    HeadPatchRegistry,
    ensure_full_instrumentation,
    cache_mode,
    patch_mode,
)
from src.metrics.syc_metric import compute_D_syc
from src.models.llama_loader import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp5: CCM Sycophancy Steering")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/exp5_ccm.yaml"),
        help="Config file",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def tokenize_prompt(tokenizer, prompt: str, device: torch.device):
    enc = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


@dataclass
class HeadEffect:
    """Stores the indirect effect of a single attention head."""
    layer: int
    head: int
    effect: float  # Average indirect effect on D_syc
    std: float     # Standard deviation
    n: int         # Number of examples


def compute_head_indirect_effect(
    model,
    tokenizer,
    registry: HeadPatchRegistry,
    examples: List[SycophancyExample],
    layer_idx: int,
    head_idx: int,
    device: torch.device,
) -> HeadEffect:
    """
    Compute the indirect effect of a single attention head.
    
    Indirect effect = D_syc(neutral) - D_syc(neutral with head patched from user_wrong)
    
    Positive effect means: patching this head with sycophantic context INCREASES sycophancy.
    This indicates the head causally mediates sycophancy.
    """
    effects = []
    
    for ex in examples:
        user_wrong_prompt = ex.user_wrong_prompt
        neutral_prompt = make_neutral_prompt(ex)
        wrong_answer = ex.wrong_answer
        right_answer = ex.correct_answer
        example_id = f"ex_{id(ex)}"
        
        # Get position index (last token position)
        inputs_uw = tokenize_prompt(tokenizer, user_wrong_prompt, device)
        inputs_n = tokenize_prompt(tokenizer, neutral_prompt, device)
        pos_idx_uw = inputs_uw["input_ids"].shape[1] - 1
        pos_idx_n = inputs_n["input_ids"].shape[1] - 1
        
        # Step 1: Cache head output from user_wrong_prompt
        with cache_mode(registry, example_id, pos_idx_uw):
            with torch.no_grad():
                _ = model(**inputs_uw)
        
        # Step 2: Compute D_syc on neutral prompt (baseline)
        base_metric = compute_D_syc(
            model=model,
            tokenizer=tokenizer,
            prompt_text=neutral_prompt,
            wrong_answer=wrong_answer,
            right_answer=right_answer,
            device=device,
        )
        base_d_syc = base_metric["D_syc"]
        
        # Step 3: Compute D_syc on neutral prompt with head patched from user_wrong
        with patch_mode(registry, example_id, layer_idx, head_idx, pos_idx_n):
            patched_metric = compute_D_syc(
                model=model,
                tokenizer=tokenizer,
                prompt_text=neutral_prompt,
                wrong_answer=wrong_answer,
                right_answer=right_answer,
                device=device,
            )
        patched_d_syc = patched_metric["D_syc"]
        
        # Indirect effect: how much does patching increase sycophancy?
        # Positive = head mediates sycophancy (patching from syc context increases D_syc)
        indirect_effect = patched_d_syc - base_d_syc
        effects.append(indirect_effect)
        
        # Clean up cache
        registry.cache.clear_example(example_id)
    
    effects_tensor = torch.tensor(effects)
    return HeadEffect(
        layer=layer_idx,
        head=head_idx,
        effect=float(effects_tensor.mean()),
        std=float(effects_tensor.std()),
        n=len(effects),
    )


def compute_all_head_effects(
    model,
    tokenizer,
    registry: HeadPatchRegistry,
    examples: List[SycophancyExample],
    device: torch.device,
) -> List[HeadEffect]:
    """Compute indirect effects for all attention heads."""
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    all_effects = []
    total_heads = num_layers * num_heads
    
    print(f"Computing indirect effects for {total_heads} heads...")
    
    with tqdm(total=total_heads, desc="Computing head effects") as pbar:
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                effect = compute_head_indirect_effect(
                    model=model,
                    tokenizer=tokenizer,
                    registry=registry,
                    examples=examples,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    device=device,
                )
                all_effects.append(effect)
                pbar.update(1)
                pbar.set_postfix({"layer": layer_idx, "head": head_idx, "effect": f"{effect.effect:.4f}"})
    
    return all_effects


def select_top_heads(
    effects: List[HeadEffect],
    top_k: int,
) -> List[HeadEffect]:
    """Select top-K heads with highest positive indirect effect (mediate sycophancy)."""
    # Sort by absolute effect (heads that strongly mediate, positive or negative)
    sorted_effects = sorted(effects, key=lambda x: abs(x.effect), reverse=True)
    return sorted_effects[:top_k]


def compute_head_steering_vectors(
    model,
    tokenizer,
    registry: HeadPatchRegistry,
    examples: List[SycophancyExample],
    selected_heads: List[HeadEffect],
    device: torch.device,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Compute steering vectors for selected heads.
    
    For each head, the steering vector is:
    mean(head_output on neutral) - mean(head_output on user_wrong)
    
    Adding this vector should DECREASE sycophancy.
    """
    print(f"Computing steering vectors for {len(selected_heads)} selected heads...")
    
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    
    # Accumulate head outputs
    head_outputs_neutral: Dict[Tuple[int, int], List[torch.Tensor]] = {
        (h.layer, h.head): [] for h in selected_heads
    }
    head_outputs_user_wrong: Dict[Tuple[int, int], List[torch.Tensor]] = {
        (h.layer, h.head): [] for h in selected_heads
    }
    
    for ex in tqdm(examples, desc="Collecting head outputs"):
        user_wrong_prompt = ex.user_wrong_prompt
        neutral_prompt = make_neutral_prompt(ex)
        example_id = f"steering_{id(ex)}"
        
        inputs_uw = tokenize_prompt(tokenizer, user_wrong_prompt, device)
        inputs_n = tokenize_prompt(tokenizer, neutral_prompt, device)
        pos_idx_uw = inputs_uw["input_ids"].shape[1] - 1
        pos_idx_n = inputs_n["input_ids"].shape[1] - 1
        
        # Cache from user_wrong
        with cache_mode(registry, example_id, pos_idx_uw):
            with torch.no_grad():
                _ = model(**inputs_uw)
        
        # Extract cached values
        for (layer, head) in head_outputs_user_wrong.keys():
            from src.hooks.head_patch_hooks import HeadKey
            key = HeadKey(example_id, layer, head)
            cached = registry.cache.get(key)
            if cached is not None:
                head_outputs_user_wrong[(layer, head)].append(cached)
        
        registry.cache.clear_example(example_id)
        
        # Cache from neutral
        with cache_mode(registry, example_id, pos_idx_n):
            with torch.no_grad():
                _ = model(**inputs_n)
        
        # Extract cached values
        for (layer, head) in head_outputs_neutral.keys():
            from src.hooks.head_patch_hooks import HeadKey
            key = HeadKey(example_id, layer, head)
            cached = registry.cache.get(key)
            if cached is not None:
                head_outputs_neutral[(layer, head)].append(cached)
        
        registry.cache.clear_example(example_id)
    
    # Compute mean difference vectors
    steering_vectors = {}
    for (layer, head) in head_outputs_neutral.keys():
        if head_outputs_neutral[(layer, head)] and head_outputs_user_wrong[(layer, head)]:
            neutral_mean = torch.stack(head_outputs_neutral[(layer, head)]).mean(dim=0)
            user_wrong_mean = torch.stack(head_outputs_user_wrong[(layer, head)]).mean(dim=0)
            # Vector to steer AWAY from sycophancy
            steering_vectors[(layer, head)] = neutral_mean - user_wrong_mean
    
    return steering_vectors


def evaluate_head_steering(
    model,
    tokenizer,
    registry: HeadPatchRegistry,
    examples: List[SycophancyExample],
    steering_vectors: Dict[Tuple[int, int], torch.Tensor],
    alpha: float,
    device: torch.device,
) -> Dict:
    """Evaluate steering by applying vectors to selected heads."""
    effects = []
    
    for ex in tqdm(examples, desc=f"Evaluating (alpha={alpha})"):
        user_wrong_prompt = ex.user_wrong_prompt
        wrong_answer = ex.wrong_answer
        right_answer = ex.correct_answer
        
        inputs = tokenize_prompt(tokenizer, user_wrong_prompt, device)
        pos_idx = inputs["input_ids"].shape[1] - 1
        
        # Baseline D_syc
        base_metric = compute_D_syc(
            model=model,
            tokenizer=tokenizer,
            prompt_text=user_wrong_prompt,
            wrong_answer=wrong_answer,
            right_answer=right_answer,
            device=device,
        )
        base_d_syc = base_metric["D_syc"]
        
        # Steered D_syc using mean ablation with steering vectors
        scaled_vectors = {k: v * alpha for k, v in steering_vectors.items()}
        registry.set_mean_ablation(scaled_vectors, pos_idx)
        
        try:
            steered_metric = compute_D_syc(
                model=model,
                tokenizer=tokenizer,
                prompt_text=user_wrong_prompt,
                wrong_answer=wrong_answer,
                right_answer=right_answer,
                device=device,
            )
            steered_d_syc = steered_metric["D_syc"]
        finally:
            registry.clear_mean_ablation()
        
        effect = base_d_syc - steered_d_syc  # Positive = reduced sycophancy
        effects.append(effect)
    
    effects_tensor = torch.tensor(effects)
    return {
        "alpha": alpha,
        "n": len(effects),
        "mean_effect": float(effects_tensor.mean()),
        "std_effect": float(effects_tensor.std()),
        "effects": [float(e) for e in effects],
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    
    # Load model
    print("Loading model...")
    model_cfg = cfg["model"]
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_cfg["name"],
        device=model_cfg.get("device", "auto"),
        dtype=model_cfg.get("dtype", "auto"),
    )
    device = next(model.parameters()).device
    
    # Instrument model
    registry = ensure_full_instrumentation(model)
    
    # Load data
    dataset_cfg = cfg["dataset"]
    ccm_cfg = cfg["ccm"]
    
    split_a = load_user_wrong_split(Path(dataset_cfg["split_a_path"]))
    split_c = load_user_wrong_split(Path(dataset_cfg["split_c_path"]))
    
    # Limit examples
    num_mediation = ccm_cfg.get("num_mediation_examples", 100)
    num_eval = ccm_cfg.get("num_eval_examples", 200)
    
    mediation_examples = split_a[:num_mediation]
    eval_examples = split_c[:num_eval]
    
    print(f"Mediation examples: {len(mediation_examples)}")
    print(f"Evaluation examples: {len(eval_examples)}")
    
    output_dir = Path(ccm_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Compute indirect effects for all heads
    print("\n" + "="*60)
    print("STEP 1: Computing indirect effects for all attention heads")
    print("="*60)
    
    all_effects = compute_all_head_effects(
        model=model,
        tokenizer=tokenizer,
        registry=registry,
        examples=mediation_examples,
        device=device,
    )
    
    # Save all effects
    effects_data = [
        {"layer": e.layer, "head": e.head, "effect": e.effect, "std": e.std, "n": e.n}
        for e in all_effects
    ]
    with (output_dir / "head_effects.json").open("w") as f:
        json.dump(effects_data, f, indent=2)
    
    # Step 2: Select top-K heads
    print("\n" + "="*60)
    print("STEP 2: Selecting top-K heads")
    print("="*60)
    
    top_k = ccm_cfg.get("top_k_heads", 50)
    selected_heads = select_top_heads(all_effects, top_k)
    
    print(f"\nTop {len(selected_heads)} heads by indirect effect:")
    print(f"{'Layer':<6} {'Head':<6} {'Effect':<12} {'Std':<12}")
    print("-" * 36)
    for h in selected_heads[:20]:  # Show top 20
        print(f"{h.layer:<6} {h.head:<6} {h.effect:+.6f}   {h.std:.6f}")
    
    # Save selected heads
    selected_data = [
        {"layer": h.layer, "head": h.head, "effect": h.effect, "std": h.std}
        for h in selected_heads
    ]
    with (output_dir / "selected_heads.json").open("w") as f:
        json.dump(selected_data, f, indent=2)
    
    # Step 3: Compute steering vectors for selected heads
    print("\n" + "="*60)
    print("STEP 3: Computing steering vectors")
    print("="*60)
    
    steering_vectors = compute_head_steering_vectors(
        model=model,
        tokenizer=tokenizer,
        registry=registry,
        examples=mediation_examples,
        selected_heads=selected_heads,
        device=device,
    )
    
    # Save steering vectors
    vectors_to_save = {f"L{k[0]}_H{k[1]}": v.tolist() for k, v in steering_vectors.items()}
    with (output_dir / "steering_vectors.json").open("w") as f:
        json.dump(vectors_to_save, f, indent=2)
    
    # Step 4: Evaluate steering
    print("\n" + "="*60)
    print("STEP 4: Evaluating steering effectiveness")
    print("="*60)
    
    alpha_values = ccm_cfg.get("alpha_sweep", [1.0, 2.0, 4.0, 8.0])
    all_results = {}
    
    for alpha in alpha_values:
        print(f"\nAlpha = {alpha}")
        result = evaluate_head_steering(
            model=model,
            tokenizer=tokenizer,
            registry=registry,
            examples=eval_examples,
            steering_vectors=steering_vectors,
            alpha=alpha,
            device=device,
        )
        all_results[str(alpha)] = result
        print(f"  Effect: {result['mean_effect']:+.4f} ± {result['std_effect']:.4f}")
    
    # Save results
    with (output_dir / "evaluation_results.json").open("w") as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Alpha':<8} {'Effect':<12} {'Std':<12}")
    print("-" * 32)
    
    best_alpha = None
    best_effect = -float("inf")
    
    for alpha_str, result in all_results.items():
        effect = result["mean_effect"]
        std = result["std_effect"]
        print(f"{alpha_str:<8} {effect:+.4f}       {std:.4f}")
        if effect > best_effect:
            best_effect = effect
            best_alpha = alpha_str
    
    print(f"\nBest alpha: {best_alpha} (effect: {best_effect:+.4f})")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
