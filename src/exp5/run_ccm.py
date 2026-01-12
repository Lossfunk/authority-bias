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
5. Steer by patching selected head outputs towards the neutral prompt activations
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

from src.data.syc_dataset import (
    ensure_splits_exist,
    load_user_wrong_split,
    make_neutral_prompt,
    SycophancyExample,
)
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


def load_head_effects(path: Path) -> List[HeadEffect]:
    with path.open("r") as f:
        data = json.load(f)
    effects = []
    for item in data:
        effects.append(
            HeadEffect(
                layer=int(item["layer"]),
                head=int(item["head"]),
                effect=float(item["effect"]),
                std=float(item.get("std", 0.0)),
                n=int(item.get("n", 0)),
            )
        )
    return effects


def load_selected_heads(path: Path) -> List[HeadEffect]:
    with path.open("r") as f:
        data = json.load(f)
    effects = []
    for item in data:
        effects.append(
            HeadEffect(
                layer=int(item["layer"]),
                head=int(item["head"]),
                effect=float(item.get("effect", 0.0)),
                std=float(item.get("std", 0.0)),
                n=int(item.get("n", 0)),
            )
        )
    return effects


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


def evaluate_head_patching_sweep(
    model,
    tokenizer,
    registry: HeadPatchRegistry,
    examples: List[SycophancyExample],
    head_tuples: List[Tuple[int, int]],
    alpha_values: List[float],
    device: torch.device,
    *,
    patch_mode: str = "interp",
) -> Dict[str, Dict]:
    """Evaluate steering by patching selected heads to cached neutral values."""
    effects_by_alpha: Dict[float, List[float]] = {float(a): [] for a in alpha_values}

    for ex in tqdm(examples, desc="Evaluating (activation patching)"):
        user_wrong_prompt = ex.user_wrong_prompt
        neutral_prompt = make_neutral_prompt(ex)
        wrong_answer = ex.wrong_answer
        right_answer = ex.correct_answer
        
        inputs_uw = tokenize_prompt(tokenizer, user_wrong_prompt, device)
        inputs_n = tokenize_prompt(tokenizer, neutral_prompt, device)
        pos_idx_uw = inputs_uw["input_ids"].shape[1] - 1
        pos_idx_n = inputs_n["input_ids"].shape[1] - 1

        # Cache neutral activations at neutral decision point
        with cache_mode(registry, ex.uid, pos_idx_n):
            with torch.no_grad():
                _ = model(**inputs_n)

        cached_all = registry.cache.get_example(ex.uid)
        if len(head_tuples) == len(cached_all):
            head_values = cached_all
        else:
            wanted = set(head_tuples)
            head_values = {k: v for k, v in cached_all.items() if k in wanted}
        registry.cache.clear_example(ex.uid)
        
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

        # Keep patch active and just sweep alpha (avoids rebuilding large dicts per alpha)
        registry.set_value_patch(head_values, pos_idx_uw, alpha=float(alpha_values[0]), mode=patch_mode)
        try:
            for alpha in alpha_values:
                registry.value_patch_alpha = float(alpha)
                steered_metric = compute_D_syc(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=user_wrong_prompt,
                    wrong_answer=wrong_answer,
                    right_answer=right_answer,
                    device=device,
                )
                steered_d_syc = steered_metric["D_syc"]
                effect = base_d_syc - steered_d_syc  # Positive = reduced sycophancy
                effects_by_alpha[float(alpha)].append(float(effect))
        finally:
            registry.clear_value_patch()

    results: Dict[str, Dict] = {}
    for alpha in alpha_values:
        vals = effects_by_alpha[float(alpha)]
        effects_tensor = torch.tensor(vals)
        results[str(alpha)] = {
            "alpha": float(alpha),
            "n": len(vals),
            "mean_effect": float(effects_tensor.mean()),
            "std_effect": float(effects_tensor.std()),
            "effects": [float(e) for e in vals],
        }
    return results


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

    raw_path = Path(dataset_cfg.get("raw_path", "external/sycophancy-eval/datasets/answer.jsonl"))
    split_a_path = Path(dataset_cfg["split_a_path"])
    split_b_path = Path(dataset_cfg.get("split_b_path", "data/split_B_user_wrong.jsonl"))
    split_c_path = Path(dataset_cfg["split_c_path"])

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing raw dataset at {raw_path}. "
            "Either set dataset.raw_path in config/exp5_ccm.yaml to a file that exists in the container, "
            "or upload answer.jsonl to the Modal volume (e.g. /volume/data/answer.jsonl) and point raw_path to data/answer.jsonl."
        )

    ensure_splits_exist(
        raw_path=raw_path,
        split_a_path=split_a_path,
        split_b_path=split_b_path,
        split_c_path=split_c_path,
        seed=seed,
    )
    
    split_a = load_user_wrong_split(split_a_path)
    split_c = load_user_wrong_split(split_c_path)
    
    # Limit examples
    num_mediation = ccm_cfg.get("num_mediation_examples", 100)
    num_eval = ccm_cfg.get("num_eval_examples", 200)
    
    mediation_examples = split_a[:num_mediation]
    eval_examples = split_c[:num_eval]
    
    print(f"Mediation examples: {len(mediation_examples)}")
    print(f"Evaluation examples: {len(eval_examples)}")
    
    output_dir = Path(ccm_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    reuse_cached = bool(ccm_cfg.get("reuse_cached", True))
    head_effects_path = output_dir / "head_effects.json"
    selected_heads_path = output_dir / "selected_heads.json"
    
    # Step 1: Compute indirect effects for all heads
    print("\n" + "="*60)
    print("STEP 1: Computing indirect effects for all attention heads")
    print("="*60)

    if reuse_cached and head_effects_path.exists():
        print(f"Reusing cached head effects from {head_effects_path}")
        all_effects = load_head_effects(head_effects_path)
    else:
        all_effects = compute_all_head_effects(
            model=model,
            tokenizer=tokenizer,
            registry=registry,
            examples=mediation_examples,
            device=device,
        )

        effects_data = [
            {"layer": e.layer, "head": e.head, "effect": e.effect, "std": e.std, "n": e.n}
            for e in all_effects
        ]
        with head_effects_path.open("w") as f:
            json.dump(effects_data, f, indent=2)
    
    # Step 2: Select top-K heads
    print("\n" + "="*60)
    print("STEP 2: Selecting top-K heads")
    print("="*60)
    
    top_k = ccm_cfg.get("top_k_heads", 50)

    if reuse_cached and selected_heads_path.exists():
        print(f"Reusing cached selected heads from {selected_heads_path}")
        selected_heads = load_selected_heads(selected_heads_path)
    else:
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
    with selected_heads_path.open("w") as f:
        json.dump(selected_data, f, indent=2)
    
    # Step 3: Evaluate steering via activation patching
    print("\n" + "="*60)
    print("STEP 3: Evaluating steering effectiveness (activation patching)")
    print("="*60)
    
    def _all_head_tuples() -> List[Tuple[int, int]]:
        return [(l, h) for l in range(model.config.num_hidden_layers) for h in range(model.config.num_attention_heads)]

    # Optional: run multiple evaluation variants without re-loading the model.
    eval_variants = ccm_cfg.get("eval_variants")
    if not eval_variants:
        eval_variants = [
            {
                "name": "selected",
                "head_set": "selected",
                "patch_mode": ccm_cfg.get("patch_eval_mode", "interp"),
                "alpha_sweep": ccm_cfg.get("alpha_sweep", [1.0, 2.0, 4.0, 8.0]),
            }
        ]

    for variant in eval_variants:
        variant_name = str(variant.get("name", "variant"))
        head_set = str(variant.get("head_set", "selected"))
        patch_eval_mode = str(variant.get("patch_mode", "interp"))
        alpha_values = [float(a) for a in variant.get("alpha_sweep", [1.0, 2.0, 4.0, 8.0])]

        if head_set == "all":
            head_tuples = _all_head_tuples()
        else:
            head_tuples = [(h.layer, h.head) for h in selected_heads]

        print("\n" + "-" * 60)
        print(f"EVAL VARIANT: {variant_name}  (head_set={head_set}, patch_mode={patch_eval_mode})")
        print("-" * 60)

        all_results = evaluate_head_patching_sweep(
            model=model,
            tokenizer=tokenizer,
            registry=registry,
            examples=eval_examples,
            head_tuples=head_tuples,
            alpha_values=alpha_values,
            device=device,
            patch_mode=patch_eval_mode,
        )

        for alpha in alpha_values:
            result = all_results[str(alpha)]
            print(f"\nAlpha = {alpha}")
            print(f"  Effect: {result['mean_effect']:+.4f} ± {result['std_effect']:.4f}")

        results_path = output_dir / f"evaluation_results_patching_{variant_name}.json"
        with results_path.open("w") as f:
            json.dump(all_results, f, indent=2)

        best_alpha = None
        best_effect = -float("inf")
        for alpha_str, result in all_results.items():
            effect = result["mean_effect"]
            if effect > best_effect:
                best_effect = effect
                best_alpha = alpha_str
        print(f"\nBest alpha for {variant_name}: {best_alpha} (effect: {best_effect:+.4f})")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
