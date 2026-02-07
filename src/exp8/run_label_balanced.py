"""Experiment 8: Order/recency vs endorsement decomposition.

Runs 5 conditions to isolate order effects and endorsement semantics:
- C0_order: neutral matched template, "A and B"
- C2_order: neutral matched template, "B and A"
- C1_contrast: endorsement with contrast, "B, not A"
- C1_plain: endorsement without contrast, "B"
- C2_fragment: diagnostic fragment, "B and A" (pragmatic leakage)

Computes decomposition (logit space):
- order          = C2_order - C0_order
- endorse        = C1_contrast - C2_order
- contrast       = C1_plain - C1_contrast
- fragment_leak  = C2_fragment - C2_order
- total_contrast = C1_contrast - C0_order
- total_plain    = C1_plain - C0_order

Usage:
    python -m src.exp8.run_label_balanced
    python -m src.exp8.run_label_balanced --max-examples 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from src.exp7.dataset_mc import MCExample, load_mc_dataset, create_mc_examples, save_mc_dataset
from src.exp7.scoring import get_ab_token_ids, score_prompt_forced_choice, ForcedChoiceResult
from src.exp8.conditions import (
    CONDITIONS,
    CONDITION_SHORT_NAMES,
    format_all_conditions,
    format_baseline,
)
from src.models.llama_loader import load_model_and_tokenizer


DEFAULT_MODELS: Tuple[str, ...] = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp8: Label-balanced endorsement decomposition"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="One or more HuggingFace model ids.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/answer.jsonl"),
        help="Path to answer.jsonl (raw data).",
    )
    parser.add_argument(
        "--mc-dataset-path",
        type=Path,
        default=Path("data/exp7_mc_dataset.jsonl"),
        help="Path to MC dataset (reuses exp7 dataset).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/exp8"),
        help="Directory to write results.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Optional cap on number of examples (0 = all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def _resolve_data_path(path: Path) -> Path:
    if path.exists():
        return path
    fallback = Path("external/sycophancy-eval/datasets/answer.jsonl")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Dataset not found at '{path}' or '{fallback}'.")


def _model_tag(model_id: str) -> str:
    return model_id.replace("/", "__")


def _summarize_values(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of values."""
    if not values:
        return {
            "mean": 0.0, "median": 0.0, "std": 0.0,
            "min": 0.0, "max": 0.0,
            "n_positive": 0, "n_negative": 0, "positive_frac": 0.0,
        }
    n_pos = sum(1 for v in values if v > 0)
    n_neg = sum(1 for v in values if v < 0)
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "std": float(stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_frac": n_pos / len(values),
    }


def compute_decomposition(
    logit_wrong: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute the order/endorsement decomposition.

    Args:
        logit_wrong: Dict mapping condition name to logit(P_wrong) for that condition

    Returns:
        Decomposition metrics:
        - order: C2_order - C0_order
        - endorse: C1_contrast - C2_order
        - contrast: C1_plain - C1_contrast
        - fragment_leak: C2_fragment - C2_order
        - total_contrast: C1_contrast - C0_order
        - total_plain: C1_plain - C0_order
    """
    c0 = logit_wrong["C0_order"]
    c2 = logit_wrong["C2_order"]
    c1c = logit_wrong["C1_contrast"]
    c1p = logit_wrong["C1_plain"]
    c2f = logit_wrong["C2_fragment"]

    return {
        "order": c2 - c0,
        "endorse": c1c - c2,
        "contrast": c1p - c1c,
        "fragment_leak": c2f - c2,
        "total_contrast": c1c - c0,
        "total_plain": c1p - c0,
        # Also include raw values
        "c0_logit_wrong": c0,
        "c2_logit_wrong": c2,
        "c1_contrast_logit_wrong": c1c,
        "c1_plain_logit_wrong": c1p,
        "c2_fragment_logit_wrong": c2f,
    }


def _order_seed(uid: str, seed: int) -> int:
    """Deterministic per-item seed to track order bookkeeping."""
    digest = hashlib.md5(f"{seed}:{uid}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def run_for_model(
    model_id: str,
    examples: List[MCExample],
    output_dir: Path,
    seed: int,
) -> Dict:
    """Run exp8 for a single model."""
    print(f"\n{'='*60}")
    print(f"Running model: {model_id}")
    print(f"{'='*60}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_id, device="auto", dtype="auto"
    )
    model.eval()
    device = next(model.parameters()).device

    # Get A/B token IDs
    token_id_a, token_id_b = get_ab_token_ids(tokenizer)
    print(f"Token A: {tokenizer.decode([token_id_a])!r} (id={token_id_a})")
    print(f"Token B: {tokenizer.decode([token_id_b])!r} (id={token_id_b})")

    # Output file
    out_path = output_dir / f"{_model_tag(model_id)}_results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect decomposition values
    order_values: List[float] = []
    endorse_values: List[float] = []
    contrast_values: List[float] = []
    fragment_leak_values: List[float] = []
    total_contrast_values: List[float] = []
    total_plain_values: List[float] = []

    # Also collect per-condition fc_wrong for baseline analysis
    per_condition_fc_wrong: Dict[str, List[float]] = {c: [] for c in CONDITIONS}

    with out_path.open("w") as f:
        for ex in tqdm(examples, desc=f"Exp8 {model_id}"):
            # Generate all condition prompts
            prompts = format_all_conditions(ex)
            prompt_dict = prompts.as_dict()

            # Also get pure baseline (no user turn)
            baseline_prompt = format_baseline(ex)

            # Score all conditions
            condition_results: Dict[str, ForcedChoiceResult] = {}
            for cond_name, prompt_text in prompt_dict.items():
                result = score_prompt_forced_choice(
                    model, tokenizer, prompt_text, device, token_id_a, token_id_b
                )
                condition_results[cond_name] = result

            # Score baseline too
            baseline_result = score_prompt_forced_choice(
                model, tokenizer, baseline_prompt, device, token_id_a, token_id_b
            )

            # Extract logit_wrong for each condition
            logit_wrong: Dict[str, float] = {}
            fc_wrong: Dict[str, float] = {}
            for cond_name, result in condition_results.items():
                if ex.wrong_label == "A":
                    logit_wrong[cond_name] = result.logit_a
                    fc_wrong[cond_name] = result.fc_a
                else:
                    logit_wrong[cond_name] = result.logit_b
                    fc_wrong[cond_name] = result.fc_b
                per_condition_fc_wrong[cond_name].append(fc_wrong[cond_name])

            # Compute decomposition
            decomp = compute_decomposition(logit_wrong)

            order_values.append(decomp["order"])
            endorse_values.append(decomp["endorse"])
            contrast_values.append(decomp["contrast"])
            fragment_leak_values.append(decomp["fragment_leak"])
            total_contrast_values.append(decomp["total_contrast"])
            total_plain_values.append(decomp["total_plain"])

            # Build record
            record = {
                "model": model_id,
                "uid": ex.uid,
                "question": ex.question,
                "correct_answer": ex.correct_answer,
                "wrong_answer": ex.wrong_answer,
                "correct_label": ex.correct_label,
                "wrong_label": ex.wrong_label,
                "endorsed_label": ex.endorsed_label,
                "order_seed": _order_seed(ex.uid, seed=seed),
                "prompts": prompt_dict,
                "baseline_prompt": baseline_prompt,
                "condition_results": {
                    cond: {
                        "logit_a": r.logit_a,
                        "logit_b": r.logit_b,
                        "fc_a": r.fc_a,
                        "fc_b": r.fc_b,
                    }
                    for cond, r in condition_results.items()
                },
                "baseline_result": {
                    "logit_a": baseline_result.logit_a,
                    "logit_b": baseline_result.logit_b,
                    "fc_a": baseline_result.fc_a,
                    "fc_b": baseline_result.fc_b,
                },
                "logit_wrong": logit_wrong,
                "fc_wrong": fc_wrong,
                "decomposition": decomp,
                "metadata": ex.metadata,
            }
            f.write(json.dumps(record) + "\n")

    # Compute summary statistics
    summary = {
        "model": model_id,
        "n_examples": len(examples),
        "token_ids": {
            "a": token_id_a,
            "b": token_id_b,
            "a_str": tokenizer.decode([token_id_a]),
            "b_str": tokenizer.decode([token_id_b]),
        },
        "decomposition": {
            "order": _summarize_values(order_values),
            "endorse": _summarize_values(endorse_values),
            "contrast": _summarize_values(contrast_values),
            "fragment_leak": _summarize_values(fragment_leak_values),
            "total_contrast": _summarize_values(total_contrast_values),
            "total_plain": _summarize_values(total_plain_values),
        },
        "per_condition_fc_wrong": {
            cond: _summarize_values(values)
            for cond, values in per_condition_fc_wrong.items()
        },
    }

    # Save summary
    summary_path = output_dir / f"{_model_tag(model_id)}_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\nResults for {model_id}:")
    print(f"  N examples: {summary['n_examples']}")
    print(f"\n  Decomposition:")
    print(f"    Order (C2-C0):           mean={summary['decomposition']['order']['mean']:.4f}")
    print(f"    Endorse (C1c-C2):        mean={summary['decomposition']['endorse']['mean']:.4f}")
    print(f"    Contrast (C1p-C1c):      mean={summary['decomposition']['contrast']['mean']:.4f}")
    print(f"    Fragment leak (C2f-C2):  mean={summary['decomposition']['fragment_leak']['mean']:.4f}")
    print(f"    Total contrast (C1c-C0): mean={summary['decomposition']['total_contrast']['mean']:.4f}")
    print(f"    Total plain (C1p-C0):    mean={summary['decomposition']['total_plain']['mean']:.4f}")
    print(f"\n  Per-condition fc_wrong means:")
    for cond in CONDITIONS:
        print(f"    {CONDITION_SHORT_NAMES[cond]}: {summary['per_condition_fc_wrong'][cond]['mean']:.4f}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return summary


def main() -> None:
    args = parse_args()

    # Load or create MC dataset (reuse from exp7)
    mc_path = args.mc_dataset_path
    if mc_path.exists():
        print(f"Loading existing MC dataset from {mc_path}")
        examples = load_mc_dataset(mc_path)
    else:
        print(f"Creating MC dataset from {_resolve_data_path(args.data_path)}")
        raw_path = _resolve_data_path(args.data_path)
        examples = create_mc_examples(raw_path, seed=args.seed)
        save_mc_dataset(examples, mc_path)
        print(f"Saved MC dataset to {mc_path}")

    # Apply max_examples limit with shuffling
    if args.max_examples > 0 and len(examples) > args.max_examples:
        rng = random.Random(args.seed)
        shuffled = list(examples)
        rng.shuffle(shuffled)
        examples = shuffled[:args.max_examples]

    print(f"\nDataset: {len(examples)} examples")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write item table (row-level bookkeeping)
    item_table_path = args.output_dir / "item_table.jsonl"
    with item_table_path.open("w") as f:
        for ex in examples:
            item = {
                "uid": ex.uid,
                "dataset": ex.metadata.get("dataset", "unknown"),
                "correct_label": ex.correct_label,
                "wrong_label": ex.wrong_label,
                "opt_a_text": ex.option_a,
                "opt_b_text": ex.option_b,
                "order_seed": _order_seed(ex.uid, seed=args.seed),
            }
            f.write(json.dumps(item) + "\n")

    # Run for each model
    summaries = []
    for model_id in args.models:
        summary = run_for_model(model_id, examples, args.output_dir, seed=args.seed)
        summaries.append(summary)

    # Save combined summary
    combined = {
        "experiment": "exp8_order_control",
        "description": "Order/recency vs endorsement decomposition (matched-template control)",
        "conditions": {
            "C0_order": "Neutral matched template: 'considering options A and B'",
            "C2_order": "Neutral matched template: 'considering options B and A'",
            "C1_contrast": "Endorsement with contrast: 'B, not A'",
            "C1_plain": "Endorsement without contrast: 'B'",
            "C2_fragment": "Fragment diagnostic: 'B and A'",
        },
        "decomposition_formulas": {
            "order": "C2_order - C0_order",
            "endorse": "C1_contrast - C2_order",
            "contrast": "C1_plain - C1_contrast",
            "fragment_leak": "C2_fragment - C2_order",
            "total_contrast": "C1_contrast - C0_order",
            "total_plain": "C1_plain - C0_order",
        },
        "n_examples": len(examples),
        "seed": args.seed,
        "models": summaries,
    }
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n{'='*60}")
    print("Experiment complete!")
    print(f"Results saved to {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
