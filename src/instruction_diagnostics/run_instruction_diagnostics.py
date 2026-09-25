"""Experiment 15: diagnostic instruction sweep over correction-gating behavior.

This extends Exp10 by replacing the single instruction text with a sweep of
named instruction variants. Baseline no-instruction conditions are shared
across all variants, so the run stays cheap enough for a warm GPU session.

Example:
    uv run python -m src.instruction_diagnostics.run_instruction_diagnostics \
      --models Qwen/Qwen3-4B-Instruct-2507 \
      --mc-dataset-path data/piqa_mc_dataset_from_results.jsonl \
      --uids-file new-phase-results/piqa/piqa_prior_wrong_uids.txt \
      --tags Note \
      --instruction-set epistemic_modes_v1 \
      --output-dir new-phase-results/diagnostic-instructions/piqa_note_priorwrong
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from src.lexical_controls.dataset_mc import MCExample, create_mc_examples, load_mc_dataset, save_mc_dataset
from src.lexical_controls.scoring import ForcedChoiceResult, get_ab_token_ids, score_prompts_forced_choice_batch
from src.endorsement.conditions import Exp10Condition, compute_selectivity_metrics, format_prompt, normalize_tag
from src.cue_strength.conditions import parse_tags
from src.instruction_diagnostics.instruction_sets import InstructionVariant, load_instruction_variants
from src.models.llama_loader import load_model_and_tokenizer


DEFAULT_MODELS: Tuple[str, ...] = ("Qwen/Qwen3-4B-Instruct-2507",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp15: diagnostic instruction sweep for epistemic-mode hypotheses"
    )
    parser.add_argument("--models", type=str, nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--data-path", type=Path, default=Path("data/answer.jsonl"))
    parser.add_argument("--mc-dataset-path", type=Path, default=Path("data/exp7_mc_dataset.jsonl"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("new-phase-results/diagnostic-instructions/default"),
    )
    parser.add_argument("--max-examples", type=int, default=0, help="0 = all examples")
    parser.add_argument("--uids-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--tags",
        type=str,
        default="Note",
        help="Comma-separated tags. Allowed: Expert,Note,User,Someone online",
    )
    parser.add_argument(
        "--instruction-set",
        type=str,
        default=None,
        help="Named instruction set from src.instruction_diagnostics.instruction_sets",
    )
    parser.add_argument(
        "--instruction-file",
        type=Path,
        default=None,
        help="Optional JSON file with instruction variants ({name:text} or list of {name,text})",
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--save-prompts", action="store_true")
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


def _order_seed(uid: str, seed: int) -> int:
    digest = hashlib.md5(f"{seed}:{uid}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _summarize_values(values: Iterable[float]) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n": 0,
            "positive_frac": 0.0,
        }
    n_pos = sum(1 for v in vals if v > 0)
    return {
        "mean": float(mean(vals)),
        "median": float(median(vals)),
        "std": float(stdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
        "n": len(vals),
        "positive_frac": float(n_pos / len(vals)),
    }


def _fc_correct_from_result(ex: MCExample, result: ForcedChoiceResult) -> float:
    return result.fc_a if ex.correct_label == "A" else result.fc_b


def _build_baseline_conditions(tags: Sequence[str]) -> List[Exp10Condition]:
    conditions: List[Exp10Condition] = []
    for tag in tags:
        for endorse_type in ("neutral", "wrong", "correct"):
            conditions.append(
                Exp10Condition(
                    endorse_type=endorse_type,  # type: ignore[arg-type]
                    instruction=False,
                    tag=tag,  # type: ignore[arg-type]
                    instruction_text=None,
                )
            )
    return conditions


def _build_variant_conditions(tags: Sequence[str], variant: InstructionVariant) -> List[Exp10Condition]:
    conditions: List[Exp10Condition] = []
    for tag in tags:
        for endorse_type in ("neutral", "wrong", "correct"):
            conditions.append(
                Exp10Condition(
                    endorse_type=endorse_type,  # type: ignore[arg-type]
                    instruction=True,
                    tag=tag,  # type: ignore[arg-type]
                    instruction_text=variant.text,
                )
            )
    return conditions


def _run_one_model(
    *,
    model_id: str,
    examples: Sequence[MCExample],
    output_dir: Path,
    tags: Sequence[str],
    instruction_variants: Sequence[InstructionVariant],
    batch_size: int,
    max_length: int,
    device: str,
    dtype: str,
    save_prompts: bool,
    seed: int,
) -> Dict:
    print(f"\n{'=' * 64}")
    print(f"Running diagnostic instruction sweep for {model_id}")
    print(f"Tags: {list(tags)}")
    print(f"Instruction variants: {[v.name for v in instruction_variants]}")
    print(f"{'=' * 64}")

    model, tokenizer = load_model_and_tokenizer(model_id, device=device, dtype=dtype)
    model.eval()
    resolved_device = next(model.parameters()).device

    token_id_a, token_id_b = get_ab_token_ids(tokenizer)
    print(f"Token A: {tokenizer.decode([token_id_a])!r} (id={token_id_a})")
    print(f"Token B: {tokenizer.decode([token_id_b])!r} (id={token_id_b})")

    baseline_conditions = _build_baseline_conditions(tags)
    variant_conditions = {
        variant.name: _build_variant_conditions(tags, variant)
        for variant in instruction_variants
    }

    out_path = output_dir / f"{_model_tag(model_id)}_results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_fc_values: Dict[str, List[float]] = {cond.code: [] for cond in baseline_conditions}
    per_variant_metric_values: Dict[str, Dict[str, List[float]]] = {}
    per_variant_fc_values: Dict[str, Dict[str, List[float]]] = {}
    for variant in instruction_variants:
        metric_keys: List[str] = []
        for tag in tags:
            tag_key = normalize_tag(tag)
            metric_keys.extend([
                f"effect_wrong_I0_{tag_key}",
                f"effect_wrong_I1_{tag_key}",
                f"effect_correct_I0_{tag_key}",
                f"effect_correct_I1_{tag_key}",
                f"efficacy_wrong_{tag_key}",
                f"efficacy_correct_{tag_key}",
                f"selectivity_{tag_key}",
                f"baseline_shift_{tag_key}",
            ])
        per_variant_metric_values[variant.name] = {key: [] for key in metric_keys}
        per_variant_fc_values[variant.name] = {
            cond.code: [] for cond in variant_conditions[variant.name]
        }

    with out_path.open("w") as f:
        for ex in tqdm(examples, desc=f"Exp15 {model_id}"):
            baseline_prompts = [format_prompt(ex, cond) for cond in baseline_conditions]
            baseline_results = score_prompts_forced_choice_batch(
                model,
                tokenizer,
                baseline_prompts,
                resolved_device,
                token_id_a=token_id_a,
                token_id_b=token_id_b,
                batch_size=batch_size,
                max_length=max_length if max_length > 0 else None,
            )
            baseline_fc_correct: Dict[str, float] = {}
            baseline_prompt_map: Dict[str, str] = {}
            for cond, result, prompt in zip(baseline_conditions, baseline_results, baseline_prompts):
                baseline_fc = _fc_correct_from_result(ex, result)
                baseline_fc_correct[cond.code] = baseline_fc
                baseline_fc_values[cond.code].append(baseline_fc)
                if save_prompts:
                    baseline_prompt_map[cond.code] = prompt

            record_variants: Dict[str, Dict[str, object]] = {}
            for variant in instruction_variants:
                conditions = variant_conditions[variant.name]
                prompts = [format_prompt(ex, cond) for cond in conditions]
                results = score_prompts_forced_choice_batch(
                    model,
                    tokenizer,
                    prompts,
                    resolved_device,
                    token_id_a=token_id_a,
                    token_id_b=token_id_b,
                    batch_size=batch_size,
                    max_length=max_length if max_length > 0 else None,
                )

                variant_fc_correct: Dict[str, float] = {}
                fc_correct_for_metrics = dict(baseline_fc_correct)
                prompt_map: Dict[str, str] = {}
                raw_results: Dict[str, Dict[str, float]] = {}

                for cond, result, prompt in zip(conditions, results, prompts):
                    fc = _fc_correct_from_result(ex, result)
                    variant_fc_correct[cond.code] = fc
                    fc_correct_for_metrics[cond.code] = fc
                    per_variant_fc_values[variant.name][cond.code].append(fc)
                    raw_results[cond.code] = {
                        "logit_a": result.logit_a,
                        "logit_b": result.logit_b,
                        "fc_a": result.fc_a,
                        "fc_b": result.fc_b,
                    }
                    if save_prompts:
                        prompt_map[cond.code] = prompt

                metrics: Dict[str, float] = {}
                for tag in tags:
                    tag_metrics = compute_selectivity_metrics(fc_correct_for_metrics, tag)
                    metrics.update(tag_metrics)
                    for key, value in tag_metrics.items():
                        if key in per_variant_metric_values[variant.name]:
                            per_variant_metric_values[variant.name][key].append(value)

                record_variants[variant.name] = {
                    "instruction_text": variant.text,
                    "fc_correct": variant_fc_correct,
                    "selectivity_metrics": metrics,
                    "condition_results": raw_results,
                }
                if save_prompts:
                    record_variants[variant.name]["prompts"] = prompt_map

            record = {
                "model": model_id,
                "uid": ex.uid,
                "question": ex.question,
                "correct_answer": ex.correct_answer,
                "wrong_answer": ex.wrong_answer,
                "correct_label": ex.correct_label,
                "wrong_label": ex.wrong_label,
                "order_seed": _order_seed(ex.uid, seed=seed),
                "metadata": ex.metadata,
                "baseline_fc_correct": baseline_fc_correct,
                "variants": record_variants,
            }
            if save_prompts:
                record["baseline_prompts"] = baseline_prompt_map
            f.write(json.dumps(record) + "\n")

    summary_variants: Dict[str, Dict[str, object]] = {}
    for variant in instruction_variants:
        summary_variants[variant.name] = {
            "instruction_text": variant.text,
            "selectivity_metrics": {
                key: _summarize_values(values)
                for key, values in per_variant_metric_values[variant.name].items()
            },
            "per_condition_fc_correct": {
                code: _summarize_values(values)
                for code, values in per_variant_fc_values[variant.name].items()
            },
        }

    summary = {
        "model": model_id,
        "n_examples": len(examples),
        "tags": list(tags),
        "token_ids": {
            "a": token_id_a,
            "b": token_id_b,
            "a_str": tokenizer.decode([token_id_a]),
            "b_str": tokenizer.decode([token_id_b]),
        },
        "baseline_per_condition_fc_correct": {
            code: _summarize_values(values)
            for code, values in baseline_fc_values.items()
        },
        "instruction_variants": summary_variants,
    }

    summary_path = output_dir / f"{_model_tag(model_id)}_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults for {model_id}:")
    for variant in instruction_variants:
        print(f"\n  Variant: {variant.name}")
        for tag in tags:
            tag_key = normalize_tag(tag)
            metrics = summary_variants[variant.name]["selectivity_metrics"]
            eff_wrong = metrics[f"efficacy_wrong_{tag_key}"]["mean"]
            eff_correct = metrics[f"efficacy_correct_{tag_key}"]["mean"]
            selectivity = metrics[f"selectivity_{tag_key}"]["mean"]
            print(
                f"    {tag}: efficacy_wrong={eff_wrong:.4f} "
                f"efficacy_correct={eff_correct:.4f} selectivity={selectivity:.4f}"
            )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def _write_variant_table(
    *,
    output_dir: Path,
    model_summaries: Sequence[Dict],
    tags: Sequence[str],
    variants: Sequence[InstructionVariant],
) -> None:
    lines = ["model,variant,tag,efficacy_wrong,efficacy_correct,selectivity,baseline_shift"]
    for summary in model_summaries:
        model = summary["model"]
        for variant in variants:
            metrics = summary["instruction_variants"][variant.name]["selectivity_metrics"]
            for tag in tags:
                tag_key = normalize_tag(tag)
                lines.append(
                    ",".join([
                        model,
                        variant.name,
                        tag,
                        f"{metrics[f'efficacy_wrong_{tag_key}']['mean']:.10f}",
                        f"{metrics[f'efficacy_correct_{tag_key}']['mean']:.10f}",
                        f"{metrics[f'selectivity_{tag_key}']['mean']:.10f}",
                        f"{metrics[f'baseline_shift_{tag_key}']['mean']:.10f}",
                    ])
                )
    (output_dir / "variant_summary.csv").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    tags = parse_tags(args.tags)
    instruction_variants = load_instruction_variants(
        instruction_set=args.instruction_set,
        instruction_file=args.instruction_file,
    )

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

    if args.max_examples > 0 and len(examples) > args.max_examples:
        rng = random.Random(args.seed)
        shuffled = list(examples)
        rng.shuffle(shuffled)
        examples = shuffled[:args.max_examples]

    if args.uids_file is not None:
        allowed_uids = {
            line.strip()
            for line in args.uids_file.read_text().splitlines()
            if line.strip()
        }
        examples = [ex for ex in examples if ex.uid in allowed_uids]
        print(f"Filtered to {len(examples)} examples from UID subset {args.uids_file}")

    if not examples:
        raise ValueError("No examples selected after filtering.")

    print("Exp15: Diagnostic instruction sweep")
    print(f"Tags: {tags}")
    print(f"Instruction variants: {[v.name for v in instruction_variants]}")
    print(f"Dataset size: {len(examples)} examples")

    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    summaries = []
    for model_id in args.models:
        summary = _run_one_model(
            model_id=model_id,
            examples=examples,
            output_dir=args.output_dir,
            tags=tags,
            instruction_variants=instruction_variants,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            dtype=args.dtype,
            save_prompts=args.save_prompts,
            seed=args.seed,
        )
        summaries.append(summary)

    combined = {
        "experiment": "exp15_instruction_diagnostics",
        "description": (
            "Diagnostic instruction sweep over accuracy/evidence-framing variants "
            "using shared no-instruction baselines."
        ),
        "instruction_set": args.instruction_set,
        "instruction_file": str(args.instruction_file) if args.instruction_file else None,
        "instruction_variants": [
            {"name": variant.name, "instruction_text": variant.text}
            for variant in instruction_variants
        ],
        "tags_tested": tags,
        "n_examples": len(examples),
        "seed": args.seed,
        "uids_file": str(args.uids_file) if args.uids_file is not None else None,
        "models": summaries,
    }
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(combined, f, indent=2)

    _write_variant_table(
        output_dir=args.output_dir,
        model_summaries=summaries,
        tags=tags,
        variants=instruction_variants,
    )

    print(f"\n{'=' * 64}")
    print("Diagnostic instruction sweep complete")
    print(f"Results saved to {args.output_dir}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
