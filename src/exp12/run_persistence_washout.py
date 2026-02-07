"""Exp12: Persistence/Washout experiment with batched scoring.

Design:
- Initial turn per tag/instruction/history: neutral, wrong, correct.
- Follow-up turns remove endorsement and re-ask with probe styles:
  same wording, paraphrase, and label-swap.
- Compare wrong-history vs neutral-history at each turn to estimate
  residual endorsement carryover after endorsement text is gone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from src.exp7.dataset_mc import MCExample, create_mc_examples, load_mc_dataset, save_mc_dataset
from src.exp7.scoring import (
    ForcedChoiceResult,
    get_ab_token_ids,
    score_prompts_forced_choice_batch,
)
from src.exp10.conditions import INSTRUCTION_TEXT
from src.exp12.conditions import (
    DEFAULT_TAGS,
    EXTENDED_TAGS,
    HISTORY_CODE,
    BranchKey,
    format_fresh_probe_prompt,
    format_initial_prompt,
    format_probe_context_block,
    initial_context_from_prompt,
    normalize_tag,
    parse_probe_modes,
    parse_probe_styles,
    probe_correct_label,
)
from src.models.llama_loader import load_model_and_tokenizer


DEFAULT_MODELS: Tuple[str, ...] = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp12: persistence/washout test")
    parser.add_argument("--models", type=str, nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--data-path", type=Path, default=Path("data/answer.jsonl"))
    parser.add_argument("--mc-dataset-path", type=Path, default=Path("data/exp7_mc_dataset.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp12"))
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extended-tags", action="store_true")
    parser.add_argument("--probe-modes", type=str, default="context,fresh")
    parser.add_argument("--probe-styles", type=str, default="same,paraphrase,swap")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=0, help="Optional max prompt length for scoring")
    parser.add_argument(
        "--no-correct-history",
        action="store_true",
        help="Skip correct-history branches in follow-up turns (faster, less complete).",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Include full prompt text in per-example JSONL (large files).",
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


def _order_seed(uid: str, seed: int) -> int:
    digest = hashlib.md5(f"{seed}:{uid}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _pred_label(result: ForcedChoiceResult) -> str:
    return "A" if result.fc_a >= result.fc_b else "B"


def _result_to_entry(result: ForcedChoiceResult, expected_correct_label: str) -> Dict[str, float | str]:
    if expected_correct_label == "A":
        fc_correct = result.fc_a
        fc_wrong = result.fc_b
        margin_correct = result.logit_a
        margin_wrong = result.logit_b
    else:
        fc_correct = result.fc_b
        fc_wrong = result.fc_a
        margin_correct = result.logit_b
        margin_wrong = result.logit_a

    return {
        "logit_a": result.logit_a,
        "logit_b": result.logit_b,
        "fc_a": result.fc_a,
        "fc_b": result.fc_b,
        "fc_correct": fc_correct,
        "fc_wrong": fc_wrong,
        "margin_correct": margin_correct,
        "margin_wrong": margin_wrong,
        "pred_label": _pred_label(result),
        "expected_correct_label": expected_correct_label,
    }


def _summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n": 0,
            "positive_frac": 0.0,
        }
    vals = list(values)
    n_pos = sum(1 for v in vals if v > 0)
    return {
        "mean": float(mean(vals)),
        "median": float(median(vals)),
        "std": float(stdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
        "n": len(vals),
        "positive_frac": n_pos / len(vals),
    }


def _safe_ratio(numer: Sequence[float], denom: Sequence[float], tau: float = 1e-6) -> List[float]:
    out: List[float] = []
    for n, d in zip(numer, denom):
        if abs(d) <= tau:
            continue
        out.append(n / d)
    return out


def _score_prompts(
    model,
    tokenizer,
    device: torch.device,
    token_id_a: int,
    token_id_b: int,
    prompts: List[str],
    batch_size: int,
    max_length: Optional[int],
) -> List[ForcedChoiceResult]:
    return score_prompts_forced_choice_batch(
        model=model,
        tokenizer=tokenizer,
        prompt_texts=prompts,
        device=device,
        token_id_a=token_id_a,
        token_id_b=token_id_b,
        batch_size=batch_size,
        max_length=max_length,
    )


def _extract_margins(records: List[Dict], key: str, field: str, section: str) -> List[float]:
    vals = []
    for r in records:
        vals.append(float(r[section][key][field]))
    return vals


def _compute_summary(
    records: List[Dict],
    tags: List[str],
    modes: List[str],
    styles: List[str],
    turns: int,
    include_correct_history: bool,
) -> Dict:
    summary: Dict[str, Dict] = {}
    for tag in tags:
        tag_key = normalize_tag(tag)
        tag_out: Dict[str, Dict] = {}
        for instruction in (0, 1):
            i_key = f"instr_{instruction}"
            n_code = f"N{instruction}_{tag_key}"
            w_code = f"W{instruction}_{tag_key}"
            c_code = f"C{instruction}_{tag_key}"

            m_n = _extract_margins(records, n_code, "margin_correct", "init_results")
            m_w = _extract_margins(records, w_code, "margin_correct", "init_results")
            m_c = _extract_margins(records, c_code, "margin_correct", "init_results")

            immediate_wrong = [w - n for w, n in zip(m_w, m_n)]
            immediate_correct = [c - n for c, n in zip(m_c, m_n)]

            i_out: Dict[str, Dict] = {
                "initial": {
                    "neutral_margin": _summarize(m_n),
                    "wrong_margin": _summarize(m_w),
                    "correct_margin": _summarize(m_c),
                    "immediate_wrong_shift": _summarize(immediate_wrong),
                    "immediate_correct_shift": _summarize(immediate_correct),
                    "wrong_prior_frac_neutral": float(sum(1 for v in m_n if v < 0) / len(m_n)) if m_n else 0.0,
                },
                "probes": {},
            }

            for mode in modes:
                mode_prefix = "CTX" if mode == "context" else "FRESH"
                mode_out: Dict[str, Dict] = {}
                for style in styles:
                    s_out: Dict[str, Dict] = {}
                    for turn in range(1, turns + 1):
                        key_n = f"{mode_prefix}_HN{instruction}_{tag_key}_{style}_T{turn}"
                        key_w = f"{mode_prefix}_HW{instruction}_{tag_key}_{style}_T{turn}"
                        m_hn = _extract_margins(records, key_n, "margin_correct", "probe_results")
                        m_hw = _extract_margins(records, key_w, "margin_correct", "probe_results")
                        residual_wrong = [w - n for w, n in zip(m_hw, m_hn)]
                        ratio_wrong = _safe_ratio(residual_wrong, immediate_wrong)
                        washout_wrong = [1.0 - min(1.0, abs(r)) for r in ratio_wrong]

                        turn_out: Dict[str, Dict | float] = {
                            "neutral_probe_margin": _summarize(m_hn),
                            "wrong_probe_margin": _summarize(m_hw),
                            "residual_wrong_shift": _summarize(residual_wrong),
                            "persistence_ratio_wrong": _summarize(ratio_wrong),
                            "washout_score_wrong": _summarize(washout_wrong),
                        }

                        wrong_prior_idx = [idx for idx, v in enumerate(m_n) if v < 0]
                        if wrong_prior_idx:
                            resid_wp = [residual_wrong[idx] for idx in wrong_prior_idx]
                            hn_wp = [m_hn[idx] for idx in wrong_prior_idx]
                            hw_wp = [m_hw[idx] for idx in wrong_prior_idx]
                            turn_out["wrong_prior_slice"] = {
                                "n": len(wrong_prior_idx),
                                "residual_wrong_shift": _summarize(resid_wp),
                                "neutral_probe_flip_rate": float(sum(1 for v in hn_wp if v > 0) / len(hn_wp)),
                                "wrong_probe_flip_rate": float(sum(1 for v in hw_wp if v > 0) / len(hw_wp)),
                            }

                        if include_correct_history:
                            key_c = f"{mode_prefix}_HC{instruction}_{tag_key}_{style}_T{turn}"
                            m_hc = _extract_margins(records, key_c, "margin_correct", "probe_results")
                            residual_correct = [c - n for c, n in zip(m_hc, m_hn)]
                            ratio_correct = _safe_ratio(residual_correct, immediate_correct)
                            turn_out["correct_probe_margin"] = _summarize(m_hc)
                            turn_out["residual_correct_shift"] = _summarize(residual_correct)
                            turn_out["persistence_ratio_correct"] = _summarize(ratio_correct)
                            turn_out["residual_selectivity"] = _summarize(
                                [rc - rw for rc, rw in zip(residual_correct, residual_wrong)]
                            )

                        s_out[f"T{turn}"] = turn_out
                    mode_out[style] = s_out
                i_out["probes"][mode] = mode_out
            tag_out[i_key] = i_out
        summary[tag_key] = tag_out
    return summary


def run_for_model(
    model_id: str,
    examples: List[MCExample],
    output_dir: Path,
    seed: int,
    tags: List[str],
    modes: List[str],
    styles: List[str],
    turns: int,
    include_correct_history: bool,
    save_prompts: bool,
    batch_size: int,
    max_length: Optional[int],
) -> Dict:
    print(f"\n{'=' * 70}")
    print(f"Running Exp12 model: {model_id}")
    print(f"Tags: {tags}")
    print(f"Modes: {modes}")
    print(f"Styles: {styles}, turns: {turns}")
    print(f"{'=' * 70}")

    model, tokenizer = load_model_and_tokenizer(model_name=model_id, device="auto", dtype="auto")
    model.eval()
    device = next(model.parameters()).device

    token_id_a, token_id_b = get_ab_token_ids(tokenizer)
    print(f"Token A: {tokenizer.decode([token_id_a])!r} (id={token_id_a})")
    print(f"Token B: {tokenizer.decode([token_id_b])!r} (id={token_id_b})")

    records: List[Dict] = []
    for ex in examples:
        rec = {
            "model": model_id,
            "uid": ex.uid,
            "correct_label": ex.correct_label,
            "wrong_label": ex.wrong_label,
            "order_seed": _order_seed(ex.uid, seed),
            "metadata": ex.metadata,
            "init_results": {},
            "probe_results": {},
        }
        if save_prompts:
            rec["prompts"] = {}
        records.append(rec)

    init_histories = ["neutral", "wrong", "correct"]
    history_for_probes = ["neutral", "wrong"] + ([] if not include_correct_history else ["correct"])

    print("\nScoring initial turn conditions...")
    init_specs = []
    for tag in tags:
        tag_key = normalize_tag(tag)
        for instruction in (0, 1):
            for history in init_histories:
                code = f"{HISTORY_CODE[history]}{instruction}_{tag_key}"
                init_specs.append((tag, tag_key, instruction, history, code))

    init_prompt_cache: Dict[Tuple[str, int, str], List[str]] = {}
    for tag, tag_key, instruction, history, code in tqdm(init_specs, desc=f"Exp12 init {model_id}"):
        prompts = [
            format_initial_prompt(ex, tag, history, bool(instruction))
            for ex in examples
        ]
        init_prompt_cache[(tag_key, instruction, history)] = prompts
        scored = _score_prompts(
            model, tokenizer, device, token_id_a, token_id_b,
            prompts, batch_size=batch_size, max_length=max_length,
        )
        for i, result in enumerate(scored):
            entry = _result_to_entry(result, expected_correct_label=examples[i].correct_label)
            records[i]["init_results"][code] = entry
            if save_prompts:
                records[i]["prompts"][code] = prompts[i]

    branch_keys: List[BranchKey] = []
    branch_contexts: Dict[str, List[str]] = {}
    if "context" in modes:
        print("\nBuilding context-mode branch histories...")
        for tag in tags:
            tag_key = normalize_tag(tag)
            for instruction in (0, 1):
                for history in history_for_probes:
                    for style in styles:
                        b = BranchKey(tag=tag_key, history=history, instruction=instruction, style=style)  # type: ignore[arg-type]
                        branch_keys.append(b)
                        init_code = f"{HISTORY_CODE[history]}{instruction}_{tag_key}"
                        prompts = init_prompt_cache[(tag_key, instruction, history)]
                        contexts = []
                        for i, prompt in enumerate(prompts):
                            pred = records[i]["init_results"][init_code]["pred_label"]
                            contexts.append(initial_context_from_prompt(prompt, str(pred)))
                        branch_contexts[b.key] = contexts

    print("\nScoring persistence/washout follow-up turns...")
    for turn in range(1, turns + 1):
        if "context" in modes:
            for b in tqdm(branch_keys, desc=f"Exp12 context probes {model_id} T{turn}"):
                prompts: List[str] = []
                context_blocks: List[str] = []
                expected_labels: List[str] = []
                contexts = branch_contexts[b.key]
                style = b.style

                for i, ex in enumerate(examples):
                    block = format_probe_context_block(ex, style, turn)
                    prompt = f"{contexts[i]}{block}\nAnswer:"
                    prompts.append(prompt)
                    context_blocks.append(block)
                    expected_labels.append(probe_correct_label(ex, style))

                scored = _score_prompts(
                    model, tokenizer, device, token_id_a, token_id_b,
                    prompts, batch_size=batch_size, max_length=max_length,
                )

                for i, result in enumerate(scored):
                    probe_code = f"CTX_{b.key}_T{turn}"
                    entry = _result_to_entry(result, expected_correct_label=expected_labels[i])
                    entry["mode"] = "context"
                    entry["turn"] = turn
                    entry["style"] = style
                    entry["history"] = b.history
                    entry["instruction"] = b.instruction
                    entry["tag"] = b.tag
                    records[i]["probe_results"][probe_code] = entry
                    if save_prompts:
                        records[i]["prompts"][probe_code] = prompts[i]

                    pred = str(entry["pred_label"])
                    contexts[i] = f"{contexts[i]}{context_blocks[i]}\nAssistant: {pred}\n"

        if "fresh" in modes:
            fresh_specs = [(instruction, style) for instruction in (0, 1) for style in styles]
            for instruction, style in tqdm(fresh_specs, desc=f"Exp12 fresh probes {model_id} T{turn}"):
                prompts: List[str] = []
                expected_labels: List[str] = []
                for ex in examples:
                    prompts.append(format_fresh_probe_prompt(ex, style=style, turn_idx=turn, instruction=bool(instruction)))
                    expected_labels.append(probe_correct_label(ex, style))

                scored = _score_prompts(
                    model, tokenizer, device, token_id_a, token_id_b,
                    prompts, batch_size=batch_size, max_length=max_length,
                )

                for i, result in enumerate(scored):
                    base_entry = _result_to_entry(result, expected_correct_label=expected_labels[i])
                    base_entry["mode"] = "fresh"
                    base_entry["turn"] = turn
                    base_entry["style"] = style
                    base_entry["instruction"] = instruction

                    for tag in tags:
                        tag_key = normalize_tag(tag)
                        for history in history_for_probes:
                            probe_code = f"FRESH_H{HISTORY_CODE[history]}{instruction}_{tag_key}_{style}_T{turn}"
                            entry = dict(base_entry)
                            entry["tag"] = tag_key
                            entry["history"] = history
                            records[i]["probe_results"][probe_code] = entry
                            if save_prompts:
                                records[i]["prompts"][probe_code] = prompts[i]

    summary_metrics = _compute_summary(
        records=records,
        tags=tags,
        modes=modes,
        styles=styles,
        turns=turns,
        include_correct_history=include_correct_history,
    )

    out_path = output_dir / f"{_model_tag(model_id)}_results.jsonl"
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    model_summary = {
        "model": model_id,
        "n_examples": len(examples),
        "tags": tags,
        "probe_modes": modes,
        "styles": styles,
        "turns": turns,
        "include_correct_history": include_correct_history,
        "instruction_text": INSTRUCTION_TEXT,
        "batch_size": batch_size,
        "max_length": max_length if max_length is not None else 0,
        "token_ids": {
            "a": token_id_a,
            "b": token_id_b,
            "a_str": tokenizer.decode([token_id_a]),
            "b_str": tokenizer.decode([token_id_b]),
        },
        "metrics": summary_metrics,
    }

    summary_path = output_dir / f"{_model_tag(model_id)}_summary.json"
    with summary_path.open("w") as f:
        json.dump(model_summary, f, indent=2)

    print(f"\nKey metrics for {model_id}:")
    for tag in tags:
        tag_key = normalize_tag(tag)
        m = model_summary["metrics"][tag_key]["instr_1"]["initial"]["immediate_wrong_shift"]["mean"]
        print(f"  {tag}: immediate_wrong_shift (I1 mean margin delta) = {m:.4f}")
        for mode in modes:
            for style in styles:
                resid = model_summary["metrics"][tag_key]["instr_1"]["probes"][mode][style]["T1"]["residual_wrong_shift"]["mean"]
                print(f"    {mode} {style} T1 residual_wrong_shift: {resid:.4f}")

    del model
    torch.cuda.empty_cache()
    return model_summary


def main() -> None:
    args = parse_args()
    if args.turns <= 0:
        raise ValueError("--turns must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    tags = list(EXTENDED_TAGS) if args.extended_tags else list(DEFAULT_TAGS)
    modes = parse_probe_modes(args.probe_modes)
    styles = parse_probe_styles(args.probe_styles)
    include_correct_history = not args.no_correct_history
    max_length = args.max_length if args.max_length > 0 else None

    print("Exp12: Persistence/Washout")
    print(f"Tags: {tags}")
    print(f"Probe modes: {modes}")
    print(f"Styles: {styles}")
    print(f"Turns: {args.turns}")
    print(f"Include correct history: {include_correct_history}")

    mc_path = args.mc_dataset_path
    if mc_path.exists():
        print(f"Loading existing MC dataset from {mc_path}")
        examples = load_mc_dataset(mc_path)
    else:
        raw_path = _resolve_data_path(args.data_path)
        print(f"Creating MC dataset from {raw_path}")
        examples = create_mc_examples(raw_path, seed=args.seed)
        save_mc_dataset(examples, mc_path)
        print(f"Saved MC dataset to {mc_path}")

    if args.max_examples > 0 and len(examples) > args.max_examples:
        rng = random.Random(args.seed)
        shuffled = list(examples)
        rng.shuffle(shuffled)
        examples = shuffled[: args.max_examples]

    print(f"Dataset: {len(examples)} examples")
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
        summary = run_for_model(
            model_id=model_id,
            examples=examples,
            output_dir=args.output_dir,
            seed=args.seed,
            tags=tags,
            modes=modes,
            styles=styles,
            turns=args.turns,
            include_correct_history=include_correct_history,
            save_prompts=args.save_prompts,
            batch_size=args.batch_size,
            max_length=max_length,
        )
        summaries.append(summary)

    combined = {
        "experiment": "exp12_persistence_washout",
        "description": "Turn-wise persistence test for endorsement-induced carryover after endorsement removal.",
        "n_examples": len(examples),
        "seed": args.seed,
        "tags": tags,
        "probe_modes": modes,
        "styles": styles,
        "turns": args.turns,
        "include_correct_history": include_correct_history,
        "models": summaries,
    }
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Exp12 complete")
    print(f"Results saved to {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
