from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Sequence

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from src.lexical_controls.dataset_mc import load_mc_dataset
from src.lexical_controls.scoring import get_ab_token_ids
from src.endorsement.conditions import generate_all_conditions
from src.authority_steering.run_dissociation_test import DEFAULT_INSTRUCTION, _format_chat_prompt, _format_exp16_prompt
from src.mechanism.hooks_v2 import setup_h100_optimizations, try_compile_model
from src.models.llama_loader import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute baseline A/B answer margins from prompt-only logits.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--dataset-path", type=Path, required=True)
    p.add_argument("--task-name", type=str, required=True)
    p.add_argument("--condition-code", type=str, default="N0_note")
    p.add_argument("--endorsement-style", type=str, default="authoritative_verified")
    p.add_argument("--answer-suffix", type=str, default="What is the answer? Reply with exactly one token: A or B.")
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-batch-tokens", type=int, default=8192, help="Dynamic token budget per batch.")
    p.add_argument("--prep-n-jobs", type=int, default=1, help="CPU workers for prompt preparation.")
    p.add_argument("--torch-threads", type=int, default=0, help="If >0, set torch intra-op CPU threads.")
    p.add_argument("--plain-prompt", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint-path", type=Path, default=None, help="JSONL checkpoint path for row-wise progress.")
    p.add_argument("--save-every-batches", type=int, default=1)
    p.add_argument("--output-path", type=Path, required=True)
    return p.parse_args()


def _condition_map() -> Dict[str, object]:
    conds = generate_all_conditions(
        tags=["Expert", "Note", "User", "Someone online"],
        instruction_text=DEFAULT_INSTRUCTION,
    )
    return {c.code: c for c in conds}


def _prompt_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    examples = load_mc_dataset(args.dataset_path)
    if args.max_items > 0:
        examples = examples[: args.max_items]
    cond = _condition_map()[args.condition_code]

    def _build(ex) -> Dict[str, object]:
        base = _format_exp16_prompt(ex=ex, condition=cond, endorsement_style=args.endorsement_style).rstrip()
        prompt_text = f"{base}\n{args.answer_suffix}".strip()
        return {
            "uid": ex.uid,
            "prompt_text": prompt_text,
            "correct_label": ex.correct_label,
            "wrong_label": ex.wrong_label,
        }

    if args.prep_n_jobs == 1:
        return [_build(ex) for ex in examples]
    return Parallel(n_jobs=args.prep_n_jobs, prefer="threads")(delayed(_build)(ex) for ex in examples)


def _batch_indices(token_lengths: Sequence[int], max_batch_size: int, max_batch_tokens: int) -> List[tuple[int, int]]:
    batches: List[tuple[int, int]] = []
    start = 0
    while start < len(token_lengths):
        end = start + 1
        max_len = token_lengths[start]
        while end < len(token_lengths):
            cand_len = max(max_len, token_lengths[end])
            cand_bs = end - start + 1
            if cand_bs > max_batch_size or cand_len * cand_bs > max_batch_tokens:
                break
            max_len = cand_len
            end += 1
        batches.append((start, end))
        start = end
    return batches


def _quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    setup_h100_optimizations()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    rows = _prompt_rows(args)
    if not rows:
        raise SystemExit("No rows to score.")

    model, tokenizer = load_model_and_tokenizer(model_name=args.model, device="auto", dtype="auto")
    model.eval()
    if not args.no_compile:
        model = try_compile_model(model)
    token_id_a, token_id_b = get_ab_token_ids(tokenizer)

    prepared = []
    for row in rows:
        model_prompt = _format_chat_prompt(tokenizer, str(row["prompt_text"]), args.plain_prompt)
        n_tok = len(tokenizer(model_prompt, add_special_tokens=False).input_ids)
        prepared.append({**row, "model_prompt": model_prompt, "n_tokens": n_tok})
    prepared.sort(key=lambda r: int(r["n_tokens"]))
    ckpt_path = args.checkpoint_path or args.output_path.with_suffix(".rows_checkpoint.jsonl")
    processed: set[str] = set()
    per_row: List[Dict[str, object]] = []
    if args.resume and ckpt_path.exists():
        for line in ckpt_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            uid = str(row.get("uid", ""))
            if uid:
                processed.add(uid)
            per_row.append(row)
        print(f"[resume] loaded {len(per_row)} rows from {ckpt_path}")
    if processed:
        prepared = [r for r in prepared if str(r["uid"]) not in processed]
    ranges = _batch_indices([int(r["n_tokens"]) for r in prepared], args.batch_size, args.max_batch_tokens)

    for batch_idx, (start, end) in enumerate(tqdm(ranges, desc="margin"), start=1):
        chunk = prepared[start:end]
        prompts = [str(r["model_prompt"]) for r in chunk]
        model_inputs = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(**model_inputs, use_cache=False)
        logits = out.logits
        attn = model_inputs.get("attention_mask")
        if attn is not None:
            last_idx = attn.sum(dim=1) - 1
        else:
            last_idx = torch.full((logits.shape[0],), logits.shape[1] - 1, dtype=torch.long, device=logits.device)
        idx = torch.arange(logits.shape[0], device=logits.device)
        final_logits = logits[idx, last_idx, :]
        log_probs = torch.log_softmax(final_logits, dim=-1)
        for i, row in enumerate(chunk):
            logp_a = float(log_probs[i, token_id_a].item())
            logp_b = float(log_probs[i, token_id_b].item())
            margin_a_minus_b = logp_a - logp_b
            corr = str(row["correct_label"])
            wrong = str(row["wrong_label"])
            margin_correct_minus_wrong = margin_a_minus_b if corr == "A" else -margin_a_minus_b
            margin_wrong_minus_correct = -margin_correct_minus_wrong
            per_row.append(
                {
                    "uid": row["uid"],
                    "task": args.task_name,
                    "condition_code": args.condition_code,
                    "correct_label": corr,
                    "wrong_label": wrong,
                    "logp_a": logp_a,
                    "logp_b": logp_b,
                    "margin_a_minus_b": margin_a_minus_b,
                    "margin_correct_minus_wrong": margin_correct_minus_wrong,
                    "margin_wrong_minus_correct": margin_wrong_minus_correct,
                }
            )
        if args.save_every_batches > 0 and (batch_idx % args.save_every_batches == 0):
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            with ckpt_path.open("w", encoding="utf-8") as f:
                for row in per_row:
                    f.write(json.dumps(row) + "\n")

    corr_margins = [float(r["margin_correct_minus_wrong"]) for r in per_row]
    wrong_margins = [float(r["margin_wrong_minus_correct"]) for r in per_row]
    summary = {
        "task": args.task_name,
        "model": args.model,
        "dataset_path": str(args.dataset_path),
        "n_rows": len(per_row),
        "mean_margin_correct_minus_wrong": float(mean(corr_margins)),
        "median_margin_correct_minus_wrong": float(median(corr_margins)),
        "mean_margin_wrong_minus_correct": float(mean(wrong_margins)),
        "median_margin_wrong_minus_correct": float(median(wrong_margins)),
        "quantiles_correct_minus_wrong": _quantiles(corr_margins),
        "quantiles_wrong_minus_correct": _quantiles(wrong_margins),
    }
    payload = {"summary": summary, "rows": per_row}
    args.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if ckpt_path.exists():
        ckpt_path.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))
    print(f"[done] wrote {args.output_path}")


if __name__ == "__main__":
    main()
