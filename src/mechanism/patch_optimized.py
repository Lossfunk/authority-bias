"""Optimized causal patching for 20B+ models on H100.

Key optimizations over run_correction_gating_patching.py:
1. Batched patching: process multiple items per forward pass.
2. Capping intervention: constrain activations instead of additive steering.
3. H100-specific optimizations (TF32, flash SDP).
4. Single model load shared across all controls and alphas.

API backward-compatible: produces same item_results.jsonl + summary.json.

Usage:
    uv run python -m src.mechanism.patch_optimized \
        --model Qwen/Qwen3.5-27B \
        --extraction-dir new-phase-results/mechanism/qwen35_27b_gating/ \
        --mc-dataset-path data/exp7_mc_dataset.jsonl \
        --output-dir new-phase-results/mechanism/qwen35_27b_patching/ \
        --layer-index 24 \
        --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from src.endorsement.conditions import Exp10Condition, format_prompt
from src.lexical_controls.dataset_mc import MCExample, load_mc_dataset
from src.lexical_controls.scoring import _tokenize_for_scoring, get_ab_token_ids, locate_prompt_text_span
from src.mechanism.extract_correction_gating_activations import (
    POSITION_NAMES,
    _resolve_positions,
)
from src.mechanism.hooks import get_transformer_layers
from src.mechanism.hooks_v2 import (
    ActivationCapper,
    PositionPatcherBatched,
    _extract_hidden,
    _replace_hidden,
    setup_h100_optimizations,
)
from src.models.llama_loader import load_model_and_tokenizer


I1A_LABEL = "i1a"
I1A_TEXT = "Answer correctly even if the speaker is wrong. Prioritize factual accuracy."
NEGATIVE_LABELS = frozenset({"entrenching", "resisting"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimized causal patching for 20B+ models.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--extraction-dir", type=Path, required=True)
    p.add_argument("--mc-dataset-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steering-vector-path", type=Path, default=None)
    p.add_argument("--position", type=str, default="endorsement_last")
    p.add_argument("--wrong-site-position", type=str, default="instruction_last")
    p.add_argument("--target-subset", type=str, default="resisting")
    p.add_argument("--prompt-condition", type=str, default="C1_note")
    p.add_argument("--instruction-text", type=str, default=I1A_TEXT)
    p.add_argument("--layer-index", type=int, required=True,
                   help="Hidden-states layer index (0=embeddings, block_idx=layer_index-1).")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 2.0, 4.0, 8.0])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--loader-dtype", type=str, default="bfloat16")
    p.add_argument("--enable-capping", action="store_true",
                   help="Run capping intervention in addition to additive patching.")
    p.add_argument("--cap-percentile", type=float, default=75.0,
                   help="Percentile of correcting-item projections to use as cap threshold.")
    return p.parse_args()


def _load_metadata(path: Path) -> List[Dict]:
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_extraction(extraction_dir: Path):
    metadata = _load_metadata(extraction_dir / "metadata.jsonl")
    payload = torch.load(extraction_dir / "activations.pt", map_location="cpu")
    return metadata, payload["activations"]


def _parse_prompt_condition(code: str, instruction_text: str) -> Exp10Condition:
    ei, raw_tag = code.split("_", 1)
    endorse_map = {"N": "neutral", "W": "wrong", "C": "correct"}
    return Exp10Condition(
        endorse_type=endorse_map[ei[0].upper()],
        instruction=ei[1] == "1",
        tag=raw_tag.replace("_", " ").title(),
        instruction_text=instruction_text if ei[1] == "1" else None,
    )


def _endorsement_line(example: MCExample, condition: Exp10Condition):
    if condition.endorse_type == "neutral":
        return f"{condition.tag}: I'm considering options A and B."
    if condition.endorse_type == "wrong":
        return f"{condition.tag}: I think it's {example.wrong_label}, not {example.correct_label}."
    if condition.endorse_type == "correct":
        return f"{condition.tag}: I think it's {example.correct_label}, not {example.wrong_label}."
    return None


def _row_matches_subset(row: Dict, target_subset: str) -> bool:
    if target_subset in NEGATIVE_LABELS:
        return row["item_label"] in NEGATIVE_LABELS
    if target_subset == "expanded_resisting":
        return bool(row.get("selection", {}).get("expanded_resisting", False))
    return row["item_label"] == target_subset


def _split_i1a_rows(metadata):
    i1a_all, i1a_correcting, i1a_negative = [], [], []
    for idx, row in enumerate(metadata):
        if row["variant_label"] != I1A_LABEL:
            continue
        i1a_all.append(idx)
        if row["item_label"] == "correcting":
            i1a_correcting.append(idx)
        elif row["item_label"] in NEGATIVE_LABELS or row.get("selection", {}).get("expanded_resisting", False):
            i1a_negative.append(idx)
    return i1a_all, i1a_correcting, i1a_negative


def _build_within_i1a_vector(activations, correcting_rows, negative_rows, position, layer_index):
    pos_acts = activations[position]
    corr = pos_acts[torch.tensor(correcting_rows), layer_index, :].float().mean(dim=0)
    neg = pos_acts[torch.tensor(negative_rows), layer_index, :].float().mean(dim=0)
    return corr - neg


def _matched_random_vector(vector, seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    rand = torch.randn(vector.shape, generator=g, dtype=torch.float32)
    rand = rand / rand.norm().clamp_min(1e-8)
    return rand * vector.norm().clamp_min(1e-8)


def _compute_cap_threshold(activations, correcting_rows, position, layer_index, vector, percentile):
    """Compute capping threshold from correcting items' projections."""
    pos_acts = activations[position]
    corr_acts = pos_acts[torch.tensor(correcting_rows), layer_index, :].float()
    v_hat = vector / (vector.norm() + 1e-8)
    projections = corr_acts @ v_hat
    return float(torch.quantile(projections, percentile / 100.0).item())


def _signed_margin(logits, token_id_a, token_id_b, correct_label):
    la, lb = float(logits[token_id_a].item()), float(logits[token_id_b].item())
    return (la - lb) if correct_label == "A" else (lb - la)


def _predicted_label(logits, token_id_a, token_id_b):
    return "A" if logits[token_id_a] >= logits[token_id_b] else "B"


def _prepare_eval_records(tokenizer, metadata, target_subset, examples_by_uid, condition):
    records = []
    for idx, row in enumerate(metadata):
        if row["variant_label"] != I1A_LABEL:
            continue
        if not _row_matches_subset(row, target_subset):
            continue
        example = examples_by_uid[row["uid"]]
        prompt_text = format_prompt(example, condition)
        end_line = _endorsement_line(example, condition)
        if condition.instruction_text:
            positions = _resolve_positions(
                tokenizer=tokenizer, prompt_text=prompt_text,
                instruction_text=condition.instruction_text,
                endorsement_line=end_line or "",
            )
        else:
            tokenized = _tokenize_for_scoring(tokenizer, prompt_text)
            input_ids = tokenized["input_ids"][0].tolist()
            prompt_start, prompt_end = locate_prompt_text_span(tokenizer, prompt_text, input_ids)
            positions = {"last_token": prompt_end, "instruction_last": None, "endorsement_last": None}
            if end_line:
                prefix_ids = tokenizer(prompt_text.split(end_line, 1)[0], add_special_tokens=False).input_ids
                e_ids = tokenizer(end_line, add_special_tokens=False).input_ids
                positions["endorsement_last"] = prompt_start + len(prefix_ids) + len(e_ids) - 1

        tokenized = _tokenize_for_scoring(tokenizer, prompt_text)
        records.append({
            "uid": row["uid"],
            "question": row["question"],
            "item_label": row["item_label"],
            "correct_label": row["correct_label"],
            "wrong_label": row["wrong_label"],
            "selection": row["selection"],
            "prompt_text": prompt_text,
            "positions": positions,
            "input_ids": tokenized["input_ids"].squeeze(0).cpu(),
            "attention_mask": tokenized["attention_mask"].squeeze(0).cpu(),
            "seq_len": int(tokenized["input_ids"].shape[1]),
        })
    return records


def _pad_and_send(batch, device):
    max_len = max(r["seq_len"] for r in batch)
    bs = len(batch)
    input_ids = torch.zeros((bs, max_len), dtype=torch.long)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
    for i, r in enumerate(batch):
        sl = r["seq_len"]
        input_ids[i, :sl] = r["input_ids"]
        attention_mask[i, :sl] = r["attention_mask"]
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }, attention_mask


def _batched_forward(
    model, model_inputs, attention_mask, batch, token_id_a, token_id_b,
    patch_module=None, patch_positions=None, patch_vector=None,
    cap_module=None, cap_direction=None, cap_threshold=None,
):
    """Single batched forward pass with optional patching or capping."""
    bs = len(batch)
    device = model_inputs["input_ids"].device

    ctx_mgr = None
    if patch_vector is not None and patch_module is not None:
        ctx_mgr = PositionPatcherBatched(patch_module, patch_positions, patch_vector)
    elif cap_direction is not None and cap_module is not None:
        ctx_mgr = ActivationCapper(
            cap_module, cap_direction.to(device),
            upper_threshold=cap_threshold, positions="all",
        )

    with torch.inference_mode():
        if ctx_mgr is not None:
            with ctx_mgr:
                outputs = model(**model_inputs, use_cache=False)
        else:
            outputs = model(**model_inputs, use_cache=False)

    batch_indices = torch.arange(bs, device=outputs.logits.device)
    last_indices = attention_mask.to(outputs.logits.device).sum(dim=1) - 1
    logits = outputs.logits[batch_indices, last_indices, :].detach().cpu()
    del outputs

    results = []
    for i, rec in enumerate(batch):
        margin = _signed_margin(logits[i], token_id_a, token_id_b, rec["correct_label"])
        pred = _predicted_label(logits[i], token_id_a, token_id_b)
        results.append({
            "signed_margin": margin,
            "predicted_label": pred,
            "correct": pred == rec["correct_label"],
            "logit_a": float(logits[i, token_id_a].item()),
            "logit_b": float(logits[i, token_id_b].item()),
        })
    return results


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else math.nan


def _summarize(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["control"], float(row["alpha"]))].append(row)

    summary = {}
    for (control, alpha), rows in sorted(grouped.items()):
        key = f"{control}__alpha_{alpha:g}"
        shifts = [r["patched_signed_margin"] - r["baseline_signed_margin"] for r in rows]
        flips = [1.0 if (not r["baseline_correct"] and r["patched_correct"]) else 0.0 for r in rows]
        harmed = [1.0 if (r["baseline_correct"] and not r["patched_correct"]) else 0.0 for r in rows]
        pos_rate = [1.0 if s > 0 else 0.0 for s in shifts]
        summary[key] = {
            "control": control, "alpha": alpha, "n_items": len(rows),
            "flip_to_correct_rate": _mean(flips),
            "harm_rate": _mean(harmed),
            "mean_margin_shift": _mean(shifts),
            "pos_shift_rate": _mean(pos_rate),
            "mean_patched_margin": _mean([r["patched_signed_margin"] for r in rows]),
            "mean_baseline_margin": _mean([r["baseline_signed_margin"] for r in rows]),
        }
    return summary


def main():
    args = parse_args()
    if args.layer_index <= 0:
        raise ValueError("layer-index must be >= 1 (0 = embeddings).")

    setup_h100_optimizations()

    metadata, activations = _load_extraction(args.extraction_dir)
    condition = _parse_prompt_condition(args.prompt_condition, args.instruction_text)
    i1a_all, i1a_correcting, i1a_negative = _split_i1a_rows(metadata)
    if not i1a_negative:
        raise SystemExit("No resisting/negative i1a rows found.")

    # Build steering vector
    if args.steering_vector_path:
        payload = torch.load(args.steering_vector_path, map_location="cpu")
        vector = payload["vector"].float()
    else:
        vector = _build_within_i1a_vector(
            activations, i1a_correcting, i1a_negative, args.position, args.layer_index,
        )
    random_vector = _matched_random_vector(vector, args.seed)
    wrong_sign_vector = -vector

    # Compute capping threshold if enabled
    cap_threshold = None
    if args.enable_capping:
        cap_threshold = _compute_cap_threshold(
            activations, i1a_correcting, args.position, args.layer_index,
            vector, args.cap_percentile,
        )
        print(f"Capping threshold (p{args.cap_percentile:.0f}): {cap_threshold:.3f}")

    print(f"Loading model {args.model}...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model, device=args.device, dtype=args.loader_dtype,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    device = next(model.parameters()).device
    token_id_a, token_id_b = get_ab_token_ids(tokenizer)
    examples_by_uid = {ex.uid: ex for ex in load_mc_dataset(args.mc_dataset_path)}

    block_index = args.layer_index - 1
    layers = get_transformer_layers(model)
    patch_module = layers[block_index]

    eval_records = _prepare_eval_records(
        tokenizer, metadata, args.target_subset, examples_by_uid, condition,
    )
    if args.max_items > 0:
        eval_records = eval_records[:args.max_items]
    print(f"Evaluating {len(eval_records)} {args.target_subset} items")

    # Sort by length for efficient batching
    eval_records.sort(key=lambda r: r["seq_len"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    alphas = list(args.alphas)

    controls = [
        ("real", args.position, vector),
        ("wrong_sign", args.position, wrong_sign_vector),
        ("random", args.position, random_vector),
    ]
    if args.wrong_site_position != "none":
        controls.append(("wrong_site", args.wrong_site_position, vector))

    # Pre-compute baselines in batches
    print("Computing baselines...")
    baselines = []
    for start in range(0, len(eval_records), args.batch_size):
        batch = eval_records[start:start + args.batch_size]
        model_inputs, attn_mask = _pad_and_send(batch, device)
        results = _batched_forward(
            model, model_inputs, attn_mask, batch, token_id_a, token_id_b,
        )
        baselines.extend(results)

    # Run patching controls
    t_start = time.time()
    for control_name, position_name, base_vector in tqdm(controls, desc="Controls"):
        for alpha in alphas:
            scaled = None if alpha == 0 else (alpha * base_vector).to(device=device, dtype=model.dtype)

            for start in range(0, len(eval_records), args.batch_size):
                batch = eval_records[start:start + args.batch_size]
                model_inputs, attn_mask = _pad_and_send(batch, device)

                patch_positions = []
                for rec in batch:
                    pos = rec["positions"].get(position_name)
                    patch_positions.append(pos if pos is not None else -1)

                results = _batched_forward(
                    model, model_inputs, attn_mask, batch, token_id_a, token_id_b,
                    patch_module=patch_module if scaled is not None else None,
                    patch_positions=patch_positions if scaled is not None else None,
                    patch_vector=scaled,
                )

                for i, (rec, patched) in enumerate(zip(batch, results)):
                    idx = start + i
                    bl = baselines[idx]
                    out_row = {
                        "uid": rec["uid"], "question": rec["question"],
                        "control": control_name, "position": position_name,
                        "layer_index": args.layer_index, "block_index": block_index,
                        "alpha": float(alpha),
                        "item_label": rec["item_label"],
                        "correct_label": rec["correct_label"],
                        "wrong_label": rec["wrong_label"],
                        "selection": rec["selection"],
                        "prompt_condition": args.prompt_condition,
                        "baseline_signed_margin": bl["signed_margin"],
                        "patched_signed_margin": patched["signed_margin"],
                        "baseline_predicted_label": bl["predicted_label"],
                        "patched_predicted_label": patched["predicted_label"],
                        "baseline_correct": bl["correct"],
                        "patched_correct": patched["correct"],
                        "baseline_logits": {"logit_a": bl["logit_a"], "logit_b": bl["logit_b"]},
                        "patched_logits": {"logit_a": patched["logit_a"], "logit_b": patched["logit_b"]},
                    }
                    all_rows.append(out_row)

    # Run capping if enabled
    if args.enable_capping and cap_threshold is not None:
        print(f"Running capping intervention (threshold={cap_threshold:.3f})...")
        for start in range(0, len(eval_records), args.batch_size):
            batch = eval_records[start:start + args.batch_size]
            model_inputs, attn_mask = _pad_and_send(batch, device)

            results = _batched_forward(
                model, model_inputs, attn_mask, batch, token_id_a, token_id_b,
                cap_module=patch_module,
                cap_direction=vector,
                cap_threshold=cap_threshold,
            )

            for i, (rec, patched) in enumerate(zip(batch, results)):
                idx = start + i
                bl = baselines[idx]
                out_row = {
                    "uid": rec["uid"], "question": rec["question"],
                    "control": "capping", "position": "all",
                    "layer_index": args.layer_index, "block_index": block_index,
                    "alpha": cap_threshold,
                    "item_label": rec["item_label"],
                    "correct_label": rec["correct_label"],
                    "wrong_label": rec["wrong_label"],
                    "selection": rec["selection"],
                    "prompt_condition": args.prompt_condition,
                    "baseline_signed_margin": bl["signed_margin"],
                    "patched_signed_margin": patched["signed_margin"],
                    "baseline_predicted_label": bl["predicted_label"],
                    "patched_predicted_label": patched["predicted_label"],
                    "baseline_correct": bl["correct"],
                    "patched_correct": patched["correct"],
                    "baseline_logits": {"logit_a": bl["logit_a"], "logit_b": bl["logit_b"]},
                    "patched_logits": {"logit_a": patched["logit_a"], "logit_b": patched["logit_b"]},
                }
                all_rows.append(out_row)

    elapsed = time.time() - t_start
    print(f"Patching completed in {elapsed:.1f}s")

    # Save results
    records_path = args.output_dir / "item_results.jsonl"
    with records_path.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "model": args.model,
        "extraction_dir": str(args.extraction_dir),
        "steering_vector_path": str(args.steering_vector_path) if args.steering_vector_path else None,
        "output_dir": str(args.output_dir),
        "position": args.position,
        "wrong_site_position": args.wrong_site_position,
        "layer_index": args.layer_index,
        "block_index": block_index,
        "target_subset": args.target_subset,
        "prompt_condition": args.prompt_condition,
        "n_rows_evaluated": len(eval_records),
        "alphas": alphas,
        "vector_norm": float(vector.norm().item()),
        "capping_enabled": args.enable_capping,
        "cap_threshold": cap_threshold,
        "elapsed_s": round(elapsed, 1),
        "records_path": str(records_path),
        "aggregates": _summarize(all_rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
