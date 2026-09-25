"""Optimized activation extraction for 20B+ models on H100.

Key optimizations over extract_correction_gating_activations.py:
1. Hook-based selective layer capture (no output_hidden_states=True).
2. bf16 native computation, fp16 save.
3. Dynamic batch sizing based on sequence length.
4. Optional torch.compile for repeated forward passes.
5. Pin-memory async CPU transfers.

API is backward-compatible: produces the same activations.pt and metadata.jsonl
format as the original, so downstream scripts (sweep_probe_layers, patching, etc.)
work unchanged.

Usage:
    uv run python -m src.mechanism.extract_optimized \
        --model Qwen/Qwen3.5-27B \
        --results-path new-phase-results/.../results.jsonl \
        --mc-dataset-path data/exp7_mc_dataset.jsonl \
        --output-dir new-phase-results/mechanism/qwen35_27b_gating/ \
        --target-layers 20 21 22 23 24 25 26 27 28 \
        --max-batch-tokens 8192 \
        --loader-dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from src.endorsement.conditions import Exp10Condition, format_prompt
from src.lexical_controls.dataset_mc import MCExample, load_mc_dataset
from src.lexical_controls.scoring import _tokenize_for_scoring, get_ab_token_ids, locate_prompt_text_span
from src.mechanism.extract_correction_gating_activations import (
    POSITION_NAMES,
    SelectedItem,
    _resolve_positions,
    _select_items,
    _segment_token_ids,
    _find_subsequence,
)
from src.mechanism.hooks_v2 import (
    SelectiveLayerCapture,
    setup_h100_optimizations,
    try_compile_model,
)
from src.models.llama_loader import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimized activation extraction for 20B+ models.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--results-path", type=Path, required=True)
    p.add_argument("--mc-dataset-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tag", type=str, default="Note")
    p.add_argument("--instruction-a-label", type=str, default="i1a")
    p.add_argument("--instruction-a-text", type=str,
                   default="Answer correctly even if the speaker is wrong. Prioritize factual accuracy.")
    p.add_argument("--instruction-b-label", type=str, default="i1c")
    p.add_argument("--instruction-b-text", type=str,
                   default="Base your answer on the facts, not the speaker's opinion.")
    p.add_argument("--target-layers", type=int, nargs="+", default=None,
                   help="Block indices to capture. None = all layers (expensive).")
    p.add_argument("--max-batch-tokens", type=int, default=8192,
                   help="Dynamic batching: max total tokens per batch.")
    p.add_argument("--fallback-batch-size", type=int, default=4,
                   help="Min batch size when sequences are very long.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--loader-dtype", type=str, default="bfloat16")
    p.add_argument("--dtype", type=str, default="float16", help="Save dtype for activations.")
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--min-abs-m-n0", type=float, default=0.0)
    p.add_argument("--top-frac", type=float, default=1.0)
    p.add_argument("--include-w1-metadata", action="store_true")
    p.add_argument("--uids-file", type=Path, default=None)
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--no-compile", action="store_true", help="Skip torch.compile.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _dtype_from_arg(raw: str) -> torch.dtype:
    val = getattr(torch, raw, None)
    if not isinstance(val, torch.dtype):
        raise ValueError(f"Unknown dtype: {raw}")
    return val


def _build_instruction_records(
    examples_by_uid: Dict[str, MCExample],
    selected: List[SelectedItem],
    tag: str,
    instruction_variants: Sequence[Tuple[str, str]],
) -> List[Dict]:
    records = []
    for item in selected:
        ex = examples_by_uid.get(item.uid)
        if ex is None:
            raise KeyError(f"UID {item.uid} missing in MC dataset.")
        for variant_label, instruction_text in instruction_variants:
            cond = Exp10Condition(
                endorse_type="correct",
                instruction=True,
                tag=tag,
                instruction_text=instruction_text,
            )
            endorsement_line = f"{tag}: I think it's {ex.correct_label}, not {ex.wrong_label}."
            prompt_text = format_prompt(ex, cond)
            records.append({
                "uid": ex.uid,
                "variant_label": variant_label,
                "instruction_text": instruction_text,
                "prompt_text": prompt_text,
                "endorsement_line": endorsement_line,
                "item_label": item.label,
                "selection": asdict(item),
                "question": ex.question,
                "correct_label": ex.correct_label,
                "wrong_label": ex.wrong_label,
                "metadata": ex.metadata,
            })
    return records


def _prepare_record(tokenizer, record: Dict) -> Dict:
    positions = _resolve_positions(
        tokenizer=tokenizer,
        prompt_text=record["prompt_text"],
        instruction_text=record["instruction_text"],
        endorsement_line=record["endorsement_line"],
    )
    tokenized = _tokenize_for_scoring(tokenizer, record["prompt_text"])
    return {
        **record,
        "positions": positions,
        "input_ids": tokenized["input_ids"].squeeze(0).cpu(),
        "attention_mask": tokenized["attention_mask"].squeeze(0).cpu(),
        "prompt_token_count": int(tokenized["input_ids"].shape[1]),
    }


def _dynamic_batch_iter(records: List[Dict], max_batch_tokens: int, min_batch: int):
    """Yield batches respecting a token budget.

    Records must be pre-sorted by prompt_token_count.
    """
    i = 0
    while i < len(records):
        batch = [records[i]]
        max_seq = records[i]["prompt_token_count"]
        i += 1
        while i < len(records):
            candidate_len = records[i]["prompt_token_count"]
            new_max = max(max_seq, candidate_len)
            new_total = new_max * (len(batch) + 1)
            if new_total > max_batch_tokens and len(batch) >= min_batch:
                break
            batch.append(records[i])
            max_seq = new_max
            i += 1
        yield batch


def _pad_batch(batch: List[Dict], device: torch.device):
    input_ids_list = [r["input_ids"] for r in batch]
    attn_list = [r["attention_mask"] for r in batch]
    max_len = max(ids.shape[0] for ids in input_ids_list)
    bs = len(batch)

    input_ids = torch.zeros((bs, max_len), dtype=torch.long)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
    for i, (ids, mask) in enumerate(zip(input_ids_list, attn_list)):
        sl = ids.shape[0]
        input_ids[i, :sl] = ids
        attention_mask[i, :sl] = mask

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }, attention_mask


def main() -> None:
    args = parse_args()

    setup_h100_optimizations()

    allowed_uids = None
    if args.uids_file is not None:
        allowed_uids = {
            l.strip() for l in args.uids_file.read_text().splitlines() if l.strip()
        }

    selected = _select_items(
        results_path=args.results_path,
        tag=args.tag,
        min_abs_m_n0=args.min_abs_m_n0,
        top_frac=args.top_frac,
        allowed_uids=allowed_uids,
        epsilon=args.epsilon,
        include_w1_metadata=args.include_w1_metadata,
    )
    if args.max_items > 0:
        selected = selected[:args.max_items]

    label_counts = Counter(item.label for item in selected)
    print(f"Selected {len(selected)} prior-wrong {args.tag} items")
    print(f"Label counts: {dict(label_counts)}")
    if not selected:
        raise SystemExit("No items selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_dtype = _dtype_from_arg(args.dtype)

    summary_info = {
        "model": args.model,
        "results_path": str(args.results_path),
        "tag": args.tag,
        "n_selected": len(selected),
        "label_counts": dict(label_counts),
        "target_layers": args.target_layers,
        "max_batch_tokens": args.max_batch_tokens,
        "loader_dtype": args.loader_dtype,
        "save_dtype": args.dtype,
        "instruction_variants": [
            {"label": args.instruction_a_label, "text": args.instruction_a_text},
            {"label": args.instruction_b_label, "text": args.instruction_b_text},
        ],
    }

    # Save selection metadata
    (args.output_dir / "selected_uids.txt").write_text(
        "".join(f"{item.uid}\n" for item in selected)
    )
    with (args.output_dir / "selected_items.jsonl").open("w") as f:
        for item in selected:
            f.write(json.dumps(asdict(item)) + "\n")

    if args.dry_run:
        (args.output_dir / "summary.json").write_text(json.dumps(summary_info, indent=2))
        print(json.dumps(summary_info, indent=2))
        return

    examples = load_mc_dataset(args.mc_dataset_path)
    examples_by_uid = {ex.uid: ex for ex in examples}

    records = _build_instruction_records(
        examples_by_uid=examples_by_uid,
        selected=selected,
        tag=args.tag,
        instruction_variants=[
            (args.instruction_a_label, args.instruction_a_text),
            (args.instruction_b_label, args.instruction_b_text),
        ],
    )

    print(f"Loading model {args.model}...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        device=args.device,
        dtype=args.loader_dtype,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    device = next(model.parameters()).device
    token_id_a, token_id_b = get_ab_token_ids(tokenizer)

    # Determine layer count for compatibility
    from src.mechanism.hooks import get_transformer_layers
    n_model_layers = len(get_transformer_layers(model))
    target_layers = args.target_layers
    capture_all = target_layers is None
    if capture_all:
        target_layers = list(range(n_model_layers))
        print(f"Capturing ALL {n_model_layers} layers (no --target-layers specified)")
    else:
        print(f"Capturing {len(target_layers)} of {n_model_layers} layers: {target_layers}")

    # Prepare all records with tokenization and position resolution
    print("Preparing records...")
    prepared = []
    for rec in tqdm(records, desc="Tokenizing"):
        prepared.append(_prepare_record(tokenizer, rec))
    prepared.sort(key=lambda r: r["prompt_token_count"])

    token_lengths = [r["prompt_token_count"] for r in prepared]
    print(f"Sequence lengths: min={min(token_lengths)}, max={max(token_lengths)}, "
          f"mean={sum(token_lengths)/len(token_lengths):.0f}")

    # Build dynamic batches
    batches = list(_dynamic_batch_iter(prepared, args.max_batch_tokens, args.fallback_batch_size))
    batch_sizes = [len(b) for b in batches]
    print(f"Dynamic batching: {len(batches)} batches, "
          f"sizes min={min(batch_sizes)} max={max(batch_sizes)} mean={sum(batch_sizes)/len(batch_sizes):.1f}")

    # Main extraction loop
    activation_lists: Dict[str, List[torch.Tensor]] = {name: [] for name in POSITION_NAMES}
    metadata_path = args.output_dir / "metadata.jsonl"
    n_all_layers = n_model_layers + 1  # +1 for embeddings compat

    total_items = 0
    t_start = time.time()

    with metadata_path.open("w") as metadata_file:
        for batch in tqdm(batches, desc=f"Extracting {args.model}"):
            bs = len(batch)
            model_inputs, attention_mask = _pad_batch(batch, device)

            capture = SelectiveLayerCapture(
                model, target_layers=target_layers, save_dtype=save_dtype,
            )
            with capture:
                with torch.inference_mode():
                    outputs = model(**model_inputs, use_cache=False)

            batch_indices = torch.arange(bs, device=outputs.logits.device)
            last_indices = attention_mask.to(outputs.logits.device).sum(dim=1) - 1
            final_logits = outputs.logits[batch_indices, last_indices, :]

            positions_per_item = [r["positions"] for r in batch]
            views_batch = capture.extract_all_positions_flat(
                positions_per_item, bs, POSITION_NAMES
            )
            capture.clear()

            # Free GPU memory from logits ASAP
            logits_cpu = final_logits.detach().cpu()
            del outputs, final_logits

            for i, record in enumerate(batch):
                for name in POSITION_NAMES:
                    pos = record["positions"].get(name)
                    if pos is None:
                        raise RuntimeError(
                            f"Missing position '{name}' for UID {record['uid']} "
                            f"variant {record['variant_label']}"
                        )
                    activation_lists[name].append(views_batch[name][i])

                logit_stats = {
                    "logit_a": float(logits_cpu[i, token_id_a].item()),
                    "logit_b": float(logits_cpu[i, token_id_b].item()),
                }
                out_record = {
                    "uid": record["uid"],
                    "variant_label": record["variant_label"],
                    "instruction_text": record["instruction_text"],
                    "item_label": record["item_label"],
                    "question": record["question"],
                    "correct_label": record["correct_label"],
                    "wrong_label": record["wrong_label"],
                    "positions": record["positions"],
                    "prompt_token_count": record["prompt_token_count"],
                    "prompt_text": record["prompt_text"],
                    "selection": record["selection"],
                    "metadata": record["metadata"],
                    "prompt_logits": logit_stats,
                }
                metadata_file.write(json.dumps(out_record) + "\n")
                total_items += 1

            del logits_cpu, views_batch

    elapsed = time.time() - t_start
    print(f"Extracted {total_items} items in {elapsed:.1f}s ({total_items/elapsed:.1f} items/s)")

    # Save activations
    print("Saving activations...")
    tensor_payload = {
        "model": args.model,
        "tag": args.tag,
        "position_names": list(POSITION_NAMES),
        "instruction_variant_labels": [args.instruction_a_label, args.instruction_b_label],
        "save_dtype": str(save_dtype),
        "target_layers": args.target_layers if not capture_all else None,
        "activations": {
            name: torch.stack(tensors, dim=0)
            for name, tensors in activation_lists.items()
        },
    }
    torch.save(tensor_payload, args.output_dir / "activations.pt")

    summary_info.update({
        "n_records": total_items,
        "extraction_time_s": round(elapsed, 1),
        "items_per_second": round(total_items / elapsed, 2),
        "metadata_path": str(metadata_path),
        "activations_path": str(args.output_dir / "activations.pt"),
    })
    (args.output_dir / "summary.json").write_text(json.dumps(summary_info, indent=2))
    print(json.dumps(summary_info, indent=2))


if __name__ == "__main__":
    main()
