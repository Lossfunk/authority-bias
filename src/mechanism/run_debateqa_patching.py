"""Causal patching for DebateQA instruction-gating mechanism.

Thin wrapper around the correction-gating patching logic. The key difference:
instead of rebuilding prompts from a uniform condition code, this script uses
the prompt text already stored in the extraction metadata (which has per-item
disagreeing endorsements).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from src.lexical_controls.dataset_mc import load_mc_dataset
from src.lexical_controls.scoring import _tokenize_for_scoring, get_ab_token_ids
from src.mechanism.extract_correction_gating_activations import _resolve_positions
from src.mechanism.hooks import get_transformer_layers
from src.mechanism.run_correction_gating_patching import (
    PositionPatcher,
    _build_within_i1a_vector,
    _load_extraction,
    _load_metadata,
    _matched_random_vector,
    _mean,
    _predicted_label_from_logits,
    _resolve_primary_device,
    _run_prompt,
    _signed_margin_from_logits,
    _summarize_records,
)
from src.models.llama_loader import load_model_and_tokenizer


I1A_TEXT = "Answer correctly even if the speaker is wrong. Prioritize factual accuracy."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal patching for DebateQA instruction-gating mechanism."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--mc-dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--position", type=str, default="endorsement_last")
    parser.add_argument("--wrong-site-position", type=str, default="instruction_last")
    parser.add_argument("--layer-index", type=int, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--loader-dtype", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.layer_index <= 0:
        raise ValueError("layer-index must be >= 1")

    metadata = _load_metadata(args.extraction_dir / "metadata.jsonl")
    _, activations = _load_extraction(args.extraction_dir)

    # Split i1a rows by label (DebateQA uses "correcting"/"resisting")
    i1a_all, i1a_correcting, i1a_negative = [], [], []
    for idx, row in enumerate(metadata):
        if row["variant_label"] != "i1a":
            continue
        i1a_all.append(idx)
        if row["item_label"] == "correcting":
            i1a_correcting.append(idx)
        else:
            i1a_negative.append(idx)

    if not i1a_negative:
        raise SystemExit("No resisting i1a rows found.")

    vector = _build_within_i1a_vector(
        activations=activations,
        correcting_rows=i1a_correcting,
        negative_rows=i1a_negative,
        position=args.position,
        layer_index=args.layer_index,
    )
    random_vector = _matched_random_vector(vector, args.seed)
    wrong_sign_vector = -vector

    # Select resisting i1a rows
    target_rows = [i for i in range(len(metadata))
                   if metadata[i]["variant_label"] == "i1a"
                   and metadata[i]["item_label"] in ("resisting", "entrenching")]
    if args.max_items > 0:
        target_rows = target_rows[:args.max_items]

    print(f"Patching {len(target_rows)} resisting i1a items at layer {args.layer_index}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        device=args.device,
        dtype=args.loader_dtype,
    )
    model.eval()
    device = _resolve_primary_device(model)
    token_id_a, token_id_b = get_ab_token_ids(tokenizer)

    block_index = args.layer_index - 1
    patch_module = get_transformer_layers(model)[block_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Use prompts directly from metadata (already have per-item endorsements)
    controls = [
        ("real", args.position, vector),
        ("wrong_sign", args.position, wrong_sign_vector),
        ("random", args.position, random_vector),
    ]
    if args.wrong_site_position != "none":
        controls.append(("wrong_site", args.wrong_site_position, vector))

    alphas = list(args.alphas)
    scaled_vectors: Dict[Tuple[str, float], Optional[torch.Tensor]] = {}
    for control_name, _, base_vector in controls:
        for alpha in alphas:
            scaled_vectors[(control_name, float(alpha))] = (
                None if alpha == 0 else (alpha * base_vector).to(device=device)
            )

    all_rows: List[Dict] = []
    records_path = args.output_dir / "item_results.jsonl"

    with records_path.open("w") as out_file:
        for row_idx in tqdm(target_rows, desc="Patching resisting items"):
            row = metadata[row_idx]
            prompt_text = row["prompt_text"]
            positions = row["positions"]
            correct_label = row["correct_label"]

            # For DebateQA, "correct_label" is arbitrary. The signed margin
            # should be computed relative to the ENDORSED option direction.
            # The extraction metadata stores which option was endorsed via
            # selection.disagree_endorse. The prompt already endorses the
            # option that disagrees with the model's prior.
            #
            # We use the existing correct_label for margin computation since
            # the vector was built with the same label convention.

            model_inputs = {
                k: v.to(device) for k, v in
                _tokenize_for_scoring(tokenizer, prompt_text).items()
            }

            # Resolve positions from stored metadata
            patch_pos = positions.get(args.position)
            if patch_pos is None:
                continue

            baseline = _run_prompt(
                model=model,
                model_inputs=model_inputs,
                token_position=int(patch_pos),
                token_id_a=token_id_a,
                token_id_b=token_id_b,
                correct_label=correct_label,
                patch_module=patch_module,
                patch_vector=None,
                device=device,
            )

            for control_name, position_name, base_vector in controls:
                ctrl_pos = positions.get(position_name)
                if ctrl_pos is None:
                    continue
                for alpha in alphas:
                    patched = _run_prompt(
                        model=model,
                        model_inputs=model_inputs,
                        token_position=int(ctrl_pos),
                        token_id_a=token_id_a,
                        token_id_b=token_id_b,
                        correct_label=correct_label,
                        patch_module=patch_module,
                        patch_vector=scaled_vectors[(control_name, float(alpha))],
                        device=device,
                    )
                    out_row = {
                        "uid": row["uid"],
                        "question": row["question"],
                        "control": control_name,
                        "position": position_name,
                        "layer_index": args.layer_index,
                        "block_index": block_index,
                        "alpha": float(alpha),
                        "item_label": row["item_label"],
                        "correct_label": correct_label,
                        "wrong_label": row["wrong_label"],
                        "selection": row["selection"],
                        "baseline_signed_margin": baseline["signed_margin"],
                        "patched_signed_margin": patched["signed_margin"],
                        "baseline_predicted_label": baseline["predicted_label"],
                        "patched_predicted_label": patched["predicted_label"],
                        "baseline_correct": baseline["correct"],
                        "patched_correct": patched["correct"],
                    }
                    out_file.write(json.dumps(out_row) + "\n")
                    all_rows.append(out_row)

    summary = {
        "model": args.model,
        "extraction_dir": str(args.extraction_dir),
        "output_dir": str(args.output_dir),
        "position": args.position,
        "wrong_site_position": args.wrong_site_position,
        "layer_index": args.layer_index,
        "block_index": block_index,
        "n_i1a_rows": len(i1a_all),
        "n_correcting_rows": len(i1a_correcting),
        "n_negative_rows": len(i1a_negative),
        "n_rows_evaluated": len(target_rows),
        "alphas": alphas,
        "vector_norm": float(vector.norm().item()),
        "aggregates": _summarize_records(all_rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
