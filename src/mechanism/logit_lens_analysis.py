"""Logit lens analysis for correction-gating mechanism.

Projects residual stream at each layer through the unembedding matrix to track
how correct vs wrong answer logit difference evolves layer-by-layer under
i1a (accuracy instruction) vs i1c (alternative instruction).

Key question: does the correct answer appear in middle layers under i1a and
get SUPPRESSED (suppression hypothesis), or does it NEVER appear (prevention)?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.exp10.conditions import Exp10Condition, format_prompt
from src.exp7.dataset_mc import MCExample, load_mc_dataset
from src.exp7.scoring import _tokenize_for_scoring, get_ab_token_ids
from src.models.llama_loader import load_model_and_tokenizer


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_EXTRACTION_DIR = Path(
    "new-phase-results/mechanism/qwen_correction_gating_i1a_i1c_note"
)
DEFAULT_MC_DATASET_PATH = Path("data/exp7_mc_dataset.jsonl")
DEFAULT_OUTPUT_DIR = Path("new-phase-results/mechanism/logit_lens_qwen")

I1A_TEXT = "Answer correctly even if the speaker is wrong. Prioritize factual accuracy."
I1C_TEXT = "Base your answer on the facts, not the speaker's opinion."


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def signed_margin(logit_a, logit_b, correct_label):
    if correct_label == "A":
        return logit_a - logit_b
    return logit_b - logit_a


def build_prompt(mc_item, instruction_text, tag="Note"):
    cond = Exp10Condition(
        endorse_type="correct",
        instruction=True,
        tag=tag,
        instruction_text=instruction_text,
    )
    return format_prompt(mc_item, cond)


def run_logit_lens(
    model,
    tokenizer,
    device,
    prompt_text: str,
    token_id_a: int,
    token_id_b: int,
    correct_label: str,
) -> np.ndarray:
    """Run forward pass, project each layer's hidden state through W_U.

    Returns array of shape [n_layers+1] with signed margin at each layer.
    Layer 0 = embedding output, layer L = final output.
    """
    inputs = _tokenize_for_scoring(tokenizer, prompt_text)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple of (1, seq_len, d_model), len = n_layers + 1

    # Get unembedding matrix
    if hasattr(model, "lm_head"):
        W_U = model.lm_head.weight  # (vocab_size, d_model)
    else:
        raise ValueError("Cannot find unembedding matrix (lm_head)")

    # Optional: layer norm before unembedding
    final_ln = None
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        final_ln = model.model.norm

    n_layers = len(hidden_states) - 1
    margins = np.zeros(n_layers + 1)

    for layer_idx, hs in enumerate(hidden_states):
        h = hs[0, -1, :]  # last token hidden state

        # Apply final layer norm (important for logit lens accuracy)
        if final_ln is not None:
            h = final_ln(h.unsqueeze(0)).squeeze(0)

        # Project through unembedding
        logits = h @ W_U.T  # (vocab_size,)
        logit_a = logits[token_id_a].item()
        logit_b = logits[token_id_b].item()
        margins[layer_idx] = signed_margin(logit_a, logit_b, correct_label)

    return margins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--extraction-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--mc-dataset-path", type=Path, default=DEFAULT_MC_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", type=str, default="Note")
    parser.add_argument("--instruction-a", type=str, default=I1A_TEXT)
    parser.add_argument("--instruction-b", type=str, default=I1C_TEXT)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load item metadata from extraction
    meta_path = args.extraction_dir / "metadata.jsonl"
    sel_path = args.extraction_dir / "selected_items.jsonl"
    meta = load_jsonl(meta_path)
    selected = load_jsonl(sel_path)

    uid_to_label = {}
    uid_to_info = {}
    for row in selected:
        # Normalize: "entrenching" -> "resisting" for consistency
        label = row["label"]
        if label == "entrenching":
            label = "resisting"
        uid_to_label[row["uid"]] = label
        uid_to_info[row["uid"]] = row

    # Get unique UIDs (each appears twice in metadata: i1a and i1c)
    seen_uids = []
    for row in meta:
        if row["uid"] not in [u for u in seen_uids]:
            seen_uids.append(row["uid"])

    if args.max_items > 0:
        seen_uids = seen_uids[:args.max_items]

    print(f"Items: {len(seen_uids)}")
    n_correcting = sum(1 for u in seen_uids if uid_to_label.get(u) == "correcting")
    n_resisting = sum(1 for u in seen_uids if uid_to_label.get(u) == "resisting")
    print(f"  correcting={n_correcting}, resisting={n_resisting}")

    # Load MC dataset
    mc_items = load_mc_dataset(args.mc_dataset_path)
    uid_to_mc = {item.uid: item for item in mc_items}

    # Load model
    print(f"Loading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(
        args.model, device=args.device, dtype="bfloat16",
        prefer_flash_attention=False,  # need output_hidden_states
    )
    model.eval()
    device = next(model.parameters()).device

    token_id_a, token_id_b = get_ab_token_ids(tokenizer)
    print(f"Token IDs: A={token_id_a}, B={token_id_b}")

    # Run logit lens for each item under both instructions
    all_results = []
    for uid in tqdm(seen_uids, desc="Logit lens"):
        mc_item = uid_to_mc.get(uid)
        if mc_item is None:
            continue

        label = uid_to_label.get(uid, "unknown")
        info = uid_to_info.get(uid, {})

        prompt_i1a = build_prompt(mc_item, args.instruction_a, tag=args.tag)
        prompt_i1c = build_prompt(mc_item, args.instruction_b, tag=args.tag)

        margins_i1a = run_logit_lens(model, tokenizer, device, prompt_i1a,
                                      token_id_a, token_id_b, mc_item.correct_label)
        margins_i1c = run_logit_lens(model, tokenizer, device, prompt_i1c,
                                      token_id_a, token_id_b, mc_item.correct_label)

        all_results.append({
            "uid": uid,
            "label": label,
            "correct_label": mc_item.correct_label,
            "m_n0": info.get("m_n0", 0),
            "margins_i1a": margins_i1a.tolist(),
            "margins_i1c": margins_i1c.tolist(),
        })

    # Save raw results
    out_path = args.output_dir / "logit_lens_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {len(all_results)} items to {out_path}")

    # Aggregate analysis
    n_layers = len(all_results[0]["margins_i1a"]) if all_results else 0
    print(f"\nLayers: {n_layers - 1} transformer layers + embedding")

    for group_name, group_label in [("correcting", "correcting"), ("resisting", "resisting")]:
        items = [r for r in all_results if r["label"] == group_label]
        if not items:
            continue

        i1a_stack = np.array([r["margins_i1a"] for r in items])
        i1c_stack = np.array([r["margins_i1c"] for r in items])

        mean_i1a = i1a_stack.mean(axis=0)
        mean_i1c = i1c_stack.mean(axis=0)

        print(f"\n{'='*60}")
        print(f"  {group_name.upper()} items (N={len(items)})")
        print(f"{'='*60}")

        # Find peak layer for i1a
        peak_layer_i1a = int(np.argmax(mean_i1a))
        peak_val_i1a = mean_i1a[peak_layer_i1a]

        # Find where correct answer first becomes positive under i1a
        first_positive_i1a = None
        for i, v in enumerate(mean_i1a):
            if v > 0:
                first_positive_i1a = i
                break

        print(f"  i1a trajectory:")
        print(f"    Peak layer: {peak_layer_i1a} (margin={peak_val_i1a:.3f})")
        print(f"    First positive: layer {first_positive_i1a}")
        print(f"    Final margin: {mean_i1a[-1]:.3f}")
        print(f"    Embedding margin: {mean_i1a[0]:.3f}")

        print(f"  i1c trajectory:")
        peak_layer_i1c = int(np.argmax(mean_i1c))
        print(f"    Peak layer: {peak_layer_i1c} (margin={mean_i1c[peak_layer_i1c]:.3f})")
        print(f"    Final margin: {mean_i1c[-1]:.3f}")

        # Key diagnostic: does margin peak then DROP under i1a for resisting items?
        if group_name == "resisting":
            drop = peak_val_i1a - mean_i1a[-1]
            print(f"\n  SUPPRESSION TEST:")
            print(f"    Peak-to-final drop (i1a): {drop:.3f}")
            if drop > 1.0 and peak_val_i1a > 0:
                print(f"    -> SUPPRESSION: correct answer appears then gets suppressed")
            elif peak_val_i1a <= 0:
                print(f"    -> PREVENTION: correct answer never clearly appears")
            else:
                print(f"    -> AMBIGUOUS: small peak ({peak_val_i1a:.3f}) with drop {drop:.3f}")

        # Layer-by-layer summary (sampled)
        sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
        print(f"\n  Layer samples (i1a / i1c):")
        for l in sample_layers:
            print(f"    L{l:2d}: {mean_i1a[l]:+7.3f} / {mean_i1c[l]:+7.3f}  (diff={mean_i1a[l]-mean_i1c[l]:+.3f})")

    # Summary stats
    summary = {
        "model": args.model,
        "n_items": len(all_results),
        "n_layers": n_layers - 1,
        "groups": {},
    }
    for group_label in ["correcting", "resisting"]:
        items = [r for r in all_results if r["label"] == group_label]
        if not items:
            continue
        i1a_stack = np.array([r["margins_i1a"] for r in items])
        i1c_stack = np.array([r["margins_i1c"] for r in items])
        mean_i1a = i1a_stack.mean(axis=0)
        mean_i1c = i1c_stack.mean(axis=0)
        summary["groups"][group_label] = {
            "n": len(items),
            "i1a_mean_trajectory": mean_i1a.tolist(),
            "i1c_mean_trajectory": mean_i1c.tolist(),
            "i1a_peak_layer": int(np.argmax(mean_i1a)),
            "i1a_peak_margin": float(mean_i1a.max()),
            "i1a_final_margin": float(mean_i1a[-1]),
            "i1c_final_margin": float(mean_i1c[-1]),
        }

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
