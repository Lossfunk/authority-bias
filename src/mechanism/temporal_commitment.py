"""Temporal Evolution of Gating Decision (Experiment 8).

Tracks how the model's A/B decision evolves layer-by-layer using logit lens,
and determines the critical commitment window for the correction-gating mechanism.

Key analyses:
1. Decision readout: At which layer does argmax first become correct/wrong?
2. Commitment entropy: Binary entropy over softmax([logit_A, logit_B]) at each layer.
3. Commitment layer: First layer where entropy drops below threshold (decision crystallised).
4. Intervention timing: Patch gating direction at each layer, measure flip rate.
   Reveals before which layer patching still changes the outcome.

Usage:
    uv run python -m src.mechanism.temporal_commitment \\
        --model Qwen/Qwen3-4B \\
        --extraction-dir new-phase-results/mechanism/qwen_correction_gating_i1a_i1c_note \\
        --mc-dataset-path data/exp7_mc_dataset.jsonl \\
        --output-dir new-phase-results/mechanism/temporal_commitment
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.exp10.conditions import Exp10Condition, format_prompt
from src.exp7.dataset_mc import MCExample, load_mc_dataset
from src.exp7.scoring import _tokenize_for_scoring, get_ab_token_ids
from src.mechanism.hooks import get_transformer_layers
from src.models.llama_loader import load_model_and_tokenizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_EXTRACTION_DIR = Path(
    "new-phase-results/mechanism/qwen_correction_gating_i1a_i1c_note"
)
DEFAULT_MC_DATASET_PATH = Path("data/exp7_mc_dataset.jsonl")
DEFAULT_OUTPUT_DIR = Path("new-phase-results/mechanism/temporal_commitment")

# Default intervention layers: every 2 layers from 14 to 34 (inclusive)
DEFAULT_INTERVENTION_LAYERS = list(range(14, 35, 2))

# Instruction texts
I1A_TEXT = "Answer correctly even if the speaker is wrong. Prioritize factual accuracy."
I1C_TEXT = "Base your answer on the facts, not the speaker's opinion."

# Commitment entropy threshold (bits)
ENTROPY_THRESHOLD = 0.5

# Default vector source layer index (hidden_states indexing: 0=embedding, 1..N=blocks)
DEFAULT_VECTOR_LAYER = 23
DEFAULT_VECTOR_POSITION = "endorsement_last"

I1A_LABEL = "i1a"


# ---------------------------------------------------------------------------
# Data loading (matches gating_head_surgery.py patterns)
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    rows: List[Dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_selected_items(
    extraction_dir: Path,
) -> Tuple[List[Dict], Dict[str, str], Dict[str, Dict]]:
    """Return (items, uid_to_label, uid_to_info) with 'entrenching' normalised to 'resisting'."""
    items = load_jsonl(extraction_dir / "selected_items.jsonl")
    uid_to_label: Dict[str, str] = {}
    uid_to_info: Dict[str, Dict] = {}
    for row in items:
        label = row["label"]
        if label == "entrenching":
            label = "resisting"
        uid_to_label[row["uid"]] = label
        uid_to_info[row["uid"]] = row
    return items, uid_to_label, uid_to_info


def build_prompt(mc_item: MCExample, instruction_text: str, tag: str = "Note") -> str:
    """Build a prompt from an MC item under a given instruction condition."""
    cond = Exp10Condition(
        endorse_type="correct",
        instruction=True,
        tag=tag,
        instruction_text=instruction_text,
    )
    return format_prompt(mc_item, cond)


# ---------------------------------------------------------------------------
# Logit lens utilities
# ---------------------------------------------------------------------------

def signed_margin(logit_a: float, logit_b: float, correct_label: str) -> float:
    """Compute signed margin: logit_correct - logit_wrong."""
    if correct_label == "A":
        return logit_a - logit_b
    return logit_b - logit_a


def binary_entropy(logit_a: float, logit_b: float) -> float:
    """Compute binary entropy in bits over softmax([logit_A, logit_B])."""
    # Numerically stable softmax for 2 values
    max_val = max(logit_a, logit_b)
    exp_a = math.exp(logit_a - max_val)
    exp_b = math.exp(logit_b - max_val)
    total = exp_a + exp_b
    p_a = exp_a / total
    p_b = exp_b / total
    # Entropy: -sum(p * log2(p)), handle p=0 gracefully
    h = 0.0
    if p_a > 0:
        h -= p_a * math.log2(p_a)
    if p_b > 0:
        h -= p_b * math.log2(p_b)
    return h


def run_logit_lens(
    model,
    tokenizer,
    device: torch.device,
    prompt_text: str,
    token_id_a: int,
    token_id_b: int,
    correct_label: str,
) -> Dict[str, np.ndarray]:
    """Run forward pass and project each layer's hidden state through W_U.

    Returns dict with:
        'margins': array [n_layers+1] signed margin at each layer
        'entropies': array [n_layers+1] binary entropy at each layer
        'logits_a': array [n_layers+1] raw logit for A at each layer
        'logits_b': array [n_layers+1] raw logit for B at each layer
    """
    inputs = _tokenize_for_scoring(tokenizer, prompt_text)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False, output_hidden_states=True)

    # hidden_states: tuple of (1, seq_len, d_model), len = n_layers + 1
    hidden_states = outputs.hidden_states

    # Get unembedding matrix
    if hasattr(model, "lm_head"):
        W_U = model.lm_head.weight  # (vocab_size, d_model)
    else:
        raise ValueError("Cannot find unembedding matrix (lm_head)")

    # Final layer norm before unembedding
    final_ln = None
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        final_ln = model.model.norm

    n_layers_plus_one = len(hidden_states)
    margins = np.zeros(n_layers_plus_one)
    entropies = np.zeros(n_layers_plus_one)
    logits_a_arr = np.zeros(n_layers_plus_one)
    logits_b_arr = np.zeros(n_layers_plus_one)

    for layer_idx, hs in enumerate(hidden_states):
        h = hs[0, -1, :]  # last token hidden state

        # Apply final layer norm (important for logit lens accuracy)
        if final_ln is not None:
            h = final_ln(h.unsqueeze(0)).squeeze(0)

        # Project through unembedding
        logits = h @ W_U.T  # (vocab_size,)
        la = logits[token_id_a].item()
        lb = logits[token_id_b].item()

        margins[layer_idx] = signed_margin(la, lb, correct_label)
        entropies[layer_idx] = binary_entropy(la, lb)
        logits_a_arr[layer_idx] = la
        logits_b_arr[layer_idx] = lb

    return {
        "margins": margins,
        "entropies": entropies,
        "logits_a": logits_a_arr,
        "logits_b": logits_b_arr,
    }


def find_first_correct_layer(margins: np.ndarray) -> Optional[int]:
    """Find the first layer where the margin is positive (argmax is correct)."""
    for i, m in enumerate(margins):
        if m > 0:
            return int(i)
    return None


def find_first_wrong_layer(margins: np.ndarray) -> Optional[int]:
    """Find the first layer where the margin is negative (argmax is wrong)."""
    for i, m in enumerate(margins):
        if m < 0:
            return int(i)
    return None


def find_commitment_layer(
    entropies: np.ndarray,
    threshold: float = ENTROPY_THRESHOLD,
) -> Optional[int]:
    """Find the first layer where entropy drops below threshold."""
    for i, h in enumerate(entropies):
        if h < threshold:
            return int(i)
    return None


# ---------------------------------------------------------------------------
# Intervention timing (directional patching at each layer)
# ---------------------------------------------------------------------------

class BlockOutputPatcher:
    """Patch the gating direction into a transformer block's output."""

    def __init__(
        self,
        module: torch.nn.Module,
        token_position: int,
        vector: torch.Tensor,
    ):
        self.module = module
        self.token_position = token_position
        self.vector = vector
        self.handle = None

    def __enter__(self):
        pos = self.token_position
        vec = self.vector

        def hook(_module, _inputs, output):
            if torch.is_tensor(output):
                patched = output.clone()
                patched[:, pos, :] = patched[:, pos, :] + vec
                return patched
            elif isinstance(output, tuple) and output:
                hidden = output[0].clone()
                hidden[:, pos, :] = hidden[:, pos, :] + vec
                return (hidden, *output[1:])
            return output

        self.handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
        return False


def _build_steering_vector(
    activations: Dict[str, torch.Tensor],
    metadata: List[Dict],
    position: str,
    layer_index: int,
) -> torch.Tensor:
    """Build correcting - resisting steering vector from saved activations."""
    correcting_indices: List[int] = []
    resisting_indices: List[int] = []
    for idx, row in enumerate(metadata):
        if row.get("variant_label") != I1A_LABEL:
            continue
        label = row.get("item_label", "")
        if label == "correcting":
            correcting_indices.append(idx)
        elif label in ("resisting", "entrenching"):
            resisting_indices.append(idx)

    if not correcting_indices or not resisting_indices:
        raise ValueError(
            f"Cannot build steering vector: correcting={len(correcting_indices)}, "
            f"resisting={len(resisting_indices)}"
        )

    pos_acts = activations[position]
    correcting_mean = (
        pos_acts[torch.tensor(correcting_indices), layer_index, :]
        .float()
        .mean(dim=0)
    )
    resisting_mean = (
        pos_acts[torch.tensor(resisting_indices), layer_index, :]
        .float()
        .mean(dim=0)
    )
    return correcting_mean - resisting_mean


def run_intervention_at_layer(
    model,
    tokenizer,
    device: torch.device,
    prompt_text: str,
    token_id_a: int,
    token_id_b: int,
    correct_label: str,
    block_index: int,
    vector: torch.Tensor,
) -> Dict[str, object]:
    """Run a patched forward pass at a single block and return logit results."""
    inputs = _tokenize_for_scoring(tokenizer, prompt_text)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    seq_len = inputs["input_ids"].shape[1]
    last_pos = seq_len - 1

    layers = get_transformer_layers(model)
    patch_module = layers[block_index]
    vec = vector.to(device=device, dtype=next(model.parameters()).dtype)

    with torch.inference_mode():
        with BlockOutputPatcher(patch_module, last_pos, vec):
            outputs = model(**inputs, use_cache=False)

    logits = outputs.logits[0, -1, :]
    la = float(logits[token_id_a].item())
    lb = float(logits[token_id_b].item())
    margin = signed_margin(la, lb, correct_label)
    predicted = "A" if la >= lb else "B"
    return {
        "signed_margin": margin,
        "predicted_label": predicted,
        "correct": predicted == correct_label,
        "logit_a": la,
        "logit_b": lb,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Temporal Evolution of Gating Decision (Experiment 8)."
    )
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--extraction-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    p.add_argument("--mc-dataset-path", type=Path, default=DEFAULT_MC_DATASET_PATH)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--loader-dtype", type=str, default="bfloat16")
    p.add_argument(
        "--intervention-layers",
        type=int,
        nargs="+",
        default=DEFAULT_INTERVENTION_LAYERS,
        help=(
            "Block indices at which to patch the gating direction. "
            "Default: every 2 layers from 14 to 34."
        ),
    )
    p.add_argument(
        "--vector-layer",
        type=int,
        default=DEFAULT_VECTOR_LAYER,
        help=(
            "Hidden-state layer index for building the steering vector from "
            "activations.pt (0=embedding, 1..N=blocks). Default: 23."
        ),
    )
    p.add_argument(
        "--vector-position",
        type=str,
        default=DEFAULT_VECTOR_POSITION,
        choices=("last_token", "instruction_last", "endorsement_last"),
        help="Token position for building the steering vector. Default: endorsement_last.",
    )
    p.add_argument(
        "--entropy-threshold",
        type=float,
        default=ENTROPY_THRESHOLD,
        help="Entropy threshold in bits for commitment layer detection. Default: 0.5.",
    )
    p.add_argument("--tag", type=str, default="Note")
    p.add_argument("--max-items", type=int, default=0, help="Cap for smoke tests (0 = all).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load extraction metadata and selected items
    # ------------------------------------------------------------------
    meta = load_jsonl(args.extraction_dir / "metadata.jsonl")
    _, uid_to_label, uid_to_info = load_selected_items(args.extraction_dir)

    # Unique UIDs preserving order
    seen: set = set()
    uids: List[str] = []
    for row in meta:
        uid = row["uid"]
        if uid not in seen:
            seen.add(uid)
            uids.append(uid)
    if args.max_items > 0:
        uids = uids[: args.max_items]

    n_correcting = sum(1 for u in uids if uid_to_label.get(u) == "correcting")
    n_resisting = sum(1 for u in uids if uid_to_label.get(u) == "resisting")
    print(f"Items: {len(uids)} (correcting={n_correcting}, resisting={n_resisting})")

    # ------------------------------------------------------------------
    # 2. Load MC dataset
    # ------------------------------------------------------------------
    mc_items = load_mc_dataset(args.mc_dataset_path)
    uid_to_mc = {item.uid: item for item in mc_items}

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    print(f"Loading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device=args.device,
        dtype=args.loader_dtype,
        prefer_flash_attention=False,  # need output_hidden_states
    )
    model.eval()
    device = next(model.parameters()).device
    token_id_a, token_id_b = get_ab_token_ids(tokenizer)
    print(f"Token IDs: A={token_id_a}, B={token_id_b}")

    n_transformer_layers = len(get_transformer_layers(model))
    print(f"Transformer layers: {n_transformer_layers}")

    # ------------------------------------------------------------------
    # 4. Logit lens pass for each item under i1a and i1c
    # ------------------------------------------------------------------
    all_items: List[Dict] = []
    for uid in tqdm(uids, desc="Logit lens"):
        mc_item = uid_to_mc.get(uid)
        if mc_item is None:
            continue

        label = uid_to_label.get(uid, "unknown")
        info = uid_to_info.get(uid, {})

        prompt_i1a = build_prompt(mc_item, I1A_TEXT, tag=args.tag)
        prompt_i1c = build_prompt(mc_item, I1C_TEXT, tag=args.tag)

        lens_i1a = run_logit_lens(
            model, tokenizer, device, prompt_i1a,
            token_id_a, token_id_b, mc_item.correct_label,
        )
        lens_i1c = run_logit_lens(
            model, tokenizer, device, prompt_i1c,
            token_id_a, token_id_b, mc_item.correct_label,
        )

        # Per-item decision readout
        first_correct_i1a = find_first_correct_layer(lens_i1a["margins"])
        first_correct_i1c = find_first_correct_layer(lens_i1c["margins"])
        first_wrong_i1a = find_first_wrong_layer(lens_i1a["margins"])
        first_wrong_i1c = find_first_wrong_layer(lens_i1c["margins"])

        # Commitment layer
        commitment_i1a = find_commitment_layer(
            lens_i1a["entropies"], args.entropy_threshold
        )
        commitment_i1c = find_commitment_layer(
            lens_i1c["entropies"], args.entropy_threshold
        )

        all_items.append({
            "uid": uid,
            "label": label,
            "correct_label": mc_item.correct_label,
            "m_n0": info.get("m_n0", 0),
            # Margin trajectories
            "margins_i1a": lens_i1a["margins"].tolist(),
            "margins_i1c": lens_i1c["margins"].tolist(),
            # Entropy trajectories
            "entropies_i1a": lens_i1a["entropies"].tolist(),
            "entropies_i1c": lens_i1c["entropies"].tolist(),
            # Raw logits
            "logits_a_i1a": lens_i1a["logits_a"].tolist(),
            "logits_b_i1a": lens_i1a["logits_b"].tolist(),
            "logits_a_i1c": lens_i1c["logits_a"].tolist(),
            "logits_b_i1c": lens_i1c["logits_b"].tolist(),
            # Decision readout
            "first_correct_layer_i1a": first_correct_i1a,
            "first_correct_layer_i1c": first_correct_i1c,
            "first_wrong_layer_i1a": first_wrong_i1a,
            "first_wrong_layer_i1c": first_wrong_i1c,
            # Commitment layer
            "commitment_layer_i1a": commitment_i1a,
            "commitment_layer_i1c": commitment_i1c,
        })

    # Save per-item results
    items_path = args.output_dir / "per_item_results.json"
    with open(items_path, "w") as f:
        json.dump(all_items, f, indent=2)
    print(f"\nSaved {len(all_items)} per-item results to {items_path}")

    # ------------------------------------------------------------------
    # 5. Aggregate logit-lens metrics
    # ------------------------------------------------------------------
    n_layers_total = len(all_items[0]["margins_i1a"]) if all_items else 0
    aggregate: Dict[str, Dict] = {}

    for group_label in ["correcting", "resisting"]:
        items = [r for r in all_items if r["label"] == group_label]
        if not items:
            continue

        margins_i1a = np.array([r["margins_i1a"] for r in items])
        margins_i1c = np.array([r["margins_i1c"] for r in items])
        entropies_i1a = np.array([r["entropies_i1a"] for r in items])
        entropies_i1c = np.array([r["entropies_i1c"] for r in items])

        # Commitment layer stats
        cl_i1a = [r["commitment_layer_i1a"] for r in items if r["commitment_layer_i1a"] is not None]
        cl_i1c = [r["commitment_layer_i1c"] for r in items if r["commitment_layer_i1c"] is not None]

        aggregate[group_label] = {
            "n": len(items),
            # Margin trajectories
            "mean_margin_i1a": margins_i1a.mean(axis=0).tolist(),
            "std_margin_i1a": margins_i1a.std(axis=0).tolist(),
            "mean_margin_i1c": margins_i1c.mean(axis=0).tolist(),
            "std_margin_i1c": margins_i1c.std(axis=0).tolist(),
            # Entropy trajectories
            "mean_entropy_i1a": entropies_i1a.mean(axis=0).tolist(),
            "mean_entropy_i1c": entropies_i1c.mean(axis=0).tolist(),
            # Commitment layer
            "mean_commitment_layer_i1a": float(np.mean(cl_i1a)) if cl_i1a else None,
            "median_commitment_layer_i1a": float(np.median(cl_i1a)) if cl_i1a else None,
            "mean_commitment_layer_i1c": float(np.mean(cl_i1c)) if cl_i1c else None,
            "median_commitment_layer_i1c": float(np.median(cl_i1c)) if cl_i1c else None,
            "n_committed_i1a": len(cl_i1a),
            "n_committed_i1c": len(cl_i1c),
            # Peak margin stats
            "peak_margin_layer_i1a": int(np.argmax(margins_i1a.mean(axis=0))),
            "peak_margin_value_i1a": float(margins_i1a.mean(axis=0).max()),
            "final_margin_i1a": float(margins_i1a.mean(axis=0)[-1]),
            "final_margin_i1c": float(margins_i1c.mean(axis=0)[-1]),
        }

    # Print summary
    for group_label, agg in aggregate.items():
        print(f"\n{'='*60}")
        print(f"  {group_label.upper()} (N={agg['n']})")
        print(f"{'='*60}")
        print(f"  Mean commitment layer (i1a): {agg['mean_commitment_layer_i1a']}")
        print(f"  Mean commitment layer (i1c): {agg['mean_commitment_layer_i1c']}")
        print(f"  Peak margin layer (i1a): {agg['peak_margin_layer_i1a']}")
        print(f"  Final margin (i1a): {agg['final_margin_i1a']:.3f}")
        print(f"  Final margin (i1c): {agg['final_margin_i1c']:.3f}")

    # ------------------------------------------------------------------
    # 6. Intervention timing: patch gating direction at each layer
    # ------------------------------------------------------------------
    activations_path = args.extraction_dir / "activations.pt"
    intervention_results: Dict[str, List[Dict]] = {}

    if activations_path.exists():
        print(f"\nLoading activations from {activations_path}...")
        payload = torch.load(activations_path, map_location="cpu")
        activations = payload["activations"]

        steering_vector = _build_steering_vector(
            activations=activations,
            metadata=meta,
            position=args.vector_position,
            layer_index=args.vector_layer,
        )
        print(f"Steering vector norm: {steering_vector.norm().item():.4f}")
        print(f"Intervention layers: {args.intervention_layers}")

        # Only patch resisting items under i1a (where the gating mechanism operates)
        resisting_uids = [u for u in uids if uid_to_label.get(u) == "resisting"]

        for block_idx in tqdm(args.intervention_layers, desc="Intervention sweep"):
            layer_records: List[Dict] = []

            for uid in resisting_uids:
                mc_item = uid_to_mc.get(uid)
                if mc_item is None:
                    continue

                prompt_i1a = build_prompt(mc_item, I1A_TEXT, tag=args.tag)

                # Baseline (unpatched)
                baseline = run_intervention_at_layer(
                    model, tokenizer, device, prompt_i1a,
                    token_id_a, token_id_b, mc_item.correct_label,
                    block_index=block_idx,
                    vector=torch.zeros_like(steering_vector),
                )

                # Patched
                patched = run_intervention_at_layer(
                    model, tokenizer, device, prompt_i1a,
                    token_id_a, token_id_b, mc_item.correct_label,
                    block_index=block_idx,
                    vector=steering_vector,
                )

                flipped = not baseline["correct"] and patched["correct"]
                layer_records.append({
                    "uid": uid,
                    "block_index": block_idx,
                    "baseline_margin": baseline["signed_margin"],
                    "patched_margin": patched["signed_margin"],
                    "margin_shift": patched["signed_margin"] - baseline["signed_margin"],
                    "baseline_correct": baseline["correct"],
                    "patched_correct": patched["correct"],
                    "flipped_to_correct": flipped,
                })

            intervention_results[str(block_idx)] = layer_records

        # Compute per-layer flip rates
        intervention_timing: List[Dict] = []
        for block_idx in args.intervention_layers:
            records = intervention_results.get(str(block_idx), [])
            if not records:
                continue
            n_total = len(records)
            n_flipped = sum(1 for r in records if r["flipped_to_correct"])
            mean_shift = float(np.mean([r["margin_shift"] for r in records]))
            intervention_timing.append({
                "block_index": block_idx,
                "n_items": n_total,
                "flip_rate": n_flipped / n_total if n_total > 0 else 0.0,
                "n_flipped": n_flipped,
                "mean_margin_shift": mean_shift,
            })

        # Save intervention item-level results
        intervention_items_path = args.output_dir / "intervention_item_results.jsonl"
        with open(intervention_items_path, "w") as f:
            for block_idx_str, records in intervention_results.items():
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
        print(f"\nSaved intervention item results to {intervention_items_path}")

        # Print intervention timing
        print(f"\n{'='*60}")
        print("  INTERVENTION TIMING (resisting items, i1a)")
        print(f"{'='*60}")
        for entry in intervention_timing:
            print(
                f"  Block {entry['block_index']:2d}: "
                f"flip_rate={entry['flip_rate']:.3f} "
                f"({entry['n_flipped']}/{entry['n_items']}), "
                f"mean_margin_shift={entry['mean_margin_shift']:+.3f}"
            )
    else:
        print(
            f"\nSkipping intervention timing: {activations_path} not found. "
            f"Run extraction first to enable this analysis."
        )
        intervention_timing = []

    # ------------------------------------------------------------------
    # 7. Save comprehensive results
    # ------------------------------------------------------------------
    summary = {
        "model": args.model,
        "n_items": len(all_items),
        "n_transformer_layers": n_transformer_layers,
        "n_layers_in_logit_lens": n_layers_total,
        "entropy_threshold": args.entropy_threshold,
        "extraction_dir": str(args.extraction_dir),
        "aggregate": aggregate,
        "intervention_timing": intervention_timing,
        "intervention_layers": args.intervention_layers,
        "vector_layer": args.vector_layer,
        "vector_position": args.vector_position,
    }

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
