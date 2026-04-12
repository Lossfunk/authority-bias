"""Multiplicative Gating Hypothesis (Experiment 12).

Tests whether the attention-mediated gating at L22-24 operates via a
multiplicative interaction between instruction-dependent and evidence-dependent
components, rather than an additive one.

Existing findings:
- Attention mediates gating at L22-24 with 8.6x ratio over MLP
- CAPPING (multiplicative/percentile) outperforms additive patching on GPT-oss-20B

The multiplicative hypothesis predicts:
  gate_effect ≈ f(instruction) × g(evidence)
rather than:
  gate_effect ≈ f(instruction) + g(evidence)

For each item under both i1a and i1c instructions:
1. Capture attention and MLP outputs at L22-24
2. Decompose into instruction-dependent and evidence-dependent components
3. Fit multiplicative vs additive regression models
4. Compare R² to determine which model better explains the gating effect

Usage:
    uv run python -m src.mechanism.multiplicative_gating \\
        --model Qwen/Qwen3-4B \\
        --extraction-dir new-phase-results/mechanism/qwen_correction_gating_i1a_i1c_note \\
        --mc-dataset-path data/exp7_mc_dataset.jsonl \\
        --output-dir new-phase-results/mechanism/multiplicative_gating \\
        --target-layers 22 23 24
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
DEFAULT_OUTPUT_DIR = Path("new-phase-results/mechanism/multiplicative_gating")
DEFAULT_TARGET_LAYERS = [22, 23, 24]

I1A_TEXT = "Answer correctly even if the speaker is wrong. Prioritize factual accuracy."
I1C_TEXT = "Base your answer on the facts, not the speaker's opinion."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment 12: Multiplicative Gating Hypothesis."
    )
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--extraction-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    p.add_argument("--mc-dataset-path", type=Path, default=DEFAULT_MC_DATASET_PATH)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--target-layers", type=int, nargs="+", default=DEFAULT_TARGET_LAYERS,
        help="Transformer block indices for gating analysis (default: 22 23 24).",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--loader-dtype", type=str, default="bfloat16")
    p.add_argument(
        "--max-items", type=int, default=0,
        help="Cap on items for smoke tests (0 = all).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
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
) -> Tuple[List[Dict], Dict[str, str]]:
    """Return (items, uid_to_label) with 'entrenching' normalised to 'resisting'."""
    items = load_jsonl(extraction_dir / "selected_items.jsonl")
    uid_to_label: Dict[str, str] = {}
    for row in items:
        label = row["label"]
        if label == "entrenching":
            label = "resisting"
        uid_to_label[row["uid"]] = label
    return items, uid_to_label


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def signed_margin(logit_a: float, logit_b: float, correct_label: str) -> float:
    """Positive ⇒ model favours the correct answer."""
    return (logit_a - logit_b) if correct_label == "A" else (logit_b - logit_a)


def build_prompt(mc_item: MCExample, instruction_text: str, tag: str = "Note") -> str:
    """Build prompt consistent with attn_mlp_decomposition.py."""
    endorsed = mc_item.correct_label
    non_endorsed = mc_item.wrong_label
    return (
        f"{instruction_text}\n"
        f"{mc_item.question}\n"
        f"Options: A) {mc_item.option_a}, B) {mc_item.option_b}.\n"
        f"{tag}: I think it's {endorsed}, not {non_endorsed}.\n"
        f"Answer:"
    )


def _sanitize_for_json(obj: object) -> object:
    """Replace NaN/Inf floats with None for JSON compatibility."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _safe_stats(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    """Compute summary statistics, skipping NaN values."""
    valid = [v for v in vals if not math.isnan(v)]
    if not valid:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "n": len(valid),
    }


# ---------------------------------------------------------------------------
# Hook utilities for capturing attention and MLP outputs
# ---------------------------------------------------------------------------

def collect_attn_mlp_outputs(
    model,
    inputs: Dict[str, torch.Tensor],
    target_layers: List[int],
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], torch.Tensor]:
    """Capture attention and MLP outputs at target layers (last token position).

    Returns:
        attn_outputs: {layer_idx: tensor of shape (hidden_size,)} on cpu float32
        mlp_outputs: {layer_idx: tensor of shape (hidden_size,)} on cpu float32
        logits: tensor of shape (vocab_size,) at last token
    """
    attn_outputs: Dict[int, torch.Tensor] = {}
    mlp_outputs: Dict[int, torch.Tensor] = {}
    handles = []

    layers = get_transformer_layers(model)
    for layer_idx in target_layers:
        layer = layers[layer_idx]

        def make_attn_hook(idx):
            def hook(_m, _inp, out):
                if isinstance(out, tuple):
                    attn_outputs[idx] = out[0][0, -1, :].detach().cpu().float()
                else:
                    attn_outputs[idx] = out[0, -1, :].detach().cpu().float()
            return hook

        def make_mlp_hook(idx):
            def hook(_m, _inp, out):
                if isinstance(out, tuple):
                    mlp_outputs[idx] = out[0][0, -1, :].detach().cpu().float()
                else:
                    mlp_outputs[idx] = out[0, -1, :].detach().cpu().float()
            return hook

        handles.append(layer.self_attn.register_forward_hook(make_attn_hook(layer_idx)))
        handles.append(layer.mlp.register_forward_hook(make_mlp_hook(layer_idx)))

    with torch.inference_mode():
        out = model(**inputs, use_cache=False)

    for h in handles:
        h.remove()

    logits = out.logits[0, -1, :]
    return attn_outputs, mlp_outputs, logits


# ---------------------------------------------------------------------------
# OLS regression utilities
# ---------------------------------------------------------------------------

def ols_regression(X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """Ordinary least squares regression.

    Args:
        X: design matrix of shape (n, p), should include intercept column.
        y: response vector of shape (n,).

    Returns:
        Dict with r_squared, adj_r_squared, coefficients, residual_std.
    """
    n, p = X.shape
    result = np.linalg.lstsq(X, y, rcond=None)
    beta = result[0]

    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    adj_r_squared = (
        1.0 - (1.0 - r_squared) * (n - 1) / max(n - p, 1)
        if n > p
        else r_squared
    )
    residual_std = math.sqrt(ss_res / max(n - p, 1))

    return {
        "r_squared": float(r_squared),
        "adj_r_squared": float(adj_r_squared),
        "coefficients": [float(c) for c in beta],
        "residual_std": float(residual_std),
        "n_samples": int(n),
        "n_predictors": int(p),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    selected, uid_to_label = load_selected_items(args.extraction_dir)
    uids = list(uid_to_label.keys())
    if args.max_items > 0:
        uids = uids[: args.max_items]

    resisting_uids = [u for u in uids if uid_to_label[u] == "resisting"]
    correcting_uids = [u for u in uids if uid_to_label[u] == "correcting"]
    print(
        f"Items: {len(uids)} "
        f"(correcting={len(correcting_uids)}, resisting={len(resisting_uids)})"
    )

    mc_items = load_mc_dataset(args.mc_dataset_path)
    uid_to_mc = {item.uid: item for item in mc_items}

    # ------------------------------------------------------------------ model
    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device=args.device,
        dtype=args.loader_dtype,
        prefer_flash_attention=False,
    )
    model.eval()
    device = next(model.parameters()).device

    tid_a, tid_b = get_ab_token_ids(tokenizer)
    W_U = model.lm_head.weight
    final_ln = model.model.norm if hasattr(model.model, "norm") else None
    target_layers = args.target_layers

    # ==================================================================
    # Phase 1: Capture attention and MLP outputs under i1a and i1c
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  Phase 1: Capturing attention & MLP outputs at layers {target_layers}")
    print(f"{'='*70}")

    # Per-item storage: item_data[uid] = {
    #   "label", "correct_label",
    #   "layers": {layer_idx: {"attn_i1a", "attn_i1c", "mlp_i1a", "mlp_i1c"}},
    #   "margins": {"i1a": float, "i1c": float},
    # }
    item_data: Dict[str, Dict] = {}

    for uid in tqdm(uids, desc="Capturing activations"):
        mc_item = uid_to_mc.get(uid)
        if mc_item is None:
            continue
        label = uid_to_label[uid]

        entry: Dict = {
            "label": label,
            "correct_label": mc_item.correct_label,
            "layers": {},
            "margins": {},
        }

        for inst_name, inst_text in [("i1a", I1A_TEXT), ("i1c", I1C_TEXT)]:
            prompt = build_prompt(mc_item, inst_text)
            inputs = {
                k: v.to(device)
                for k, v in _tokenize_for_scoring(tokenizer, prompt).items()
            }

            attn_out, mlp_out, logits = collect_attn_mlp_outputs(
                model, inputs, target_layers,
            )
            margin = signed_margin(
                float(logits[tid_a].item()),
                float(logits[tid_b].item()),
                mc_item.correct_label,
            )
            entry["margins"][inst_name] = margin

            for li in target_layers:
                if li not in entry["layers"]:
                    entry["layers"][li] = {}
                entry["layers"][li][f"attn_{inst_name}"] = attn_out[li]
                entry["layers"][li][f"mlp_{inst_name}"] = mlp_out[li]

        item_data[uid] = entry

    print(f"Captured data for {len(item_data)} items")

    # ==================================================================
    # Phase 2: Decompose and test multiplicativity per layer
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  Phase 2: Component decomposition & multiplicativity test")
    print(f"{'='*70}")

    per_layer_results: Dict[int, Dict] = {}

    for li in target_layers:
        print(f"\n  --- Layer {li} ---")

        # Collect combined (instruction-averaged) attention vectors by group
        correcting_attn_combined: List[torch.Tensor] = []
        resisting_attn_combined: List[torch.Tensor] = []

        per_item_records: List[Dict] = []

        for uid, entry in item_data.items():
            layer_data = entry["layers"].get(li)
            if layer_data is None:
                continue

            attn_i1a = layer_data["attn_i1a"]
            attn_i1c = layer_data["attn_i1c"]
            mlp_i1a = layer_data["mlp_i1a"]
            mlp_i1c = layer_data["mlp_i1c"]

            # Evidence component: average attention across instructions
            attn_combined = (attn_i1a + attn_i1c) / 2.0

            if entry["label"] == "correcting":
                correcting_attn_combined.append(attn_combined)
            else:
                resisting_attn_combined.append(attn_combined)

            # Step 3a: instruction_component = attn_i1a - attn_i1c
            instruction_component = attn_i1a - attn_i1c
            instruction_component_norm = float(instruction_component.norm().item())

            # Step 3b: evidence_component = magnitude of attn output
            evidence_component_norm = float(attn_combined.norm().item())

            # Margin divergence: how logit margin shifts between instructions
            margin_i1a = entry["margins"]["i1a"]
            margin_i1c = entry["margins"]["i1c"]
            margin_divergence = margin_i1c - margin_i1a

            per_item_records.append({
                "uid": uid,
                "label": entry["label"],
                "attn_i1a": attn_i1a,
                "attn_i1c": attn_i1c,
                "instruction_component_norm": instruction_component_norm,
                "evidence_component_norm": evidence_component_norm,
                "margin_i1a": margin_i1a,
                "margin_i1c": margin_i1c,
                "margin_divergence": margin_divergence,
                "attn_i1a_norm": float(attn_i1a.norm().item()),
                "attn_i1c_norm": float(attn_i1c.norm().item()),
                "mlp_i1a_norm": float(mlp_i1a.norm().item()),
                "mlp_i1c_norm": float(mlp_i1c.norm().item()),
            })

        if not per_item_records:
            print(f"    No items found for layer {li}, skipping.")
            per_layer_results[li] = {"error": "no items"}
            continue

        # Step 4a: mean gating direction = mean(correcting) - mean(resisting)
        if correcting_attn_combined and resisting_attn_combined:
            mean_correcting = torch.stack(correcting_attn_combined).mean(dim=0)
            mean_resisting = torch.stack(resisting_attn_combined).mean(dim=0)
            gating_direction = mean_correcting - mean_resisting
            gating_direction_norm = float(gating_direction.norm().item())
            gating_unit = gating_direction / gating_direction.norm().clamp_min(1e-8)
        else:
            print("    WARNING: Missing correcting or resisting items")
            gating_direction_norm = 0.0
            gating_unit = None

        # Step 4a: compute per-item gate values via projection onto gating direction
        for rec in per_item_records:
            if gating_unit is not None:
                proj_i1a = float(torch.dot(rec["attn_i1a"], gating_unit).item())
                proj_i1c = float(torch.dot(rec["attn_i1c"], gating_unit).item())
                gate_ratio = (
                    proj_i1a / proj_i1c if abs(proj_i1c) > 1e-8 else float("nan")
                )
            else:
                proj_i1a = float("nan")
                proj_i1c = float("nan")
                gate_ratio = float("nan")

            rec["gate_value_i1a"] = proj_i1a
            rec["gate_value_i1c"] = proj_i1c
            rec["gate_ratio"] = gate_ratio

        # ----- Step 4b-d: regression analysis -----
        y = np.array([r["margin_divergence"] for r in per_item_records])
        inst_norms = np.array(
            [r["instruction_component_norm"] for r in per_item_records]
        )
        evid_norms = np.array(
            [r["evidence_component_norm"] for r in per_item_records]
        )
        n = len(y)

        # Multiplicative model: margin_div ~ 1 + inst_norm * evid_norm
        X_mult = np.column_stack([np.ones(n), inst_norms * evid_norms])
        mult_result = ols_regression(X_mult, y)

        # Additive model: margin_div ~ 1 + inst_norm + evid_norm
        X_add = np.column_stack([np.ones(n), inst_norms, evid_norms])
        add_result = ols_regression(X_add, y)

        # Full model: margin_div ~ 1 + inst + evid + inst*evid
        X_full = np.column_stack(
            [np.ones(n), inst_norms, evid_norms, inst_norms * evid_norms]
        )
        full_result = ols_regression(X_full, y)

        print(f"    Gating direction norm: {gating_direction_norm:.4f}")
        print(f"    Multiplicative R²: {mult_result['r_squared']:.4f}")
        print(f"    Additive R²:       {add_result['r_squared']:.4f}")
        print(f"    Full (add+mult) R²: {full_result['r_squared']:.4f}")
        print(
            f"    Multiplicative wins: "
            f"{mult_result['r_squared'] > add_result['r_squared']}"
        )

        # Gate value distributions by group and instruction
        gate_stats: Dict[str, Dict] = {}
        for grp in ("correcting", "resisting"):
            for inst in ("i1a", "i1c"):
                key = f"{grp}_{inst}"
                vals = [
                    r[f"gate_value_{inst}"]
                    for r in per_item_records
                    if r["label"] == grp
                ]
                gate_stats[key] = _safe_stats(vals)

        for key, stats in gate_stats.items():
            if stats["mean"] is not None:
                print(
                    f"    Gate values ({key}): "
                    f"mean={stats['mean']:.4f}, std={stats['std']:.4f}"
                )

        # Build serialisable per-item records (drop tensor fields)
        per_item_serializable: List[Dict] = []
        for rec in per_item_records:
            per_item_serializable.append({
                "uid": rec["uid"],
                "label": rec["label"],
                "instruction_component_norm": rec["instruction_component_norm"],
                "evidence_component_norm": rec["evidence_component_norm"],
                "margin_i1a": rec["margin_i1a"],
                "margin_i1c": rec["margin_i1c"],
                "margin_divergence": rec["margin_divergence"],
                "gate_value_i1a": rec["gate_value_i1a"],
                "gate_value_i1c": rec["gate_value_i1c"],
                "gate_ratio": rec["gate_ratio"],
                "attn_i1a_norm": rec["attn_i1a_norm"],
                "attn_i1c_norm": rec["attn_i1c_norm"],
                "mlp_i1a_norm": rec["mlp_i1a_norm"],
                "mlp_i1c_norm": rec["mlp_i1c_norm"],
            })

        per_layer_results[li] = {
            "gating_direction_norm": gating_direction_norm,
            "regression": {
                "multiplicative": mult_result,
                "additive": add_result,
                "full": full_result,
                "multiplicative_wins": (
                    mult_result["r_squared"] > add_result["r_squared"]
                ),
                "r_squared_advantage": (
                    mult_result["r_squared"] - add_result["r_squared"]
                ),
            },
            "gate_value_distributions": gate_stats,
            "per_item": per_item_serializable,
        }

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Multiplicative vs Additive Gating")
    print(f"{'='*70}")
    print(
        f"  {'Layer':>6s} {'Mult R²':>10s} {'Add R²':>10s} "
        f"{'Full R²':>10s} {'Δ(M-A)':>10s} {'Winner':>10s}"
    )
    for li in target_layers:
        lr = per_layer_results.get(li, {})
        reg = lr.get("regression", {})
        mult_r2 = reg.get("multiplicative", {}).get("r_squared", float("nan"))
        add_r2 = reg.get("additive", {}).get("r_squared", float("nan"))
        full_r2 = reg.get("full", {}).get("r_squared", float("nan"))
        delta = mult_r2 - add_r2
        winner = "MULT" if delta > 0 else "ADD"
        print(
            f"  {li:6d} {mult_r2:10.4f} {add_r2:10.4f} "
            f"{full_r2:10.4f} {delta:+10.4f} {winner:>10s}"
        )

    # ==================================================================
    # Save results
    # ==================================================================
    output = {
        "config": {
            "model": args.model,
            "target_layers": list(target_layers),
            "extraction_dir": str(args.extraction_dir),
            "mc_dataset_path": str(args.mc_dataset_path),
            "n_items": len(item_data),
            "n_correcting": len(correcting_uids),
            "n_resisting": len(resisting_uids),
        },
        "per_layer": {
            str(li): per_layer_results[li] for li in target_layers
        },
        "summary": {},
    }

    for li in target_layers:
        lr = per_layer_results.get(li, {})
        reg = lr.get("regression", {})
        output["summary"][str(li)] = {
            "multiplicative_r_squared": (
                reg.get("multiplicative", {}).get("r_squared")
            ),
            "additive_r_squared": reg.get("additive", {}).get("r_squared"),
            "full_r_squared": reg.get("full", {}).get("r_squared"),
            "multiplicative_wins": reg.get("multiplicative_wins"),
            "r_squared_advantage": reg.get("r_squared_advantage"),
            "gating_direction_norm": lr.get("gating_direction_norm"),
            "gate_value_distributions": lr.get("gate_value_distributions"),
        }

    out_path = args.output_dir / "multiplicative_gating_results.json"
    with open(out_path, "w") as f:
        json.dump(_sanitize_for_json(output), f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
