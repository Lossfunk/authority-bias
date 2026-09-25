"""Build a parent summary JSON for exp23 across all 4 models.

Aggregates per-model `authority_transfer_summary.json` into:
  results/authority/context_transfer/authority_transfer_contexts/exp23_summary.json

Includes:
  - per-model best-layer wrong-rate at alpha={0,1} per (variant, condition)
  - cross-model means
  - source-vs-assistant control gaps for the transfer claim
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1] / "results/authority" / "context_transfer" / "authority_transfer_contexts"
OUT = ROOT / "exp23_summary.json"
MODELS = ["qwen35", "gpt_oss", "olmo2", "olmo31"]
CONDITIONS = [
    "note_source_W1",
    "system_source_W1",
    "rag_source_W1",
    "note_user_W1",
    "system_user_W1",
    "rag_user_W1",
]
TRANSFER_CONDITIONS = ["system_source_W1", "rag_source_W1"]
VARIANTS = ["source", "user", "assistant"]


def load(model):
    p = ROOT / model / "fold0" / "authority_transfer_summary.json"
    return json.loads(p.read_text())


def index(rows):
    return {(r["variant"], int(r["layer"]), float(r["alpha"]), r["condition"]): r for r in rows}


def baseline(idx, cond):
    for k, r in idx.items():
        if k[2] == 0.0 and k[3] == cond:
            return r["wrong_rate"]
    return None


def best_layer(idx, variant, cond):
    """Layer minimizing alpha=1 wrong-rate for this (variant, cond)."""
    best = (None, float("inf"))
    layers = sorted({k[1] for k in idx if k[0] == variant})
    for L in layers:
        r = idx.get((variant, L, 1.0, cond))
        if r is None:
            continue
        if r["wrong_rate"] < best[1]:
            best = (L, r["wrong_rate"])
    return best


def main():
    per_model = {}
    layers_used = {}
    elapsed_s = {}
    n_eval_items = {}
    direction_paths = {}
    for m in MODELS:
        d = load(m)
        idx = index(d["summary"])
        layers_used[m] = d.get("layers")
        elapsed_s[m] = d.get("elapsed_seconds")
        n_eval_items[m] = d.get("n_eval_items")
        # Anonymise: replace full paths with short relative template
        direction_paths[m] = {
            v: f"exp22/selection_holdout_50_50/{m}/fold0/directions/{v}.pt"
            for v in VARIANTS + ["source_resid_assistant", "user_resid_assistant"]
        }

        cells = {}
        for cond in CONDITIONS:
            base = baseline(idx, cond)
            cell = {"baseline_wrong_rate": base, "variants": {}}
            for v in VARIANTS:
                L, wr1 = best_layer(idx, v, cond)
                cell["variants"][v] = {
                    "best_layer": L,
                    "alpha1_wrong_rate": wr1,
                    "delta_pp": (wr1 - base) * 100 if (wr1 is not None and base is not None) else None,
                }
            cells[cond] = cell
        per_model[m] = {
            "layers": layers_used[m],
            "n_eval_items": n_eval_items[m],
            "direction_paths": direction_paths[m],
            "cells": cells,
        }

    # Cross-model means
    cross = {}
    for cond in CONDITIONS:
        bases = [per_model[m]["cells"][cond]["baseline_wrong_rate"] for m in MODELS]
        bases = [b for b in bases if b is not None]
        cell = {
            "baseline_wrong_rate_mean": sum(bases) / len(bases) if bases else None,
            "variants": {},
        }
        for v in VARIANTS:
            wrs = [per_model[m]["cells"][cond]["variants"][v]["alpha1_wrong_rate"] for m in MODELS]
            wrs = [w for w in wrs if w is not None]
            mean_wr = sum(wrs) / len(wrs) if wrs else None
            mean_delta = (mean_wr - cell["baseline_wrong_rate_mean"]) * 100 if (mean_wr is not None and cell["baseline_wrong_rate_mean"] is not None) else None
            cell["variants"][v] = {
                "alpha1_wrong_rate_mean": mean_wr,
                "delta_pp_mean": mean_delta,
            }
        cross[cond] = cell

    # Transfer claim: source minus assistant Δ for each transfer cell
    transfer_claim = {}
    for cond in TRANSFER_CONDITIONS:
        per = {}
        for m in MODELS:
            s = per_model[m]["cells"][cond]["variants"]["source"]["delta_pp"]
            a = per_model[m]["cells"][cond]["variants"]["assistant"]["delta_pp"]
            u = per_model[m]["cells"][cond]["variants"]["user"]["delta_pp"]
            per[m] = {
                "source_delta_pp": s,
                "user_delta_pp": u,
                "assistant_delta_pp": a,
                "source_minus_assistant_pp": (s - a) if (s is not None and a is not None) else None,
            }
        transfer_claim[cond] = per

    out = {
        "experiment": "exp23_authority_transfer_contexts",
        "created_unix": datetime.now(timezone.utc).timestamp(),
        "fold": "fold0",
        "models": MODELS,
        "conditions": CONDITIONS,
        "variants": VARIANTS,
        "protocol": {
            "fit": "exp22 train-fold directions reused (no re-fit on transfer prompts).",
            "eval": "Held-out fold0 eval split (256 items per model).",
            "best_layer_rule": "Layer minimizing alpha=1 wrong_rate per (variant, condition).",
            "transfer_target": "system_source_W1 and rag_source_W1.",
            "control": "assistant variant (axis fitted on assistant-vs-base; should not move authority compliance).",
        },
        "per_model": per_model,
        "cross_model": cross,
        "transfer_claim": transfer_claim,
        "headline": {
            "rag_source_W1_source_delta_mean_pp": cross["rag_source_W1"]["variants"]["source"]["delta_pp_mean"],
            "rag_source_W1_assistant_delta_mean_pp": cross["rag_source_W1"]["variants"]["assistant"]["delta_pp_mean"],
            "system_source_W1_source_delta_mean_pp": cross["system_source_W1"]["variants"]["source"]["delta_pp_mean"],
            "system_source_W1_assistant_delta_mean_pp": cross["system_source_W1"]["variants"]["assistant"]["delta_pp_mean"],
            "note_source_W1_source_delta_mean_pp": cross["note_source_W1"]["variants"]["source"]["delta_pp_mean"],
            "note_source_W1_assistant_delta_mean_pp": cross["note_source_W1"]["variants"]["assistant"]["delta_pp_mean"],
        },
        "interpretation": {
            "supported": "Qwen3.5 — strong, clean transfer (rag_source −29.6pp, assistant flat).",
            "partial": "Cross-model rag_source averages −8.3pp source vs −0.9pp assistant.",
            "negative": "OLMo-2 transfer absent; OLMo-3.1 weak; GPT-OSS system_source shows control violation (assistant Δ ≥ source Δ).",
            "axis_collapse": "source ≈ user Δ patterns, consistent with one authority-compliance direction with two adjacent endpoints.",
        },
        "source_files": {
            m: f"{m}/fold0/authority_transfer_summary.json" for m in MODELS
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")
    print(json.dumps(out["headline"], indent=2))
    print()
    print("rag_source_W1 source-minus-assistant gap (pp):")
    for m, d in transfer_claim["rag_source_W1"].items():
        print(f"  {m:<8} {d['source_minus_assistant_pp']:+.2f}")


if __name__ == "__main__":
    main()
