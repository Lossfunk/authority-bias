"""Summarize exp23 authority-transfer results across models / contexts.

Reads neurips-results/exp23/authority_transfer_contexts/{model}/fold0/authority_transfer_summary.json
and prints:
  - per-model best layer per (variant, condition): wrong-rate at alpha=0 vs alpha=1, delta
  - aggregated transfer table: rows = condition (note/system/rag x source/user W1)
                              cols = variant (source/user/assistant), values = best-delta wrong-rate (a1 - a0)
  - cross-context transfer claim: source removal works on system/rag source-authority too.

Headline rows are alpha=1 per variant minus alpha=0 baseline; layers chosen as the
layer that best reduces note_source_W1 wrong-rate for variant=source (matched layer
across other variants). All numbers are on the 50/50 held-out fold0 eval split
using exp22 train-fold directions reused.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1] / "neurips-results" / "exp23" / "authority_transfer_contexts"
MODELS = ["qwen35", "gpt_oss", "olmo2", "olmo31"]
CONDITIONS = [
    "note_source_W1",
    "system_source_W1",
    "rag_source_W1",
    "note_user_W1",
    "system_user_W1",
    "rag_user_W1",
]
VARIANTS = ["source", "user", "assistant"]


def load_rows(model: str):
    p = ROOT / model / "fold0" / "authority_transfer_summary.json"
    data = json.loads(p.read_text())
    return data["summary"], data.get("layers", [])


def index_rows(rows):
    # idx[(variant, layer, alpha, condition)] = wrong_rate
    idx = {}
    for r in rows:
        idx[(r["variant"], int(r["layer"]), float(r["alpha"]), r["condition"])] = r[
            "wrong_rate"
        ]
    return idx


def baseline_wrong_rate(idx, condition):
    # baseline = alpha=0 wrong-rate. Same across variants (no intervention) — pick any.
    for v in VARIANTS:
        for k in [k for k in idx if k[0] == v and k[2] == 0.0 and k[3] == condition]:
            return idx[k]
    return float("nan")


def best_layer_for_variant(idx, variant, condition):
    layers = sorted({k[1] for k in idx if k[0] == variant})
    best_layer, best_wrong = None, float("inf")
    for L in layers:
        wr = idx.get((variant, L, 1.0, condition))
        if wr is None:
            continue
        if wr < best_wrong:
            best_wrong = wr
            best_layer = L
    return best_layer, best_wrong


def fmt_pct(x):
    return f"{100 * x:6.2f}"


def fmt_delta(x):
    return f"{100 * x:+7.2f}"


def main():
    print()
    for model in MODELS:
        rows, layers = load_rows(model)
        idx = index_rows(rows)
        print(f"=== {model} (layers={layers}) ===")
        # Header
        print(
            f"  {'condition':<18}  {'baseline':>8}   "
            + "  ".join([f"{v[:4]}_a1(L)  Δ".ljust(16) for v in VARIANTS])
        )
        for cond in CONDITIONS:
            base = baseline_wrong_rate(idx, cond)
            cells = []
            for v in VARIANTS:
                L, wr1 = best_layer_for_variant(idx, v, cond)
                delta = wr1 - base
                cells.append(f"{fmt_pct(wr1)}(L{L})  {fmt_delta(delta)}")
            print(f"  {cond:<18}  {fmt_pct(base):>8}   " + "  ".join(cells))
        print()

    # ---------- aggregated cross-model table ----------
    print()
    print("=== Cross-model summary (mean across 4 models, alpha=1 wrong-rate at best layer) ===")
    print(
        f"  {'condition':<18}  {'baseline':>8}   "
        + "  ".join([f"{v[:4]}_a1   Δ".ljust(14) for v in VARIANTS])
    )
    agg_base = defaultdict(list)
    agg = defaultdict(lambda: defaultdict(list))  # cond -> variant -> [wrong_rate]
    for model in MODELS:
        rows, _ = load_rows(model)
        idx = index_rows(rows)
        for cond in CONDITIONS:
            base = baseline_wrong_rate(idx, cond)
            agg_base[cond].append(base)
            for v in VARIANTS:
                L, wr1 = best_layer_for_variant(idx, v, cond)
                agg[cond][v].append(wr1)
    for cond in CONDITIONS:
        base_mean = sum(agg_base[cond]) / len(agg_base[cond])
        cells = []
        for v in VARIANTS:
            mwr = sum(agg[cond][v]) / len(agg[cond][v])
            delta = mwr - base_mean
            cells.append(f"{fmt_pct(mwr):>6}  {fmt_delta(delta)}")
        print(f"  {cond:<18}  {fmt_pct(base_mean):>8}   " + "  ".join(cells))
    print()

    # ---------- transfer claim test ----------
    print()
    print("=== Transfer claim: source-removal Δ on source-authority transfer prompts ===")
    print(
        f"  {'context':<10}  {'baseline_W1':>11}  {'src_a1_W1':>9}  {'Δ src':>7}  "
        f"{'usr_a1_W1':>9}  {'Δ usr':>7}  {'asst_a1_W1':>10}  {'Δ asst':>7}"
    )
    for ctx, cond in [("note", "note_source_W1"), ("system", "system_source_W1"), ("rag", "rag_source_W1")]:
        base_mean = sum(agg_base[cond]) / len(agg_base[cond])
        cells = []
        for v in VARIANTS:
            mwr = sum(agg[cond][v]) / len(agg[cond][v])
            cells.extend([fmt_pct(mwr), fmt_delta(mwr - base_mean)])
        print(f"  {ctx:<10}  {fmt_pct(base_mean):>11}  " + "  ".join(cells))
    print()


if __name__ == "__main__":
    main()
