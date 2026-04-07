"""CMAP control: compare instruction-sensitivity profiles for correcting vs resisting.

Uses saved logit lens data. The key test: if the CMAP forward sweep's
monotonically increasing profile is just "later layers = more processed info",
then correcting items should show the SAME shape as resisting items.
If resisting items show a qualitatively different profile (e.g., a bump at 
L22-24 not present for correcting), then the sweep IS gating-specific.

Additionally re-analyzes saved CMAP sweep JSON with per-group breakdown.
"""
import json
import sys
from pathlib import Path

import numpy as np


def load_logit_lens_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_cmap_sweep(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def analyze_logit_lens_profiles(summary: dict):
    """Compare instruction-sensitivity profile (i1c - i1a) per layer."""
    print("=" * 70)
    print("  LOGIT LENS: Instruction Sensitivity Profile (i1c - i1a)")
    print("=" * 70)

    groups = summary["groups"]
    n_layers = summary["n_layers"]

    for group_name in ["correcting", "resisting"]:
        g = groups[group_name]
        i1a = np.array(g["i1a_mean_trajectory"])
        i1c = np.array(g["i1c_mean_trajectory"])
        div = i1c - i1a  # instruction sensitivity at each layer

        print(f"\n  {group_name.upper()} (N={g['n']})")
        print(f"  {'Layer':>6s} {'i1a':>8s} {'i1c':>8s} {'div(i1c-i1a)':>14s}")
        for l in range(len(i1a)):
            flag = ""
            if 22 <= l <= 24:
                flag = " <-- L22-24"
            print(f"  {l:6d} {i1a[l]:+8.3f} {i1c[l]:+8.3f} {div[l]:+14.3f}{flag}")

    # Compute normalized profiles for shape comparison
    corr_i1a = np.array(groups["correcting"]["i1a_mean_trajectory"])
    corr_i1c = np.array(groups["correcting"]["i1c_mean_trajectory"])
    res_i1a = np.array(groups["resisting"]["i1a_mean_trajectory"])
    res_i1c = np.array(groups["resisting"]["i1c_mean_trajectory"])

    corr_div = corr_i1c - corr_i1a
    res_div = res_i1c - res_i1a

    # Normalize to max=1 for shape comparison
    corr_div_norm = corr_div / (np.abs(corr_div).max() + 1e-8)
    res_div_norm = res_div / (np.abs(res_div).max() + 1e-8)

    print(f"\n{'='*70}")
    print(f"  SHAPE COMPARISON: Normalized instruction sensitivity")
    print(f"  (If shapes are identical, CMAP profile is trivial)")
    print(f"{'='*70}")

    print(f"\n  {'Layer':>6s} {'Corr(norm)':>12s} {'Resist(norm)':>14s} {'Diff':>10s}")
    shape_diffs = []
    for l in range(len(corr_div)):
        diff = res_div_norm[l] - corr_div_norm[l]
        shape_diffs.append(diff)
        flag = ""
        if abs(diff) > 0.1:
            flag = " ***"
        if 22 <= l <= 24:
            flag += " [L22-24]"
        print(f"  {l:6d} {corr_div_norm[l]:+12.4f} {res_div_norm[l]:+14.4f} {diff:+10.4f}{flag}")

    # Key diagnostic metrics
    shape_diffs = np.array(shape_diffs)
    corr_peak = np.argmax(np.abs(corr_div))
    res_peak = np.argmax(np.abs(res_div))

    # Where does the divergence first become large?
    threshold = 0.3 * np.abs(res_div).max()
    corr_onset = next((i for i, v in enumerate(np.abs(corr_div)) if v > threshold), -1)
    res_onset = next((i for i, v in enumerate(np.abs(res_div)) if v > threshold), -1)

    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC METRICS")
    print(f"{'='*70}")
    print(f"  Correcting: peak div at L{corr_peak} ({corr_div[corr_peak]:+.3f}), "
          f"onset (30% of max) at L{corr_onset}")
    print(f"  Resisting:  peak div at L{res_peak} ({res_div[res_peak]:+.3f}), "
          f"onset (30% of max) at L{res_onset}")
    print(f"  Onset difference: {res_onset - corr_onset} layers")

    # Direction of divergence
    print(f"\n  Direction check (sign of i1c - i1a):")
    for l in [20, 22, 23, 24, 28, 32, 35]:
        if l < len(corr_div):
            print(f"    L{l:2d}: corr={corr_div[l]:+.3f}, resist={res_div[l]:+.3f}")

    # Correlation between profiles
    from scipy.stats import pearsonr, spearmanr
    r_pearson, p_pearson = pearsonr(corr_div, res_div)
    r_spearman, _ = spearmanr(corr_div, res_div)
    print(f"\n  Profile correlation:")
    print(f"    Pearson r = {r_pearson:.4f} (p = {p_pearson:.2e})")
    print(f"    Spearman r = {r_spearman:.4f}")
    if r_pearson > 0.95:
        print(f"    --> SAME SHAPE: CMAP profile is likely trivial (later = more info)")
    elif r_pearson > 0.8:
        print(f"    --> SIMILAR SHAPE: mostly trivial but some group-specific signal")
    else:
        print(f"    --> DIFFERENT SHAPES: gating-specific signal present")

    # The most important test: is the RESISTING profile more concentrated at L22-24?
    # Compute fraction of total divergence in L22-24 window
    l22_24_corr = np.abs(corr_div[22:25]).sum() / (np.abs(corr_div).sum() + 1e-8)
    l22_24_res = np.abs(res_div[22:25]).sum() / (np.abs(res_div).sum() + 1e-8)

    print(f"\n  Fraction of total divergence in L22-24:")
    print(f"    Correcting: {l22_24_corr:.4f} ({l22_24_corr*100:.1f}%)")
    print(f"    Resisting:  {l22_24_res:.4f} ({l22_24_res*100:.1f}%)")
    if l22_24_res > l22_24_corr * 1.5:
        print(f"    --> RESISTING has disproportionate signal at L22-24 ({l22_24_res/l22_24_corr:.1f}x)")
    else:
        print(f"    --> No special concentration at L22-24 for resisting items")

    # Also check: does resisting show SIGN REVERSAL that correcting doesn't?
    corr_sign_changes = np.sum(np.diff(np.sign(corr_div)) != 0)
    res_sign_changes = np.sum(np.diff(np.sign(res_div)) != 0)
    print(f"\n  Sign changes in divergence profile:")
    print(f"    Correcting: {corr_sign_changes}")
    print(f"    Resisting:  {res_sign_changes}")

    return {
        "profile_correlation_pearson": float(r_pearson),
        "profile_correlation_spearman": float(r_spearman),
        "correcting_peak_layer": int(corr_peak),
        "resisting_peak_layer": int(res_peak),
        "correcting_onset_layer": int(corr_onset),
        "resisting_onset_layer": int(res_onset),
        "l22_24_fraction_correcting": float(l22_24_corr),
        "l22_24_fraction_resisting": float(l22_24_res),
        "correcting_div_profile": corr_div.tolist(),
        "resisting_div_profile": res_div.tolist(),
    }


def main():
    base = Path("new-phase-results/mechanism")

    # Load logit lens summary
    ll_summary = load_logit_lens_summary(base / "logit_lens_qwen/summary.json")
    results = analyze_logit_lens_profiles(ll_summary)

    out_path = base / "cmap_control_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
