"""Logit-space metrics and bootstrap infrastructure for Exp11.

Implements:
- Log-odds margin computation (from raw logits or probabilities)
- Per-item metrics: effects, efficacy, relative suppression, normalized selectivity
- Coherent bootstrap with single-index-resample
- Three-layer reporting: r_w (Layer 1), r_c (Layer 2), dr (Layer 3 intersection)
- τ-sensitivity analysis with full CIs
- Endorsement susceptibility analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# Default parameters
TAU_DEFAULT = 1e-3
TAU_VALUES = [1e-4, 1e-3, 1e-2]
EPS = 1e-10  # Epsilon for numerical stability
TRIM_FRACTION_DEFAULT = 0.1  # 10% symmetric trim for robust mean


@dataclass
class ItemMetrics:
    """Per-item metrics in logit space."""

    # Raw margins
    m_N0: float  # Neutral, no instruction
    m_N1: float  # Neutral, with instruction
    m_W0: float  # Wrong endorsement, no instruction
    m_W1: float  # Wrong endorsement, with instruction
    m_C0: float  # Correct endorsement, no instruction
    m_C1: float  # Correct endorsement, with instruction

    # Effects (positive = endorsement worked)
    effect_w_I0: float  # Wrong effect without instruction
    effect_w_I1: float  # Wrong effect with instruction
    effect_c_I0: float  # Correct effect without instruction
    effect_c_I1: float  # Correct effect with instruction

    # Efficacy (positive = instruction helped)
    efficacy_w: float
    efficacy_c: float

    # Item metadata
    question_id: str


def trimmed_mean(arr: np.ndarray, trim_fraction: float = TRIM_FRACTION_DEFAULT) -> float:
    """Compute symmetric trimmed mean for 1D arrays."""
    if arr.size == 0:
        return np.nan
    if trim_fraction <= 0:
        return float(np.mean(arr))

    k = int(np.floor(trim_fraction * arr.size))
    if 2 * k >= arr.size:
        return float(np.mean(arr))

    sorted_arr = np.sort(arr)
    return float(np.mean(sorted_arr[k: arr.size - k]))


def compute_margin_from_logits(l_c: float, l_w: float) -> float:
    """Compute log-odds margin from raw logits.

    This is the preferred method when logits are available.

    Args:
        l_c: Logit for correct answer
        l_w: Logit for wrong answer

    Returns:
        m = l_c - l_w (log-odds margin)
    """
    return l_c - l_w


def compute_margin_from_probs(p_c: float, eps: float = EPS) -> float:
    """Compute log-odds margin from forced-choice probability.

    Fallback method when only probabilities are available.
    Assumes p_c + p_w = 1 (forced-choice).

    CRITICAL: Clip p_c first, then derive p_w = 1 - p_c to preserve identity.
    DO NOT clip both independently.

    Args:
        p_c: Probability of correct answer (assumed p_c + p_w = 1)
        eps: Epsilon for numerical stability

    Returns:
        m = log(p_c / p_w) = logit(p_c)
    """
    p_c_clipped = np.clip(p_c, eps, 1 - eps)
    p_w = 1 - p_c_clipped  # Preserves p_c + p_w = 1
    m = np.log(p_c_clipped / p_w)
    return m


def compute_item_metrics(
    m_N0: float,
    m_N1: float,
    m_W0: float,
    m_W1: float,
    m_C0: float,
    m_C1: float,
    question_id: str = "",
) -> ItemMetrics:
    """Compute all metrics for a single item.

    Args:
        m_N0, m_N1: Neutral margins (no instruction, with instruction)
        m_W0, m_W1: Wrong endorsement margins
        m_C0, m_C1: Correct endorsement margins
        question_id: Item identifier

    Returns:
        ItemMetrics with all computed values
    """
    # Effects (positive = endorsement worked in intended direction)
    effect_w_I0 = m_N0 - m_W0  # Positive when wrong endorsement shifts toward wrong
    effect_w_I1 = m_N1 - m_W1
    effect_c_I0 = m_C0 - m_N0  # Positive when correct endorsement shifts toward correct
    effect_c_I1 = m_C1 - m_N1

    # Efficacy (positive = instruction reduced effect)
    efficacy_w = effect_w_I0 - effect_w_I1
    efficacy_c = effect_c_I0 - effect_c_I1

    return ItemMetrics(
        m_N0=m_N0,
        m_N1=m_N1,
        m_W0=m_W0,
        m_W1=m_W1,
        m_C0=m_C0,
        m_C1=m_C1,
        effect_w_I0=effect_w_I0,
        effect_w_I1=effect_w_I1,
        effect_c_I0=effect_c_I0,
        effect_c_I1=effect_c_I1,
        efficacy_w=efficacy_w,
        efficacy_c=efficacy_c,
        question_id=question_id,
    )


def bootstrap_all_metrics_coherent(
    items: List[ItemMetrics],
    n_boot: int = 10000,
    use_sign_consistent_mask: bool = True,
    tau: float = TAU_DEFAULT,
) -> Dict[str, Tuple[float, Tuple[float, float]]]:
    """Bootstrap all metrics coherently with single index resample per replicate.

    CRITICAL: Single index resample drives ALL metrics (conditional and unconditional).
    This ensures coherent CIs across all reported statistics.

    Args:
        items: List of ItemMetrics
        n_boot: Number of bootstrap replicates
        use_sign_consistent_mask: If True, use sign-consistent mask (primary).
                                   If False, use absolute mask (robustness).
        tau: Threshold for validity mask

    Returns:
        Dict mapping metric names to (point_estimate, (ci_low, ci_high))
    """
    n_total = len(items)

    # Extract arrays (all items)
    effect_w_I0 = np.array([item.effect_w_I0 for item in items])
    effect_c_I0 = np.array([item.effect_c_I0 for item in items])
    effect_w_I1 = np.array([item.effect_w_I1 for item in items])
    effect_c_I1 = np.array([item.effect_c_I1 for item in items])
    efficacy_w = np.array([item.efficacy_w for item in items])
    efficacy_c = np.array([item.efficacy_c for item in items])
    m_N0 = np.array([item.m_N0 for item in items])
    m_N1 = np.array([item.m_N1 for item in items])

    # Precompute ratio metrics with NaN for mask-invalid items
    r_w = np.full(n_total, np.nan)
    r_c = np.full(n_total, np.nan)
    dr = np.full(n_total, np.nan)
    norm_sel = np.full(n_total, np.nan)  # Only used for absolute mask robustness

    # EXPLICIT MASK ARRAYS (structural validity)
    if use_sign_consistent_mask:
        mask_w = effect_w_I0 > tau         # Mask for Layer 1
        mask_c = effect_c_I0 > tau         # Mask for Layer 2
        mask_both = mask_w & mask_c        # Mask for Layer 3
        mask_abs = None                    # Not used under sign-consistent
    else:
        mask_abs = (np.abs(effect_w_I0) > tau) & (np.abs(effect_c_I0) > tau)
        mask_w = mask_c = mask_both = None  # Not used under absolute

    # Track ratio non-finite counts (will be set after ratio computation)
    n_ratio_nonfinite_w = 0
    n_ratio_nonfinite_c = 0
    n_ratio_nonfinite_both = 0
    n_ratio_nonfinite_abs = 0

    if use_sign_consistent_mask:
        # Layer 1: Compute r_w for mask-valid items
        for i in np.where(mask_w)[0]:
            r_w[i] = 1 - (effect_w_I1[i] / effect_w_I0[i])
        # Count and sanitize non-finite
        nonfinite_w = mask_w & ~np.isfinite(r_w)
        n_ratio_nonfinite_w = int(np.sum(nonfinite_w))
        r_w[nonfinite_w] = np.nan

        # Layer 2: Compute r_c for mask-valid items
        for i in np.where(mask_c)[0]:
            r_c[i] = 1 - (effect_c_I1[i] / effect_c_I0[i])
        # Count and sanitize non-finite
        nonfinite_c = mask_c & ~np.isfinite(r_c)
        n_ratio_nonfinite_c = int(np.sum(nonfinite_c))
        r_c[nonfinite_c] = np.nan

        # Layer 3: Compute dr for intersection
        for i in np.where(mask_both)[0]:
            dr[i] = r_w[i] - r_c[i]
        # dr non-finite if either component is non-finite
        nonfinite_both = mask_both & ~np.isfinite(dr)
        n_ratio_nonfinite_both = int(np.sum(nonfinite_both))
        dr[nonfinite_both] = np.nan

    else:
        # Absolute mask: Compute norm_selectivity only
        for i in np.where(mask_abs)[0]:
            norm_sel[i] = (efficacy_w[i] / abs(effect_w_I0[i])) - \
                          (efficacy_c[i] / abs(effect_c_I0[i]))
        # Count and sanitize non-finite
        nonfinite_abs = mask_abs & ~np.isfinite(norm_sel)
        n_ratio_nonfinite_abs = int(np.sum(nonfinite_abs))
        norm_sel[nonfinite_abs] = np.nan

    # Baseline shift
    delta_baseline = m_N1 - m_N0

    # Initialize stats dict based on mask type
    # ALL proportions bootstrap cleanly for CIs
    if use_sign_consistent_mask:
        stats = {
            # Layer 1: r_w only
            "r_w_mean": [], "r_w_median": [],
            # Layer 2: r_c only
            "r_c_mean": [], "r_c_median": [],
            # Layer 3: dr (intersection)
            "dr_mean": [], "dr_median": [], "frac_dr_pos": [],
            # Unconditional metrics (on full resample)
            "frac_eff_w_gt_c": [],
            "eff_diff_mean": [],
            "eff_diff_median": [],
            "eff_diff_trimmed_mean": [],
            "baseline_shift_mean": [],
            # Diagnostics with bootstrap CIs (proportions)
            "pct_r_w_outside": [], "pct_r_c_outside": [],
            "pct_mask_valid_w": [], "pct_mask_valid_c": [], "pct_mask_valid_both": [],
            "pct_ratio_nonfinite_given_valid_w": [], "pct_ratio_nonfinite_given_valid_c": [],
            "pct_ratio_nonfinite_given_valid_both": []
        }
    else:
        # Absolute mask: only norm_selectivity + unconditional (no r_w/r_c/dr)
        stats = {
            "norm_sel_mean": [], "norm_sel_median": [],
            "frac_eff_w_gt_c": [],
            "eff_diff_mean": [],
            "eff_diff_median": [],
            "eff_diff_trimmed_mean": [],
            "baseline_shift_mean": [],
            # Diagnostics with bootstrap CIs
            "pct_mask_valid_abs": [],
            "pct_ratio_nonfinite_given_valid_abs": []
        }

    # Track how often bootstrap has zero MASK-VALID items (structural)
    n_boot_zero_mask_w = 0
    n_boot_zero_mask_c = 0
    n_boot_zero_mask_both = 0
    n_boot_zero_mask_abs = 0

    # Track how often bootstrap has zero FINITE items (zero finite-after-ratio)
    # "Finite" means mask-valid AND ratio didn't blow up
    n_boot_zero_finite_w = 0
    n_boot_zero_finite_c = 0
    n_boot_zero_finite_both = 0
    n_boot_zero_finite_abs = 0

    for _ in range(n_boot):
        # SINGLE index resample for ALL metrics
        boot_idx = np.random.choice(n_total, n_total, replace=True)

        # Unconditional metrics on full resample (always computed)
        eff_diff_boot = efficacy_w[boot_idx] - efficacy_c[boot_idx]
        stats["frac_eff_w_gt_c"].append(np.mean(eff_diff_boot > 0))
        stats["eff_diff_mean"].append(np.mean(eff_diff_boot))
        stats["eff_diff_median"].append(np.median(eff_diff_boot))
        stats["eff_diff_trimmed_mean"].append(trimmed_mean(eff_diff_boot))
        stats["baseline_shift_mean"].append(np.mean(delta_baseline[boot_idx]))

        # Get resampled arrays
        boot_r_w = r_w[boot_idx]
        boot_r_c = r_c[boot_idx]
        boot_dr = dr[boot_idx]
        boot_norm_sel = norm_sel[boot_idx]

        if use_sign_consistent_mask:
            # Resample mask arrays for diagnostic CIs
            boot_mask_w = mask_w[boot_idx]
            boot_mask_c = mask_c[boot_idx]
            boot_mask_both = mask_both[boot_idx]

            # Compute mask counts for this resample
            n_mask_w_boot = np.sum(boot_mask_w)
            n_mask_c_boot = np.sum(boot_mask_c)
            n_mask_both_boot = np.sum(boot_mask_both)

            # Track zero mask-valid replicates (structural)
            if n_mask_w_boot == 0:
                n_boot_zero_mask_w += 1
            if n_mask_c_boot == 0:
                n_boot_zero_mask_c += 1
            if n_mask_both_boot == 0:
                n_boot_zero_mask_both += 1

            # Diagnostic CIs: mask coverage (proportions bootstrap cleanly)
            stats["pct_mask_valid_w"].append(100 * np.mean(boot_mask_w))
            stats["pct_mask_valid_c"].append(100 * np.mean(boot_mask_c))
            stats["pct_mask_valid_both"].append(100 * np.mean(boot_mask_both))

            # VALIDITY VIA isfinite (finite-after-ratio)
            # Layer 1: r_w
            finite_w_boot = np.isfinite(boot_r_w)
            n_finite_w_boot = np.sum(finite_w_boot)

            # Diagnostic CI: ratio-nonfinite given mask-valid
            if n_mask_w_boot > 0:
                n_nonfinite_w_boot = n_mask_w_boot - n_finite_w_boot
                stats["pct_ratio_nonfinite_given_valid_w"].append(100 * n_nonfinite_w_boot / n_mask_w_boot)
            else:
                stats["pct_ratio_nonfinite_given_valid_w"].append(np.nan)

            if n_finite_w_boot > 0:
                finite_r_w = boot_r_w[finite_w_boot]
                stats["r_w_mean"].append(np.mean(finite_r_w))
                stats["r_w_median"].append(np.median(finite_r_w))
                stats["pct_r_w_outside"].append(100 * np.mean((finite_r_w < 0) | (finite_r_w > 1)))
            else:
                n_boot_zero_finite_w += 1
                stats["r_w_mean"].append(np.nan)
                stats["r_w_median"].append(np.nan)
                stats["pct_r_w_outside"].append(np.nan)

            # Layer 2: r_c
            finite_c_boot = np.isfinite(boot_r_c)
            n_finite_c_boot = np.sum(finite_c_boot)

            if n_mask_c_boot > 0:
                n_nonfinite_c_boot = n_mask_c_boot - n_finite_c_boot
                stats["pct_ratio_nonfinite_given_valid_c"].append(100 * n_nonfinite_c_boot / n_mask_c_boot)
            else:
                stats["pct_ratio_nonfinite_given_valid_c"].append(np.nan)

            if n_finite_c_boot > 0:
                finite_r_c = boot_r_c[finite_c_boot]
                stats["r_c_mean"].append(np.mean(finite_r_c))
                stats["r_c_median"].append(np.median(finite_r_c))
                stats["pct_r_c_outside"].append(100 * np.mean((finite_r_c < 0) | (finite_r_c > 1)))
            else:
                n_boot_zero_finite_c += 1
                stats["r_c_mean"].append(np.nan)
                stats["r_c_median"].append(np.nan)
                stats["pct_r_c_outside"].append(np.nan)

            # Layer 3: dr (intersection mask + finite)
            finite_both_boot = np.isfinite(boot_dr)
            n_finite_both_boot = np.sum(finite_both_boot)

            if n_mask_both_boot > 0:
                n_nonfinite_both_boot = n_mask_both_boot - n_finite_both_boot
                stats["pct_ratio_nonfinite_given_valid_both"].append(100 * n_nonfinite_both_boot / n_mask_both_boot)
            else:
                stats["pct_ratio_nonfinite_given_valid_both"].append(np.nan)

            if n_finite_both_boot > 0:
                finite_dr = boot_dr[finite_both_boot]
                stats["dr_mean"].append(np.mean(finite_dr))
                stats["dr_median"].append(np.median(finite_dr))
                stats["frac_dr_pos"].append(np.mean(finite_dr > 0))
            else:
                n_boot_zero_finite_both += 1
                stats["dr_mean"].append(np.nan)
                stats["dr_median"].append(np.nan)
                stats["frac_dr_pos"].append(np.nan)

        else:
            # Absolute mask: only norm_selectivity (no Layer 1/2/3 language)
            boot_mask_abs = mask_abs[boot_idx]
            n_mask_abs_boot = np.sum(boot_mask_abs)

            # Track zero mask-valid replicates (structural)
            if n_mask_abs_boot == 0:
                n_boot_zero_mask_abs += 1

            # Diagnostic CIs: mask coverage
            stats["pct_mask_valid_abs"].append(100 * np.mean(boot_mask_abs))

            finite_abs_boot = np.isfinite(boot_norm_sel)
            n_finite_abs_boot = np.sum(finite_abs_boot)

            # Diagnostic CI: ratio-nonfinite given mask-valid
            if n_mask_abs_boot > 0:
                n_nonfinite_abs_boot = n_mask_abs_boot - n_finite_abs_boot
                stats["pct_ratio_nonfinite_given_valid_abs"].append(100 * n_nonfinite_abs_boot / n_mask_abs_boot)
            else:
                stats["pct_ratio_nonfinite_given_valid_abs"].append(np.nan)

            if n_finite_abs_boot > 0:
                finite_norm_sel = boot_norm_sel[finite_abs_boot]
                stats["norm_sel_mean"].append(np.mean(finite_norm_sel))
                stats["norm_sel_median"].append(np.median(finite_norm_sel))
            else:
                n_boot_zero_finite_abs += 1
                stats["norm_sel_mean"].append(np.nan)
                stats["norm_sel_median"].append(np.nan)

    # Build results with point estimates and CIs
    results = {}

    # Extract bootstrap CIs for all stats keys
    for key in stats:
        arr = np.array(stats[key])
        arr = arr[~np.isnan(arr)]
        if len(arr) > 0:
            results[key] = (float(np.mean(arr)), tuple(np.percentile(arr, [2.5, 97.5]).tolist()))
        else:
            results[key] = (np.nan, (np.nan, np.nan))

    # Unconditional metrics point estimates
    results["frac_eff_w_gt_c"] = (float(np.mean(efficacy_w - efficacy_c > 0)),
                                  results["frac_eff_w_gt_c"][1])
    eff_diff = efficacy_w - efficacy_c
    results["eff_diff_mean"] = (float(np.mean(eff_diff)),
                                results["eff_diff_mean"][1])
    results["eff_diff_median"] = (float(np.median(eff_diff)),
                                  results["eff_diff_median"][1])
    results["eff_diff_trimmed_mean"] = (float(trimmed_mean(eff_diff)),
                                        results["eff_diff_trimmed_mean"][1])
    results["baseline_shift_mean"] = (float(np.mean(delta_baseline)),
                                      results["baseline_shift_mean"][1])
    results["n_total"] = n_total

    if use_sign_consistent_mask:
        # DISJOINT DIAGNOSTICS:
        # 1) Mask coverage (structural): how many items pass the mask
        # 2) Ratio non-finite given mask-valid (numerical): among mask-valid, how many blow up

        n_mask_valid_w = int(np.sum(mask_w))
        n_mask_valid_c = int(np.sum(mask_c))
        n_mask_valid_both = int(np.sum(mask_both))

        # Finite counts (mask-valid AND ratio is finite)
        finite_r_w_mask = np.isfinite(r_w)
        finite_r_c_mask = np.isfinite(r_c)
        finite_dr_mask = np.isfinite(dr)

        n_finite_w = int(np.sum(finite_r_w_mask))
        n_finite_c = int(np.sum(finite_r_c_mask))
        n_finite_both = int(np.sum(finite_dr_mask))

        # Layer 1: r_w point estimates (on finite items)
        if n_finite_w > 0:
            finite_r_w = r_w[finite_r_w_mask]
            results["r_w_mean"] = (float(np.mean(finite_r_w)), results["r_w_mean"][1])
            results["r_w_median"] = (float(np.median(finite_r_w)), results["r_w_median"][1])
            results["pct_r_w_outside"] = (float(100 * np.mean((finite_r_w < 0) | (finite_r_w > 1))),
                                          results["pct_r_w_outside"][1])

        # Layer 2: r_c point estimates (on finite items)
        if n_finite_c > 0:
            finite_r_c = r_c[finite_r_c_mask]
            results["r_c_mean"] = (float(np.mean(finite_r_c)), results["r_c_mean"][1])
            results["r_c_median"] = (float(np.median(finite_r_c)), results["r_c_median"][1])
            results["pct_r_c_outside"] = (float(100 * np.mean((finite_r_c < 0) | (finite_r_c > 1))),
                                          results["pct_r_c_outside"][1])

        # Layer 3: dr point estimates (on finite items)
        if n_finite_both > 0:
            finite_dr = dr[finite_dr_mask]
            results["dr_mean"] = (float(np.mean(finite_dr)), results["dr_mean"][1])
            results["dr_median"] = (float(np.median(finite_dr)), results["dr_median"][1])
            results["frac_dr_pos"] = (float(np.mean(finite_dr > 0)), results["frac_dr_pos"][1])

        # STRUCTURAL: Mask coverage per layer (preserve CIs from bootstrap)
        results["n_mask_valid_w"] = n_mask_valid_w
        results["n_mask_valid_c"] = n_mask_valid_c
        results["n_mask_valid_both"] = n_mask_valid_both

        # CRITICAL: Preserve CI tuples! Do NOT overwrite with bare floats.
        # WRONG: results["pct_mask_valid_w"] = float(100 * n_mask_valid_w / n_total)
        # RIGHT: Compute point estimate, keep bootstrap CI from earlier
        results["pct_mask_valid_w"] = (float(100 * n_mask_valid_w / n_total), results["pct_mask_valid_w"][1])
        results["pct_mask_valid_c"] = (float(100 * n_mask_valid_c / n_total), results["pct_mask_valid_c"][1])
        results["pct_mask_valid_both"] = (float(100 * n_mask_valid_both / n_total), results["pct_mask_valid_both"][1])

        # NUMERICAL: Ratio non-finite given mask-valid (preserve CIs from bootstrap)
        # Same pattern: compute point estimate, preserve bootstrap CI
        point_w = float(100 * n_ratio_nonfinite_w / n_mask_valid_w) if n_mask_valid_w > 0 else np.nan
        point_c = float(100 * n_ratio_nonfinite_c / n_mask_valid_c) if n_mask_valid_c > 0 else np.nan
        point_both = float(100 * n_ratio_nonfinite_both / n_mask_valid_both) if n_mask_valid_both > 0 else np.nan
        results["pct_ratio_nonfinite_given_valid_w"] = (point_w, results["pct_ratio_nonfinite_given_valid_w"][1])
        results["pct_ratio_nonfinite_given_valid_c"] = (point_c, results["pct_ratio_nonfinite_given_valid_c"][1])
        results["pct_ratio_nonfinite_given_valid_both"] = (point_both, results["pct_ratio_nonfinite_given_valid_both"][1])

        # Finite counts (for reporting)
        results["n_finite_w"] = n_finite_w
        results["n_finite_c"] = n_finite_c
        results["n_finite_both"] = n_finite_both

        # Bootstrap diagnostics: zero mask-valid (structural - no items pass threshold)
        results["pct_boot_zero_mask_w"] = float(100 * n_boot_zero_mask_w / n_boot)
        results["pct_boot_zero_mask_c"] = float(100 * n_boot_zero_mask_c / n_boot)
        results["pct_boot_zero_mask_both"] = float(100 * n_boot_zero_mask_both / n_boot)

        # Bootstrap diagnostics: zero finite-after-ratio (numerical - ratio blew up everywhere)
        results["pct_boot_zero_finite_w"] = float(100 * n_boot_zero_finite_w / n_boot)
        results["pct_boot_zero_finite_c"] = float(100 * n_boot_zero_finite_c / n_boot)
        results["pct_boot_zero_finite_both"] = float(100 * n_boot_zero_finite_both / n_boot)

    else:
        # Absolute mask: only norm_selectivity (no Layer 1/2/3 language)
        n_mask_valid_abs = int(np.sum(mask_abs))

        # Finite counts
        finite_norm_sel_mask = np.isfinite(norm_sel)
        n_finite_abs = int(np.sum(finite_norm_sel_mask))

        if n_finite_abs > 0:
            finite_norm_sel = norm_sel[finite_norm_sel_mask]
            results["norm_sel_mean"] = (float(np.mean(finite_norm_sel)), results["norm_sel_mean"][1])
            results["norm_sel_median"] = (float(np.median(finite_norm_sel)), results["norm_sel_median"][1])

        # STRUCTURAL: Mask coverage (preserve CIs from bootstrap)
        results["n_mask_valid_abs"] = n_mask_valid_abs
        results["pct_mask_valid_abs"] = (float(100 * n_mask_valid_abs / n_total), results["pct_mask_valid_abs"][1])

        # NUMERICAL: Ratio non-finite given mask-valid (preserve CIs from bootstrap)
        point_abs = float(100 * n_ratio_nonfinite_abs / n_mask_valid_abs) if n_mask_valid_abs > 0 else np.nan
        results["pct_ratio_nonfinite_given_valid_abs"] = (point_abs, results["pct_ratio_nonfinite_given_valid_abs"][1])

        # Finite count
        results["n_finite_abs"] = n_finite_abs

        # Bootstrap diagnostics: zero mask-valid (structural)
        results["pct_boot_zero_mask_abs"] = float(100 * n_boot_zero_mask_abs / n_boot)

        # Bootstrap diagnostics: zero finite-after-ratio (numerical)
        results["pct_boot_zero_finite_abs"] = float(100 * n_boot_zero_finite_abs / n_boot)

    return results


def run_tau_sensitivity_with_ci(
    items: List[ItemMetrics],
    tau_values: List[float] = TAU_VALUES,
    n_boot: int = 5000,
    use_sign_consistent_mask: bool = True
) -> Dict[str, Dict]:
    """Run analysis for multiple τ values with bootstrap CIs.

    Args:
        items: List of ItemMetrics
        tau_values: List of τ thresholds to test
        n_boot: Number of bootstrap replicates
        use_sign_consistent_mask: If True, use sign-consistent mask (primary)

    Returns:
        Dict mapping "tau_{tau}" to metrics dict
    """
    results = {}
    for tau in tau_values:
        # Use the same coherent bootstrap routine
        metrics = bootstrap_all_metrics_coherent(
            items,
            n_boot=n_boot,
            use_sign_consistent_mask=use_sign_consistent_mask,
            tau=tau
        )
        if use_sign_consistent_mask:
            results[f"tau_{tau}"] = {
                "tau": tau,
                "n_total": metrics["n_total"],
                # STRUCTURAL: Mask coverage per layer
                "n_mask_valid_w": metrics["n_mask_valid_w"],
                "n_mask_valid_c": metrics["n_mask_valid_c"],
                "n_mask_valid_both": metrics["n_mask_valid_both"],
                "pct_mask_valid_w": metrics["pct_mask_valid_w"],
                "pct_mask_valid_c": metrics["pct_mask_valid_c"],
                "pct_mask_valid_both": metrics["pct_mask_valid_both"],
                # NUMERICAL: Ratio non-finite given mask-valid
                "pct_ratio_nonfinite_given_valid_w": metrics["pct_ratio_nonfinite_given_valid_w"],
                "pct_ratio_nonfinite_given_valid_c": metrics["pct_ratio_nonfinite_given_valid_c"],
                "pct_ratio_nonfinite_given_valid_both": metrics["pct_ratio_nonfinite_given_valid_both"],
                # Finite counts
                "n_finite_w": metrics["n_finite_w"],
                "n_finite_c": metrics["n_finite_c"],
                "n_finite_both": metrics["n_finite_both"],
                # Layer 1 & 2 metrics
                "r_w_mean": metrics["r_w_mean"],
                "r_c_mean": metrics["r_c_mean"],
                # Layer 3 (intersection)
                "dr_mean": metrics["dr_mean"],
                "dr_median": metrics["dr_median"],
                "frac_dr_pos": metrics["frac_dr_pos"],
                # Bootstrap diagnostics: zero mask-valid
                "pct_boot_zero_mask_w": metrics["pct_boot_zero_mask_w"],
                "pct_boot_zero_mask_c": metrics["pct_boot_zero_mask_c"],
                "pct_boot_zero_mask_both": metrics["pct_boot_zero_mask_both"],
                # Bootstrap diagnostics: zero finite-after-ratio
                "pct_boot_zero_finite_w": metrics["pct_boot_zero_finite_w"],
                "pct_boot_zero_finite_c": metrics["pct_boot_zero_finite_c"],
                "pct_boot_zero_finite_both": metrics["pct_boot_zero_finite_both"]
            }
        else:
            # Absolute mask: only norm_selectivity (no Layer 1/2/3 language)
            results[f"tau_{tau}"] = {
                "tau": tau,
                "n_total": metrics["n_total"],
                # STRUCTURAL: Mask coverage
                "n_mask_valid_abs": metrics["n_mask_valid_abs"],
                "pct_mask_valid_abs": metrics["pct_mask_valid_abs"],
                # NUMERICAL: Ratio non-finite given mask-valid
                "pct_ratio_nonfinite_given_valid_abs": metrics["pct_ratio_nonfinite_given_valid_abs"],
                # Finite count
                "n_finite_abs": metrics["n_finite_abs"],
                # Metrics
                "norm_sel_mean": metrics["norm_sel_mean"],
                "norm_sel_median": metrics["norm_sel_median"],
                # Bootstrap diagnostics: zero mask-valid
                "pct_boot_zero_mask_abs": metrics["pct_boot_zero_mask_abs"],
                # Bootstrap diagnostics: zero finite-after-ratio
                "pct_boot_zero_finite_abs": metrics["pct_boot_zero_finite_abs"]
            }
    return results


def compute_slice_m_N0_stats(items: List[ItemMetrics]) -> Dict[str, float]:
    """Compute m_N0 distribution statistics for a slice.

    Args:
        items: List of ItemMetrics

    Returns:
        Dict with mean, median, min, max of m_N0
    """
    if not items:
        return {
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "n": 0
        }

    m_N0_arr = np.array([item.m_N0 for item in items])

    return {
        "mean": float(np.mean(m_N0_arr)),
        "median": float(np.median(m_N0_arr)),
        "min": float(np.min(m_N0_arr)),
        "max": float(np.max(m_N0_arr)),
        "n": len(items)
    }


def compute_endorsement_susceptibility(
    items: List[ItemMetrics],
    tau: float = TAU_DEFAULT
) -> Dict[str, float]:
    """Compute endorsement susceptibility metrics for a slice.

    Returns p(effect_c_I0 > τ) and distribution stats for effect_c_I0.
    This quantifies WHY intersection shrinkage happens.

    Args:
        items: List of ItemMetrics
        tau: Threshold for validity

    Returns:
        Dict with susceptibility metrics
    """
    if not items:
        return {
            "p_effect_c_works": 0.0,
            "effect_c_I0_mean": np.nan,
            "effect_c_I0_median": np.nan,
            "effect_c_I0_q25": np.nan,
            "effect_c_I0_q75": np.nan,
            "n": 0
        }

    effect_c_I0_arr = np.array([item.effect_c_I0 for item in items])

    return {
        "p_effect_c_works": float(np.mean(effect_c_I0_arr > tau)),
        "effect_c_I0_mean": float(np.mean(effect_c_I0_arr)),
        "effect_c_I0_median": float(np.median(effect_c_I0_arr)),
        "effect_c_I0_q25": float(np.percentile(effect_c_I0_arr, 25)),
        "effect_c_I0_q75": float(np.percentile(effect_c_I0_arr, 75)),
        "n": len(items)
    }
