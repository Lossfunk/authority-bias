"""Experiment 11: Logit-Space Analysis + Inverted-Prior Test (Truth-Tracking vs Prior-Consistency Control).

Extends Exp10 to distinguish whether instruction-based selectivity is:
1. Truth-tracking: Model evaluates correctness and selectively suppresses wrong endorsements
2. Prior-consistency control: Model suppresses endorsements that conflict with its prior belief

Key Innovation: Inverted-prior test on items where m_N0 < 0 (model's neutral prediction is WRONG).

Design:
- Uses existing Exp10 data (no new model runs needed for Parts A-D)
- Analyzes in log-odds margin space (logit-space) for proper ratio metrics
- Three-layer reporting: r_w (Layer 1), r_c (Layer 2), dr (Layer 3 intersection)
- Bootstrap with single-index-resample for coherent CIs across all metrics

Core Metrics (Logit-Space):
- Log-odds margin: m = log(p_c / p_w) = logit(p_c) when p_c + p_w = 1
- Effects: effect_w_I0 = m_N0 - m_W0, effect_c_I0 = m_C0 - m_N0 (per instruction state)
- Efficacy: efficacy_w = effect_w_I0 - effect_w_I1, efficacy_c = effect_c_I0 - effect_c_I1
- Relative suppression: r_w = 1 - (effect_w_I1 / effect_w_I0), r_c = 1 - (effect_c_I1 / effect_c_I0)
- Differential suppression: dr = r_w - r_c (equals norm_selectivity under sign-consistent mask)

Critical Test (Part B - Inverted-Prior):
On items where m_N0 < 0 (model's neutral prediction is WRONG):
- Correct endorsement CONFLICTS with model's prior
- Wrong endorsement AGREES with model's prior

Slices:
1. Full inverted-prior set (m_N0 < 0)
2. Low-confidence wrong (bottom 50% of |m_N0|)
3. High-confidence wrong (top 25% of |m_N0|) - CRITICAL discriminant
4. Top-10% confidence wrong (maximum discriminant power)

If truth-tracking:
- r_w stays positive in high-confidence-wrong slice
- Instruction still suppresses wrong endorsements even when they agree with prior

If prior-consistency control:
- r_w collapses toward zero or negative in high-confidence-wrong slice
- Model protects its (wrong) prior more strongly when confident

Implementation Structure:
- Part A: Logit-space replication of Exp10 with proper bootstrap
- Part B: Inverted-prior test with multi-slice analysis (PRIMARY RESULT)
- Part C: Confidence-binned analysis (descriptive)
- Part D: Baseline instruction effect panel
- Part E: Extended tags (Priority 4)
- Part F: Cross-model validation (Priority 5)

Technical Safeguards:
- Three-layer reporting prevents "artifact of conditioning" criticism
- Validity via isfinite() masks (not re-thresholding in bootstrap)
- Disjoint diagnostics: structural (n_mask_valid_*), numerical (pct_ratio_nonfinite_given_valid_*)
- Bootstrap conditioning diagnostics: pct_boot_zero_mask_*, pct_boot_zero_finite_*
- τ-sensitivity analysis with CIs
- Absolute mask robustness check
"""
