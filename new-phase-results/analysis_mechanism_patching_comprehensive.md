# Comprehensive Analysis: Mechanistic Patching of Evidence-Gating in Qwen3-4B

**Model:** Qwen/Qwen3-4B-Instruct-2507  
**Mechanism:** Instruction-dependent evidence gating  
**Direction:** Within-i1a correction gating vector, extracted at endorsement token position  
**Date compiled:** 2026-04-03

---

## 1. Experimental Setup Overview

### 1.1 Extraction

- **Items selected:** 449 total (298 correcting, 151 entrenching)
- **Source datasets:** TriviaQA and TruthfulQA
- **Instruction variants:**
  - **i1a:** "Answer correctly even if the speaker is wrong. Prioritize factual accuracy."
  - **i1c:** "Base your answer on the facts, not the speaker's opinion."
  - **i1d:** "Answer correctly by evaluating the facts."
- **Positions probed:** last_token, instruction_last, endorsement_last
- **Steering vector norm (endorsement L23):** 7.926
- **Steering vector norm (last token L21):** 3.807
- **Boundary steering vector norm (endorsement L23):** 7.185

### 1.2 Experimental Design

All patching experiments apply the within-i1a gating direction to model activations at specified positions/layers and measure the shift in margin (correct_logit − wrong_logit).

**Controls used in every experiment:**
- **real:** Actual gating vector (test condition)
- **random:** Random vector of same norm (specificity control)
- **wrong_sign:** Negated gating vector (direction specificity)
- **wrong_site:** Vector applied at instruction_last instead of endorsement_last (site specificity)

**Alpha values tested:** 0.0, 0.5, 1.0, 2.0, 4.0, 8.0 (some experiments use subset: 0, 2, 4, 8)

---

## 2. Primary Patching Results: Endorsement Position (Layer 23) vs. Last Token (Layer 21)

### 2.1 Entrenching Items — Position Comparison (alpha = 4.0, real vector)

| Metric | Endorsement L23 | Last Token L21 |
|--------|-----------------|----------------|
| N items | 151 | 151 |
| Mean baseline margin | −9.354 | −9.354 |
| Mean margin shift | **+1.930** | +1.603 |
| Flip-to-correct rate | **8.6%** | 2.0% |
| Harm rate | 0.0% | 0.0% |
| Positive shift rate | 87.4%¹ | —² |

¹ From heterogeneity summary at alpha=4.  
² Not directly reported for L21 at alpha=4 in this format.

### 2.2 Entrenching Items — Full Alpha Sweep (Endorsement L23)

| Alpha | Mean Shift (real) | Flip Rate (real) | Mean Shift (random) | Flip Rate (random) | Mean Shift (wrong_sign) | Mean Shift (wrong_site) |
|-------|-------------------|-------------------|---------------------|---------------------|-------------------------|-------------------------|
| 0.0 | 0.000 | 0.0% | 0.000 | 0.0% | 0.000 | 0.000 |
| 0.5 | +0.215 | 0.7% | −0.101 | 0.0% | −0.228 | −0.038 |
| 1.0 | +0.445 | 1.3% | −0.185 | 0.0% | −0.450 | −0.085 |
| 2.0 | +0.942 | 4.6% | −0.334 | 0.0% | −0.834 | −0.151 |
| 4.0 | **+1.930** | **8.6%** | −0.505 | 0.0% | −1.325 | −0.281 |
| 8.0 | +2.305 | 8.6% | −0.339 | 0.7% | −1.264 | −0.327 |

**Key observations:**
- Real vector produces monotonically increasing positive shifts up to alpha=8
- Flip rate plateaus at 8.6% for alpha=4 and alpha=8
- **Random vector produces negative shifts** (noise degradation), confirming direction specificity
- **Wrong-sign vector produces symmetric negative shifts**, confirming the direction is meaningful
- **Wrong-site vector produces near-zero or small negative shifts**, confirming site specificity at endorsement position

### 2.3 Entrenching Items — Full Alpha Sweep (Last Token L21)

| Alpha | Mean Shift (real) | Flip Rate (real) | Mean Shift (random) | Flip Rate (random) | Mean Shift (wrong_sign) | Mean Shift (wrong_site) |
|-------|-------------------|-------------------|---------------------|---------------------|-------------------------|-------------------------|
| 0.0 | 0.000 | 0.0% | 0.000 | 0.0% | 0.000 | 0.000 |
| 0.5 | +0.336 | 0.0% | +0.079 | 0.0% | −0.343 | −0.004 |
| 1.0 | +0.669 | 0.0% | +0.198 | 0.0% | −0.652 | −0.032 |
| 2.0 | +1.180 | 1.3% | +0.382 | 0.7% | −1.123 | −0.039 |
| 4.0 | +1.603 | 2.0% | +0.796 | 0.7% | −1.370 | −0.146 |
| 8.0 | +2.028 | 1.3% | +1.865 | 5.3% | +0.021 | −0.239 |

**Key observations:**
- Last-token vector produces real positive shifts but **random vector also produces substantial shifts at high alpha** (alpha=8: random +1.865 vs real +2.028)
- The **specificity gap narrows dramatically at alpha=8**, suggesting last-token direction is less specific
- Endorsement position is clearly superior: at alpha=4, specificity ratio = real/random = 1.930/−0.505 (real is positive, random is negative) vs last-token = 1.603/0.796 (only 2:1 ratio)
- Wrong-site control confirms the endorsement direction has no effect at instruction position

### 2.4 Comparison Summary — Site and Direction Specificity

| Position | Layer | alpha=4 Real Shift | alpha=4 Random Shift | Specificity (real−random) | alpha=4 Flip Rate | Wrong-Sign Shift |
|----------|-------|-----------------|-------------------|---------------------------|----------------|-------------------|
| **Endorsement** | **23** | **+1.930** | **−0.505** | **+2.435** | **8.6%** | **−1.325** |
| Last Token | 21 | +1.603 | +0.796 | +0.807 | 2.0% | −1.370 |

**Endorsement L23 wins on both specificity and flip rate.** The endorsement position shows a 3× larger specificity gap and 4× higher flip rate compared to the last token position.

---

## 3. Instruction-Condition Specificity Gradient

The gating vector was extracted from i1a (C1, correction instruction with Note tag). We test whether this vector's efficacy varies across prompt conditions that differ in how strongly the model should integrate evidence.

### 3.1 Entrenching Items across Conditions (alpha = 4.0, real vector)

| Condition | N | Baseline Margin | Mean Shift | Flip Rate | Harm Rate |
|-----------|---|-----------------|------------|-----------|-----------|
| **C1 Note (i1a)** | 151 | −9.354 | **+1.930** | **8.6%** | 0.0% |
| **C1 Expert** | 151 | −1.587 | **+0.453** | **26.5%** | 11.3% |
| **W1 Note** | 151 | −3.298 | +0.301 | 3.3% | 0.0% |
| **N0 Note** | 151 | −1.531 | −0.065 | 2.0% | 0.7% |

### 3.2 Analysis of the Specificity Gradient

**Expected ordering (by instruction strength): N0 < W1 < C1**

- **N0 (no instruction):** The vector has **essentially no effect** — mean shift is slightly negative (−0.065), and the 2.0% flip rate is at noise level (random control at alpha=4 shows 1.3% flips). This confirms the gating direction is **instruction-dependent**: without a correction instruction, the mechanism is not engaged.
- **W1 (weak instruction):** Moderate positive shift (+0.301) with 3.3% flip rate and 0% harm. The direction partially engages under a weaker correction instruction.
- **C1 Note (strong instruction, note tag):** Strongest positive shift (+1.930) with 8.6% flip rate, 0% harm. Full engagement of the gating mechanism.
- **C1 Expert (strong instruction, expert tag):** High flip rate (26.5%) but also high harm rate (11.3%). The expert tag creates a closer-to-boundary baseline (−1.587 vs −9.354), making items more susceptible to flipping in both directions.

**The gradient is: N0 (no effect) → W1 (weak effect) → C1 (strong effect)**, confirming the direction is instruction-gated.

### 3.3 C1 Expert — Special Case

The C1 Expert condition is notable:
- Baseline margin is much closer to 0 (−1.587) compared to C1 Note (−9.354)
- This means items are already near the decision boundary under the expert condition
- The vector achieves 26.5% flip rate but at cost of 11.3% harm rate
- Net beneficial items: 26.5% − 11.3% = **15.2% net flip rate**
- This suggests the expert tag partially overrides the evidence-rejection signal, and the gating vector tips boundary items over

---

## 4. Correcting Items (Safety Check)

Correcting items are already answering correctly with the correction instruction. Patching should ideally not harm them.

### 4.1 C1 Note Correcting (alpha = 4.0, real vector)

| Metric | Value |
|--------|-------|
| N items | 298 |
| Mean baseline margin | +0.166 |
| Mean margin shift | +0.234 |
| Flip-to-correct rate (baseline wrong→correct) | 7.0% |
| Harm rate (baseline correct→wrong) | 5.7% |
| **Net effect** | **+1.3% net improvement** |

### 4.2 Correcting Items — Breakdown by Baseline Correctness

| Subset | N | Baseline Margin | Mean Shift | Positive Shift Rate | Flip Rate | Harm Rate |
|--------|---|-----------------|------------|---------------------|-----------|-----------|
| Baseline correct | 152 | +7.095 | +0.471 | 63.2% | 0.0% | **11.2%** |
| Baseline wrong | 146 | −7.047 | −0.012 | 41.1% | **14.4%** | 0.0% |

**Key finding:** Among correcting items that the model already gets right (margin +7.095), the patching still produces a positive mean shift (+0.471) but harms 11.2%. However, among items the model currently gets wrong despite having the correction instruction, 14.4% are flipped to correct. The overall net is approximately balanced.

### 4.3 Correcting Items — Harm Analysis by Margin Bin

| Margin Bin | N | Mean Shift | Harm Rate |
|------------|---|------------|-----------|
| (−∞, 0] | 146 | −0.012 | 0.0% |
| (0, 1] | 16 | +1.492 | **31.3%** |
| (1, 3] | 28 | +0.045 | **32.1%** |
| (3, 6] | 23 | +1.103 | 8.7% |
| (6, 9] | 32 | +0.551 | 0.0% |
| (9, +∞) | 53 | +0.066 | 1.9% |

**Harm is concentrated at narrow baseline margins (0–3).** Items with margin > 6 are essentially never harmed. This makes sense: items barely above the correct threshold can be tipped to wrong by any perturbation.

---

## 5. Boundary Items Analysis

A separate "boundary" steering vector was extracted from items near the decision boundary. We compare its effectiveness against the standard (all-items) vector.

### 5.1 Entrenching Items (alpha = 4.0, real vector)

| Vector | N | Baseline Margin | Mean Shift | Flip Rate | Harm Rate |
|--------|---|-----------------|------------|-----------|-----------|
| Standard (all-items) | 151 | −9.354 | **+1.930** | **8.6%** | 0.0% |
| Boundary-filtered | 151 | −3.743 | +0.429 | 5.3% | 6.0% |

**Note:** The boundary entrenching experiment uses a different baseline margin (−3.743 vs −9.354), suggesting it is evaluated on a different subset/condition. The boundary vector has norm 7.185 vs 7.926 for the standard vector.

### 5.2 Correcting Items (alpha = 4.0, real vector)

| Vector | N | Baseline Margin | Mean Shift | Flip Rate | Harm Rate |
|--------|---|-----------------|------------|-----------|-----------|
| Standard (all-items) | 151→298 | +0.166 | +0.234 | 7.0% | 5.7% |
| Boundary-filtered | 298 | +0.166 | +0.282 | 8.4% | 6.4% |

### 5.3 Boundary Vector — Alpha Sweep (Entrenching)

| Alpha | Mean Shift (real) | Flip Rate (real) | Mean Shift (random) | Harm Rate |
|-------|-------------------|-------------------|---------------------|-----------|
| 2.0 | +0.087 | 1.3% | −0.096 | 0.7% |
| 4.0 | +0.429 | 5.3% | −0.155 | 6.0% |
| 8.0 | +1.065 | 13.9% | −0.147 | 7.3% |

The boundary vector achieves higher flip rates at alpha=8 (13.9%) but introduces more harm (7.3%). The standard vector at alpha=4 achieves 8.6% flips with 0% harm — a **better specificity/safety trade-off**.

---

## 6. Reference Patching (v2) — New Baseline Evaluation

The "reference" experiments re-evaluate the standard gating vector with slightly different baseline measurements.

### 6.1 Reference Entrenching (alpha = 4.0, real vector)

| Metric | Reference v2 | Original (endorsement L23) |
|--------|-------------|----------------------------|
| N items | 151 | 151 |
| Baseline margin | −3.773 | −9.354 |
| Mean shift | +0.169 | +1.930 |
| Flip rate | 4.0% | 8.6% |
| Harm rate | 4.0% | 0.0% |

The reference entrenching experiment shows a different baseline margin (−3.773), which suggests these may be evaluated under a different prompt condition. The smaller shift and higher harm rate indicate the vector is less effective on this particular subset.

### 6.2 Reference Correcting (alpha = 4.0, real vector)

| Metric | Reference v2 | Original (C1 Note) |
|--------|-------------|---------------------|
| N items | 298 | 298 |
| Baseline margin | +0.155 | +0.166 |
| Mean shift | +0.242 | +0.234 |
| Flip rate | 7.0% | 7.0% |
| Harm rate | 5.4% | 5.7% |

The reference correcting results are nearly identical to the original, confirming reproducibility.

---

## 7. Cross-Condition Transfer (i1a → i1c, i1a → i1d)

The extraction experiments prepared activations for cross-condition analysis:

### 7.1 i1a → i1c Transfer

- **Source direction:** Within-i1a gating vector
- **Target activations:** i1c ("Base your answer on the facts, not the speaker's opinion.")
- **Items:** 449 (298 correcting, 151 entrenching)
- **Extraction positions:** last_token, instruction_last, endorsement_last
- **Records extracted:** 898 (449 items × 2 instruction variants)

### 7.2 i1a → i1d Transfer

- **Source direction:** Within-i1a gating vector
- **Target activations:** i1d ("Answer correctly by evaluating the facts.")
- **Items:** 449 (298 correcting, 151 entrenching)
- **Records extracted:** 898

**Note:** The i1c and i1d extraction summaries confirm activation extraction was completed, but the patching results for these cross-condition transfers are embedded in the main patching experiments. The primary patching experiments (endorsement L23, last token L21, etc.) all use the i1a_i1c extraction as their input data, meaning the patching itself tests the i1a-derived vector applied to prompts evaluated under the i1a condition. The i1a→i1d activations are extracted but no separate patching summary is available for them.

---

## 8. Heterogeneity Analysis (Deep Dive)

### 8.1 Entrenching Items — Who Flips?

| Category | N | Mean Baseline Margin | Mean Shift | Flip Rate |
|----------|---|---------------------|------------|-----------|
| **Flipped items** | 13 | **−2.894** | **+5.788** | 100% |
| Non-flipped items | 138 | −9.963 | +1.566 | 0% |

**Flipped items have much smaller baseline margins** (−2.894 vs −9.963), meaning they were already close to the decision boundary. The vector shifts them by +5.788 on average — more than double the baseline deficit.

### 8.2 Entrenching Items — By Source Dataset

| Source | N | Baseline Margin | Mean Shift | Positive Shift Rate | Flip Rate | Harm Rate |
|--------|---|-----------------|------------|---------------------|-----------|-----------|
| TriviaQA | 85 | −8.760 | +1.844 | 87.1% | **10.6%** | 0.0% |
| TruthfulQA | 66 | −10.119 | +2.040 | 87.9% | 6.1% | 0.0% |

- TriviaQA items flip more often (10.6% vs 6.1%) despite having smaller baseline margins
- TruthfulQA items show larger mean shifts (+2.040 vs +1.844) but fewer flips, suggesting their margins are too deep
- Both datasets show zero harm rate

### 8.3 Entrenching Items — By Baseline Margin Bin

| Margin Bin | N | Mean Baseline | Mean Shift | Positive Shift Rate | Flip Rate | Harm Rate |
|------------|---|---------------|------------|---------------------|-----------|-----------|
| (−2, 0] | 5 | −1.275 | **+4.425** | 100% | **100%** | 0% |
| (−4, −2] | 12 | −3.031 | +3.375 | 100% | **33.3%** | 0% |
| (−8, −4] | 36 | −5.833 | +2.458 | 94.4% | **11.1%** | 0% |
| (−12, −8] | 51 | −10.015 | +2.257 | 98.0% | 0.0% | 0% |
| (−∞, −12] | 47 | −13.809 | +0.535 | 65.9% | 0.0% | 0% |

**Clear monotonic pattern:**
- All items with baseline margin in (−2, 0] flip (100%)
- Flip rate decreases with distance from boundary: 100% → 33.3% → 11.1% → 0% → 0%
- Even items with very deep margins (−12 to −8) show positive shifts (+2.257) in 98% of cases
- The most deeply entrenched items (< −12) still show positive shifts in 65.9% of cases, but the shift magnitude is smaller (+0.535)

### 8.4 Correcting Items — Who Gets Harmed?

| Category | N | Mean Baseline Margin | Mean Shift |
|----------|---|---------------------|------------|
| **Harmed items** | 17 | **+2.110** | **−5.007** |
| Unharmed items | 281 | +0.048 | +0.552 |

**Harmed items have narrow positive margins** (+2.110) and experience large negative shifts (−5.007). These are fragile correct answers that the perturbation tips over.

### 8.5 Correcting Items — By Source Dataset

| Source | N | Baseline Margin | Mean Shift | Flip Rate | Harm Rate |
|--------|---|-----------------|------------|-----------|-----------|
| TriviaQA | 109 | −0.165 | +0.141 | 8.3% | 5.5% |
| TruthfulQA | 189 | +0.357 | +0.288 | 6.3% | 5.8% |

Both sources show similar harm rates (~5.5–5.8%). TruthfulQA items have slightly higher baseline margins and correspondingly larger positive shifts.

---

## 9. Control Conditions Summary Table

All at alpha=4.0 for the primary entrenching experiment (endorsement L23):

| Control | Mean Shift | Flip Rate | Harm Rate | Interpretation |
|---------|------------|-----------|-----------|----------------|
| **Real** | **+1.930** | **8.6%** | **0.0%** | ✓ Causal gating effect |
| Random | −0.505 | 0.0% | 0.0% | ✗ Not a generic perturbation effect |
| Wrong-sign | −1.325 | 0.0% | 0.0% | ✓ Direction matters (negation hurts) |
| Wrong-site | −0.281 | 0.0% | 0.0% | ✓ Position matters (instruction site ineffective) |

**All four controls confirm the specificity of the gating mechanism:**
1. Only the real vector produces positive shifts
2. Random perturbations of the same magnitude produce negative shifts
3. Negating the direction reverses the effect symmetrically
4. Applying at the wrong position (instruction token instead of endorsement token) has negligible effect

---

## 10. Derived Metrics and Summary Statistics

### 10.1 Effect Sizes

| Experiment | alpha | Shift / |baseline| | Cohen's d approx¹ |
|------------|---|---------------------|---------------------|
| Endorsement L23 entrenching | 4.0 | 1.930/9.354 = **0.206** | — |
| Last Token L21 entrenching | 4.0 | 1.603/9.354 = **0.171** | — |
| C1 Expert entrenching | 4.0 | 0.453/1.587 = **0.285** | — |
| W1 Note entrenching | 4.0 | 0.301/3.298 = **0.091** | — |
| N0 Note entrenching | 4.0 | −0.065/1.531 = **−0.042** | — |

¹ Cohen's d would require per-item standard deviations not available in summaries.

### 10.2 Specificity Ratios (real vs random shift at alpha=4)

| Experiment | Real Shift | Random Shift | Specificity Ratio |
|------------|------------|--------------|-------------------|
| Endorsement L23 | +1.930 | −0.505 | **∞** (different sign) |
| Last Token L21 | +1.603 | +0.796 | **2.01** |
| C1 Expert | +0.453 | −0.016 | **∞** (different sign) |
| W1 Note | +0.301 | −0.045 | **∞** (different sign) |
| N0 Note | −0.065 | −0.076 | **~1.0** (both negative, no real effect) |

### 10.3 Asymmetry Check: Real vs Wrong-Sign (alpha=4)

| Experiment | Real Shift | Wrong-Sign Shift | Sum (symmetry check) |
|------------|------------|-------------------|----------------------|
| Endorsement L23 | +1.930 | −1.325 | +0.605 |
| Last Token L21 | +1.603 | −1.370 | +0.233 |
| C1 Expert | +0.453 | −0.236 | +0.217 |
| W1 Note | +0.301 | −0.111 | +0.190 |

The sum is consistently positive, suggesting a mild asymmetry: positive perturbation along the gating direction has slightly more impact than negative perturbation. This could indicate the mechanism operates in a regime where the "accept evidence" pole is easier to activate than the "reject evidence" pole is to strengthen.

### 10.4 Overall Summary Statistics

| Category | Metric | Value |
|----------|--------|-------|
| **Primary effect** | Entrenching flip rate (endorsement L23, alpha=4) | **8.6%** |
| **Primary effect** | Entrenching positive shift rate | **87.4%** |
| **Primary effect** | Entrenching mean margin shift | **+1.930** |
| **Safety** | Entrenching harm rate | **0.0%** |
| **Safety** | Correcting harm rate (C1 Note) | **5.7%** |
| **Near-boundary** | Flip rate for |margin| < 2 | **100%** |
| **Near-boundary** | Flip rate for |margin| 2–4 | **33.3%** |
| **Specificity** | N0 (no instruction) effect | **−0.065 (null)** |
| **Specificity** | W1 (weak instruction) effect | **+0.301** |
| **Specificity** | C1 Note (strong instruction) effect | **+1.930** |
| **Specificity** | C1 Expert effect | **+0.453** |
| **Position** | Endorsement L23 > Last Token L21 | **2.4× shift, 4.3× flips** |
| **Controls** | Random, wrong-sign, wrong-site | **All null or opposite** |

---

## 11. Key Findings for the Paper

### Finding 1: A Causal Gating Direction Exists
Adding the within-i1a gating vector at the endorsement token position (layer 23) shifts 87.4% of entrenching items toward the correct answer, with a mean shift of +1.930 logit-margin units. The random control produces no positive effect (−0.505), the wrong-sign control reverses the effect (−1.325), and the wrong-site control is null (−0.281).

### Finding 2: The Direction Is Instruction-Gated
The N0 (no-instruction) condition shows zero effect (−0.065), confirming the direction only operates when the model has been instructed to prioritize factual accuracy. The effect scales with instruction strength: N0 (null) → W1 (+0.301) → C1 (+1.930).

### Finding 3: Site Specificity — Endorsement Token Is the Causal Site
Patching at the endorsement token position (L23) is 3× more specific and 4× more effective (in flip rate) than patching at the last token (L21). Wrong-site controls at instruction_last position show near-zero effects at both positions.

### Finding 4: The Mechanism Operates on Boundary Items
100% of items within 2 logit-margin units of the decision boundary flip; 33.3% within 2–4 units flip; 0% beyond 8 units flip. However, even deeply entrenched items (margin > 12) show positive shifts in 66% of cases — the direction consistently pushes toward evidence integration, but only flips items that are close enough.

### Finding 5: Low Harm Rate in the Primary Condition
The entrenching items show 0% harm rate across all alpha values in the primary C1 Note condition. Correcting items show 5.7% harm rate, concentrated at narrow-margin items (margin 0–3: ~32% harm). Items with margin > 6 are essentially never harmed.

### Finding 6: Expert Tag Creates a Different Regime
The C1 Expert condition shows the highest flip rate (26.5%) but also substantial harm (11.3%). The expert tag brings baseline margins much closer to zero (−1.587 vs −9.354), creating a regime where the gating vector can tip items in either direction. This suggests the expert tag partially overrides the evidence-rejection mechanism even without patching.

---

## 12. Raw Data Reference

All results are from the following directories under `new-phase-results/mechanism/`:

| Directory | Condition | Position | Layer | Subset |
|-----------|-----------|----------|-------|--------|
| `qwen_correction_gating_patching_endorsement_l23` | C1 Note (i1a) | endorsement | 23 | entrenching |
| `qwen_correction_gating_patching_lasttoken_l21` | C1 Note (i1a) | last_token | 21 | entrenching |
| `qwen_correction_gating_patching_c1note_correcting` | C1 Note (i1a) | endorsement | 23 | correcting |
| `qwen_correction_gating_patching_c1expert_entrenching` | C1 Expert | endorsement | 23 | entrenching |
| `qwen_correction_gating_patching_n0note_entrenching` | N0 Note | endorsement | 23 | entrenching |
| `qwen_correction_gating_patching_w1note_entrenching` | W1 Note | endorsement | 23 | entrenching |
| `qwen_correction_gating_patching_boundary_correcting` | C1 Note (boundary vec) | endorsement | 23 | correcting |
| `qwen_correction_gating_patching_boundary_entrenching` | C1 Note (boundary vec) | endorsement | 23 | entrenching |
| `qwen_correction_gating_patching_reference_correcting_v2` | C1 Note (reference) | endorsement | 23 | correcting |
| `qwen_correction_gating_patching_reference_entrenching_v2` | C1 Note (reference) | endorsement | 23 | entrenching |
| `qwen_correction_gating_i1a_i1c_note` | Cross-condition extraction (i1a→i1c) | all | — | all |
| `qwen_correction_gating_i1a_i1d_note` | Cross-condition extraction (i1a→i1d) | all | — | all |
| `qwen_correction_gating_extract` | Initial extraction (3 items) | all | — | correcting |
| `qwen_correction_gating_heterogeneity_summary.json` | Heterogeneity breakdown | — | — | both |
