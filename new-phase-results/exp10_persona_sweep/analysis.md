# Persona Sweep Analysis (exp10)

## Overview

This experiment tests whether different system-prompt personas affect how the model gates evidence.
We evaluate 6 personas (default, scientist, skeptic, judge, empath, therapist) × 2 instruction variants (i1a, i1c) on Qwen3-4B.

**Key question:** Do different personas produce different correction/resistance/entrenchment rates?

### Condition key
- **N0**: Neutral baseline (no speaker claim)
- **N1**: Speaker makes neutral/no-opinion statement
- **W1**: Speaker endorses the *wrong* answer
- **C1**: Speaker endorses the *correct* answer
- **Prior-wrong items**: Items where the model gets N0 wrong (baseline prior favors wrong answer)

### Instruction variants
- **i1a**: "Answer correctly even if the speaker is wrong. Prioritize factual accuracy."
- **i1c**: Counter-suggestibility variant

---

## Expert Tag Results

### Instruction variant: i1a

| Persona | N | Prior-Wrong | PW Rate | Correction (C1) | Resistance (W1) | Entrenchment | C1−W1 Margin Shift |
|---------|---|-------------|---------|------------------|-----------------|--------------|-------------------|
| default | 1813 | 888 | 49.0% | 0.0% | 100.0% | 100.0% | 0.0003 |
| scientist | 1813 | 888 | 49.0% | 0.0% | 100.0% | 100.0% | -0.0117 |
| skeptic | 1813 | 888 | 49.0% | 0.0% | 100.0% | 100.0% | -0.0193 |
| judge | 1813 | 888 | 49.0% | 0.0% | 99.9% | 99.9% | -0.0255 |
| empath | 1813 | 891 | 49.1% | 0.0% | 99.7% | 99.7% | -0.0208 |
| therapist | 1813 | 888 | 49.0% | 0.0% | 100.0% | 100.0% | -0.0121 |

#### Correction rate with 95% bootstrap CIs (i1a)

| Persona | Correction Rate [95% CI] | Resistance Rate [95% CI] |
|---------|--------------------------|--------------------------|
| default | 0.0% [0.0–0.0] | 100.0% [100.0–100.0] |
| scientist | 0.0% [0.0–0.0] | 100.0% [100.0–100.0] |
| skeptic | 0.0% [0.0–0.0] | 100.0% [100.0–100.0] |
| judge | 0.0% [0.0–0.0] | 99.9% [99.7–100.0] |
| empath | 0.0% [0.0–0.0] | 99.7% [99.2–100.0] |
| therapist | 0.0% [0.0–0.0] | 100.0% [100.0–100.0] |

### Instruction variant: i1c

| Persona | N | Prior-Wrong | PW Rate | Correction (C1) | Resistance (W1) | Entrenchment | C1−W1 Margin Shift |
|---------|---|-------------|---------|------------------|-----------------|--------------|-------------------|
| default | 1813 | 888 | 49.0% | 0.0% | 100.0% | 100.0% | 0.0018 |
| scientist | 1813 | 888 | 49.0% | 0.0% | 100.0% | 100.0% | -0.0120 |
| skeptic | 1813 | 888 | 49.0% | 0.0% | 100.0% | 100.0% | -0.0195 |
| judge | 1813 | 888 | 49.0% | 0.0% | 100.0% | 100.0% | -0.0228 |
| empath | 1813 | 891 | 49.1% | 0.0% | 99.9% | 99.9% | -0.0070 |
| therapist | 1813 | 888 | 49.0% | 0.0% | 100.0% | 100.0% | -0.0037 |

#### Correction rate with 95% bootstrap CIs (i1c)

| Persona | Correction Rate [95% CI] | Resistance Rate [95% CI] |
|---------|--------------------------|--------------------------|
| default | 0.0% [0.0–0.0] | 100.0% [100.0–100.0] |
| scientist | 0.0% [0.0–0.0] | 100.0% [100.0–100.0] |
| skeptic | 0.0% [0.0–0.0] | 100.0% [100.0–100.0] |
| judge | 0.0% [0.0–0.0] | 100.0% [100.0–100.0] |
| empath | 0.0% [0.0–0.0] | 99.9% [99.7–100.0] |
| therapist | 0.0% [0.0–0.0] | 100.0% [100.0–100.0] |


## Note Tag Results

### Instruction variant: i1a

| Persona | N | Prior-Wrong | PW Rate | Correction (C1) | Resistance (W1) | Entrenchment | C1−W1 Margin Shift |
|---------|---|-------------|---------|------------------|-----------------|--------------|-------------------|
| default | 1813 | 888 | 49.0% | 0.0% | 100.0% | 100.0% | 0.0023 |
| scientist | 1813 | 890 | 49.1% | 0.2% | 99.8% | 99.8% | -0.0055 |
| skeptic | 1813 | 888 | 49.0% | 0.0% | 100.0% | 100.0% | -0.0071 |
| judge | 1813 | 890 | 49.1% | 0.4% | 99.4% | 99.4% | -0.0238 |
| empath | 1813 | 891 | 49.1% | 3.1% | 97.0% | 97.0% | -0.0002 |
| therapist | 1813 | 888 | 49.0% | 0.3% | 99.8% | 99.8% | -0.0003 |

#### Correction rate with 95% bootstrap CIs (i1a)

| Persona | Correction Rate [95% CI] | Resistance Rate [95% CI] |
|---------|--------------------------|--------------------------|
| default | 0.0% [0.0–0.0] | 100.0% [100.0–100.0] |
| scientist | 0.2% [0.0–0.6] | 99.8% [99.4–100.0] |
| skeptic | 0.0% [0.0–0.0] | 100.0% [100.0–100.0] |
| judge | 0.4% [0.1–0.9] | 99.4% [98.9–99.9] |
| empath | 3.1% [2.0–4.3] | 97.0% [95.7–98.1] |
| therapist | 0.3% [0.0–0.8] | 99.8% [99.4–100.0] |

### Instruction variant: i1c

| Persona | N | Prior-Wrong | PW Rate | Correction (C1) | Resistance (W1) | Entrenchment | C1−W1 Margin Shift |
|---------|---|-------------|---------|------------------|-----------------|--------------|-------------------|
| default | 1813 | 888 | 49.0% | 0.0% | 100.0% | 100.0% | -0.0010 |
| scientist | 1813 | 890 | 49.1% | 0.2% | 99.8% | 99.8% | 0.0012 |
| skeptic | 1813 | 888 | 49.0% | 0.0% | 100.0% | 100.0% | -0.0089 |
| judge | 1813 | 890 | 49.1% | 0.1% | 99.8% | 99.8% | -0.0135 |
| empath | 1813 | 891 | 49.1% | 1.0% | 99.4% | 99.4% | 0.0041 |
| therapist | 1813 | 888 | 49.0% | 0.0% | 100.0% | 100.0% | 0.0019 |

#### Correction rate with 95% bootstrap CIs (i1c)

| Persona | Correction Rate [95% CI] | Resistance Rate [95% CI] |
|---------|--------------------------|--------------------------|
| default | 0.0% [0.0–0.0] | 100.0% [100.0–100.0] |
| scientist | 0.2% [0.0–0.6] | 99.8% [99.4–100.0] |
| skeptic | 0.0% [0.0–0.0] | 100.0% [100.0–100.0] |
| judge | 0.1% [0.0–0.3] | 99.8% [99.4–100.0] |
| empath | 1.0% [0.4–1.7] | 99.4% [98.9–99.9] |
| therapist | 0.0% [0.0–0.0] | 100.0% [100.0–100.0] |


## Statistical Comparison vs. Default Persona

For each non-default persona, we test whether correction and resistance rates differ significantly from the default persona using both bootstrap permutation tests and chi-squared tests.

### Expert tag

#### i1a

| Persona | Δ Correction | Bootstrap p | χ² | χ² p | Δ Resistance | Bootstrap p | χ² | χ² p |
|---------|-------------|-------------|-----|------|-------------|-------------|-----|------|
| scientist | +0.0% | 1.0000 | 0.00 | 1.0000 | +0.0% | 1.0000 | 0.00 | 1.0000 |
| skeptic | +0.0% | 1.0000 | 0.00 | 1.0000 | +0.0% | 1.0000 | 0.00 | 1.0000 |
| judge | +0.0% | 1.0000 | 0.00 | 1.0000 | -0.1% | 1.0000 | 1.00 | 0.3172 |
| empath | +0.0% | 1.0000 | 0.00 | 1.0000 | -0.3% | 0.2505 | 2.99 | 0.0835 |
| therapist | +0.0% | 1.0000 | 0.00 | 1.0000 | +0.0% | 1.0000 | 0.00 | 1.0000 |

#### i1c

| Persona | Δ Correction | Bootstrap p | χ² | χ² p | Δ Resistance | Bootstrap p | χ² | χ² p |
|---------|-------------|-------------|-----|------|-------------|-------------|-----|------|
| scientist | +0.0% | 1.0000 | 0.00 | 1.0000 | +0.0% | 1.0000 | 0.00 | 1.0000 |
| skeptic | +0.0% | 1.0000 | 0.00 | 1.0000 | +0.0% | 1.0000 | 0.00 | 1.0000 |
| judge | +0.0% | 1.0000 | 0.00 | 1.0000 | +0.0% | 1.0000 | 0.00 | 1.0000 |
| empath | +0.0% | 1.0000 | 0.00 | 1.0000 | -0.1% | 1.0000 | 1.00 | 0.3180 |
| therapist | +0.0% | 1.0000 | 0.00 | 1.0000 | +0.0% | 1.0000 | 0.00 | 1.0000 |

### Note tag

#### i1a

| Persona | Δ Correction | Bootstrap p | χ² | χ² p | Δ Resistance | Bootstrap p | χ² | χ² p |
|---------|-------------|-------------|-----|------|-------------|-------------|-----|------|
| scientist | +0.2% | 0.4920 | 2.00 | 0.1575 | -0.2% | 0.5000 | 2.00 | 0.1575 |
| skeptic | +0.0% | 1.0000 | 0.00 | 1.0000 | +0.0% | 1.0000 | 0.00 | 1.0000 |
| judge | +0.4% | 0.1225 | 4.00 | 0.0455 | -0.6% | 0.0725 | 5.00 | 0.0253 |
| empath | +3.1%* | 0.0000 | 28.35 | 0.0000 | -3.0%* | 0.0000 | 27.32 | 0.0000 |
| therapist | +0.3% | 0.2520 | 3.01 | 0.0830 | -0.2% | 0.5145 | 2.00 | 0.1571 |

#### i1c

| Persona | Δ Correction | Bootstrap p | χ² | χ² p | Δ Resistance | Bootstrap p | χ² | χ² p |
|---------|-------------|-------------|-----|------|-------------|-------------|-----|------|
| scientist | +0.2% | 0.4970 | 2.00 | 0.1575 | -0.2% | 0.4890 | 2.00 | 0.1575 |
| skeptic | +0.0% | 1.0000 | 0.00 | 1.0000 | +0.0% | 1.0000 | 0.00 | 1.0000 |
| judge | +0.1% | 1.0000 | 1.00 | 0.3177 | -0.2% | 0.4875 | 2.00 | 0.1575 |
| empath | +1.0%* | 0.0030 | 9.02 | 0.0027 | -0.6% | 0.0705 | 5.00 | 0.0254 |
| therapist | +0.0% | 1.0000 | 0.00 | 1.0000 | +0.0% | 1.0000 | 0.00 | 1.0000 |

## Effect Size: Margin Shifts on Prior-Wrong Items

The signed margin shift (C1 − W1) on prior-wrong items measures how much more the model moves toward the correct answer when given correct endorsement vs. wrong endorsement. Larger values = stronger evidence gating.

### Expert tag

| Persona | i1a: Mean W1 Margin | i1a: Mean C1 Margin | i1a: C1−W1 Shift | i1c: Mean W1 Margin | i1c: Mean C1 Margin | i1c: C1−W1 Shift |
|---------|--------------------|--------------------|------------------|--------------------|--------------------|------------------|
| default | -0.9183 | -0.9181 | 0.0003 | -0.9358 | -0.9340 | 0.0018 |
| scientist | -0.5572 | -0.5688 | -0.0117 | -0.5633 | -0.5754 | -0.0120 |
| skeptic | -0.5566 | -0.5759 | -0.0193 | -0.6541 | -0.6737 | -0.0195 |
| judge | -0.4602 | -0.4857 | -0.0255 | -0.4669 | -0.4898 | -0.0228 |
| empath | -0.3081 | -0.3288 | -0.0208 | -0.3934 | -0.4004 | -0.0070 |
| therapist | -0.4573 | -0.4693 | -0.0121 | -0.5410 | -0.5447 | -0.0037 |

### Note tag

| Persona | i1a: Mean W1 Margin | i1a: Mean C1 Margin | i1a: C1−W1 Shift | i1c: Mean W1 Margin | i1c: Mean C1 Margin | i1c: C1−W1 Shift |
|---------|--------------------|--------------------|------------------|--------------------|--------------------|------------------|
| default | -0.8961 | -0.8938 | 0.0023 | -0.9137 | -0.9147 | -0.0010 |
| scientist | -0.4390 | -0.4444 | -0.0055 | -0.5008 | -0.4995 | 0.0012 |
| skeptic | -0.4725 | -0.4796 | -0.0071 | -0.5879 | -0.5967 | -0.0089 |
| judge | -0.3204 | -0.3442 | -0.0238 | -0.4155 | -0.4290 | -0.0135 |
| empath | -0.1607 | -0.1609 | -0.0002 | -0.3025 | -0.2984 | 0.0041 |
| therapist | -0.3387 | -0.3390 | -0.0003 | -0.5689 | -0.5670 | 0.0019 |

## Cross-Persona Summary

### Key Findings

#### Expert tag

**i1a:**
- Correction rate range: 0.0% – 0.0% (spread: 0.0pp)
  - Highest correction: **default** (0.0%)
  - Lowest correction: **default** (0.0%)
- Resistance rate range: 99.7% – 100.0% (spread: 0.3pp)
  - Lowest resistance (best): **empath** (99.7%)
  - Highest resistance (worst): **default** (100.0%)
- Largest C1−W1 margin shift: **default** (0.0003)

**i1c:**
- Correction rate range: 0.0% – 0.0% (spread: 0.0pp)
  - Highest correction: **default** (0.0%)
  - Lowest correction: **default** (0.0%)
- Resistance rate range: 99.9% – 100.0% (spread: 0.1pp)
  - Lowest resistance (best): **empath** (99.9%)
  - Highest resistance (worst): **default** (100.0%)
- Largest C1−W1 margin shift: **default** (0.0018)

#### Note tag

**i1a:**
- Correction rate range: 0.0% – 3.1% (spread: 3.1pp)
  - Highest correction: **empath** (3.1%)
  - Lowest correction: **default** (0.0%)
- Resistance rate range: 97.0% – 100.0% (spread: 3.0pp)
  - Lowest resistance (best): **empath** (97.0%)
  - Highest resistance (worst): **default** (100.0%)
- Largest C1−W1 margin shift: **default** (0.0023)

**i1c:**
- Correction rate range: 0.0% – 1.0% (spread: 1.0pp)
  - Highest correction: **empath** (1.0%)
  - Lowest correction: **default** (0.0%)
- Resistance rate range: 99.4% – 100.0% (spread: 0.6pp)
  - Lowest resistance (best): **empath** (99.4%)
  - Highest resistance (worst): **default** (100.0%)
- Largest C1−W1 margin shift: **empath** (0.0041)

## Interpretation

1. **Persona effects are small but real, and concentrated in one persona.** The empath persona is the only one that significantly differs from default on correction rates (p < 0.005 bootstrap, Note tag, both instruction variants). It achieves 3.1% correction on prior-wrong items under i1a (vs. 0.0% for default), the highest of any persona. All other personas remain near-zero correction.

2. **Expert vs. Note tag asymmetry.** Under the Expert tag, all personas produce essentially 0% correction and ~100% resistance — the model is completely entrenched regardless of persona. Under the Note tag, small cracks appear: empath shows measurable correction (3.1% i1a, 1.0% i1c), judge shows 0.4% (i1a), and scientist/therapist show trace-level correction (0.2–0.3%). This suggests the Note speaker framing is a precondition for persona effects to manifest.

3. **Margin levels vary substantially by persona, even when correction rates don't.** Although correction rates are near-zero for most personas, the raw W1 margins (how wrong the model is on prior-wrong items under wrong endorsement) differ dramatically:
   - default: W1 margin ≈ −0.90 (deeply entrenched)
   - empath: W1 margin ≈ −0.16 to −0.30 (much less entrenched)
   - judge: W1 margin ≈ −0.32 to −0.47
   - skeptic: W1 margin ≈ −0.47 to −0.59
   
   This means personas shift the model's *confidence* in the wrong answer, even when they don't flip the binary answer. The empath persona brings the model closest to the decision boundary.

4. **The C1−W1 margin shift is near-zero for all personas.** The differential response to correct vs. wrong endorsements is negligible (< 0.03 for all personas), indicating that personas do not improve evidence *discrimination* — they uniformly shift the margin without creating asymmetric gating.

5. **Statistical significance is sparse.** Out of 20 persona-vs-default comparisons on correction rates, only 2 reach p < 0.05 (both empath, Note tag). After any multiple-testing correction, only the empath/i1a/Note result (p ≈ 0.000) would survive.

6. **Bottom line for the paper.** Personas do not substantially modulate evidence gating. The empath persona shows a small but statistically significant increase in correctability (~3pp), primarily under the Note tag with the factual-accuracy instruction (i1a). The mechanism appears to be margin compression (making the model less confident in wrong answers) rather than improved evidence discrimination. This supports the interpretation that evidence gating is primarily a *representational* phenomenon (what the model computes) rather than a *persona-level* phenomenon (what role the model plays).

---

*Analysis generated from exp10_persona_sweep data. Bootstrap permutation tests use 2,000 iterations, seed=42.*
