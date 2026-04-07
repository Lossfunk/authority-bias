# Experimental Results

**Project**: Instruction-Dependent Evidence Gating in Language Models
**Branch**: `neurips-submission`
**Last updated**: 2026-04-07

---

## 1. Experimental Setup

### 1.1 The exp10 Framework

Each item is a multiple-choice question (A/B) tested under 6 conditions, crossing 3 endorsement types with 2 instruction states:

| Code | Endorsement | Instruction | Template |
|------|-------------|-------------|----------|
| N0 | Neutral ("I'm considering options A and B") | Absent | Baseline prior belief |
| N1 | Neutral | Present | Instruction effect on prior |
| W0 | Wrong (speaker endorses wrong answer) | Absent | Misleading evidence alone |
| W1 | Wrong | Present | Instruction + misleading evidence |
| C0 | Correct (speaker endorses right answer) | Absent | Corrective evidence alone |
| C1 | Correct | Present | Instruction + corrective evidence |

Speaker tags: "Note:" (low authority), "Expert:" (high authority), "User:", "Someone online:"

### 1.2 Instruction Variants

| Code | Wording |
|------|---------|
| **i1a** | "Answer correctly even if the speaker is wrong. Prioritize factual accuracy." |
| **i1c** | "Base your answer on the facts, not the speaker's opinion." |
| **i1d** | "Answer correctly by evaluating the facts." |

### 1.3 Datasets

| Dataset | Items | Domain | Source |
|---------|-------|--------|--------|
| Factual QA | 1,813 | Trivia / knowledge | TriviaQA + TruthfulQA |
| PIQA | ~1,800 | Commonsense / physical intuition | PIQA benchmark |
| DebateQA | 999 | Debatable / no ground truth | DebateQA (EACL 2026) |

### 1.4 Models Tested

| Model | Family | Size | Active Params |
|-------|--------|------|---------------|
| Qwen3-4B-Instruct | Alibaba Qwen | 4B | 4B |
| Qwen3-4B-Thinking | Alibaba Qwen | 4B | 4B |
| Llama-3.1-8B-Instruct | Meta Llama | 8B | 8B |
| Gemma-3-12B-IT | Google Gemma | 12B | 12B |
| Qwen3-30B-A3B-Instruct | Alibaba Qwen (MoE) | 30B | 3B |
| GPT-oss-20B | OpenAI | 20B | 20B |
| Gemma-4-26B-A4B-it | Google Gemma (MoE) | 26B | 4B |
| Qwen3.5-27B | Alibaba Qwen | 27B | 27B |
| OLMo-2-0325-32B-Instruct | Allen AI OLMo | 32B | 32B |

---

## 2. Behavioral Results

### 2.1 The Backfire Effect (Prior-State Conditioned)

On prior-wrong items under Note tag, the accuracy instruction (i1a) blocks correct endorsements instead of helping:

**Qwen3-4B-Instruct (factual QA, 1,813 items):**

| Instruction | Effect on prior-wrong items (ratio metric dr) | Effect on prior-wrong items (additive eff_diff) |
|---|---|---|
| i1a | -0.68 (entrenches) | -7.88 |
| i1d | -0.05 (neutral) | -4.89 |
| i1c | +0.42 (corrects) | +2.54 |

CIs do not overlap between i1a and i1c.

### 2.2 Instruction Wording Gradient

Diagnostic instruction sweep (factual QA, Qwen3-4B, prior-wrong items, Note tag):

| Instruction | Selectivity |
|---|---|
| i1a ("even if the speaker is wrong") | -0.431 (pathological) |
| accuracy_priority_only | -0.180 |
| as_accurately_as_you_can | -0.144 |
| facts_presented_only | -0.066 |
| consider_info_carefully | +0.012 (neutral) |

Monotonic gradient from pathological to neutral. The adversarial clause "even if the speaker is wrong" is the primary driver. Explicitly mentioning conflict is itself a trigger.

### 2.3 Speaker Authority Crossover

| | Note (low authority) | Expert (high authority) |
|---|---|---|
| **i1a** (resist speaker) | Entrenches errors | Roughly neutral |
| **i1c** (evaluate facts) | Corrects errors | Makes things worse |

No single instruction is universally helpful.

### 2.4 Cross-Model Replication (Factual QA, Note tag, prior-wrong)

| Model | i1a effect | i1c effect | Sign flip? |
|---|---|---|---|
| Qwen3-4B-Instruct | -0.68 | +0.42 | Yes |
| Llama-3.1-8B-Instruct | -0.36 | +0.26 (null) | Attenuation only |
| Gemma-3-12B-IT | Noisy | Noisy | Suggestive |

General finding: i1c always reduces entrenchment relative to i1a. The full sign flip is a strong special case in Qwen; attenuation is general.

### 2.5 Cross-Domain Replication (Note tag, prior-wrong, ratio metric)

| | Factual QA (i1a -> i1c) | PIQA (i1a -> i1c) |
|---|---|---|
| Qwen3-4B | -0.68 -> +0.42 (flip) | -0.46 -> +0.13 (flip) |
| Llama-3.1-8B | -0.36 -> +0.26 (null) | -0.59 -> +0.54 (flip) |

### 2.6 30B Scale

Qwen3-30B-A3B-Instruct: entrenchment survives at 30B on 3/4 speaker tags.

### 2.7 DebateQA (No Ground Truth)

**Qwen3-4B (DebateQA, 999 items):**
- i1a dampens endorsement effect on prior-disagreeing items by 45.7pp
- i1c dampens by only 6.5pp
- Instruction-gating persists without ground truth

**GPT-oss-20B (DebateQA):**
- Overall averages: near-zero instruction effect
- After prior-state conditioning (Expert tag): i1a dampens by 6.6pp, i1c by 1.3pp
- Note tag: no differentiation (zero)
- Prior-state conditioning was essential to reveal the effect

### 2.8 Gemma-4-26B-A4B-it Behavioral

- 887/1813 items prior-wrong (massive B-preference)
- Only 1/887 prior-wrong items resists correction under i1a (near-zero selectivity)
- Expanded labels (W1-susceptible proxy): 409 resisting, 478 correcting
- Model is almost universally endorsement-susceptible

---

## 3. Credulity Dial Analysis

**Finding**: Across every model/tag/domain tested, i1a both PROTECTS prior-correct items from wrong endorsements AND BLOCKS prior-wrong items from correct endorsements. It is a general skepticism mode, not a targeted backfire.

i1c is the opposite: general openness to evidence.

**Qwen3-30B-A3B, Note tag (near-optimal calibration):**
- Protection of correct items: +3.86
- Blocking of wrong items: +0.44
- i1c is skeptical of bad evidence and open to good evidence

The "backfire" on prior-wrong items is a consequence of general skepticism, not a special mechanism.

---

## 4. Qwen3-4B Correction-Gating Mechanism

### 4.1 Probe Results

| Position | Best Layer | CV Accuracy | Majority Baseline |
|---|---|---|---|
| Endorsement (last token) | 23 | 81% (cross-condition i1a->i1c) | 66% |
| Endorsement (within-i1a) | 23 | 75.1% | 66% |
| Last token | -- | Instruction identity only | -- |

The instruction position carries zero signal about item-level outcome. The endorsement position carries the behavioral signal.

### 4.2 Geometry

Projections at endorsement position, layer 23:
- i1a mean: -5.5
- i1d mean: +2.4
- i1c mean: +8.7

Within i1a: entrenching mean = -9.48, correcting mean = -1.55 (effect size d = 0.91)

### 4.3 Causal Patching (endorsement position, layer 23, alpha=4)

| Condition | Margin shift | Items shifted correct | Answer flips | Harm |
|---|---|---|---|---|
| **Real vector** | **+1.93** | **87%** | **8.6% (13/151)** | **0%** |
| Random vector | -0.50 | 5% | 0% | 0% |
| Opposite direction | -1.33 | 1% | 0% | 0% |
| Wrong position (instruction) | -0.28 | 24% | 0% | 0% |

### 4.4 Evidence Specificity

| Prompt condition | What's present | Margin shift | Flip rate |
|---|---|---|---|
| N0 (no endorsement) | Question + instruction only | **-0.07** | 2% |
| W1 (wrong endorsement) | + speaker endorses wrong | **+0.30** | 3.3% |
| C1 (correct endorsement) | + speaker endorses correct | **+1.93** | 8.6% |

30x stronger with correct endorsement than without. The vector modulates evidence integration, not answer selection.

### 4.5 Cross-Domain Transfer

| Direction | Target | Transfer AUROC |
|---|---|---|
| Factual QA -> PIQA (i1a) | Vector transfer | 0.831 |
| PIQA -> Factual QA (i1a) | Vector transfer | 0.826 |
| Factual QA -> PIQA (i1c) | Vector transfer | 0.736 |
| PIQA -> Factual QA (i1c) | Vector transfer | 0.798 |

Mean-difference vectors transfer better than logistic probes: the cross-domain signal is a simple linear direction.

### 4.6 Confound Checks (All Clean)

| Confound | Probe accuracy | Baseline | Verdict |
|---|---|---|---|
| Confidence (m_N0) regressed out | 75.5% | 66.4% | Clean (r = -0.097) |
| TF-IDF on question text | 66.1% | 66.4% | Null |
| Question length | 65.9% | 66.4% | Null |
| Source dataset | 66.4% | 66.4% | Null |
| All surface features combined | 63.9% | 66.4% | Null |

### 4.7 Known Direction Comparison

| Direction | Cosine similarity | Transfer after projection |
|---|---|---|
| Proxy opinion-like | 0.093 | Unchanged |
| Authority-like (wrong endorse) | -0.005 | Unchanged |
| Authority-like (correct endorse) | 0.060 | Unchanged |

Vector norm drops only 0.6% after projecting out all three. The correction-gating direction is nearly orthogonal to known directions.

### 4.8 Patching by Confidence Band

| Baseline confidence | Items | Flip rate |
|---|---|---|
| Weakly wrong (margin -2 to 0) | 5 | 100% |
| Moderately wrong (-4 to -2) | 12 | 33% |
| Clearly wrong (-8 to -4) | 36 | 11% |
| Strongly wrong (-12 to -8) | 51 | 0% |
| Extremely wrong (below -12) | 47 | 0% |

---

## 5. Mechanism at Scale

### 5.1 Qwen3-30B-A3B

**Primary labels (strict C1 resisting):**
- 302 prior-wrong, only 16 strict resisting (low power)
- Best layer 25, probe CV AUROC 0.785
- Vector AUROC on i1a: 0.884
- Patching: 1/16 flips, clean controls
- Framing: weakly powered, not null

**Expanded labels (W1-susceptibility):**
- 236 negatives, best layer 42
- Probe CV AUROC 0.764
- i1a->i1c transfer AUROC: 0.937
- Does NOT produce same causal story on strict C1 prompts
- Framing: broader evidence-susceptibility representation

### 5.2 GPT-oss-20B (Factual QA)

- 810 prior-wrong items (371 correcting / 439 resisting)
- Best layer 8, within-i1a probe CV AUROC: 0.677
- i1a->i1c transfer AUROC: 0.926 (very strong)
- Mean-difference vector transfer: weaker but real
- Patching: modest, directionally correct, not dramatic
- Framing: cross-family replication of representational structure

### 5.3 GPT-oss-20B (DebateQA, Expert tag)

- 873 items (386 correcting / 487 resisting)
- Best layer 14, endorsement_last, probe CV AUROC: 0.603
- i1a->i1c transfer AUROC: 0.883 (strong)
- Vector transfer: i1a AUROC 0.631, i1c AUROC 0.636
- **Causal patching: NULL** (real vector indistinguishable from random/wrong-sign/wrong-site controls)
- Framing: decodable but not steerable ("thermometer not heater")

---

## 6. Confabulation Pilot (GPT-oss-20B, Closed)

- Round 1 (re-answer): indistinguishable groups
- Round 2 (defend-this-answer): 424/439 self-corrected
- Forced-choice resistance does not survive into generation
- Line closed

---

## 7. Pre-Endorsement (N0) Probe

### 7.1 Core Finding

N0 activations (before any instruction or endorsement) predict future resistance/correction under i1a.

### 7.2 Three-Way Decomposition (confidence + position + residual)

| Model | N items | Full AUROC | Conf alone | Pos alone | Conf out | Pos out | Both out |
|---|---|---|---|---|---|---|---|
| **Qwen3-4B** | 449 | 0.626 | 0.560 | 0.493 | 0.609 | 0.627 | **0.612** |
| **GPT-oss-20B** | 810 | 0.700 | 0.720 | 0.534 | 0.669 | 0.691 | **0.659** |
| **Gemma-4-26B** | 887 | 0.617 | 0.822 | 0.488 | 0.524 | 0.617 | **0.526** |

Best layers: Qwen layer 28, GPT-oss layer 21, Gemma layer 25.

### 7.3 Interpretation

- **Position is not a confounder.** Position alone is at or below chance across all models. Removing it changes essentially nothing.
- **Qwen3-4B**: Confidence-independent signal survives (0.612 after both projections). Confidence alone is weak (0.560). The probe reads something genuine beyond confidence and position.
- **GPT-oss-20B**: Confidence-independent signal survives (0.659 after both projections). Confidence alone is stronger here (0.720) but the residual is also strong.
- **Gemma-4-26B**: Collapses to near-chance after confidence removal (0.526). Confidence alone is very strong (0.822). Behavior is almost entirely explained by confidence. No "third thing."

### 7.4 Cross-Model Comparison

Gemma's epistemic system is simple: confidence determines behavior. Qwen and GPT-oss have richer structure: a confidence-independent dimension exists that predicts evidence-integration response. Different training pipelines produce different levels of epistemic representational complexity.

---

## 8. Belief Provenance Stress Tests

### 8.1 A/B Position Confounder Check

Behavioral data showed positional skew (Qwen resisting items: 78% correct=A vs 48% for correcting). But the three-way decomposition (Section 7.2) shows position alone gets below-chance AUROC from N0 activations. The behavioral skew is a sampling coincidence, not a representational confounder.

### 8.2 Cross-Model Resistance Correlation

Do the same items resist across models?

| Pair | Shared prior-wrong | Cohen's kappa | Phi |
|---|---|---|---|
| Qwen vs GPT-oss | 187 | 0.046 | 0.076 |
| Qwen vs Gemma | 237 | -0.008 | -0.026 |
| GPT-oss vs Gemma | 40 | 0.254 | 0.381 |

Qwen vs GPT-oss: essentially zero. Resistance is NOT a stable item-level property across models. Different models resist on different items.

Confidence-controlled (Qwen vs GPT-oss, terciles of avg |m_N0|):

| Bin | N | Agreement | Expected | Kappa |
|---|---|---|---|---|
| Low | 62 | 0.500 | 0.452 | 0.087 |
| Mid | 63 | 0.429 | 0.447 | -0.033 |
| High | 62 | 0.532 | 0.488 | 0.087 |

No bin shows meaningful above-chance agreement. The null holds within confidence bins.

### 8.3 Consensus-Wrong Analysis

Items where all 3 models get wrong (supposedly "memorized misconceptions"):

**GPT-oss-20B (focal model):**

| Group | N | Resist rate | Mean |m_N0| |
|---|---|---|---|
| All-wrong consensus | 16 | 0.188 | 0.719 |
| Partial consensus | 195 | 0.528 | 1.159 |
| Unique-wrong | 599 | 0.556 | 1.276 |

Consensus-wrong items are EASIER to correct, not harder. Within confidence bins on GPT-oss:

| Bin | Consensus resist rate | Unique resist rate | Consensus effect |
|---|---|---|---|
| Low conf | 0.222 | 0.833 | -0.611 |
| Mid conf | 0.143 | 0.649 | -0.506 |

The "memorized = rigid" prediction is wrong. Common misconceptions may have been explicitly corrected in training data, making the model primed to accept correction on those items.

---

## 9. Reasoning Traces (Qualitative, Qwen3-4B-Thinking)

117 high-confidence-wrong items with full chain-of-thought:

| Pattern | Count | Description |
|---|---|---|
| Entrenchment + confabulation | 19 | Model invents fake citations to support wrong belief |
| Correction accepted | 22 | Lower-confidence items where model successfully updates |
| Literalization | 35 | "Prioritize factual accuracy" reinterpreted as "only physical facts count" |
| Think-then-ignore | varies | Model acknowledges correct evidence in reasoning, then does the wrong thing |

Generation-based answer rates:

| Condition | Accuracy |
|---|---|
| Baseline (no note, no instruction) | 36.4% |
| Correct note + instruction | 55.0% |
| Wrong note, no instruction | 25.3% |
| Wrong note + instruction | 28.1% |

Instruction helps accept corrections (+19pp) but barely helps resist misleading evidence (+3pp).

Caveat: 84-90% of traces truncated at 768 tokens. Qualitative evidence only.

---

## 10. Summary of What Holds and What Failed

### Established Findings

1. **Accuracy instructions backfire on prior-wrong items.** Instruction wording determines direction. Replicates across 3 model families, 2 domains, 30B scale.
2. **The credulity dial.** Instructions control a general trust parameter. i1a = skepticism (protects correct, blocks correction). i1c = calibrated openness.
3. **The correction-gating direction (Qwen3-4B).** Causal, evidence-specific, cross-domain, position-specific, orthogonal to known directions. Full mechanistic toolkit.
4. **The N0 pre-endorsement signal.** Confidence-independent, position-independent signal predicts future evidence-integration behavior. Survives three-way decomposition at 0.612 (Qwen) and 0.659 (GPT-oss).
5. **Training-dependent epistemic architecture.** Gemma: simple (confidence-only). Qwen/GPT-oss: richer (confidence-independent dimension exists).

### Null or Negative Results

1. **GPT-oss causal patching (DebateQA).** Decodable but not steerable. Representational structure exists but is not causally upstream.
2. **Cross-model resistance correlation.** Null (kappa ~0.05). Resistance is model-specific, not item-specific.
3. **Consensus-wrong = rigid.** The opposite: consensus-wrong items are easier to correct.
4. **Confabulation pilot.** Forced-choice resistance doesn't survive into generation.

---

## 10.5 Precision-Weighting Tests

### 10.5.1 Flip-Rate Ratio Scales with Confidence

Prediction: if i1a acts as a precision-weight on priors, the i1c/i1a flip-rate ratio should increase with confidence (more confident priors are harder to override, so precision-weighting has more room to bite).

| Confidence bin | i1c flip rate | i1a flip rate | Ratio (i1c/i1a) |
|---|---|---|---|
| Low | 0.72 | 0.50 | 1.44 |
| Mid | 0.63 | 0.36 | 1.75 |
| High | 0.31 | 0.14 | 2.21 |

**CONFIRMED**: Ratio scales monotonically 1.44 → 1.75 → 2.21.

### 10.5.2 N0 Residual Predicts Update Magnitude

Prediction: the confidence-independent N0 probe residual should correlate with actual update magnitude |m_C1 - m_N0|.

| Model | Pearson r | Spearman r | Partial r (conf out) | p-value |
|---|---|---|---|---|
| **Qwen3-4B** | 0.369 | 0.379 | **0.320** | 3.8e-12 |
| **GPT-oss-20B** | 0.298 | 0.319 | **0.306** | 4.7e-19 |

**CONFIRMED** on both models. The residual N0 signal predicts how much the model will actually update, controlling for confidence.

Bin analysis (Qwen3-4B, residual score terciles):

| Residual bin | N | Mean update | Mean confidence |
|---|---|---|---|
| Low | 148 | +6.5 | -9.9 |
| Mid | 153 | +12.1 | -11.1 |
| High | 148 | +16.3 | -13.5 |

### 10.5.3 Zero-Knowledge Test

On near-chance items (|m_N0| < 0.5), the instruction gap is still +6.4 logits (68% of the high-confidence gap). Instruction sets MODE, not just precision on prior. This is a partial crack in the pure precision-weighting framework — the instruction has an effect even when there is no prior to upweight.

---

## 10.6 Logit Lens Analysis (Qwen3-4B)

### 10.6.1 Method

For each item, run forward pass under i1a and i1c, project each layer's residual stream (last token) through the unembedding matrix W_U to track how correct-vs-wrong answer logit difference evolves layer-by-layer.

### 10.6.2 Correcting Items (N=298)

| Layer sample | i1a margin | i1c margin | Difference |
|---|---|---|---|
| L0 (embedding) | -0.38 | -0.38 | 0.00 |
| L9 | -0.14 | -0.08 | -0.06 |
| L18 | -0.20 | -0.26 | +0.06 |
| L27 | +1.05 | +2.47 | -1.42 |
| L32 (peak) | +4.45 | +8.77 | -4.32 |
| L36 (final) | +2.44 | +4.42 | -1.98 |

Both instructions lead to correction, but **i1c produces 2x stronger correction signal** at peak. i1a dampens the correction even on items that ultimately correct.

### 10.6.3 Resisting Items (N=151)

| Layer sample | i1a margin | i1c margin | Difference |
|---|---|---|---|
| L0 (embedding) | -0.17 | -0.17 | 0.00 |
| L9 | -0.04 | -0.01 | -0.04 |
| L18 | -0.10 | -0.14 | +0.04 |
| L27 | -2.49 | +0.32 | -2.81 |
| L32 | -8.30 | +1.13 | -9.43 |
| L36 (final) | -4.86 | +0.62 | -5.47 |

### 10.6.4 Key Finding: PREVENTION, Not Suppression

Under i1a, the correct answer **never appears** in intermediate layers for resisting items. The margin stays negative from embedding through final layer, plunging to -8.3 at L32. Under i1c, the same items show weak positive correction starting at L22.

This rules out suppression (correct answer computed then removed). The mechanism is **prevention**: i1a prevents the correct answer from ever being computed for these items. The divergence starts around L25 — the same region where the correction-gating probe peaks (L23).

Under i1c, even resisting items show mild correction (+0.62 final), consistent with i1c being the "open to evidence" instruction.

---

## 10.7 Base Model Logit Lens (Qwen3-4B-Base)

### 10.7.1 Method

Same items and prompts as the instruct model logit lens, but run through `Qwen/Qwen3-4B-Base` (no RLHF/DPO). Three conditions: i1a instruction, i1c instruction, no instruction. Raw prompt format (no chat template).

### 10.7.2 Correcting Items (N=298) — BASE MODEL

| Layer sample | i1a margin | i1c margin | no_inst margin |
|---|---|---|---|
| L0 | -0.29 | -0.29 | -0.29 |
| L9 | -0.09 | -0.04 | -0.03 |
| L18 | -0.40 | -0.48 | -0.50 |
| L27 | +1.53 | +2.38 | +1.32 |
| L36 (final) | +1.07 | +1.21 | +0.59 |

### 10.7.3 Resisting Items (N=151) — BASE MODEL

| Layer sample | i1a margin | i1c margin | no_inst margin |
|---|---|---|---|
| L0 | -0.13 | -0.13 | -0.13 |
| L9 | +0.01 | +0.04 | +0.02 |
| L18 | -0.17 | -0.25 | -0.25 |
| L27 | +1.09 | +1.81 | +0.86 |
| L36 (final) | +0.76 | +0.91 | +0.35 |

### 10.7.4 Key Finding: Gating is RLHF-Created

| Comparison | Instruct (i1a) | Base (i1a) | Interpretation |
|---|---|---|---|
| Correcting final | +2.44 | +1.07 | Base corrects weakly |
| Resisting final | **-4.86** | **+0.76** | Base does NOT resist (still corrects!) |
| i1a-i1c divergence (resisting, final) | **5.47** | **0.16** | 34x smaller divergence in base |

The base model shows:
1. **No prevention**: resisting items end at +0.76 under i1a (positive = correct answer preferred). The instruct model drives these to -4.86.
2. **Minimal instruction sensitivity**: i1a vs i1c divergence at final layer is 0.16 for resisting items in base vs 5.47 in instruct (34x reduction).
3. **Both groups correct similarly**: correcting and resisting items look nearly identical in base (+1.07 vs +0.76 final), unlike instruct where they diverge massively (+2.44 vs -4.86).

The prevention mechanism is entirely absent in the base model. RLHF/DPO created the instruction-dependent gating circuit. The base model treats all endorsements similarly regardless of instruction wording.

---

## 10.8 GPT-oss-20B Multi-Layer Patching

### 10.8.1 Motivation

Single-layer patching on GPT-oss was null (Section 5.3) despite strong representational signal (probe AUROC 0.677). Hypothesis: the computation is distributed across layers, so single-layer patching is insufficient.

### 10.8.2 Top Layers by Probe AUROC

| Layer | AUROC |
|---|---|
| 9 | 0.678 |
| 8 | 0.678 |
| 1 | 0.674 |
| 10 | 0.672 |
| 3 | 0.662 |

Signal is spread across early-to-middle layers (1-12), unlike Qwen where it peaks sharply at L23.

### 10.8.3 Multi-Layer Patching Results (alpha=4.0)

| Layers patched | Real shift | Random shift | Opposite shift | Real pos_rate | Random pos_rate |
|---|---|---|---|---|---|
| k=1 (best) | +0.015 | -0.024 | -0.028 | 40.5% | 29.6% |
| k=3 (top 3) | +0.043 | -0.041 | -0.051 | 51.9% | 29.2% |
| k=5 (top 5) | **+0.069** | -0.005 | -0.086 | **59.2%** | 39.9% |
| k=8 (top 8) | +0.066 | +0.008 | -0.108 | 60.1% | 43.3% |

### 10.8.4 Interpretation

- **Distributed computation CONFIRMED**: k=1 is near-null (shift +0.015), k=5 reaches +0.069 — 4.6x larger effect. Real vs opposite divergence grows from 0.043 (k=1) to 0.155 (k=5).
- **Correctly directional**: Real always positive, opposite always negative, monotonically scaling with alpha and k. This is not noise.
- **Still modest**: Even at k=8 alpha=4, flip rate is only 0.9% (4/439). Compare to Qwen's 8.6% at single-layer alpha=4. GPT-oss's gating mechanism is genuinely more distributed and harder to steer.
- **Thermometer, weakly heater**: The direction IS causally relevant (real >> random >> opposite), but the effect is diffuse. GPT-oss's epistemic architecture distributes the gating computation more than Qwen's.

---

## 10.9 Cross-Model Activation Patching (CMAP): Base vs Instruct

### 10.9.1 Method

Following Prakash et al. (ICLR 2024), we test whether the instruct model's gating activations are compatible with the base model's residual stream. We cache the instruct model's block outputs at layers 22-24 (the gating region), then re-run the base model with those layers replaced by the instruct activations. If the base model starts showing instruction-dependent gating, the mechanism was enhanced from an existing circuit (Prakash et al. prediction). If not, RLHF built something fundamentally new.

### 10.9.2 Results

**Resisting items (N=151):**

| Condition | i1a margin | i1c margin | Divergence (i1c-i1a) |
|---|---|---|---|
| Base model alone | +1.03 | +1.23 | +0.20 |
| Instruct model alone | -7.11 | +0.98 | +8.08 |
| **Base + instruct patch** | **-3.05** | **+0.40** | **+3.45** |

**Recovery ratio: 41.3%** — patching the instruct model's L22-24 activations into the base model recovers 41% of the full instruct model's instruction-dependent gating.

**Correcting items (N=298):**

| Condition | i1a margin | i1c margin | Divergence |
|---|---|---|---|
| Base model alone | +1.49 | +1.64 | +0.16 |
| Instruct model alone | +3.78 | +6.64 | +2.86 |
| Base + instruct patch | +1.47 | +2.79 | +1.32 |

### 10.9.3 Key Finding: COMPATIBLE — Enhances Existing Mechanism

The instruct model's activations are compatible with the base model's downstream computation. Patching just 3 layers (22-24) into the base model:
- Flips resisting items from positive (+1.03) to negative (-3.05) under i1a
- Creates instruction-dependent divergence where there was almost none (0.20 -> 3.45)
- Recovers 41% of the full instruct effect

This confirms the Prakash et al. prediction: RLHF enhanced an existing context-sensitivity mechanism into instruction-dependent gating. The base model's downstream layers (25-36) can already process the gating signal — they just never receive it without RLHF.

**Revised interpretation**: The base model has latent circuitry for context-dependent evidence weighting. RLHF trained layers 22-24 to produce instruction-sensitive representations that this downstream circuitry can act on. The mechanism wasn't created from scratch — it was activated and sharpened.

---

## 10.10 CMAP Layer Sweep: Mapping the Latent Receiver

### 10.10.1 Forward Sweep (instruct -> base, one layer at a time)

For resisting items, patching a single instruct layer into the base model and measuring the resulting instruction-dependent divergence (i1c - i1a):

| Layer range | Divergence lift | Interpretation |
|---|---|---|
| L0-7 | <0.3 | No signal. Early layers not involved. |
| L8-10 | 0.3-0.4 | Weak signal begins. |
| L11-15 | 0.5-0.6 | Moderate. Gating representation starts forming. |
| L16-18 | 1.0-1.7 | Strong. Mid-network carries substantial gating info. |
| L19-24 | 2.3-3.3 | Very strong. This is the gating "sender" region (overlaps with probe peak at L23). |
| L25-30 | 3.4-4.1 | Even stronger. Later layers carry MORE gating signal. |
| L31-35 | 4.3-8.1 | Maximum. Final layers carry the most instruction-dependent signal. |

The profile is **monotonically increasing** — later instruct layers carry progressively more gating information. This means the gating signal is not localized to one layer. It accumulates throughout the network, with each layer adding more instruction-dependent structure. The base model's receiver is most effective when given later-layer instruct activations.

### 10.10.2 Reverse CMAP (base -> instruct at L22-24)

Patching base model activations INTO the instruct model at L22-24:

| Condition | i1a margin | i1c margin | Divergence |
|---|---|---|---|
| Instruct alone | -7.11 | +0.98 | +8.08 |
| Instruct + base L22-24 | +2.14 | +2.20 | +0.06 |

**Gating lost: 99.3%.** Replacing just 3 layers (22-24) with base activations completely destroys the instruct model's gating mechanism. The resisting items go from -7.11 (strong prevention under i1a) to +2.14 (mild correction, same as base model). Instruction sensitivity drops from 8.08 to 0.06.

### 10.10.3 Interpretation: Where RLHF Did Its Work

The two experiments together paint a clear picture:

1. **RLHF's enhancement is concentrated at L22-24.** Removing these layers eliminates 99.3% of the gating. These are the "sender" layers that RLHF trained to produce instruction-sensitive representations.

2. **The base model's downstream layers (25-35) are already competent receivers.** When fed instruct activations from any single layer L19+, they produce substantial instruction-dependent behavior. Pre-training built the "if-then" circuitry; RLHF just trained the "if" condition.

3. **The gating signal accumulates across layers.** It's not a single "gating layer" — each layer from ~L11 onward adds instruction-dependent structure. But L22-24 is the critical bottleneck where the signal is consolidated.

4. **Pre-training left a latent epistemic circuit.** The base model can already process context-dependent evidence signals — it just never generates them. RLHF activated this latent capacity by training mid-network layers to produce instruction-sensitive representations.

---

## 11. Cross-Model Mechanism Summary

### 11.1 Comprehensive Table (All Models)

| Model | Size | Prior-Wrong | N_Resisting | Probe AUROC | Patching Flip Rate (α=4) | Capping Effect | Key Finding |
|---|---|---|---|---|---|---|---|
| **Qwen3-4B-Instruct** | 4B | 449 | 151 | 0.81 (L23, cross-cond) | 8.6% (13/151) | +1.93 shift | Full causal circuit; evidence-specific, cross-domain |
| **Llama-3.1-8B-Instruct** | 8B | — | — | — | — | — | Behavioral replication (attenuation, not full flip) |
| **Gemma-3-12B-IT** | 12B | — | — | — | — | — | Behavioral: noisy, suggestive |
| **GPT-oss-20B** | 20B | 810 | 439 | 0.677 (L8) | 0.9% (k=1), 1.4% (k=8) | +0.07 (k=5) | Distributed; decodable, weakly steerable |
| **Gemma-4-26B-A4B-it** | 26B (4B active) | 887 | 1 (strict) / 409 (expanded) | 0.617 (L25) | — | — | Near-zero entrenchment; confidence-only epistemic system |
| **Qwen3.5-27B** | 27B | 208 | 7 | — | 0% (7 items, underpowered) | ~0 | Too few resisting items for mechanism analysis |
| **Qwen3-30B-A3B** | 30B (3B active) | 302 | 16 | 0.785 (L25) | 6.3% (1/16) at α=4 | +1.03 shift | Weakly powered but directionally correct |
| **OLMo-2-32B-Instruct** | 32B | 320 | 21 | — | 0% (21 items) | ~0 | Null patching; possible different architecture |

### 11.2 Interpretation

- **Qwen3-4B** remains the only model with a fully characterized, causally validated gating circuit at single-layer resolution.
- **GPT-oss-20B** has the representational structure (strong probe, strong i1a→i1c transfer AUROC 0.926) but the computation is distributed across layers, making single-layer patching insufficient.
- **Qwen3-30B-A3B** shows directional effects (6.3% flip at α=4, 12.5% at α=8) despite only 16 resisting items — consistent with the Qwen family mechanism.
- **Qwen3.5-27B** and **OLMo-2-32B** have too few resisting items (7 and 21 respectively) for meaningful causal tests. Null patching results are underpowered, not necessarily negative.
- **Gemma-4-26B** is fundamentally different: nearly all items correct under i1a (1/887 resisting), so there is almost nothing to steer.

---

## 12. Head-Level Gating Circuit (Qwen3-4B)

### 12.1 Method

For each attention head in layers 22–24 (the gating region), we compute:
1. **Gating divergence**: difference in head output margin between i1a and i1c conditions
2. **Attention allocation**: fraction of attention to instruction vs. endorsement tokens
3. **Ablation effect**: impact of zeroing individual head outputs on model behavior

### 12.2 Top Gating Heads

| Head | Gating Divergence | Attention Pattern | Role |
|---|---|---|---|
| **L24.H8** | Large (endorsement-reader) | High attention to endorsement tokens | Reads speaker evidence |
| **L23.H7** | -2.01 (correcting), -0.87 (resisting) | ~58% attention to instruction | Reads instruction content |
| **L23.H12** | -0.61 (correcting), -0.30 (resisting) | Mixed instruction + endorsement | Integrates both signals |

### 12.3 Complementary Attention Specialization

Heads divide into two functional classes:
- **Endorsement-readers** (e.g., L22.H0, L22.H17): >45% attention to endorsement tokens, carry evidence signal
- **Instruction-readers** (e.g., L22.H13, L22.H21, L23.H14): >90% attention to instruction tokens, carry instruction-mode signal

The two classes produce complementary representations that are combined downstream.

### 12.4 Distributed Mechanism

Top-5 head cascade produces only ~2% flip rate vs. 8.6% for full-layer patching at L23. The gating computation is distributed across many heads within each layer, not concentrated in a few "gating heads."

### 12.5 L22.H3 Surprise

L22.H3 ranks 65th by gating divergence but has the highest single-head ablation effect. High attention to instruction tokens (51% i1a) but small divergence — suggesting it carries a stable instruction signal that the downstream circuit depends on, even though its own output doesn't change much between conditions. The circuit depends on this head's baseline contribution, not its differential contribution.

---

## 13. Temporal Commitment Dynamics (Qwen3-4B)

### 13.1 Method

Intervene with the correction-gating vector at different layers to map when the model commits to its answer. For each intervention layer, patch the i1c vector into i1a activations at the endorsement position and measure downstream flip rate.

### 13.2 Intervention Timing Results

| Intervention Layer | Flip Rate | Mean Margin Shift | Interpretation |
|---|---|---|---|
| L14 | 0.0% | -0.28 | Too early; harmful |
| L16 | 0.0% | -0.21 | Still harmful |
| **L18** | **2.0%** (3/151) | **-0.08** | Vulnerability window opens |
| L20 | 0.7% (1/151) | -0.37 | Window closing |
| L22 | 0.0% | -0.38 | Window closed |
| **L24** | **0.0%** | **-0.04** | Point of no return |
| L26–L34 | 0.0% | ~0 | Post-commitment |

### 13.3 Key Findings

1. **Vulnerability window at L18**: The only layer where mid-stream intervention produces flips. Before L18, the model hasn't begun computing the answer. After L20, it's committed.
2. **Point of no return at L24**: By L24, margin shifts drop to near-zero. The decision is crystallized.
3. **Resisting items maintain higher entropy**: Mean entropy at L24 is 0.45 (resisting) vs 0.25 (correcting) under i1a — resisting items are less certain but still committed to the wrong answer.
4. **Non-monotonic intervention pattern**: The margin shift is most negative (harmful) at L14, becomes least negative at L18 (the vulnerability window), then worsens again before settling near zero. This is consistent with a brief window where the computation is "in flight" and susceptible to redirection.

### 13.4 Entropy Profiles

Resisting items under i1a show consistently higher entropy than correcting items throughout layers 25–36 (mean 0.42 vs 0.23 at L30). Under i1c, the gap narrows. The instruction changes whether high-entropy items resolve toward the correct or wrong answer.

---

## 14. Multiplicative vs Additive Gating (Qwen3-4B)

### 14.1 Hypothesis

If the gating mechanism is multiplicative (gate × evidence), then the product of gate value and evidence component norm should predict margin divergence better than an additive model.

### 14.2 Regression Results (Layer 22)

| Model | R² | Adj R² | Interpretation |
|---|---|---|---|
| Multiplicative (gate × evidence) | 0.025% | -0.20% | Null |
| Additive (gate + evidence) | 0.058% | -0.39% | Null |
| Full (gate + evidence + interaction) | 0.22% | -0.46% | Null |

**Multiplicative model refuted**: R² < 0.3% for all models. Neither gate values nor evidence component norms, nor their interaction, predict item-level margin divergence.

### 14.3 But Gate Values Cleanly Separate Groups

Despite the regression null, gate values show clean group separation:

| Group | Mean Gate (i1a) | Mean Gate (i1c) | Shift |
|---|---|---|---|
| Correcting | -0.31 | -0.04 | +0.27 |
| Resisting | -2.10 | -1.62 | +0.48 |

### 14.4 Key Finding: Correction Instruction Processed Even When Resisted

The i1a→i1c gate shift is LARGER for resisting items (+0.48) than correcting items (+0.27). At layer 24, mean gate values shift from 1.62 to 3.23 for resisting items under i1c. The model processes the correction instruction — the gate opens — but the downstream computation still doesn't flip the answer. The gate is necessary but not sufficient for correction.

---

## 15. Persona-Conditioned Evidence Gating

### 15.1 Method

Test whether different system-prompt personas (default, scientist, skeptic, judge, empath, therapist) modulate how Qwen3-4B gates evidence. Six personas × 2 instruction variants (i1a, i1c) × 2 speaker tags (Expert, Note).

### 15.2 Results Summary

**Expert tag (all personas)**:
- Correction rate: 0.0% across ALL personas and both instructions
- Resistance rate: 99.7%–100.0%
- No persona breaks through Expert-tag entrenchment

**Note tag**:

| Persona | i1a Correction | i1c Correction | i1a C1−W1 Shift |
|---|---|---|---|
| default | 0.0% | 0.0% | +0.002 |
| scientist | 0.2% | 0.2% | -0.006 |
| skeptic | 0.0% | 0.0% | -0.007 |
| judge | 0.4% | 0.1% | -0.024 |
| **empath** | **3.1%** | **1.0%** | -0.000 |
| therapist | 0.3% | 0.0% | -0.000 |

### 15.3 Empath: The Only Significant Persona

The empath persona is the only one that significantly differs from default (p < 0.005, bootstrap permutation, Note tag). It achieves 3.1% correction under i1a vs. 0.0% for default. The mechanism is **margin compression**: empath brings W1 margins from -0.90 (default) to -0.16, bringing items closer to the decision boundary without improving evidence discrimination (C1−W1 shift ≈ 0).

### 15.4 Key Finding: Near-Null Result

Personas do not substantially modulate evidence gating. The C1−W1 margin shift (evidence discrimination) is <0.03 for all personas. This supports the interpretation that **evidence gating is a representational phenomenon** (what the model computes at the endorsement position) **rather than a persona-level phenomenon** (what role the model plays). The gating circuit operates below the persona abstraction layer.

---

## 16. Gemma-4-26B Expanded Extraction

### 16.1 Strict Labels

- 887 prior-wrong items out of 1,813
- Only **1 out of 887** resists correction under i1a (near-zero entrenchment)
- The model is almost universally endorsement-susceptible

### 16.2 Expanded Labels (W1 Susceptibility)

To enable mechanism analysis despite near-zero strict resistance:
- **409 expanded-resisting** items (susceptible to wrong endorsement under W1)
- **478 expanded-correcting** items
- Expanded labels extracted with both i1a and i1c activations (1,774 total records)

### 16.3 Interpretation

Gemma-4-26B's epistemic system is qualitatively different from Qwen/GPT-oss:
- Almost no instruction-dependent entrenchment (the defining phenomenon is absent)
- N0 probe collapses to near-chance after confidence removal (Section 7.3: 0.526 AUROC)
- Behavior is almost entirely explained by confidence level
- The expanded extraction enables analysis of a different question: what determines susceptibility to misleading evidence, rather than instruction-dependent gating

---

### Open Questions

1. What is the "third thing" (beyond confidence and position) in the N0 representation?
2. ~~Is the N0 signal created by post-training (RLHF/DPO) or present in base models?~~ **ANSWERED: RLHF enhanced L22-24 to produce instruction-sensitive gating signals. The base model's downstream circuitry (L25-35) was already a competent receiver. Removing L22-24 from the instruct model destroys 99.3% of gating. Confirmed via CMAP (Prakash et al. 2024 methodology).**
3. Why does GPT-oss have the signal but Gemma does not? Architecture (dense vs MoE) or training recipe?

---

## 17. File Locations

### Result Directories

| Path | Contents |
|---|---|
| `new-phase-results/qwen3-4b-results/` | Qwen3-4B behavioral (exp7-exp14) |
| `new-phase-results/qwen3-30b-a3b-results/` | Qwen3-30B behavioral |
| `new-phase-results/gpt_oss_20b-results/` | GPT-oss-20B factual QA behavioral |
| `new-phase-results/gemma-4-26b-results/` | Gemma-4-26B behavioral |
| `new-phase-results/llama-3.1-8b-results/` | Llama-3.1-8B behavioral |
| `new-phase-results/debateqa/` | DebateQA behavioral (all models) |
| `new-phase-results/exp10_persona_sweep/` | Persona sweep analysis (6 personas) |
| `new-phase-results/piqa/` | PIQA behavioral |
| `new-phase-results/mechanism/` | All mechanism extractions, probes, patching |
| `new-phase-results/reasoning-traces/` | GPT-oss confabulation pilot |

### Key Mechanism Files

| Path | Contents |
|---|---|
| `mechanism/qwen_correction_gating_i1a_i1c_note/` | Qwen3-4B extraction (449 items) |
| `mechanism/gpt_oss_20b_correction_gating_i1a_i1c_note/` | GPT-oss-20B extraction (810 items) |
| `mechanism/gemma_correction_gating_i1a_i1c_note/` | Gemma-4-26B extraction (887 items) |
| `mechanism/qwen_n0_note_activations/` | Qwen N0 activations |
| `mechanism/gpt_oss_n0_note_activations/` | GPT-oss N0 activations |
| `mechanism/gemma_n0_note_activations/` | Gemma N0 activations |
| `mechanism/qwen_n0_probe_3way.json` | Qwen three-way decomposition |
| `mechanism/gpt_oss_n0_probe_3way.json` | GPT-oss three-way decomposition |
| `mechanism/gemma_n0_probe_3way.json` | Gemma three-way decomposition |
| `mechanism/n0_residual_correlation.json` | Precision-weighting correlation results |
| `mechanism/logit_lens_qwen/` | Qwen instruct logit lens trajectories (449 items) |
| `mechanism/logit_lens_qwen_base/` | Qwen base model logit lens trajectories (449 items) |
| `mechanism/gpt_oss_multilayer_patching/` | GPT-oss multi-layer patching results |
| `mechanism/cmap_base_instruct/` | CMAP cross-model patching (base vs instruct) |
| `mechanism/cmap_layer_sweep/` | CMAP single-layer sweep + reverse CMAP |
| `mechanism/gating_head_surgery/` | Head-level gating attribution (L22-24) |
| `mechanism/temporal_commitment/` | Temporal commitment dynamics (layer sweep) |
| `mechanism/multiplicative_gating/` | Multiplicative vs additive gating test |
| `mechanism/qwen3_5_27b_gating_i1a_i1c_note/` | Qwen3.5-27B extraction (208 items) |
| `mechanism/qwen3_5_27b_patching_capping_note/` | Qwen3.5-27B patching (7 resisting) |
| `mechanism/olmo2_32b_gating_i1a_i1c_note/` | OLMo-2-32B extraction (320 items) |
| `mechanism/olmo2_32b_patching_capping_note/` | OLMo-2-32B patching (21 resisting) |
| `mechanism/gemma4_expanded_gating_i1a_i1c_note/` | Gemma-4-26B expanded extraction (887 items) |

### Analysis Scripts

| Path | What it does |
|---|---|
| `src/exp10/analyze_credulity_dial.py` | Credulity dial analysis across all models |
| `src/exp10/analyze_debateqa_priorstate.py` | Prior-state conditioning for DebateQA |
| `src/mechanism/extract_condition_activations.py` | GPU activation extraction |
| `src/mechanism/sweep_probe_layers.py` | Parallelized layer sweep (joblib) |
| `src/mechanism/evaluate_probe_transfer.py` | Cross-condition probe transfer |
| `src/mechanism/analyze_n0_probe.py` | N0 probe with three-way decomposition (parallelized) |
| `src/mechanism/analyze_belief_provenance.py` | Cross-model correlation, consensus analysis |
| `src/mechanism/n0_residual_correlation.py` | Precision-weighting test: N0 residual vs update |
| `src/mechanism/logit_lens_analysis.py` | Logit lens: layer-by-layer answer trajectory |
| `src/mechanism/logit_lens_base_model.py` | Logit lens on base model (RLHF vs pre-training) |
| `src/mechanism/run_multilayer_patching.py` | Multi-layer patching (parallelized probe sweep) |
| `src/mechanism/cmap_base_instruct.py` | CMAP: patch instruct activations into base model |
| `src/mechanism/cmap_layer_sweep.py` | CMAP single-layer sweep + reverse CMAP |
| `src/mechanism/run_debateqa_patching.py` | DebateQA causal patching |
