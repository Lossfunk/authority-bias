# Experimental Results

**Project**: Instruction-Dependent Evidence Gating in Language Models
**Branch**: `neurips-submission`
**Last updated**: 2026-04-05

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

### Open Questions

1. What is the "third thing" (beyond confidence and position) in the N0 representation?
2. Is the N0 signal created by post-training (RLHF/DPO) or present in base models?
3. Why does GPT-oss have the signal but Gemma does not? Architecture (dense vs MoE) or training recipe?

---

## 11. File Locations

### Result Directories

| Path | Contents |
|---|---|
| `new-phase-results/qwen3-4b-results/` | Qwen3-4B behavioral (exp7-exp14) |
| `new-phase-results/qwen3-30b-a3b-results/` | Qwen3-30B behavioral |
| `new-phase-results/gpt_oss_20b-results/` | GPT-oss-20B factual QA behavioral |
| `new-phase-results/gemma-4-26b-results/` | Gemma-4-26B behavioral |
| `new-phase-results/llama-3.1-8b-results/` | Llama-3.1-8B behavioral |
| `new-phase-results/debateqa/` | DebateQA behavioral (all models) |
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
| `src/mechanism/run_debateqa_patching.py` | DebateQA causal patching |
