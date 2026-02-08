# Experiment Plan: Regroup and Next Steps (Feb 2026)

## TL;DR Decisions

1. **PsiloQA**: not a fit for this paper.
2. **CBRN / safety-bypass framing**: not recommended for current submission scope.
3. **Immediate priority**: fix Exp12 validity controls, then run targeted novelty experiments.
4. **Write now, but do not freeze claims yet**.

---

## Where We Are

Completed core sequence: `exp7 -> exp12` on Llama/Qwen (with Exp12 on Instruct variants).

What is strong:
1. Endorsement effects are real and authority-sensitive.
2. Instruction can strongly reduce initial wrong-endorsement shift.
3. Multi-turn carryover exists in context mode.

What is not yet fully settled:
1. **Fresh-mode zero residual in Exp12 is not a fully independent negative control** in current implementation.
2. Probe-style comparisons are partly confounded by prompt-form differences (imperativeness/constraints).
3. Llama and Qwen do **not** look cleanly “qualitatively identical” in Exp12 dynamics; heterogeneity is substantial.

---

## Dataset Recommendations

## Keep

1. `external/sycophancy-eval` MCQ setup as the base benchmark (continuity with prior experiments).

## Add (high value)

1. **TruthfulQA** (misconception-heavy): best fit for your prior-consistency story.
2. **MMLU subset** (domain/difficulty spread): best fit for external-validity claims.

## Optional

1. **ARC-Challenge** (hard science MCQ) if compute allows.

## Avoid for this paper

1. **PsiloQA**: span-level hallucination detection task; misaligned with forced-choice logit-decomposition design.
2. **CBRN custom dataset**: high confound risk (safety refusal vs anti-sycophancy), high ethics/reviewer risk, likely scope creep.

---

## Priority Experiments

## P0: Validity Gate (must run before strong claims)

### Exp12-R: Strict Fresh Control
- Run fresh probes independently for each history branch (`HN/HW/HC`), no result-sharing.
- Goal: test whether transcript-independent residuals are truly zero.

### Exp12-M: Matched Probe Styles
- Make `same/paraphrase/swap` matched in imperativeness, output constraint, and length.
- Goal: isolate style effects from instruction-strength artifacts.

### Exp12-T: Prevention vs Cure Timing
- Instruction at `T0` vs only added at `T1` vs only added at `T2`.
- Goal: causal separation of prevention from recovery.

### Exp12-K: Repeated Endorsement (Badgering)
- K-turn wrong endorsement with `K in {1,2,5,10,20}`.
- Probe context and fresh after each K.
- Goal: accumulation, saturation, or resistance dynamics.

---

## P1: Novelty Layer (high upside for workshop/main)

### Exp-I: Imperativeness Sweep
- Factor: low/medium/high imperativeness with matched semantics.
- Apply to both endorsement and correction prompts.
- Goal: connect compliance dynamics to command strength (and explain instruction effects mechanistically).

### Exp-A: Authority × Order Arbitration
- Conflict setup: wrong-high-authority then correct-low-authority, and reverse.
- Goal: disentangle authority, recency, and contradiction resolution policy.

### Exp-G: Representation Dynamics Over Turns
- Track hidden-state drift and recovery under repeated pressure.
- Include targeted interventions at follow-up turns.
- Goal: move from behavioral evidence to mechanistic temporal evidence.

---

## P2: External Validity Layer

### Exp-D1: TruthfulQA Transfer
- Same decomposition pipeline, with misconception alignment analysis.

### Exp-D2: MMLU Transfer
- Stratify by baseline confidence and domain cluster.

---

## Internal-State Poisoning Program

Use explicit competing hypotheses:

1. **H0 (Context-only carryover)**: effects vanish when transcript is removed.
2. **H1 (Transient internal adaptation)**: small transcript-independent residual appears after repeated pressure but decays quickly.
3. **H2 (Durable poisoning)**: transcript-independent residual persists across new prompts/items.

Discriminating tests:

1. **Independent fresh probes** (Exp12-R) -> tests H0 vs (H1/H2).
2. **Delayed fresh retest** after filler turns/new chat -> tests transient vs durable.
3. **Cross-item transfer** after repeated badgering on one set -> tests whether bias generalizes.
4. **Representation recovery curves** (Exp-G) -> tests whether internal geometry returns to neutral.

Decision rule for strong “no poisoning” claim:
1. Near-zero independent fresh residual across K.
2. No delayed fresh residual.
3. No cross-item transfer.
4. Hidden-state drift that is reversible after context removal.

---

## Submission Strategy

## ICML Workshop (target now)

Minimum package:
1. Existing `exp7-12` story (with corrected caveats).
2. `P0` validity gate experiments (`Exp12-R`, `Exp12-M`, `Exp12-T`).
3. One novelty module (`Exp-I` or `Exp12-K`).

## NeurIPS Main (target after workshop)

Needed additions:
1. Full `P0 + P1` set.
2. Cross-dataset replication (`TruthfulQA`, `MMLU`).
3. Mechanistic temporal evidence (`Exp-G`).
4. Strong statistics and ablations for cross-model heterogeneity.

---

## Practical Notes

1. Keep claims tiered in drafts: `established`, `likely`, `pending controls`.
2. Avoid “safety bypass” framing in this paper; focus on compliance-control dynamics and epistemic reliability.
3. Treat Qwen/Llama differences as a feature, not noise: model-family heterogeneity is publishable if analyzed cleanly.
