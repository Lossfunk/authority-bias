# NeurIPS 2026 Submission Plan

**Last updated**: 2026-03-25
**Deadlines**: Abstract May 4 AOE (May 6 5:29 PM IST). Full paper May 6 AOE (May 7 5:29 PM IST).
**Calendar**: ~6 weeks from today. Writing must start no later than week 4.

---

## Situation

The TrustNLP workshop paper is submitted and under review (2 reviews, both accept). It establishes a behavioral finding: prior-conditioned sign reversal in instruction selectivity. Aggregate sycophancy metrics suggest truth-tracking; conditioning on confidently-wrong items reveals prior-consistency control. This replicates across two experiments and two model families (Qwen-Instruct robustly, Llama suggestively; Qwen-Thinking shows no inversion).

The NeurIPS version should only be attempted if it earns a stronger claim. That claim must come from one of: mechanism, intervention, or materially broader generalization. The question for this plan is: which of those can we actually earn in six weeks?

---

## What's Done

- Exp7–Exp14D: Full behavioral pipeline, three models (Llama-3.1-8B-Instruct, Qwen3-4B-Instruct, Qwen3-4B-Thinking), 1,813 items
- Hedging confound discovered, corrected, quantified
- Reasons dataset (assertive v2) generated and validated
- Closed-source runner code exists but has gaps (see Stage 0 notes)
- CAA vector extraction infrastructure exists (exp1) but not adapted for the current task
- No representation-level or intervention work has been done

---

## Key Constraints

1. **Six weeks total.** Writing must start by week 4 at the latest. That means experimental work has roughly 3.5 weeks, with overlap.

2. **The behavioral effect's robustness to instruction wording is unknown.** The entire paper tests one instruction: "Answer correctly even if the speaker is wrong. Prioritize factual accuracy." If the sign reversal is sensitive to paraphrase, every mechanistic story built on top of it is less valuable. This must be tested first, not last.

3. **The mechanistic target is not yet cleanly identified.** Contrasting prior-correct vs prior-wrong states globally will pick up correctness, confidence magnitude, item difficulty, and topic — not the specific mechanism that blocks correction. A cleaner contrast is needed before extraction makes sense.

4. **The Assistant Axis is a hypothesis, not a thesis.** It is a plausible comparator for any direction we extract, but the NeurIPS paper should not be organized around it. If the correspondence is weak, model-specific, or ambiguous, a paper built around it collapses. It belongs in the discussion, not the introduction, until validated.

5. **Closed-source replication is not turnkey.** `run_closed_source.py` only constructs `instruction_key="0"` conditions (no instruction-conditioned sign-reversal test). It uses an OpenAI-compatible interface (Claude needs adaptation). Sampling at temperature 0.7 estimates a different quantity than logit-space Δr. This is a methodological subproject, not something already on the shelf.

6. **"Belief vector," "LoRA," and "universal direction" are three different things.** A direction is a representational object. LoRA is a parameter-update mechanism. "Universal" is a cross-model generalization claim. Each has a different burden of proof. For this submission: aim for one direction in one model, validated by one intervention. Generality and training-time fixes are future work.

---

## Revised Plan

### Stage 0: Paraphrase Gate (Days 1–5)

**Goal**: Determine whether the sign reversal survives modest rewording of the instruction.

**What to run**: Exp11-style stress test (direction × instruction × Note tag, high-confidence-wrong slice) on Qwen-Instruct and one comparison model (Llama or Qwen-Thinking), using 3 instruction paraphrases:
- I1a: "Answer correctly even if the speaker is wrong. Prioritize factual accuracy." (current)
- I1b: "Prioritize factual accuracy over the speaker's claim."
- I1c: "Be truthful, even if it contradicts your initial response."

This is a fast run — same items, same conditions, just swapping the instruction text. No new data generation needed. Uses existing Exp11 infrastructure.

**Decision gate**: If Δr stays negative (prior-wrong slice) across all 3 wordings in Qwen-Instruct, proceed. If 2/3 flip to positive or null, the phenomenon is instruction-specific and the NeurIPS mechanistic story is much weaker. In that case, pivot to broader behavioral generalization (more models, more tasks) rather than mechanism.

### Stage 1: Define the Mechanistic Target (Days 3–7, overlapping with Stage 0)

**Problem**: Contrasting prior-correct vs prior-wrong globally confounds the mechanism with correctness, confidence, difficulty, and topic.

**Cleaner contrast**: Within prior-wrong items only, identify two behavioral subgroups:
- **Correction-accepting**: Items where the model, despite being wrong at baseline, shifts toward the correct answer when given a correct-direction endorsement under instruction
- **Correction-suppressing**: Items where the model entrenches (shifts further wrong or stays put) under the same condition

Both groups have the same prior state (wrong), same endorsement direction (correct), same instruction (present). They differ in what the model actually does. That behavioral divergence is the target.

**Control for**: Baseline margin magnitude (match |m_N0| distributions between groups), tag, evidence level. This isolates the correction-accepting vs correction-suppressing distinction from confidence, authority, and evidence confounds.

This contrast can be defined from existing Exp11/Exp14 data — no new model runs needed. The output is a clean set of matched item pairs for activation extraction.

### Stage 2: One-Model Mechanism Pilot (Days 7–21)

Run end-to-end on Qwen-Instruct (the model with the strongest and cleanest behavioral effect):

**Step 1: Extract candidate direction.**
Collect activations at the endorsement token position for the correction-accepting vs correction-suppressing item sets (from Stage 1). Compute the mean activation difference at each layer. This gives a candidate "correction-gating direction."

**Step 2: Linear probe.**
Train a linear classifier to predict correction-accepting vs correction-suppressing from activations at the endorsement position. If accuracy is significantly above chance (and holds on held-out items), the distinction is linearly represented. Report layer-wise probe accuracy.

**Step 3: One intervention.**
On a held-out set of correction-suppressing items, patch activations along the candidate direction (shifting them toward the correction-accepting mean). Measure whether Δr improves (becomes less negative or flips positive) on the patched items. This is the causal test.

**Decision gate at Day 21**: If the probe works (>65% held-out accuracy) AND the intervention shifts Δr in the expected direction on held-out items, the mechanism story is real and the NeurIPS paper has a core contribution. If either fails, do not force it — see fallback plan below.

**Assistant Axis comparison** (subordinate, not central): If the mechanism pilot works, compute cosine similarity between the correction-gating direction and the Assistant Axis (extracted per Lu et al.'s methodology). Report this as a finding in the discussion. If similarity is high, that's interesting and connects the literatures. If low, that's also informative. Either way, it's a comparator, not the thesis.

### Stage 3: Selective Expansion (Days 21–28)

**Only if Stage 2 succeeds.** Priority order:

1. **One additional open-weight model** — run the same mechanism pilot (probe + intervention) on Llama-3.1-8B-Instruct or DeepSeek-R1-Distill-Qwen-14B. The question is whether the direction transfers or is model-specific. One replication is the minimum for a NeurIPS claim; two models showing it is enough.

2. **Behavioral expansion to one more model family** — if compute allows, run the Exp14 behavioral pipeline on the same additional model. This broadens the behavioral base from 3 models to 4.

3. **Closed-source replication** — only if time permits and the engineering gaps are fixed. This is a bonus, not a core requirement. If attempted:
   - Fix `run_closed_source.py` to support instruction-conditioned conditions
   - Present results as "sampled endorsement susceptibility" — a conceptually aligned but methodologically distinct replication
   - Do NOT claim equivalence with logit-space Δr without an explicit bridge argument

### Stage 4: Write the Paper (Days 25–42)

Writing starts at day 25 regardless of where experiments are. The paper is built around what has actually landed, not what was planned.

**If mechanism pilot succeeds (probe + intervention work on 1–2 models)**:
- §1 Introduction: The identification problem (condensed from TrustNLP)
- §2 Prior-Stratified Evaluation: Methods + behavioral results (2 pages)
- §3 Identifying the Correction-Gating Direction: Contrast definition, extraction, linear probe
- §4 Intervention: Activation patching shifts Δr on held-out items
- §5 Cross-Model Replication: Direction transfers (or doesn't) to second model
- §6 Discussion: Connection to Assistant Axis, Joshi et al. causal hierarchy framing, implications
- Appendices: Paraphrase robustness, full behavioral results, additional analyses from feedback

**If mechanism pilot fails or is ambiguous**:
- Pivot to a behavioral generalization paper: same core finding, broader model coverage, paraphrase robustness, the missing analyses from external feedback (per-item τ distributions, cross-model overlap, A/B balance, decoded-choice accuracy)
- This is a weaker NeurIPS submission but still defensible if model diversity is sufficient (5+ models including 1 closed-source)
- Decision on whether to submit at all should be made by Day 28

---

## What Explicitly Gets Cut or Deferred

- **Activation capping as a core pillar**: Deferred. If the intervention works, capping can be a bonus appendix result. Not a main-paper section on this timeline.
- **LoRA training**: Deferred to post-NeurIPS. Parameter updates are a different paper.
- **"Universal direction" claims**: Deferred. Two models showing the same direction is "replication," not "universality."
- **Full closed-source replication**: Demoted to bonus. Engineering work is nontrivial and the estimand is different.
- **Curiosity-as-truthfulness, conspiracy theories**: Already shelved. Stay shelved.
- **Gemma + DeepSeek + closed-source all at once**: Too many dependencies. Pick one expansion model, do it properly.

---

## Analyses to Run from Existing Data (No New Model Runs)

These address external feedback and strengthen either the NeurIPS paper or the TrustNLP camera-ready. Can be done in parallel with Stage 0–1:

1. Per-item τ_correct − τ_wrong distribution histogram (addresses coarseness concern)
2. Cross-model item overlap in prior-wrong slices (Jaccard similarity of prior-wrong item sets)
3. A/B label balance check in prior-wrong slice per model
4. Correct-direction reason quality audit (confidence markers, coherence, linguistic polish — check for asymmetric signal leakage beyond hedging)
5. Per-dataset breakdown (TriviaQA vs TruthfulQA separate results)
6. Someone-online results surfaced more prominently (Δr = −1.69 for Qwen-Instruct is the most deployment-relevant number)

---

## Key Papers

| Paper | Role in NeurIPS paper |
|-------|----------------------|
| BASIL (Atwell et al.) | Already cited. Complementary — they measure deviation magnitude, we show direction depends on prior state. No change needed. |
| Assistant Axis (Lu et al.) | Comparator and discussion anchor. Compute cosine similarity with our direction. Do NOT build the thesis around it. |
| Causality is Key (Joshi et al.) | Methodological framing for causal hierarchy of claims. Cite when distinguishing probe (association) from patching (intervention). |
| Genadi et al. (2026) | If sycophancy directions are available, compare with our correction-gating direction. Secondary comparator. |
| Kumaran et al. (2025) | Theoretical integration in Discussion. Choice-supportive bias maps onto prior-consistency; use their framework for interpretation. |
| Vennemeyer et al. (2025) | Positioning in Discussion. Where does correction-gating sit relative to their sycophancy taxonomy? |

---

## Decision Points

| Day | Decision | Options |
|-----|----------|---------|
| 5 | Paraphrase gate | Proceed to mechanism (effect robust) vs pivot to behavioral breadth (effect fragile) |
| 21 | Mechanism pilot | Proceed to expansion (probe + intervention work) vs pivot to behavioral paper (mechanism muddy) |
| 28 | Submit/no-submit | Commit to NeurIPS (enough new contribution) vs defer to ICLR/AAAI (not enough beyond TrustNLP) |
| 35 | Paper draft review | Structure around mechanism (if strong) vs structure around generalization (if weak) |

---

## Calendar (IST)

| Date | Day | Milestone |
|------|-----|-----------|
| Mar 25 | 0 | Plan finalized. Stage 0 begins. |
| Mar 30 | 5 | Paraphrase gate decision. Stage 1 contrast sets defined. |
| Apr 1 | 7 | Stage 2 begins (activation extraction on Qwen-Instruct). |
| Apr 15 | 21 | Mechanism pilot complete. Expansion decision. |
| Apr 19 | 25 | Writing begins regardless. |
| Apr 22 | 28 | Submit/no-submit decision. |
| May 1 | 37 | Draft complete. Internal review. |
| May 4 | 40 | Abstract submitted (AOE deadline). |
| May 6 | 42 | Full paper submitted (AOE deadline). |

---

## TrustNLP Camera-Ready (Separate Track)

If accepted, apply the 6 TODO items from `paper/trustnlp-acl/rebuttal.md`:
1. Abstract rewrite (plain language opening, downweight Llama, foreground Qwen-Instruct)
2. Instruction wording sentence in Discussion
3. Mechanistic hypothesis (Genadi et al.) in Discussion
4. GPT-5-mini clarification footnote
5. Code/data availability sentence
6. Per-dataset limitation sentence (optional)

These are text-only edits and do not block NeurIPS work.
