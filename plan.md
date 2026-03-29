# NeurIPS 2026 Submission Plan

**Last updated**: 2026-03-26 (v5 — added identity controls, stop-loss rule, tightened claims)
**Deadlines**: Abstract May 4 AOE (May 6 5:29 PM IST). Full paper May 6 AOE (May 7 5:29 PM IST).
**Calendar**: ~6 weeks from March 25. Writing must start no later than April 16.

---

## Situation

The TrustNLP workshop paper is submitted (2 reviews, both accept). It establishes a behavioral finding: prior-conditioned sign reversal in instruction selectivity.

Since submission, we have completed:
- Paraphrase gate (3 instruction wordings on Qwen)
- Llama cross-family replication (factual QA)
- Gemma-3-12B third-family replication (factual QA)
- PIQA cross-domain generalization (Qwen + Llama)
- i1d linguistic ablation (Qwen, factual QA)
- Qwen-Thinking reasoning traces (117 high-conf-wrong items, 4 conditions)

The NeurIPS paper has three possible shapes depending on whether the mechanistic work lands:
1. **Best case**: Behavioral finding + mechanistic explanation (extraction → probe → patching) + practical intervention
2. **Good case**: Behavioral finding + preliminary mechanistic evidence (extraction + probe, patching inconclusive)
3. **Fallback**: Strong behavioral/methodological contribution alone (still viable but harder at NeurIPS)

**The mechanistic work is the swing factor.** Everything else is supporting evidence.

---

## What's Done

### Completed Experiments

| Experiment | Models | Datasets | Status |
|-----------|--------|----------|--------|
| Paraphrase gate (i1a, i1b, i1c) | Qwen-Instruct | Factual QA | ✅ (i1b on 112 items only) |
| Full i1a vs i1c | Qwen-Instruct | Factual QA | ✅ |
| Full i1a vs i1c | Llama-3.1-8B | Factual QA | ✅ |
| Full i1a vs i1c | Gemma-3-12B | Factual QA | ✅ |
| Full i1a vs i1c | Qwen-Instruct | PIQA | ✅ |
| Full i1a vs i1c | Llama-3.1-8B | PIQA | ✅ |
| i1d linguistic ablation | Qwen-Instruct | Factual QA | ✅ |
| Reasoning traces | Qwen-Thinking | Factual QA (117 items) | ✅ (qualitative) |

### Key Results Summary

**Note tag (primary cell), prior-wrong items, dr:**

| Dataset | Model | i1a | i1c | Direction |
|---------|-------|-----|-----|-----------|
| Factual QA | Qwen (4B) | −0.68 | **+0.42** | Sign flip |
| Factual QA | Llama (8B) | −0.36 | +0.26 (null) | Attenuation |
| Factual QA | Gemma (12B) | −7.29 (pathological) | +1.52 (pathological) | Metric-dependent |
| PIQA | Qwen (4B) | −0.46 | **+0.13** | Sign flip |
| PIQA | Llama (8B) | −0.59 | **+0.54** | Sign flip |

**Graded ablation (Qwen, Note, factual QA):**

| Instruction | dr (full) | eff_diff (full) |
|-------------|-----------|-----------------|
| i1a (resist-speaker) | −0.68 | −7.88 |
| i1d (evaluate, no adversarial) | −0.05 | −4.89 |
| i1c (facts-framing) | +0.42 | +2.54 |

**Expert crossover**: Expert shows a reliable dissociation from Note, strongest in Qwen factual QA. Elsewhere more mixed, smaller, or metric-dependent.

**Reasoning traces**: 19 entrenchment cases, 22 correction-accepted, 35 literalization, 19 confabulation (preliminary qualitative coding — no formal rubric or inter-annotator reliability yet). "Think-then-ignore" pattern independently matches MATS findings.

---

## Revised Priority Stack

### Priority 0: Draft Fallback Paper Structure (Day 0–1)

Before any more experiments, draft the behavioral-only paper skeleton. This ensures we're not scrambling if the mechanistic work is inconclusive.

**Fallback paper structure** (behavioral/methodological contribution):
- §1 Introduction: The identification problem + why instruction form matters
- §2 Prior-Stratified Evaluation Framework: Methods, metrics, prior-state conditioning
- §3 Experiment 1 — Prior-Conditioned Sign Reversal: i1a results (condensed from TrustNLP)
- §4 Experiment 2 — Instruction Form Sensitivity: i1a vs i1c factorial + i1d ablation
  - Central cells: Note × Expert
  - Graded ablation: i1a → i1d → i1c (the best single figure in the paper)
  - Cross-model: Qwen (sign flip) + Llama (attenuation) + Gemma (partial)
  - Cross-domain: Factual QA + PIQA
- §5 Experiment 3 — Reasoning Traces: Qualitative evidence from Qwen-Thinking showing "think-then-ignore"
- §6 Discussion: Instruction-as-authority-signal, connection to consistency training (GDM), MATS constitution auditing, implications
- §7 Limitations
- Appendices: Full tag results, evidence-quality scaling (Exp14), hedging confound, repeated pressure

**Full paper structure** (if mech work lands):
- Same as above, but §5 becomes the mechanistic section (extraction → probe → patching)
- Reasoning traces move to §6 or appendix
- Claim 3 (mechanistic) becomes load-bearing

### Priority 1: Mechanistic Pilot — THE GATING EXPERIMENT (Days 1–14)

This is the single most important thing to do next. Everything else is secondary.

**Step 1: Define contrastive pairs (Day 1)**

On Qwen-Instruct, factual QA, Note tag, prior-wrong items:
- i1a (resist-speaker) vs i1c (evaluate-facts) on the same items
- Both produce measurable endorsement effects but in opposite behavioral directions
- This controls for: item content, prior state, endorsement direction, tag. Only instruction form varies.

Within i1a, further split:
- Correction-suppressing items (i1a makes the model more wrong)
- Correction-accepting items (i1a doesn't entrench, or model still shifts correct)

This gives the cleanest possible contrast for activation extraction.

**Step 2: Extract activations (Days 2–4)**

Collect residual stream activations at the instruction/endorsement token positions for both i1a and i1c conditions. Use existing CAA infrastructure from exp1, adapted for the current task.

Compute mean activation difference at each layer. This gives a candidate "correction-gating direction."

**Step 3: Linear probe (Days 4–7)**

Train a linear classifier to predict i1a-entrenchment vs i1c-correction from activations at the endorsement position. Layer-wise held-out accuracy.

**Critical: compare against baselines.** Shuffled-label probes and random-direction controls. The >65% threshold is only meaningful relative to these baselines.

**Decision gate at Day 7**: If probe accuracy >65% held-out at any layer AND significantly above shuffled-label baseline, proceed to patching. If probe fails (<55% everywhere or not above baseline), the behavioral divergence is not linearly represented. Pivot to fallback paper.

**Step 3b: "Not just instruction identity" controls (Days 5–7, concurrent with probe)**

The most obvious confound: a reviewer will say "of course you can linearly separate i1a and i1c — the prompts are lexically different. That doesn't mean the direction is about correction-gating." Three controls address this:

1. **i1d projection test**: Project i1d activations onto the extracted direction. Under Note, i1d should land between i1a and i1c (it's behaviorally intermediate). Under Expert, i1d should NOT land between them (i1d behaves like i1a on Expert, not intermediate). If the direction tracks behavioral outcome rather than lexical similarity, the projection should be tag-dependent. If it just separates instruction wording, projections would be tag-invariant.

2. **Within-i1a outcome probe**: Within items that all received i1a, can activations predict which items the model entrenches on vs which it (partially) corrects? This uses the same instruction everywhere, so any separability is about the item-level processing, not instruction identity. This is the strongest control — if it works, the direction is genuinely about correction-gating.

3. **Cross-dataset transfer**: Extract the direction on factual QA, test probe accuracy on PIQA (or vice versa). If the direction is about lexical instruction features, it should transfer perfectly. If it's about correction-gating mechanisms, transfer should be good but not identical (the mechanism is the same, the items differ).

All three are cheap once activations are extracted. They share the same activation data. The i1d projection test is the fastest (no training, just dot products). The within-i1a probe is the most convincing.

**Step 4: Activation patching (Days 7–12)**

On held-out correction-suppressing items (under i1a), patch activations along the candidate direction (shifting toward the i1c/correction-accepting mean).

**Critical DV choice**: Do NOT measure patching success via Δr (too noisy — second-order metric, noise propagates through ratio computation). Use both:
- **Primary: Per-item answer change**: Does the model's forced-choice answer flip from wrong to correct after patching? (Clean binary signal, but may be sparse.)
- **Secondary: Logit margin shift**: Does the margin move toward the correct answer? (Continuous signal, captures sub-threshold improvements even when answer doesn't flip.)

**Step 5: Characterize the direction (Days 10–14)**

Compare the correction-gating direction against multiple known directions:
- Assistant Axis (Lu et al., 2026) — is correction-gating just persona reinforcement?
- Sycophancy direction (Panickssery/Genadi) — is it a subtype of sycophantic agreement?
- Random baselines — is the cosine similarity meaningful?

**Do NOT commit to one comparison in advance.** Any outcome is informative:
- High cosine with Assistant Axis → instruction-induced entrenchment is persona reinforcement
- High cosine with sycophancy direction → correction-gating is a specific subtype of sycophancy
- Low cosine with all → genuinely new representational axis
- Present all comparisons and let the data speak

**If patching succeeds AND the direction passes at least 2/3 identity controls**: attempt activation capping as a bonus (Days 12–14). Not load-bearing.

**HARD STOP-LOSS RULE**: If probe is weak (below baseline or <60% after one serious pass) OR patching is inconclusive (answer-flip rate not significantly above chance after one round of iteration), STOP. Do not iterate on the mech section. Pivot immediately to fallback paper and redirect remaining time to 30B+ model, i1b full run, and writing. The mech section must not eat the paper.

### Priority 2: Parallel Quick Wins (Days 1–7, concurrent with mech work)

These require no new model runs and directly address reviewer concerns:

1. **m_N0 distribution comparison**: Factual QA vs PIQA baseline margins. Tests the "softer priors explain cleaner PIQA results" hypothesis. If confirmed, explains cross-dataset variation. If not, need a different explanation before writing.

2. **Baseline endorsement susceptibility by dataset and tag**: Compute the raw endorsement effect (no instruction) per tag on both datasets. The "PIQA is cleaner because priors are softer" story may be only half the explanation — PIQA items may also differ in how susceptible they are to endorsement in general. This goes alongside m_N0 as a complementary analysis.

3. **Ratio metric pathology characterization**: For Gemma Note (dr = −11.35, +9.22), examine the per-item ratio distribution. Is it bimodal? Heavy-tailed? Driven by a few extreme items with tiny denominators? This helps readers trust the non-pathological cells.

3. **i1b full run on Qwen** (factual QA, Note + Expert tags): Cheap — one model, existing pipeline. Fills the conspicuous gap in the graded ablation. If i1b fits the i1a → i1b → i1d → i1c gradient, you have four-point dose-response from two independent ablation axes.

4. **Qwen-Thinking logit vs generation disconnect**: Report explicitly that 36.4% baseline accuracy under generation on "high-confidence-wrong" items reflects the reasoning model's capacity to override its first-token prior through extended CoT. Support with one empirical check: compare generated final answer vs first-token forced-choice on the same subset (or compare generation at temp=0). One short table kills this objection cleanly.

### Priority 3: 30B+ Model (Days 5–10, if H100 available)

**Single biggest reviewer vulnerability.** One condition would close it.

**What to run**: Note tag, i1a vs i1c, factual QA, full prior-wrong slice.
**Model options** (in order of preference):
- Qwen3-30B-A3B (MoE, same family as our cleanest model, ~30B active)
- Llama-3.1-70B (if H100 80GB available — needs ~140GB for bf16, so either quantize or use 2×H100)

Even a null result is informative: if larger models don't show the effect, that's a scaling finding. If they do, it's a massive strengthener.

### Priority 4: Write the Paper (Days 14–42)

Writing starts Day 14 if mech pilot probe results are in. Day 16 at latest regardless.

**The single summary figure**: 2×2 faceted plot where columns = Note/Expert, rows = aggregate/prior-wrong, with i1a and i1c as grouped bars showing Δr. This communicates the entire triple interaction in one glance. Must be Figure 1 or Figure 2.

**Abstract framing** (must be honest):
- "In the strongest case (Qwen-Instruct), instruction framing flips the sign of the effect on prior-wrong items; across all three model families, it attenuates the wrong-direction advantage by 34–55%."
- Do NOT say "we show instruction framing reverses error entrenchment" without the scope condition.

---

## Revised Claim Hierarchy

### Claim 1: Methodological (strongest, most general)
Prior-stratified evaluation is necessary for trust-relevant assessment of LLM instruction fidelity. Aggregate metrics are structurally blind to the failure modes we demonstrate.

*Stands regardless of any particular model's behavior.*

### Claim 2: Behavioral (moderate generality)
Instruction framing and authority context jointly determine whether prior-wrong states are corrected or entrenched. We demonstrate this with a graded ablation (i1a → i1d → i1c), across two datasets and three model families. The general finding is attenuation; the sign flip is a strong special case in Qwen.

*Model-dependent in magnitude. Directionally consistent across families.*

### Claim 3: Mechanistic (CONDITIONAL — depends on pilot results)
The behavioral divergence between resist-speaker and evaluate-facts instructions corresponds to a linearly extractable direction. Patching along this direction shifts correction-suppressing items toward correction. The direction's relationship to known axes (Assistant Axis, sycophancy directions) characterizes what representational structure the instruction manipulation is operating on.

*Only included if probe + patching succeed. If inconclusive, becomes "preliminary mechanistic evidence" or is dropped entirely.*

### Claim 4: Practical (scoped)
In this evaluation regime, no instruction wording is universally helpful across authority contexts on prior-wrong items. What helps under low authority can hurt under high authority.

*Scoped to: 4B–12B models, two-choice QA, tested instruction phrasings. Explicitly does not claim generality to 70B+ models, open-ended generation, or multi-turn dialogue.*

---

## Decision Points

| Day | Decision | Gate condition | If pass | If fail |
|-----|----------|---------------|---------|---------|
| 7 | Probe accuracy | >65% held-out at any layer | Proceed to patching | Pivot to fallback paper; prioritize 30B+ and i1b |
| 12 | Patching effect | Per-item answer change significantly above baseline | Include mech section as core contribution | Probe-only as "preliminary evidence"; paper is primarily behavioral |
| 14 | Direction characterization | Cosine similarities computed against ≥3 known axes | Include comparison; strongest correlation anchors discussion | Report as new axis; still valuable |
| 16 | Writing start | N/A | Mandatory regardless of experiment status | — |
| 30 | Submit decision | Paper draft is coherent and makes contribution beyond TrustNLP | Submit | Defer to ICLR 2027 |

---

## Calendar (IST)

| Date | Day | Milestone |
|------|-----|-----------|
| Mar 26 | 1 | Plan v4 finalized. Start mech pilot (define contrastive pairs). Start parallel analyses. |
| Mar 27 | 2 | Activation extraction begins on Qwen-Instruct. i1b full run starts. |
| Mar 30 | 5 | m_N0 distribution analysis complete. Gemma ratio pathology characterized. |
| Apr 1 | 7 | **PROBE GATE.** Linear probe results in. Go/no-go on patching. |
| Apr 1 | 7 | 30B+ model run starts (if H100 available). |
| Apr 5 | 10 | i1b full run complete. Patching experiments running (if probe passed). |
| Apr 8 | 14 | **PATCHING GATE.** Direction characterization complete. Paper structure locked. |
| Apr 10 | 16 | **WRITING STARTS** regardless. |
| Apr 20 | 26 | Draft v1 complete. Internal review. |
| Apr 27 | 33 | Draft v2. All figures finalized. |
| May 1 | 37 | Final revision. |
| May 4 | 40 | Abstract submitted (AOE). |
| May 6 | 42 | Full paper submitted (AOE). |

---

## Addressing Specific Reviewer Concerns

### "The Expert crossover is rhetorically inflated"

**Rebuttal**: The i1d result strengthens the Expert story beyond null → mildly-negative. Under Expert: i1a = null (+0.11), i1d = strongly positive (+1.26 on high-conf), i1c = negative (−0.73). This is a **non-monotonic** pattern where the intermediate instruction is the best for Expert. The full ablation on Expert is: removing the adversarial clause alone (i1d) HELPS Expert, but adding the facts-framing (i1c) HURTS it. This is a genuine dissociation, not a weak effect.

**In the paper**: Present the i1d Expert result as a finding in its own right, not just as evidence for the Note story. "The adversarial clause and the facts-evaluation framing have independent and sometimes opposing effects depending on authority context."

### "Qwen-Thinking 36.4% baseline accuracy is a leak"

**Rebuttal**: This is not a leak. Logit-based forced-choice at the first token and temperature-0.6 generation are genuinely different measurement modalities. The model's first-token logit can favor the wrong answer while the full generation (with chain-of-thought reasoning over hundreds of tokens) arrives at the correct answer. This is consistent with Qwen-Thinking's partial resistance in the behavioral data — the reasoning process partially overcomes the logit-level prior.

**In the paper**: State this explicitly. "The 36.4% generation-mode accuracy on logit-defined high-confidence-wrong items reflects the reasoning model's capacity to override its first-token prior through extended chain-of-thought."

### "Ratio vs additive divergence — is Llama attenuation real or just general sensitivity reduction?"

**Rebuttal**: Honest answer is we cannot cleanly distinguish these for Llama because the ratio decomposition has poor mask validity. The additive metric confirms attenuation exists but cannot attribute it to selective channel modulation vs general sensitivity reduction.

**In the paper**: Acknowledge directly. "For Llama, the additive metric confirms attenuation, but we cannot cleanly decompose this into channel-specific effects due to mask validity constraints (35% on high-confidence-wrong). Llama is therefore a weaker datapoint than Qwen for the selectivity claim, though directionally consistent."

### "Why are User and Someone-online tags noisier?"

**Hypothesis**: These tags produce more variable endorsement effects because the perceived authority is ambiguous — "User" and "Someone online" are genuinely uncertain authority levels, leading to bimodal or heavy-tailed per-item effect distributions. Expert and Note are at the extremes (high/low authority) and produce more consistent processing.

**In the paper**: Characterize the per-item effect distributions for each tag. Show that User/Someone-online have higher variance, which explains the metric instability.

---

## What Gets Cut or Deferred

**Cut from NeurIPS scope**:
- LoRA fine-tuning interventions (parameter updates are a different paper)
- "Universal direction" cross-model transfer claims
- Full closed-source replication
- Curiosity-as-truthfulness, conspiracy theories
- Gemma on PIQA (unless mask validity analysis suggests it would help)
- Open-ended generation format testing

**Included as conditional**:
- Activation capping (only if patching succeeds first)
- 30B+ model (only if H100 available)

---

## Key Papers to Cite

| Paper | Role |
|-------|------|
| BASIL (Atwell et al.) | Complementary: they measure deviation magnitude, we show direction depends on prior state + instruction form |
| Assistant Axis (Lu et al., 2026) | One of several comparators for the correction-gating direction. Not the thesis. |
| Causality is Key (Joshi et al.) | Methodological framing: extraction = association, patching = intervention |
| GDM Consistency Training (Irpan & Turner, 2025) | We show where their clean-prompt assumption breaks |
| MATS Constitution Auditing (aryaj et al., March 2026) | Independent discovery of "think-then-ignore" pattern; our traces show the same in a different domain |
| Kumaran et al. (2025) | Choice-supportive bias maps onto prior-consistency |
| Vennemeyer et al. (2025) | Sycophancy taxonomy; where does correction-gating sit? |
| Shapira et al. (2026) | RLHF amplifies sycophancy; we add: it's prior-conditioned and instruction-form-dependent |

---

## TrustNLP Camera-Ready (Separate Track)

If accepted, apply the 6 TODO items from `paper/trustnlp-acl/rebuttal.md`. These are text-only edits and do not block NeurIPS work. The paraphrase gate evidence now strengthens the Discussion section on instruction wording.
