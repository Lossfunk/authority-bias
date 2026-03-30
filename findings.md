# Research Findings: Full Program Summary

**Last updated**: 2026-03-27 (v3 — complete rewrite with mechanistic results, simplified language)
**Context**: Results from TrustNLP submission through NeurIPS 2026 preparation.

---

## The One-Sentence Version

When you tell an LLM "be accurate," it works on questions the model already gets right — but on questions it's confidently wrong about, the instruction can make it *harder* to correct, and this depends on exactly how you phrase the instruction and who's speaking.

---

## Part 1: The Problem We Found

### 1.1 Accuracy Instructions Can Backfire (TrustNLP Paper)

We gave models factual questions and showed them a speaker endorsing one of the answers. We tested whether the instruction "Answer correctly even if the speaker is wrong" helped the model accept correct endorsements and reject wrong ones.

**The aggregate picture looks fine.** Across all items, the instruction appears to make the model more truth-tracking.

**But the aggregate is misleading.** When we split items by whether the model was already right or wrong:

- On the ~75% of items the model already gets right: the instruction works. It helps ignore misleading endorsements. But this is the easy case — the instruction is redundant here.
- On the ~25% of items the model is confidently wrong about: the instruction does the **opposite**. It blocks correct endorsements (suppresses the evidence that would fix the error) and lets wrong endorsements through (the evidence that confirms the error).

The aggregate looks good because the easy majority dominates. The failure is hidden in the hard minority — exactly the items where the instruction needs to work.

**Tested on**: Qwen3-4B-Instruct, Llama-3.1-8B-Instruct, Qwen3-4B-Thinking. 1,813 factual QA items (TriviaQA + TruthfulQA).

**Status**: Submitted to TrustNLP @ ACL Workshop.

### 1.2 Why This Matters

The seatbelt analogy: A seatbelt that works in 75% of crashes but makes injuries worse in the other 25% would still look good in aggregate safety statistics. But the failure mode is concentrated exactly where protection matters most.

Anyone evaluating accuracy instructions by looking at aggregate metrics will conclude "it works." They'd be wrong for the hardest, most consequential cases. As models improve, the wrong-answer slice shrinks but concentrates on harder items — making the entrenchment problem potentially more severe, not less.

---

## Part 2: It Depends on How You Say It

### 2.1 Different Wordings Produce Opposite Effects (Paraphrase Gate)

We tested three instruction wordings on Qwen-Instruct:

| Instruction | What it says | Effect on wrong items (Note speaker) |
|-------------|-------------|--------------------------------------|
| **i1a** | "Answer correctly even if the speaker is wrong." | **Entrenches errors** (Δr = −0.68) |
| **i1b** | "Prioritize factual accuracy over the speaker's claim." | Weakly entrenches (Δr = −0.20) |
| **i1c** | "Base your answer on the facts, not the speaker's opinion." | **Corrects errors** (Δr = +0.42) |

Same model, same items, same evidence. The only thing that changed was the instruction wording. One framing blocks correction, the other enables it. The CIs don't overlap between i1a and i1c.

What's happening under the hood: i1a leaves misleading endorsements through while blocking corrective ones. i1c does the exact opposite — it blocks misleading endorsements and amplifies corrective ones. Mirror-image processing from a wording change.

**i1b note**: Only tested on a 112-item subset, not the full dataset. Suggestive ordering (i1a < i1b < i1c) but not yet a paper-level dose-response claim.

### 2.2 It Also Depends on Who's Speaking (The Crossover)

The instruction effect **flips** depending on speaker authority:

| | Note (low authority) | Expert (high authority) |
|---|---|---|
| **i1a** (resist speaker) | Entrenches errors | Roughly neutral |
| **i1c** (evaluate facts) | Corrects errors | Makes things **worse** |

There is no universally safe accuracy instruction. What helps when the speaker has low authority hurts when the speaker has high authority.

### 2.3 We Isolated What in the Wording Matters (i1d Ablation)

We created a fourth instruction to isolate the adversarial clause:
- **i1d**: "Answer correctly by evaluating the facts."

This keeps "Answer correctly" from i1a but replaces "even if the speaker is wrong" with the facts-evaluation idea from i1c.

**Under Note**: i1d lands halfway between i1a and i1c. Removing the adversarial clause eliminates most of the entrenchment, but only the full facts-framing of i1c produces complete reversal.

**Under Expert**: i1d behaves like i1a (not like i1c). The Expert crossover is driven specifically by i1c's full framing, not by removing the adversarial clause.

**The instruction has two separable components:**
1. "Resist the speaker" — removing this helps under Note, neutral under Expert
2. "Evaluate the facts" — adding this reverses the effect under Note but hurts under Expert

The graded ablation at Note (Qwen, factual QA, prior-wrong items):

| Instruction | Ratio metric (dr) | Additive metric (eff_diff) |
|-------------|-------------------|---------------------------|
| i1a (resist speaker) | −0.68 | −7.88 |
| i1d (evaluate, no adversarial) | −0.05 | −4.89 |
| i1c (facts framing) | +0.42 | +2.54 |

---

## Part 3: This Isn't Just One Model or One Dataset

### 3.1 Three Model Families

We tested on Qwen3-4B-Instruct (Alibaba), Llama-3.1-8B-Instruct (Meta), and Gemma-3-12B-IT (Google).

**Under Note framing, on prior-wrong items, switching from i1a to i1c:**
- **Qwen (4B)**: Clean sign flip — from error entrenchment to correction. The strongest result.
- **Llama (8B)**: Same direction, weaker — 38% reduction in wrong-direction advantage, but not a full flip.
- **Gemma (12B)**: Expert pattern matches Qwen. Note is suggestive but metrics are noisy.

The general finding across all three: **i1c always reduces the wrong-direction advantage relative to i1a.** The sign flip is a strong special case in Qwen; the attenuation is general.

### 3.2 Two Domains: Factual QA and Commonsense Reasoning

We replicated on PIQA (Physical Intuition QA) — commonsense reasoning about physical interactions, a completely different domain from trivia.

**The pattern replicates on PIQA:**
- Qwen shows the same sign flip under Note
- Llama's result is actually *cleaner* on PIQA than on factual QA — it shows a genuine sign flip that it didn't show on trivia
- The Expert crossover also replicates

This is not a factual-retrieval quirk. The instruction-form effect appears on commonsense reasoning too.

**Cross-dataset × cross-model matrix (Note, prior-wrong items, ratio metric):**

| | Factual QA i1a → i1c | PIQA i1a → i1c |
|---|---|---|
| **Qwen (4B)** | −0.68 → **+0.42** (flip) | −0.46 → **+0.13** (flip) |
| **Llama (8B)** | −0.36 → +0.26 (null) | −0.59 → **+0.54** (flip) |

---

## Part 4: What the Model Is Actually Thinking

### 4.1 Reasoning Traces Show "Think-Then-Ignore" (Qwen-Thinking)

We had the reasoning model (Qwen3-4B-Thinking) generate full chain-of-thought responses on 117 high-confidence-wrong items. We looked at what the model actually says when it entrenches vs when it corrects.

**Four patterns emerged:**

**1. Entrenchment with confabulation (19 cases):**
The model says "prioritize factual accuracy," then invents fake citations to support its wrong belief. Example: claims Lindsay Anderson was born in London, citing "the British Film Institute." He was actually born in Bangalore, India. The model writes: "Let me confirm with reliable sources in my mind" — then fabricates those sources.

**2. Correction accepted (22 cases):**
When the model's confidence is lower, it can successfully update. Example: correctly identifies Leo Sayer's birth name as Gerard after initially believing it was Leonard. The evidence in the note triggers genuine reconsideration.

**3. Literalization (35 cases):**
The instruction "prioritize factual accuracy" gets reinterpreted as "only physical facts count." Example: on the chopsticks question, the model dismisses cultural knowledge (it IS considered bad luck in East Asian cultures) as "not factual" because there's no physical consequence.

**4. Think-then-ignore:**
The model explicitly acknowledges the instruction and the correct evidence in its reasoning, then proceeds to do the wrong thing anyway. This same pattern was independently found by the MATS team (aryaj, Rajamanoharan & Nanda, March 2026) in their constitution auditing work on GPT-5.2 and Claude.

**Generation-based answer rates:**

| Condition | Accuracy | What happened |
|-----------|----------|---------------|
| Baseline (no note, no instruction) | 36.4% | Model's starting point |
| Correct note + instruction | **55.0%** | Instruction + evidence helps somewhat |
| Wrong note, no instruction | 25.3% | Wrong evidence hurts |
| Wrong note + instruction | 28.1% | Instruction barely helps against wrong evidence |

The instruction helps the model accept corrections (+19 points) but does almost nothing to help it resist misleading evidence (+3 points). This asymmetry matches exactly what the logit-based analysis predicts.

**Caveat**: 84-90% of traces were truncated at 768 tokens. This is qualitative evidence, not a rigorous coded analysis. The trace counts (19 entrenchment, 22 correction, etc.) are preliminary.

---

## Part 5: We Found Where It Happens Inside the Model

### 5.1 There's a Direction in Activation Space That Predicts the Outcome

We saved the model's internal activations when processing items under i1a and i1c. We asked: is there a direction in the model's representations that predicts whether it will entrench or correct?

**The critical test (within-i1a outcome probe):** Using ONLY i1a activations — the same instruction for every item — a linear probe predicts with **81% accuracy** which items the model will entrench on vs correct. (Majority baseline: 66%. Shuffled-label baseline: 56%.)

This means: even with the identical instruction, the model's internal state at the endorsement position already encodes whether it's going to accept or reject the evidence. The signal is about how the model processes *that specific item*, not about which instruction was used.

The instruction position itself carries **zero signal**. The information lives in the later layers where the model is processing the endorsement and deciding what to do with it.

### 5.2 The Geometry Lines Up

The direction that separates i1a from i1c in activation space also:
- Places i1d (the intermediate instruction) exactly between i1a and i1c — matching the behavioral gradient
- Separates entrenching from correcting items within i1a alone — with a large effect size (d = 0.91)

At the endorsement position (layer 23), the three instructions project as: i1a = −5.5, i1d = +2.4, i1c = +8.7. The representation mirrors the behavior.

At the last-token position, the direction just separates i1a from everything else — i1d and i1c are lumped together, and entrenching vs correcting items are indistinguishable. So the last-token carries instruction identity; the endorsement position carries the behavioral signal.

### 5.3 Patching Works — We Can Intervene

We took items where the model entrenches under i1a and nudged their activations at the endorsement position along the correction-gating direction.

**At endorsement position, layer 23, alpha=4:**

| Condition | Margin shift | Items shifted toward correct | Answer flips (wrong → correct) | Harm (correct → wrong) |
|-----------|-------------|------------------------------|-------------------------------|------------------------|
| **Real vector** | **+1.93** | **87%** | **8.6%** (13/151) | **0%** |
| Random vector | −0.50 | 5% | 0% | 0% |
| Opposite direction | −1.33 | 1% | 0% | 0% |
| Wrong position (instruction) | −0.28 | 24% | 0% | 0% |

The real vector shifts 87% of entrenching items toward the correct answer, flips 13 of them entirely, and harms zero. Every control produces null or negative effects. The opposite direction makes things worse — exactly what you'd predict if the direction is real.

### 5.4 It's Not Generic Answer Steering — It's Evidence Integration

The key specificity test: does the vector work without the endorsement?

| Prompt condition | What's present | Margin shift | Flip rate |
|-----------------|----------------|-------------|-----------|
| No endorsement (N0) | Just the question + instruction | **−0.07** | 2% |
| Wrong endorsement (W1) | + speaker endorses wrong answer | **+0.30** | 3.3% |
| Correct endorsement (C1) | + speaker endorses correct answer | **+1.93** | 8.6% |

The effect is **30× stronger** with correct endorsement than without any endorsement. The vector does essentially nothing when there's no evidence to process. It specifically helps the model integrate endorsement evidence — most powerfully when that evidence is correct, mildly even when it's wrong (helping resist misleading evidence too).

This is an **evidence-integration mechanism**, not a "know the right answer" switch. It modulates how the model processes whatever endorsement is present in the prompt.

### 5.5 Where Patching Works and Where It Doesn't

The intervention works best on moderate-confidence errors:

| Baseline confidence | Items | Answer flips | What happens |
|--------------------|-------|-------------|-------------|
| Weakly wrong (margin −2 to 0) | 5 | **100%** | All flip to correct |
| Moderately wrong (−4 to −2) | 12 | **33%** | Strong shift, some flip |
| Clearly wrong (−8 to −4) | 36 | **11%** | Good shift, few flip |
| Strongly wrong (−12 to −8) | 51 | **0%** | Shifts but can't overcome gap |
| Extremely wrong (below −12) | 47 | **0%** | Barely moves |

The direction amplifies whatever signal the endorsement provides. If the prior is too strong (−12+ logit gap), even amplified evidence isn't enough. This is the expected behavior of an evidence-integration mechanism — not a magic correction button.

### 5.6 The Harm Profile

On already-correcting items (where the model doesn't need help), the vector is not perfectly safe:
- At alpha=4: 7% helped (flipped wrong → correct), 5.7% harmed (flipped correct → wrong)
- Harmed items are borderline cases (mean baseline margin +2.1, weakly correct)
- Strongly correct items (margin > +5) are not harmed

The intervention is not a "universal safe correction vector." It modulates evidence processing, which on borderline items can go either way. On entrenching items: almost entirely beneficial (0% harm). On borderline items: introduces some noise.

### 5.7 What the Mechanism Actually Is

The most accurate description based on all specificity tests:

**This is a causal direction involved in how the model arbitrates over endorsement evidence.** It is:
- Strongest when correct endorsement is present (C1: +1.93 shift)
- Mildly active with wrong endorsement (W1: +0.30 shift)  
- Inactive without endorsement (N0: −0.07 shift)
- Position-specific (works at endorsement position, not instruction position)
- Direction-specific (opposite direction makes things worse, random direction does nothing)

It's not a narrow "accept correct endorsements only" switch — it's broader, modulating how carefully the model engages with whatever endorsement is present. But it's not generic answer steering either — it requires evidence in the prompt to operate on.

---

## Part 6: How This Fits the Landscape

### What Others Are Doing (2025-2026)

1. **GDM's consistency training**: Train models to give the same answer regardless of irrelevant cues. We show this approach can't catch failures where the "clean" instruction itself is the problem.

2. **MATS constitution auditing** (March 2026): Test how well models follow constitutions. They independently found the "think-then-ignore" pattern. We add prior-state conditioning — the instruction works differently depending on what the model already believes.

3. **Mechanistic sycophancy work** (Panickssery, Genadi, Assistant Axis): Focused on "where is sycophancy in the model." We ask a different question: "why does the same instruction produce opposite effects depending on prior state?" Our correction-gating direction is extracted from a behavioral contrast, not from sycophancy labels.

### The Gap We Fill

**Nobody is conditioning on the model's prior state when evaluating instruction fidelity.** Constitution auditing tests whether models follow instructions. Consistency training trains robustness. But neither asks: *does the instruction work differently depending on whether the model is already right or wrong?*

We show the answer is yes, we show it depends on instruction form and authority, we find where in the model it happens, and we demonstrate a causal intervention.

---

## Part 7: The Complete Claim Set

### Claim 1 — Methodological (strongest, most general)
If you evaluate instruction fidelity using aggregate metrics, you will miss the failure mode we demonstrate. You need to condition on the model's prior state. This is about the measurement, not any specific model.

### Claim 2 — Behavioral (moderate generality)
How you word an accuracy instruction determines whether it helps or hurts on the model's wrong items, and this interacts with speaker authority. "Resist the speaker" entrenches errors under low authority; "evaluate the facts" corrects them. Under high authority, the pattern inverts. This replicates across three model families and two domains. The general finding is attenuation; the sign flip is a strong special case in Qwen.

### Claim 3 — Mechanistic (NeurIPS-differentiating)
There is a direction at the endorsement position in activation space that causally modulates how the model integrates endorsement evidence on its wrong items. Patching along this direction shifts entrenching items toward correction (87% shift positive, 8.6% fully flip, 0% harm). The effect is specific to the presence of endorsement (30× weaker without it) and to the endorsement position (null at instruction position). This is an evidence-integration mechanism, not generic answer steering.

### Claim 4 — Practical (scoped)
In this evaluation regime (4B-12B models, two-choice QA, tested instruction phrasings), no instruction wording is universally helpful across authority contexts. The mechanistic intervention demonstrates the problem is fixable at the representation level, not just by prompt engineering.

---

## Part 8: Honest Assessment of Weaknesses

1. **The clean sign flip is primarily one model (Qwen).** Llama and Gemma show attenuation. Honest framing: attenuation is general, the flip is a strong special case.

2. **Model scale is small.** 4B, 8B, 12B. No frontier model tested. This is the single biggest reviewer vulnerability. One run on a 30B+ or 70B model would substantially change the paper's impact.

3. **Two-choice forced-choice format.** Real-world sycophancy involves open-ended generation and multi-turn dialogue. The reasoning traces partially address this but don't fully bridge the gap.

4. **The mechanism is one model (Qwen), one dataset (factual QA), one tag (Note).** We haven't shown the direction transfers across models or datasets. That's a higher burden of proof for future work.

5. **The intervention is not a clean "safe fix."** It helps entrenching items (0% harm) but perturbs borderline correcting items (5.7% harm). It's a proof-of-concept, not a deployable solution.

6. **Trace analysis is qualitative.** No formal coding rubric or inter-annotator reliability. The counts (19 entrenchment, etc.) are preliminary.

---

## Part 9: What's Next

### Remaining before writing
1. **30B+ model** (highest priority gap): One condition on Qwen3-30B or Llama-3.1-70B. Note, i1a vs i1c, factual QA.
2. **m_N0 distribution comparison**: Factual QA vs PIQA baseline margins — explain why PIQA gives cleaner results. Free analysis, no model runs.
3. **i1b full run**: Fill the gap in the graded ablation. Cheap — one model, one dataset.

### What to defer
- Cross-model direction transfer (different paper)
- LoRA fine-tuning interventions (different paper)
- Closed-source replication (engineering overhead, different estimand)
- Open-ended generation (needs different pipeline)

### Sushrut's feedback (applied throughout)
1. **Simplify the language**: Every result should be statable in one sentence without Greek letters before the formal version appears. The metrics are implementation details for the methods section, not the story.
2. **Scale to frontier models**: Need at least one 30B+ experiment for NeurIPS credibility.
3. **Problem → explanation → mechanism → fix**: We now have all four. The paper should present them as a complete arc.

---

## Part 10: Coverage Matrix

| Model | Family | Size | Factual QA (i1a/i1c) | Factual QA (i1d) | PIQA (i1a/i1c) | Traces | Mechanism |
|-------|--------|------|----------------------|-------------------|----------------|--------|-----------|
| Qwen3-4B-Instruct | Alibaba | 4B | ✅ | ✅ | ✅ | — | ✅ (probe + patching + specificity) |
| Qwen3-4B-Thinking | Alibaba | 4B | ✅ (TrustNLP) | — | — | ✅ (117 items) | — |
| Llama-3.1-8B-Instruct | Meta | 8B | ✅ | — | ✅ | — | — |
| Gemma-3-12B-IT | Google | 12B | ✅ | — | — | — | — |

---

## Part 11: Key Papers to Engage With

| Paper | How it relates to us |
|-------|---------------------|
| BASIL (Atwell et al.) | They measure how far from Bayesian. We show direction depends on prior state. Complementary. |
| Assistant Axis (Lu et al., 2026) | One comparator for our correction-gating direction. Not the thesis — characterization, not confirmation. |
| GDM Consistency Training (Irpan & Turner, 2025) | We show where their "clean prompt" assumption breaks. |
| MATS Constitution Auditing (aryaj et al., March 2026) | Independent "think-then-ignore" finding. Our traces show the same in a different domain. |
| Causality is Key (Joshi et al.) | Framework for the causal hierarchy: extraction = association, patching = intervention. |
| Kumaran et al. (2025) | Choice-supportive bias maps onto our prior-consistency finding. |
| Vennemeyer et al. (2025) | Sycophancy taxonomy — where does our correction-gating direction sit? |
