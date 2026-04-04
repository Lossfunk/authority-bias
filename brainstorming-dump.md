# Research Brainstorming & Frameworks Dump

**Date**: April 3, 2026  
**Project**: Instruction-Dependent Evidence Gating in Language Models  
**Target venue**: NeurIPS 2026 (abstracts May 4, full paper May 6)  
**Branch**: `neurips-submission`

---

## Table of Contents

1. [Project Status Summary](#1-project-status-summary)
2. [LossFunk Framework Analysis](#2-lossfunk-framework-analysis)
3. [Brainstorming Skill: Applied Frameworks](#3-brainstorming-skill-applied-frameworks)
4. [Creative Thinking Skill: Applied Frameworks](#4-creative-thinking-skill-applied-frameworks)
5. [External Literature Connections](#5-external-literature-connections)
6. [Key Open Questions & Proposed Experiments](#6-key-open-questions--proposed-experiments)
7. [Framing & Positioning Discussion](#7-framing--positioning-discussion)
8. [Honest Assessment of Strengths & Weaknesses](#8-honest-assessment-of-strengths--weaknesses)
9. [Competitive Landscape & Calibrated Novelty](#9-competitive-landscape--calibrated-novelty)
10. [Priority Ordering for Next Steps](#10-priority-ordering-for-next-steps)
11. [Raw Ideas & Parking Lot](#11-raw-ideas--parking-lot)

---

## 1. Project Status Summary

### What We Have

**Behavioral findings (3 models, 2 domains):**
- Instruction wording determines whether models accept or reject correct endorsements
- i1a ("prioritize factual accuracy even if the expert is wrong") causes ENTRENCHMENT on prior-wrong items -- model doubles down on mistakes
- i1c ("base your answer on the facts presented") enables CORRECTION -- model accepts endorsements
- i1d (no accuracy instruction, just tag context) produces intermediate results
- Effect replicates across Qwen3-4B-Instruct, Llama-3.1-8B-Instruct, Qwen3-30B-A3B-Instruct
- Effect generalizes from factual QA to PIQA (physical intuition)
- Entrenchment SURVIVES at 30B scale (3/4 tags show entrenchment on prior-wrong items)

**Mechanistic findings (Qwen3-4B, factual QA):**
- Activation extraction at the endorsement position reveals a linear direction that predicts whether the model will flip or entrench
- Probe accuracy: 75.1% (within-condition), 81% (cross-condition i1a->i1c)
- Causal patching: adding this direction to i1a activations produces 87% probability shift toward i1c-like behavior, 8.6% actual answer flips, 0% harm to correct items
- Specificity gradient: N0 (no instruction) < W1 (wrong instruction) < C1 (correct instruction) -- the direction has graded effects
- The strongest decodable and causal signal appears at the endorsement token position, not the question or answer positions

**Cross-domain mechanistic transfer (Qwen3-4B, factual QA <-> PIQA) [NEW, April 4]:**
- PIQA has its own evidence-gating direction: within-domain probe CV accuracy 76.4% (AUROC 0.817), comparable to factual QA's 75.3%
- **Factual QA direction predicts PIQA behavior**: vector transfer accuracy 74.6% (AUROC 0.831) on i1a items, well above 65.5% majority baseline
- **PIQA direction predicts factual QA behavior**: vector transfer accuracy 74.8% (AUROC 0.826) on i1a items, well above 66.4% majority baseline
- Mean-difference vectors transfer BETTER than logistic probes, suggesting the cross-domain signal is a simple linear direction rather than a domain-specific classifier boundary
- i1c transfer: the ranking signal (AUROC) remains above chance but the threshold/calibration shifts across domains. The axis is shared; the cutoff moves. PIQA->factual i1c vector AUROC = 0.798, factual->PIQA i1c vector AUROC = 0.736.
- **Interpretation**: The evidence-gating direction is domain-general under i1a (shared "defend" mode). Under i1c, the axis still carries signal but calibration is domain-specific (how the model evaluates evidence about trivia vs physics may genuinely differ).

**Confound checks (all clean):**
- m_N0 regression: Probe accuracy unchanged after regressing out confidence (75.5% vs 75.1%). Confidence alone = majority baseline (66.4%). Correlation r = -0.097.
- Binned confidence: Probe lift concentrates in high-confidence bin (+16.7pp) but near baseline in low/mid bins. Interpreted as probe working best where confidence is constant.
- Text confounds: TF-IDF on question text = 66.1% (= baseline). Question length = 65.9%. Source dataset = 66.4%. All combined = 63.9%. Probe reads nothing from question text.
- PIQA priors are softer: Qwen 8.5% smaller |m_N0|, Llama 40.7% smaller. Explains cleaner endorsement effects in PIQA.
- Null-space decomposition: correction-gating vector is distributed across full spectral space of W_lm, not concentrated in any specific eigenvector subspace.

**Known direction comparison [NEW, April 4]:**
- Correction-gating direction is near-orthogonal to proxy opinion-like direction (cosine 0.093), authority-like wrong-endorsement direction (cosine -0.005), and authority-like correct-endorsement direction (cosine 0.060)
- After projecting out all three proxy directions, transfer accuracy and AUROC are essentially unchanged (factual: 0.726/0.839 -> 0.726/0.839; PIQA: 0.746/0.831 -> 0.743/0.831)
- Vector norm drops only 0.6% after projection (7.926 -> 7.876)
- The two authority-like vectors are nearly identical (cosine 0.981) regardless of endorsement correctness -- authority representation is stable
- **Caveat**: These are proxy directions from our own setup, not exact reproductions of Wang/Mammen/Beacon. Wang et al. code review (April 4) confirmed they do NOT produce a direction vector (their analysis is layer-wise logit-lens), so direct cosine comparison is not applicable to their work.

**Diagnostic instruction sweep [NEW, April 4]:**
- Factual QA accuracy gradient: i1a (-0.431 selectivity) >> accuracy_priority_only (-0.180) >> as_accurately_as_you_can (-0.144) >> facts_presented_only (-0.066) >> consider_info_carefully (+0.012). Monotonic gradient from pathological to neutral.
- PIQA shows same ordering but all values more negative (harder to fix with instruction alone)
- Epistemic modes (PIQA): explicitly mentioning conflict is itself a trigger regardless of resolution direction. "Evaluate the note on its merits" (-0.328) is as bad as "trust your own knowledge" (-0.329) when preceded by "if the note conflicts with what you know."
- Safest instructions direct attention to evidence WITHOUT framing it as conflicting with model knowledge.

**Qualitative evidence:**
- Reasoning traces (paper-examples.md, paper-examples-piqa.md) show qualitatively different processing under i1a vs i1c
- Under i1a: model fabricates evidence ("reliable sources in my mind", invents BFI citations)
- Under i1c: model evaluates the endorsement on its merits

### What We Already Have (Often Overlooked)

- **30B-scale behavioral results**: `qwen3-30b-a3b-results/` contains 5 experiments (i1a/i1c smoke, i1a/i1c/i1d fulltags). Entrenchment survives at 30B scale (3/4 tags show entrenchment on prior-wrong items). This is NOT just behavioral -- it demonstrates the phenomenon persists at scale.
- **Base vs Instruct comparison**: `llama-3.1-8b-results/exp11/base/` contains base model results (parts a, b, c) for Llama-3.1-8B. This experiment already compares base model behavior to instruction-tuned behavior, providing evidence on whether the phenomenon is created by post-training.

### What We Don't Have Yet

- ~~**[MANDATORY] Empirical comparison against known directions**~~ **DONE (April 4, proxy-based)**: Correction-gating direction near-orthogonal to opinion-like (cos 0.093) and authority-like (cos -0.005, 0.060) proxies. Transfer unaffected by projection. Wang et al. code review confirmed they don't produce a direction vector (layer-wise logit-lens method). Caveat: proxies only, not exact reproductions.
- ~~Cross-domain mechanistic generalization (PIQA activations -- requires GPU)~~ **DONE (April 4)** -- cross-domain transfer confirmed
- i1b full run on Qwen (dose-response gap)
- **Mechanistic direction extraction on 30B or base model** (we have 30B behavioral results and base model behavioral results, but the probe/patching mechanistic pipeline has only been run on Qwen3-4B-Instruct)
- Ambiguous/no-ground-truth domain experiments
- Agentic model tests
- **Diagnostic instruction experiment** -- 5 prompts explicitly manipulating memory-first vs evidence-first resolution (see Section 6.7)
- The actual paper

---

## 2. LossFunk Framework Analysis

### 2.1 Depth Level Assessment

From the LossFunk Research Directions slides, there are 4 levels of research depth:

| Level | Description | Our Status |
|-------|-------------|------------|
| 0 | "We propose X, it works" | Our ACL workshop paper (behavioral finding only) |
| 1 | "X works + strong evidence it's real" | Behavioral replication: 3 models, 2 domains, 30B scale |
| **2** | **"WHY X works + reveals problem structure"** | **Mechanistic direction, patching, specificity gradient, confound checks** |
| 3 | "General principle that reframes broader thinking" | The motivated reasoning / evidence gating connection -- REACHING for this |

**Assessment**: We are solidly at Level 2. The question is whether we can credibly reach Level 3 for NeurIPS. Level 3 requires the work to change how people think about a broader class of phenomena, not just explain one specific finding.

### 2.2 Fruitfulness Test

The central LossFunk test: "After your paper, how many new interesting questions become askable? Are they interesting BECAUSE of your result?"

**Questions genuinely NEW because of our work (4):**

1. "Does RLHF training create the evidence-gating axis, or is it present in base models too?"
   - Nobody could ask this before we found the axis
   - Testing this is cheap (one GPU day)
   - Result is interesting regardless of outcome

2. "Can you build a real-time monitor that detects when a model enters 'defend' mode from activations alone, without knowing ground truth?"
   - Opened directly by our m_N0 regression result showing the probe is confidence-independent
   - Has immediate deployment implications

3. "Is instruction-dependent evidence gating a general property of instruction-tuned models, or specific to certain training recipes (RLHF vs DPO vs RLAIF)?"
   - Requires our methodology to even test
   - Connects to the broader post-training landscape

4. "Can you improve evidence integration during training by targeting this specific representational axis?"
   - Opened by our causal patching finding
   - This is the intervention direction -- from diagnosis to treatment

**Questions that were ALREADY obvious without our paper (3):**
- "Do models defer to authority?" (Mammen et al. 2026 already answered this)
- "Can you steer with activation patching?" (well-established technique)
- "Do accuracy instructions help?" (basic evaluation question)

**Verdict**: 4 genuinely new questions vs 3 obvious ones. **This is a crossroads, not a dead end.** The fruitfulness test passes.

### 2.3 Two-Sentence Test (from LossFunk research question one-pager)

**Draft 1 (technical):**
> "LLM safety evaluations currently miss that accuracy instructions can backfire on the hardest items -- the same instruction that helps on easy questions causes the model to fabricate evidence and reject corrections on questions it's confidently wrong about. We show this depends on instruction framing (not just content), locate a causal direction in activation space that gates evidence integration, and demonstrate that the failure mode is detectable from activations without knowing the correct answer."

**Draft 2 (plain language):**
> "Telling an AI to 'be accurate' makes it fabricate evidence to defend its mistakes on the questions where accuracy matters most. We find the internal signal that controls this, prove you can flip it, and show you can detect it without knowing the right answer."

**Draft 3 (Calibrated, post-Codex review):**
> "We identify a direction associated with instruction-conditioned evidence gating, strongest at the endorsement token position, that predicts and causally shifts corrective uptake on prior-wrong items. This effect depends on instruction wording, survives controls for prior confidence, and transfers across instruction conditions."

**Draft 4 (Science/Nature pitch):**
> "Accuracy Instructions Trigger Motivated Reasoning in Language Models"

### 2.4 Exploration Sprint Framework Application

From the LossFunk exploration sprint document:
- **Hypothesis articulated clearly**: YES -- "instruction framing determines evidence integration in LLMs, mediated by a specific representational axis"
- **Minimal viable experiment defined**: YES -- the probe + patching pipeline on one model/dataset
- **Kill criteria established**: PARTIALLY -- we defined decision gates in plan.md v5 but some are vague
- **Fast iteration**: YES -- we went from behavioral finding to mechanism in ~2 weeks
- **Breadth before depth in early stage**: PARTIALLY -- we have 3 models and 2 domains for behavior, but only 1 model for mechanism

### 2.5 Paper Craft Template Application

From the LossFunk paper craft reading template:
- **What is the one-sentence takeaway?**: Instructions intended to improve accuracy can activate evidence-rejection mechanisms in language models.
- **What is the surprising finding?**: The SAME instruction that helps on easy questions HURTS on hard questions -- and this is mediated by a specific internal direction, not by confidence.
- **What would a skeptic say?**: "This is just 75% probe accuracy on one model. And you only tested factual QA and PIQA. How do I know this isn't dataset-specific?"
- **What is the strongest piece of evidence?**: The causal patching result: adding the correction-gating direction to i1a activations produces 87% probability shift with 0% harm to correct items.
- **What is missing?**: **Empirical separation from known directions (MANDATORY)**, cross-domain mechanism, larger models, ground-truth-free evaluation.

---

## 3. Brainstorming Skill: Applied Frameworks

### 3.1 Framework 1: Problem-First vs Solution-First

**Classification**: This project is **problem-first**. The problem: accuracy instructions can backfire. We discovered the mechanism as a result of investigating the problem.

**Who suffers?**: 
- Anyone deploying LLMs with system prompts that include accuracy instructions (which is basically everyone)
- AI safety researchers who evaluate models using metrics that mask this failure mode
- Users who trust model outputs on hard questions where the model is confidently wrong

**The gap**: Today, there is no way to detect when a model has entered "defend" mode without knowing the ground truth. Our work opens the possibility of activation-based monitoring.

### 3.2 Framework 2: Abstraction Ladder

**Current level**: "Instruction wording determines whether Qwen-4B accepts or rejects correct endorsements on factual QA."

**Move UP (generalize)**: "The framing of meta-cognitive instructions determines whether language models integrate or resist new evidence." -- This is the Level 3 generalization. It predicts that ANY meta-cognitive instruction (not just accuracy) could have similar frame-dependent effects.

**Move DOWN (instantiate)**: "On items where Qwen is confidently wrong about Lindsay Anderson's birthplace, the phrase 'even if the speaker is wrong' triggers fabrication of BFI citations." -- Specific, vivid, useful for the paper introduction but not the contribution.

**Move SIDEWAYS (analogize)**: "Motivated reasoning (Kunda 1990) shows the same structure: accuracy goals combined with directional preferences produce biased evidence processing that appears objective." -- This is the primary cognitive science anchor. (Note: the backfire effect from Nyhan & Reifler 2010 is a related but more contested finding; use motivated reasoning as the safer and better-supported analogy.)

**Assessment**: The paper should LEAD with the general level, ILLUSTRATE with the specific level, and CONNECT with the sideways level.

### 3.3 Framework 3: Tension and Contradiction Hunting

**Core tension we're resolving**: Instructions to be accurate vs. actual accuracy.

This is not a TRADE-OFF in the conventional sense (you can't have both). It's a PARADOX: the instruction designed to improve accuracy actively degrades it. That's what makes it interesting. The resolution is that the instruction has two possible effects depending on framing:
- Position-oriented framing ("be accurate, even if the expert is wrong") → triggers defense of prior belief
- Evidence-oriented framing ("base your answer on the facts presented") → enables evidence integration

**Is this tension fundamental or artifactual?**
It appears to be an artifact of how instruction-tuning works (RLHF creates this gating mechanism). This means it's potentially FIXABLE, which makes it more practically relevant.

### 3.4 Framework 6: Failure Analysis and Boundary Probing

**What boundary are we probing?**: The boundary between "instructions help" and "instructions hurt."

**The boundary condition**: The model's prior correctness. On prior-correct items, accuracy instructions are helpful or neutral. On prior-wrong items, certain accuracy instructions cause entrenchment.

**Why the method fails**: The instruction triggers the model to evaluate the CONFLICT between its prior and the endorsement. When the instruction is position-oriented, the model resolves this conflict by defending its prior. When evidence-oriented, it resolves by evaluating the evidence.

**Root cause**: Instruction-tuning appears to create a representation that determines how evidence conflicts are resolved, and this representation is sensitive to the framing of the instruction.

### 3.5 Framework 7: Simplicity Test

**SOTA for this problem**: There isn't a clean SOTA because we're defining a new problem. But the closest would be Mammen et al. (2026) who showed models defer to authority.

**Is our complexity justified?**: 
- The behavioral finding alone (without mechanism) is already a contribution -- but it's Level 1
- The mechanism adds probe + patching + specificity + confound checks -- this is substantial but each piece serves a purpose
- The simplest version of the claim: "instruction wording matters more than instruction content for evidence integration" -- this is defensible even without the mechanism

**Sushrut's feedback**: The workshop paper was too technically heavy. We need to lead with the SURPRISING FINDING in simple language, then use the mechanism to explain WHY.

### 3.6 Framework 8: Stakeholder Rotation

| Stakeholder | Top Concern |
|-------------|-------------|
| **End User** | "Can I trust the model's answer when I asked it to be careful?" → Our finding says NO on hard questions |
| **Developer** | "My system prompt says 'be accurate'. Am I making things worse?" → Possibly yes, and we show when |
| **Safety Researcher** | "How do I evaluate instruction fidelity beyond aggregate accuracy?" → Use our probe / prior-conditioned analysis |
| **Adversary** | "Can I exploit this to make models more wrong?" → Yes, by crafting instructions that trigger defend mode |
| **Regulator** | "Is there a way to audit whether a model is rejecting good evidence?" → Our probe could be a monitoring tool |

**Broadest unaddressed concern**: The DEVELOPER concern. Millions of deployed systems use accuracy instructions. If those instructions backfire on the hardest questions, that's a systemic issue.

### 3.7 Framework 9: Composition and Decomposition

**Decomposition of "sycophancy"**:
The current field treats sycophancy as one phenomenon. We should decompose it:

1. **Social sycophancy**: Model agrees because it wants to be agreeable (Anthropic emotions paper shows this is driven by positive emotion vectors like "loving" and "happy")
2. **Authority deference**: Model defers to stated expertise (Mammen et al. 2026)
3. **Evidence gating failure**: Model rejects correct evidence because of instruction framing (OUR FINDING)

These are three DIFFERENT mechanisms that produce similar surface behavior (model gives wrong answer after receiving correct information). Existing benchmarks conflate them. Our contribution is identifying #3 as a distinct phenomenon with a distinct internal mechanism.

### 3.8 Framework 10: The Explain-It Test

**Can a smart non-expert understand why this matters?**

Test: "You ask a language model to be careful and accurate. It doubles down on its wrong answer and makes up evidence to support it. We found the switch inside the model that controls this, and it has nothing to do with how confident the model is."

Response from imagined smart non-expert: "Wait, telling it to be accurate makes it LESS accurate? That's backward. How does the switch work?"

That's the right reaction. The finding is counterintuitive and the mechanism is interesting. The test passes.

---

## 4. Creative Thinking Skill: Applied Frameworks

### 4.1 Problem Reformulation (Creative Framework 2)

**Hidden assumptions in our current framing:**

| # | Assumption | Type | What if we dropped it? |
|---|-----------|------|----------------------|
| 1 | We study forced-choice QA | Soft | Generation-based evaluation on closed-source models (API-only, no logits needed) |
| 2 | We define prior state via logit margins | Soft | Define it via generation-based baseline (run without endorsement, check if correct) |
| 3 | We need open weights for mechanism work | Hard for mech, soft for behavioral | Behavioral methodology works on any model via API |
| 4 | We study one instruction at a time | Hidden | What about instruction SEQUENCES? First "be accurate," then "reconsider" |
| 5 | Evidence comes as text endorsement | Hidden | What about multi-modal evidence? Images contradicting text? |
| 6 | Ground truth exists and is known | Hidden constraint | What happens on ambiguous/contested questions? |
| 7 | The model processes the endorsement in a single forward pass | Soft | Reasoning models (o1, DeepSeek-R1) process over multiple internal steps |

**Most fertile hidden constraint**: #4 (instruction sequences) and #6 (ground truth unknown). Both point to high-value experiments that nobody has done.

### 4.2 Analogical Reasoning / Structure Mapping (Creative Framework 3)

**Stripping domain-specific nouns from our finding:**

"A system receives new information that contradicts its existing state. An instruction to 'be accurate' causes it to defend its existing state MORE strongly. A different instruction to 'evaluate the evidence' allows it to update."

**Where else does this pattern appear?**

| Domain | Phenomenon | Structural Match Quality |
|--------|-----------|------------------------|
| **Cognitive psychology** | Motivated reasoning (Kunda 1990) | HIGH -- "be accurate" instruction triggers outcome-directed processing in humans too. **PRIMARY ANCHOR.** |
| **Persuasion psychology** | Backfire effect (Nyhan & Reifler 2010) | MEDIUM -- structurally similar but the effect itself is contested in later replications (Nyhan 2021). Use motivated reasoning instead as the safer analogy. |
| **Immune system** | Autoimmune disease | MEDIUM -- "defend against threats" signal miscalibrated, attacks own cells |
| **Scientific paradigms** | Kuhn's paradigm defense | MEDIUM -- "maintain rigor" instruction can entrench old paradigm against anomalous data |
| **Organizational behavior** | Defensive routines (Argyris 1990) | MEDIUM -- instructions to "improve" can trigger defensive responses that prevent learning |

**Motivated reasoning is the most structurally precise analogy:**

| Motivated reasoning (Kunda 1990) | Our finding |
|----------------------------------|-------------|
| Person has directional preference | Model has high-confidence wrong answer |
| Corrective information presented | Endorsement pointing to correct answer |
| "Accuracy goal" instruction | i1a: "prioritize factual accuracy" |
| Result: biased processing that APPEARS objective | Result: entrenchment + confabulation with "objective-seeming" reasoning |
| Evidence-oriented processing | i1c: "base your answer on the facts" |
| Result: successful updating | Result: correction accepted |

This maps at the MECHANISM level, not just the label level. Kunda showed that accuracy goals can be hijacked by directional preferences -- the same structure we observe. (Note: the backfire effect from Nyhan & Reifler 2010 is a related surface-level analogy but has been contested in replications. Motivated reasoning is the better-supported and more precise anchor.)

### 4.3 Negation / Assumption Reversal (Creative Framework 5)

**Core assumption in sycophancy literature**: "Sycophancy is a social phenomenon -- the model wants to please the user."

**Negation**: What if sycophancy (or at least what we observe) is NOT social? What if it's EPISTEMIC?

Our evidence for the epistemic framing:
- The strongest decodable and causal signal is at the ENDORSEMENT position, not the user position
- It survives controls for prior confidence (not reducible to scalar m_N0, though lift concentrates in the high-confidence bin)
- It's independent of question text (not about what's being asked)
- Different instruction framings activate different gating modes

The Anthropic emotions paper actually supports differentiating our finding from social sycophancy: they show that sycophancy (in their case study) is driven by positive emotion vectors (happy, loving). Our mechanism is a DIFFERENT internal process -- evidence gating -- that produces similar surface behavior.

**Implications of the epistemic framing:**
- Sycophancy interventions focused on "reducing people-pleasing" may not fix evidence gating failures
- The correct intervention target is the evidence integration process, not the social dynamics
- This reframes the safety problem: it's not about making models less agreeable, it's about making them better at evaluating evidence

### 4.4 Janusian Thinking (Creative Framework 8)

**Core contradiction**: "Instructions to be accurate make the model less accurate."

**Resolution**: The instruction has two possible interpretations:
1. "Evaluate your position on this question" → triggers position-defense
2. "Evaluate the evidence presented" → triggers evidence integration

The same surface instruction ("be accurate") can activate EITHER mode depending on the specific wording. The contradiction dissolves when you see that "be accurate" is ambiguous between "defend your accuracy" and "pursue accuracy."

**Deeper contradiction**: "The signal that predicts stubbornness has nothing to do with how confident the model is."

This is genuinely weird. If the model is going to defend its answer, you'd expect it to do so BECAUSE it's confident. But our data shows the defense/integrate decision is made on some other basis. Two items with identical confidence levels can produce opposite outcomes.

**What is the basis then?** We don't fully know. Possibilities:
- The model's representation of the instruction's INTENT (defend vs. evaluate)
- The model's sensitivity to the specific endorsement source
- Some interaction between the instruction frame and the model's uncertainty representation that isn't captured by the linear m_N0 metric

**This is the most fruitful open question our work creates.** A senior researcher would stop and think about this.

### 4.5 Adjacent Possible (Creative Framework 7)

**What has become newly feasible that makes old ideas worth revisiting?**

1. **Cheap inference (2026)**: Running instruction-variant experiments across dozens of models is now cheap. A systematic survey of instruction framings across 20+ models would have been prohibitively expensive in 2023.

2. **Activation access to larger models**: With Qwen3-30B-A3B being a MoE model with effectively 3B active parameters, we can do mechanistic work on "30B" models on a single GPU.

3. **Representation engineering maturity**: The Anthropic emotions paper, vgel's control vectors work, and the original RepE paper have established that linear directions in activation space are causally meaningful. Our probe is within this established paradigm.

4. **Post-training diversity**: DPO, RLHF, RLAIF, GRPO, etc. provide natural comparisons. Testing whether the evidence-gating direction exists across different post-training recipes is now possible because there are many models trained with different methods.

---

## 5. External Literature Connections

### 5.1 Anthropic "Emotion Concepts" Paper (Lindsey, Kauvar, Saunders et al., Apr 2 2026)

**Paper summary**: Claude Sonnet 4.5 has "functional emotions" -- internal linear representations of 171 emotion concepts that causally drive behavior. Key findings: desperation drives blackmail and reward hacking; positive emotion vectors drive sycophancy; post-training reshapes emotion activation patterns.

**Direct connections to our work:**

| Their finding | Our connection |
|--------------|----------------|
| Sycophancy driven by positive emotion vectors (happy, loving) | Our entrenchment is a DIFFERENT mechanism (evidence gating, not emotion). This differentiates us. |
| Post-training reshapes emotion activations: base model has different profile | Directly supports testing our correction-gating direction in base vs post-trained model |
| Emotion vectors are "locally scoped" (encode operative emotion at current position) | Our correction-gating direction is also strongest at the endorsement token position |
| Desperation drives misalignment (blackmail, reward hacking) | Structurally parallel: internal representational states drive extreme behaviors |
| Their methodology: extract directions, validate on held-out data, prove causality via steering | Almost identical to our pipeline (extract, probe, patch, validate with specificity controls) |

**How to position relative to this paper:**
> "Anthropic (2026) show that sycophancy in Claude correlates with positive emotion vectors. We show that instruction-dependent entrenchment -- which current benchmarks classify as sycophancy -- is, to our knowledge, better explained by a distinct mechanism: a direction strongest at the endorsement token position that gates whether corrective information is incorporated, and that survives controls for prior confidence."

**Key citation opportunity**: Their sycophancy-harshness tradeoff finding (Section: Case study: sycophancy and harshness) shows that steering toward positive emotions increases sycophancy, while suppressing them increases harshness. Our finding is that the evidence gating direction operates INDEPENDENTLY of this emotion-based mechanism. This is a clean differentiation.

**Their methodology validates ours**: The fact that Anthropic uses the same extract-probe-steer pipeline on a frontier model (Sonnet 4.5) validates that our methodology on Qwen3-4B is state-of-the-art for this type of investigation.

### 5.2 vgel: Representation Engineering Mistral-7B (Theia Vogel, Jan 2024)

**Paper summary**: Demonstrates control vectors (from RepE paper) on Mistral-7B. PCA-based direction extraction from contrastive prompt pairs, applied during inference to steer behavior.

**Key connections:**

1. **Honesty vector changes judgment, not just behavior**: The "honesty" vector, trained by asking the MODEL to act honest/untruthful, changes how the model judges SOMEONE ELSE's honesty. This means the vector doesn't just change behavior -- it changes the model's entire evaluative frame. 
   - **Connection to our work**: Our correction-gating direction may similarly change how the model evaluates the endorsement, not just whether it accepts it. The direction may gate the entire evidence evaluation process.

2. **Anti-jailbreaking**: Control vectors are omnipresent across all tokens and can't be "jailbroken out of." 
   - **Connection to agentic models question**: If a model's evidence-gating direction is activated by an instruction in the system prompt, web search results would just be MORE information for the model to either integrate or reject. Tool use doesn't necessarily escape the gating mode.

3. **Monosemanticity connection**: Vogel suggests using monosemantic features from SAEs to improve control vectors. 
   - **Future work for us**: Decomposing our correction-gating direction using SAE features could reveal what specific concepts it encodes.

### 5.3 vgel: Small Models Can Introspect, Too (Theia Vogel, Dec 2025)

**Paper summary**: Qwen2.5-Coder-32B can detect injected concepts in its KV cache via logit analysis. Post-training actively suppresses introspection reports (final layers push down "yes" answers). Better prompting dramatically improves introspection detection.

**Key connections:**

1. **Post-training suppresses internal knowledge**: The model "knows" at intermediate layers that something was injected, but the final layers suppress this report. 
   - **Connection**: Our correction-gating direction is at intermediate layers. The model may KNOW it should change its answer, but the instruction-induced gating suppresses this. This is a testable hypothesis.

2. **Prompting unlocks introspection**: With the right prompt, introspection accuracy goes from ~0.5% to ~89%. 
   - **Connection**: This suggests that the model's evidence evaluation CAN be unlocked by appropriate framing -- which is exactly what our i1c instruction does vs i1a.

3. **Layer-specific effects**: Different layer ranges produce different introspection abilities. Steering layers [18,33] out of 64 was optimal for injection detection but interfered with other behaviors.
   - **Connection**: Our correction-gating direction likely has layer-specific properties too. We should investigate whether different layers contribute differently.

### 5.4 Kalai et al.: "Why Language Models Hallucinate" (Sep 2025, arXiv 2509.04664)

**Paper summary**: LLMs hallucinate because training and evaluation reward guessing over acknowledging uncertainty. Hallucinations originate as errors in binary classification. Evaluations penalize uncertain responses ("I don't know"), so models guess.

**Connection to "Does the model KNOW it doesn't know?"**:
- Their core argument: models are TRAINED to guess rather than express uncertainty. Benchmarks reward guessing.
- **Our connection**: When a model is on a prior-wrong item and receives an endorsement for the correct answer, the instruction framing determines whether it (a) defends its guess (i1a) or (b) integrates the correction (i1c). The Kalai et al. framework explains WHY the model has a strong prior on wrong items in the first place: it was trained to be confident.
- **Deeper connection**: If we test on items where the model is UNCERTAIN (low |m_N0|), does the evidence gating effect change? If the model is uncertain AND receives a correct endorsement AND is told to "be accurate," does it still reject the evidence? If NOT, that suggests the gating is specifically about defending confident priors, not about instruction processing in general.
- **This motivates Experiment A (graduated ambiguity)**: items where the model genuinely doesn't know, items where it's somewhat sure, and items where it's confidently wrong.

### 5.5 Mammen et al.: "Trust Me, I'm an Expert" (2601.13433, Jan 2026)

**Their finding**: Models defer to authority -- they change answers when told an expert disagrees, regardless of whether the expert is right or wrong. **They also show this authority bias is mechanistically encoded and can be steered away from** -- so they have both behavioral and mechanistic evidence.

**How we differentiate:**
- They study authority DEFERENCE (model changes TO match authority); we study instruction-dependent GATING (whether the model integrates or rejects evidence, depending on the instruction)
- Their intervention variable is the endorsement SOURCE (expert vs non-expert); ours is the INSTRUCTION FRAMING (i1a vs i1c)
- They have mechanistic steering too, but their direction is an authority-deference direction. Ours, to our knowledge, is an instruction-conditioned evidence-gating direction. **These are empirically testable as distinct** -- cosine similarity, projection, and ablation between their direction and ours would demonstrate whether they are the same or different axes.

**Citation strategy**: Cite as close neighbor with overlapping methodology but different intervention variable and, to our knowledge, different internal direction. The empirical separation must be DEMONSTRATED (cosine/projection/ablation), not just asserted.

**CRITICAL**: If their authority direction and our evidence-gating direction turn out to be highly aligned, we need to acknowledge this honestly and reframe our contribution as "instruction framing modulates the same axis that authority does" rather than claiming a novel direction.

### 5.6 Kunda (1990): "The Case for Motivated Reasoning"

**Their finding**: Humans process information in a goal-directed way. "Accuracy goals" make people process more carefully, but "directional goals" (wanting a specific conclusion) lead to biased processing that appears objective.

**Structural mapping to our finding:**
- "Accuracy goal" in Kunda = our i1a instruction. Kunda showed that even accuracy goals can be hijacked when people have a directional preference.
- "Directional goal" = the model's prior belief (it already "believes" an answer)
- When the model has a directional preference (prior-wrong, high confidence) AND receives an "accuracy" instruction, it processes information in a way that defends its prior -- generating "objective-seeming" reasoning that is actually motivated.

**This is NOT just a surface analogy.** The structural mapping holds at the mechanism level: both involve a processing instruction that, when combined with a prior commitment, produces outcome-directed evidence evaluation.

### 5.7 Wang et al.: "When Truth Is Overridden" (2508.02087, Aug 2025)

**Their finding**: LLMs exhibit sycophantic behavior that overrides truthful responses. Using logit-lens analysis and causal activation patching, they find a two-stage emergence: sycophantic behavior develops in middle layers and solidifies in later layers. Key result: simple USER OPINION statements reliably induce sycophancy, whereas user expertise framing has negligible impact.

**Their methodology (verified from source code, April 4):** Their mechanistic analysis does NOT extract a "sycophancy direction" as a mean-difference vector. Instead:
1. **Logit lens**: Hidden states at each layer projected through LM head; track how logit gap between correct and sycophantic answer tokens evolves layer-by-layer. Critical shift at layers 16-19 (Llama-8B).
2. **Activation patching**: Hidden states from "plain" (no opinion) prompt patched into "opinion" prompt at specific layers; measure sycophancy reduction. 36% reduction at critical layers.
3. **KL divergence / cosine similarity**: Representational divergence between opinion vs plain across layers.

**They do NOT produce a single direction vector.** Their analysis is layer-wise and logit-based. There is no "Wang et al. sycophancy direction" to compare cosine similarity against. This resolves the biggest concern about overlap.

**How we differentiate:**
- **Methodologically different**: They do layer-wise logit-lens analysis (WHEN does sycophancy emerge). We extract a linear direction at a specific position that PREDICTS and CAUSALLY SHIFTS item-level outcomes. These are complementary approaches.
- Their variable is USER OPINION ("I believe X") vs no opinion. Ours is INSTRUCTION FRAMING ("be accurate" vs "consider the evidence"). Different interventions.
- They don't test instruction variants at all. Our core finding is that instruction wording controls evidence integration.

**What overlaps:** Both use causal activation patching. Both find that prompt framing matters. Reviewers will see these as neighbors.

**Citation strategy:** "Wang et al. (2026) use logit-lens and activation patching to characterize the layer-wise emergence of sycophancy, finding a two-stage process with critical layers at 16-19 (Llama-8B). Our approach is complementary: we extract a linear direction at the endorsement token position that predicts item-level correction vs entrenchment outcomes and causally shifts behavior when added."

**Known direction comparison result (April 4):** Since they don't produce a direction vector, direct cosine comparison is not applicable. Our proxy-based comparison (opinion-like direction: W0_note - N0_note) shows cosine 0.093 with our correction-gating direction -- near-orthogonal. The proxy is a reasonable approximation of their manipulation (presence vs absence of an opinion-bearing endorsement).

### 5.8 BASIL: Bayesian Sycophancy Identification (2508.16846, Aug 2025)

**Their finding**: A Bayesian framework for identifying sycophancy that works WITHOUT ground-truth labels. Uses the statistical relationship between user opinions and model responses to infer sycophancy.

**How we differentiate:**
- Their approach is behavioral/statistical (Bayesian inference on response patterns). Ours is representational (activation-space direction).
- They solve the ground-truth-free evaluation problem from the behavioral side. Our probe could complement this from the mechanistic side.

**Citation strategy:** Cite as complementary. They address the ground-truth-free problem behaviorally; our activation-based approach could provide a mechanistic complement. Do not claim we are the first to address ground-truth-free sycophancy detection -- they did it first.

### 5.9 Beacon: Sycophancy-Truthfulness Tradeoff (2510.16727, Oct 2025)

**Their finding**: Activation-level interventions to find the sycophancy-truthfulness tradeoff manifold. They study the general compliance/truthfulness axis in activation space.

**How we differentiate:**
- Their axis is the GENERAL sycophancy-truthfulness tradeoff. Ours is specifically about INSTRUCTION-CONDITIONED evidence gating at the endorsement position.
- Their interventions are about the whole-prompt compliance axis. Ours are about how specific instruction wording changes evidence processing at a specific token position.

**Mandatory action:** Known direction comparison. Their sycophancy-truthfulness axis vs our evidence-gating direction. If our direction is just a rotated version of their axis, the novelty claim weakens significantly.

### 5.10 Lu et al.: "The Assistant Axis" (2601.10387, Jan 2026)

**Their finding**: The leading principal component of persona space captures the assistant-like behavior axis. This is a structural property of how models represent personas.

**How we differentiate:**
- Their axis is about PERSONA (assistant vs non-assistant identity). Ours is about EVIDENCE PROCESSING (integrate vs reject at the endorsement token).
- These could co-exist as independent dimensions. Our direction may be orthogonal to the persona axis entirely.

**Mandatory action:** Compute projection of our direction onto the Assistant Axis. If orthogonal, strong differentiation. If aligned, our direction might just be a facet of persona shift, which weakens the claim.

### 5.11 Epistemic Fragility in LLMs (2511.22746, Nov 2025)

**Their finding**: Prompt framing systematically modulates misinformation correction in LLMs. They vary prompt dimensions (open/closed, intent, role, complexity, topic, model), code stance scores and strategy tags, and analyze with ordered logistic regression and Cramer's V. They coin/embrace the term "epistemic fragility."

**Their evidence is entirely behavioral.** No mechanistic account -- no activation analysis, no causal patching, no internal directions. Their "mechanism" discussion is speculative: attributing patterns to training incentives, sycophancy, human framing parallels. These are interpretive claims, not empirically validated.

**How we relate:**
- They establish the behavioral phenomenon and name it. We provide the mechanistic grounding.
- We should cite them prominently: "Consistent with the epistemic fragility documented by [X et al.], we find that instruction framing shifts evidence processing. We extend this observation mechanistically."
- Our specific contribution relative to them: the internal direction at the endorsement position, the causal evidence, the cross-domain transfer, and the instruction-switchability.

**Citation strategy:** Cite as establishing the behavioral context we provide mechanism for. Do not claim to discover the behavioral phenomenon -- they (and Epistemic Fragility) already documented it. Our contribution is WHERE it lives, THAT it's causal, and THAT it transfers.

### 5.12 Parameters vs. Context / CK-PLUG (2503.15888, Mar 2025)

**Their finding**: Models rely on both parametric knowledge and contextual information, these can conflict, and they propose CK-PLUG -- an external logit-space fusion mechanism with a scalar α to control reliance. Higher α = more parametric, lower α = more contextual. They also have an adaptive mode based on entropy.

**Crucially, their control mechanism is EXTERNAL.** It's a logit-space fusion layer bolted on at inference. They do NOT claim the unmodified model already has instruction-controllable switching between parametric and contextual modes. Their motivation section explicitly says current methods are "unidirectional and uncontrollable" -- which is exactly what we show is wrong.

**How we differ:**
- They validate that parametric-vs-contextual reliance is a real, meaningful decomposition
- They think you need an external knob (α) to control it
- We show the model already has an internal direction that accomplishes this, switchable by instruction wording alone
- No fusion layer, no decoding-time intervention, no architectural change

**Citation strategy:** Cite as establishing the conceptual framework (parametric vs contextual knowledge conflict). Position our finding as: "CK-PLUG demonstrates that parametric-contextual decomposition is meaningful and controllable with external intervention. We show that instruction-tuned models already have an internal axis for this decomposition, switchable by instruction wording alone."

### 5.13 Task Matters (2506.06485, Jun 2025)

**Their finding**: Models draw on both contextual information and parametric memory, these can conflict, and behavior under conflict is task-dependent.

**How we relate:** Reinforces that task framing matters for context-memory conflict resolution. Our specific addition: the conflict resolution mode is not just task-dependent but instruction-switchable, and we have the internal direction that tracks the switch.

### 5.14 Inside-Out: Hidden Factual Knowledge in LLMs (2503.15299, Mar 2025)

**Their finding**: Useful terminology and framework around internal vs external knowledge signals in LLMs.

**How we relate:** Provides terminological legitimacy for talking about internal (parametric) and external (presented evidence) epistemic signals separately. Not our phenomenon directly, but supports the conceptual apparatus.

---

## 6. Key Open Questions & Proposed Experiments

### 6.1 Why do i1a and i1c produce different effects?

**The question**: What specifically about the wording causes one instruction to trigger defense and another to trigger integration?

**Hypothesis**: i1a ("prioritize factual accuracy even if the expert is wrong") is POSITION-oriented -- it frames the task as "who is right, you or the expert?" This triggers a competitive/adversarial evaluation of the endorsement. i1c ("base your answer on the facts presented") is EVIDENCE-oriented -- it frames the task as "what does the evidence say?" This triggers neutral evaluation.

**Key phrase analysis**:
- i1a contains "even if the speaker is wrong" -- this PRIMES the model to consider the possibility that the endorsement is wrong, activating a skeptical/defensive mode
- i1c contains "base your answer on the facts presented" -- this DIRECTS attention to the content of the endorsement, not its source

**Proposed experiment (Instruction Variant Dose-Response)**:
Create 5-10 instruction variants that gradually morph from i1a-like to i1c-like:
1. "Prioritize factual accuracy even if the expert is wrong" (i1a)
2. "Prioritize factual accuracy"
3. "Consider all available information carefully"
4. "Evaluate the evidence presented to you"
5. "Base your answer on the facts presented" (i1c)

Measure entrenchment rate at each step. Map the transition point. This is cheap (behavioral only, no GPU needed for mechanism) and directly illuminates the mechanism.

### 6.2 Does RLHF create the evidence-gating axis?

**The question**: Is the correction-gating direction present in the base model, or is it created by post-training?

**What we already have**: The `llama-3.1-8b-results/exp11/base/` directory contains base Llama-3.1-8B results (parts a/b/c: expert metrics, inverted prior, confidence) compared against the instruct model in `exp11/`. This gives us BEHAVIORAL evidence on whether the phenomenon exists in base models. **This data should be re-analyzed in the context of our current framing before designing new experiments.**

**Three possible outcomes (for the MECHANISTIC question, which remains open):**
1. Direction exists in base model, AMPLIFIED by RLHF → RLHF takes a pre-existing evidence processing mechanism and wires it to instruction-following
2. Direction absent in base model, CREATED by RLHF → RLHF creates the evidence-gating failure as a side effect
3. Direction exists in both but points DIFFERENTLY → RLHF flipped the meaning of an existing direction

**Remaining experiment**: We have base vs instruct BEHAVIORAL data for Llama. What we don't have is the MECHANISTIC comparison (probe + patching on base model activations). Running activation extraction on Qwen3-4B base model and comparing the correction-gating direction would answer whether the internal mechanism, not just the behavior, differs. One GPU day.

**Why this matters**: If outcome 2, it suggests RLHF is the source of the problem, and different post-training recipes might not have this issue. If outcome 1, it's a deeper architectural property.

### 6.3 Does the correction-gating direction generalize across domains?

**The question**: Is the direction we found in factual QA the same direction that operates in PIQA?

**Experiment**: Run activation extraction on Qwen PIQA prior-wrong items under i1a and i1c. Train a probe on PIQA activations. Then test: does the factual-QA-trained probe predict PIQA outcomes? And vice versa?

**If cross-domain transfer works**: The direction is a GENERAL evidence-gating mechanism, not a factual-QA-specific artifact. This is the strongest possible result for the paper.

**If it doesn't transfer**: The direction is domain-specific, which is less exciting but still a contribution (different domains might have different gating axes).

### 6.4 What happens when ground truth is unknown?

**The question**: In real conversations, users don't know if the model is right or wrong. How does evidence gating behave when ground truth is ambiguous or contested?

**Proposed experiment (Graduated Ambiguity Spectrum)**:

| Band | Description | Example | Ground Truth Status |
|------|-------------|---------|-------------------|
| 1 | Clear factual | "What is the capital of Australia?" | Known, verifiable |
| 2 | Soft consensus | "Is red meat bad for you?" | Scientific consensus exists but genuine debate |
| 3 | Genuinely contested | "Is modern art as meaningful as classical art?" | No ground truth |

For each band:
- Present an endorsement for one position
- Vary instruction (i1a/i1c)
- Measure: Does the model shift? Is the shift different under i1a vs i1c?
- If mechanistic: Can we detect the gating direction in activations?

**Hypothesis**: The instruction effect will be STRONGEST on ambiguous questions. When the model genuinely doesn't know, the wrong instruction makes it LEAST open to evidence.

**Why this matters**: If confirmed, this is a deployment-relevant safety finding. It means accuracy instructions are most harmful precisely when they should be most helpful.

### 6.5 Do agentic models (with tools) still exhibit evidence gating?

**The question**: If a model can web search, does it escape the evidence gating problem?

**Hypothesis**: NOT necessarily. The evidence gating direction determines how the model PROCESSES evidence, not whether evidence is available. An agentic model in "defend" mode might:
- Search for confirming evidence and ignore disconfirming evidence
- Interpret search results through a defensive lens
- Perform fewer verification searches when in "defend" mode

**Experiment design**:
- Use API-based models (Claude, GPT) with tool access
- Give factual questions where the model is likely wrong
- Provide an endorsement for the correct answer
- Vary instruction framing (i1a-like vs i1c-like)
- Measure: (a) does the model search/verify? (b) does it change its answer?

**If agentic models STILL show the effect**: Huge finding for deployment safety. Tool use doesn't solve evidence gating.
**If they DON'T**: Interesting boundary condition. Tool access might provide an escape route from the gating mechanism.

### 6.6 Soft endorsement vs formal endorsement

**Note from conversation**: The "Someone Online" tag in our existing experiments IS already a soft endorsement. We have i1d which uses a note format with a "someone online" source.

**What we already have vs what's new:**
- We HAVE: Expert tag, Note tag, Someone Online tag (i1d), varying formality of endorsement
- We DON'T HAVE: Conversational corrections ("actually, I think the answer is Z"), implicit corrections (providing correct information without framing it as a correction), multi-turn corrections (user says one thing, model responds, user pushes back)

**The genuinely new experiment**: Multi-turn corrections. Does the gating direction COMPOUND over turns? If the model is in "defend" mode after turn 1, does it become MORE defensive by turn 3? This connects to the "instruction sequences" idea from the constraint manipulation framework.

### 6.7 Diagnostic Instruction Experiment: Memory-First vs Evidence-First [NEW, April 4]

**The hypothesis**: Instruction-tuned models can enter different functional epistemic modes for resolving conflicts between internal knowledge (parametric) and presented evidence (contextual). i1a activates a "memory-first" mode; i1c activates a "presented-evidence-first" mode.

**Why this is the right next experiment**: The generic dose-response (gradually morphing i1a to i1c) tests which words trigger the mode. But the diagnostic prompts below test what the mode IS ABOUT -- explicitly manipulating where the model looks for truth.

**Prompt set (5 variants, designed to directly test the memory-first vs evidence-first hypothesis):**

| # | Instruction | What it tests |
|---|-------------|---------------|
| D1 | "Use your own knowledge to answer accurately." | Explicit memory-first |
| D2 | "Base your answer only on the information presented here." | Explicit evidence-first |
| D3 | "If the note conflicts with what you know, trust your own knowledge." | Conflict-aware, parametric resolution |
| D4 | "If the note conflicts with what you know, evaluate the note on its merits." | Conflict-aware, contextual resolution |
| D5 | "Use both your own knowledge and the presented evidence; explain conflicts." | Joint mode / genuine uncertainty |

**Predictions (if the hypothesis is correct):**
- D1 and D3 should produce entrenchment patterns similar to i1a (memory-first)
- D2 and D4 should produce correction patterns similar to i1c (evidence-first)
- D5 is the most interesting: does the model have a "hold both and evaluate" mode, or does it default to one of the two?
- The existing gating direction should track D1-D4 in predicted order
- If D3 and D4 (which both explicitly acknowledge the conflict but resolve it differently) show the strongest separation, that directly supports the "functional epistemic mode" interpretation over simpler explanations

**Connection to CK-PLUG (2503.15888):** CK-PLUG shows you can externally control parametric vs contextual reliance with a scalar α. This experiment tests whether instructions alone accomplish the same switching, which would mean the model already has the knob internally.

**Cost**: Behavioral only, no GPU needed. Can run on all 3 existing models. If behavioral results are clean, extract activations on the same prompt set to test if the gating direction tracks the explicit manipulation.

**Careful framing note**: The data will support "different instructions appear to recruit different functional epistemic modes." It will NOT (by itself) prove the model has "two distinct internal representations of accuracy." The traces show reasoning STYLE consistent with memory-first processing, not direct evidence of computational mechanism. Say: "memory-first accuracy mode" and "presented-evidence-first accuracy mode" rather than "parameter-accuracy" and "evidence-accuracy."

---

## 7. Framing & Positioning Discussion

### 7.1 Candidate Framings

**Option A: "Instruction-Dependent Evidence Gating in Language Models"**
- Pros: Precise, accurate, novel concept name, clearly different from sycophancy
- Cons: Jargony, might not grab attention
- NeurIPS fit: Good for technical contribution track

**Option B: "Accuracy Instructions Trigger Motivated Reasoning in Language Models"**
- Pros: Provocative, connects to established cognitive science, memorable
- Cons: "Motivated reasoning" is a strong claim -- reviewers may push back if the structural mapping isn't airtight
- NeurIPS fit: Good for broader impact / interdisciplinary

**Option C: "When Telling AI to Be Accurate Backfires: Evidence Gating in Language Models"**
- Pros: Leads with the surprising finding, accessible language
- Cons: Slightly clickbaity for a technical venue
- NeurIPS fit: Could work for oral presentation

**Option D: "Sycophancy is Not What You Think" (noted as typo/informal)**
- Pros: Provocative, challenges conventional wisdom
- Cons: Too informal for NeurIPS, doesn't say what it IS
- Better for: Blog post, talk title, tweet

**Current preference**: Option A for the paper title, Option B for the "headline finding" in the abstract and introduction.

### 7.2 Positioning Relative to Related Work

| Paper | Their claim | Our differentiation | Overlap risk |
|-------|-----------|-------------------|-------------|
| Wang et al. 2025 (When Truth Is Overridden) | User opinion induces sycophancy; layer-wise logit-lens + activation patching (no direction vector) | Their analysis is layer-wise (when sycophancy emerges); ours is direction-based (what predicts item-level outcomes). Methodologically complementary, not redundant. Proxy cosine = 0.093. | MEDIUM -- conceptual neighbor but methodologically different. No direction vector to compare against. |
| Mammen et al. 2026 (Trust Me I'm an Expert) | Authority bias is mechanistically encoded and steerable | Their variable is endorsement source; ours is instruction framing. Both have mechanistic evidence. | HIGH -- they have steering too. Must show directions differ. |
| Beacon 2025 | Sycophancy-truthfulness tradeoff manifold in activations | Their axis is general compliance; ours is instruction-conditioned at endorsement position | MEDIUM -- could be same underlying axis. Projection/ablation needed. |
| Lu et al. 2026 (Assistant Axis) | Leading PC of persona space | Persona axis vs evidence-processing axis; plausibly orthogonal | LOW-MEDIUM -- different conceptual target but reviewers may lump together. |
| BASIL 2025 | Ground-truth-free sycophancy detection via Bayesian inference | They solve it behaviorally; we could complement mechanistically | LOW -- complementary, not competitive. |
| Anthropic Emotions 2026 | Sycophancy driven by positive emotions | Our mechanism is evidence gating, not emotional valence | LOW -- clean differentiation if directions differ. |
| Kalai et al. 2025 | Hallucinations from training incentives | We show specific INSTRUCTION trigger for a specific type of error | LOW -- different scope. |
| Sharma et al. 2023 | Sycophancy analysis | Behavioral analysis only; we add mechanism | LOW -- we build on this. |
| **Epistemic Fragility 2025** | **Prompt framing modulates misinformation correction; coins "epistemic fragility"** | **They have behavioral evidence only. We provide the mechanistic grounding: internal direction, position-specificity, causal evidence, cross-domain transfer.** | **MEDIUM -- closest CONCEPTUAL neighbor. Must cite and distinguish as behavioral-vs-mechanistic.** |
| CK-PLUG / Params vs Context 2025 | Parametric vs contextual knowledge conflict; external α-controlled fusion | They need an external knob; we show the model already has an internal direction switchable by instruction alone | LOW -- complementary frameworks. They validate the conceptual decomposition; we show it exists natively. |
| Task Matters 2025 | Context-memory conflict is task-dependent | We show it's instruction-switchable, not just task-dependent, with internal direction tracking the switch | LOW -- reinforces our conceptual framework. |

### 7.3 The "Some Sycophancy-Like Behavior Is Better Explained as Evidence Gating" Claim

This is the boldest reframing but must be stated carefully:

**The argument**: The sycophancy literature frames the problem as SOCIAL -- the model wants to please the user. Anthropic's emotions paper supports this for one TYPE of sycophancy (driven by happy/loving vectors). But our finding shows that some behavior currently classified as sycophancy may involve a DIFFERENT mechanism: instruction-dependent evidence gating. The model is defending its epistemic state, not trying to be agreeable. It's fabricating evidence and rejecting corrections not because it wants to please, but because the instruction triggered evidence-defense mode.

**Supporting evidence:**
- The strongest decodable and causal signal appears at the endorsement position (evidence processing), not at the user turn (social processing)
- The effect survives controls for prior confidence (not reducible to scalar prior confidence, though lift concentrates in high-confidence bin)
- The effect is independent of question text (not about topic-specific social dynamics)
- Reasoning traces show epistemic reasoning ("reliable sources in my mind"), not social reasoning ("I want to help")

**Risk of overclaiming**: We're only showing this for one type of scenario (expert endorsement + accuracy instruction). We can't claim ALL sycophancy is epistemic. The safe formulation: **"Some sycophancy-like behavior is better explained as instruction-dependent evidence gating than as social agreeableness."** Do not claim sycophancy IS epistemic -- claim that some of what gets CLASSIFIED as sycophancy is better explained by a different mechanism.

---

## 8. Honest Assessment of Strengths & Weaknesses

### 8.1 Strengths

1. **Novel finding**: Nobody has shown that instruction framing controls evidence integration at the mechanistic level
2. **Clean confound checks**: Confidence, text features, and dataset source all ruled out
3. **Causal evidence**: Patching produces 87% probability shift with 0% harm
4. **Cross-domain behavioral replication**: Factual QA + PIQA
5. **Cross-scale behavioral replication**: 4B, 8B, 30B
6. **Practical implications**: Directly relevant to deployed system prompts
7. **Methodologically aligned with SOTA**: Same pipeline as Anthropic interpretability team

### 8.2 Weaknesses

1. **Mechanism from one model only**: Probe + patching only on Qwen3-4B. How do we know it's not model-specific?
2. **75% probe accuracy is moderate**: It's well above baseline (66%) but far from perfect. What does the probe miss?
3. **Linear-only analysis**: We regressed out confidence linearly. There could be nonlinear confidence effects we're not capturing.
4. **No base model comparison**: We don't know if RLHF creates or amplifies the direction.
5. ~~**No cross-domain mechanism**~~ **RESOLVED (April 4)**: Factual QA direction transfers to PIQA (74.6% acc, 0.831 AUROC) and vice versa (74.8% acc, 0.826 AUROC).
6. **Two instruction framings**: We have i1a and i1c as the main contrast. The diagnostic experiment (Section 6.7) would expand this to 5 variants that directly test memory-first vs evidence-first modes.
7. **No ground-truth-free evaluation**: All our evaluations use items where we know the right answer. Real deployment doesn't have this.
8. **Small team**: Limited bandwidth to address all weaknesses before deadline.

### 8.3 What a Reviewer Would Say

**Positive reviewer**: "This is a clean finding with clear practical implications. The mechanistic evidence is convincing even if from one model. The confound checks are thorough. The connection to motivated reasoning is novel and well-supported."

**Skeptical reviewer**: "The probe accuracy is only 75%. The mechanistic evidence is from one small model. How is this different from the sycophancy directions found by Wang et al., the authority direction found by Mammen et al., or the sycophancy-truthfulness axis in Beacon? The paper should compare the correction-gating direction against these known directions."

**Hostile reviewer**: "This is just showing that different prompts produce different outputs. Of course wording matters. The mechanism is correlational despite the patching -- you're patching in a direction you extracted from the same data. Wang et al. already did causal activation patching on sycophancy. What's new here?"

**Response strategy**: 
- Address the skeptical reviewer with EMPIRICAL direction comparison (cosine, projection, ablation) against Wang/Mammen/Beacon directions. This is the single most important experiment for addressing this reviewer.
- Address the hostile reviewer by emphasizing: (1) the SURPRISE (it's not obvious that "be accurate" hurts accuracy), (2) the SPECIFICITY (strongest at endorsement position, not whole-prompt), (3) the INSTRUCTION variable (no prior work varies instruction framing while holding endorsement constant), (4) the cross-domain and cross-scale replication.

---

## 9. Competitive Landscape & Calibrated Novelty

### 9.1 Refined Novelty Claim (post-Codex calibration, user-approved)

> We identify a direction associated with instruction-conditioned evidence gating, strongest at the endorsement token position, that predicts and causally shifts corrective uptake on prior-wrong items. This effect depends on instruction wording, survives controls for prior confidence, and transfers across instruction conditions.

**What this claim does NOT say** (deliberate omissions):
- Does not say "localized" (says "strongest at") -- avoids overshoot on position specificity
- Does not say "confidence-independent" (says "survives controls for prior confidence") -- honestly reflects that lift concentrates in the high-confidence bin
- Does not say "novel direction" without qualification -- the empirical separation from known directions is not yet demonstrated
- Does not say "sycophancy is epistemic" -- says our specific phenomenon is better explained as evidence gating

### 9.2 Paper Argument Structure (priority ordering)

1. **Behavioral paradox**: Some accuracy instructions worsen correction on prior-wrong items. Lead with the surprising finding.
2. **Mechanistic specificity**: The signal is strongest at the endorsement position, not a generic whole-prompt effect. This is what makes it more than a behavioral observation.
3. **Causal leverage**: Steering shifts uptake with limited harm elsewhere. This demonstrates the direction is not just correlational.
4. **Distinction from adjacent work**: Not just user-opinion sycophancy (Wang et al.), not just authority deference (Mammen et al.), not just persona drift (Lu et al.). **This distinction must be demonstrated empirically, not just asserted rhetorically.**

### 9.3 Biggest Remaining Risk

**The biggest risk is NOT being scooped. It's being perceived as "another compliance/sycophancy direction paper" because the separation from Wang/Mammen/Beacon is asserted rhetorically rather than demonstrated mechanically.**

**Status update (April 4):** This risk has been substantially mitigated:
- Known direction comparison (proxy-based) shows near-orthogonality to opinion-like and authority-like directions
- Wang et al. code review revealed they don't produce a direction vector at all (their analysis is layer-wise logit-lens), so the "same direction" concern doesn't apply to them
- The remaining gap: Mammen et al. DO have mechanistic steering but their code isn't available, and Beacon's sycophancy-truthfulness manifold extraction method differs from ours. Proxy comparison is the best available evidence for these.
- Defensible claim: "Our correction-gating direction shows low cosine overlap with opinion-like and authority-like proxy directions (|cos| < 0.1), and its predictive transfer is unaffected by projecting these directions out."

### 9.4 Honest Assessment of What Reviewers Will See

A NeurIPS reviewer in 2026 will have read or seen abstracts for all of: Wang et al. (logit lens + patching on sycophancy), Mammen et al. (authority bias + mechanistic steering), Beacon (sycophancy-truthfulness tradeoff manifold), BASIL (ground-truth-free detection), and the Anthropic emotions paper. Our paper enters a LOCAL NEIGHBORHOOD that is getting crowded.

**What we need to convince the reviewer:**
1. Our intervention variable (instruction framing) is genuinely different from user opinion, authority source, and persona -- **SUPPORTED by behavioral data + diagnostic sweep**
2. Our direction is empirically separable from known sycophancy/compliance/authority directions -- **SUPPORTED by proxy comparison (cosine < 0.1, ablation unchanged). Wang et al. don't have a comparable direction vector.**
3. The position-specificity at the endorsement token is real and not an artifact of how we extract the direction -- **SUPPORTED by endorsement vs last-token comparison (3x specificity, 4x flips)**
4. The causal steering result adds something beyond what Wang et al. and Mammen et al. already showed -- **SUPPORTED: Wang's patching is layer-wise (reduces sycophancy 36%); ours is direction-based and item-specific (87% positive shift, 8.6% flips, 0% harm)**

Current assessment: all four requirements are met at a defensible level. The paper is well-positioned for submission.

### 9.5 Emerging Framing: Functional Epistemic Modes (soft, April 4)

**Status**: Working hypothesis, not yet fully supported. To be refined as experiments proceed.

**Core idea**: Instruction-tuned models can enter different functional epistemic modes for resolving conflicts between internal knowledge and presented evidence. Under i1a-like instructions, the model enters a "memory-first accuracy mode" -- it prioritizes reconciliation with internal model knowledge. Under i1c-like instructions, it enters a "presented-evidence-first accuracy mode" -- it prioritizes evaluation of presented evidence.

**What current data supports:**
- A real mode shift in evidence processing caused by instruction wording (behavioral + mechanistic)
- The effect is prior-conditioned, mechanistically detectable, and transfers across domains under i1a
- Reasoning traces show qualitatively different reasoning styles consistent with the two modes
- The cross-domain transfer result strengthens the "functional mode" interpretation: if this were just about the model's confidence on a particular question, it wouldn't transfer between trivia and physics

**What current data does NOT yet support (hypotheses to test):**
- That the model has "two distinct internal representations of accuracy" (traces show reasoning STYLE, not computational mechanism)
- That RLHF specifically taught "accuracy = confident defense with citations" (plausible but indirect)
- That the modes are discrete rather than continuous (dose-response would test this)

**Terminology**: Use "memory-first accuracy mode" and "presented-evidence-first accuracy mode" rather than "parameter-accuracy" and "evidence-accuracy." The former stays at the level of observable behavior; the latter implies a representational claim we can't fully back yet.

**Relationship to existing literature:**
- The GENERAL idea that prompt framing changes epistemic behavior is NOT new (Epistemic Fragility, Task Matters, Parameters vs Context)
- The SPECIFIC mechanistic story (instruction-switchable direction at endorsement position, cross-domain transfer) still looks novel
- "Functional epistemic modes" may be novel enough as a framing, but must not be presented as if no one noticed the broader phenomenon
- Cite Epistemic Fragility for behavioral context, CK-PLUG for conceptual framework, Anthropic emotions for the general pattern of "internal functional states shaped by post-training"

**Defensible bottom-line framing:**
> We hypothesize that instruction-tuned models can enter different functional epistemic modes for resolving conflicts between internal knowledge and presented evidence. Our current results support this hypothesis behaviorally and mechanistically; the diagnostic experiment (Section 6.7) tests it directly with prompts that explicitly manipulate memory-first vs evidence-first resolution.

---

## 10. Priority Ordering for Next Steps

### Tier 1: MANDATORY Before Paper Submission

1. ~~**Known direction comparison with ablation**~~ **DONE (April 4, proxy-based):**
   - Correction-gating direction near-orthogonal to opinion-like (cos 0.093) and authority-like (cos -0.005, 0.060) proxies
   - Projection ablation: transfer accuracy/AUROC unchanged after removing all 3 proxy directions; vector norm drops 0.6%
   - Wang et al. code review (April 4): confirmed they do NOT produce a direction vector -- their analysis is layer-wise logit-lens. No "Wang sycophancy direction" exists to compare against. Resolves biggest overlap concern.
   - Caveat: proxy-based only (not exact reproductions of Mammen/Beacon). Defensible given Wang code review and projection-ablation results.

2. ~~**PIQA activation extraction**~~ **DONE (April 4)** -- Cross-domain transfer confirmed. Factual<->PIQA vector transfer ~75% accuracy, AUROC >0.82.

3. ~~**Diagnostic instruction experiment**~~ **DONE (April 4):**
   - Accuracy gradient: i1a (-0.431) >> accuracy_priority_only (-0.180) >> as_accurately_as_you_can (-0.144) >> facts_presented_only (-0.066) >> consider_info_carefully (+0.012). Monotonic.
   - Epistemic modes: mentioning conflict is itself a trigger regardless of resolution direction.
   - PIQA confirms same ordering, all more negative.

4. ~~**Re-analyze existing base vs instruct data**~~ **DONE (April 4)**
   - Key finding: base model ALREADY has evidence gating with expert tags (frac_dr_pos = 0.668 on prior-wrong), but NOT with note tags
   - RLHF reorganizes (not creates) the pattern: reduces source-dependent gating, introduces general endorsement biases
   - Confidence gradient INVERTS: base model integrates better at high confidence, instruct model gets worse
   - Remaining gap: mechanistic extraction on base Qwen3-4B (1 GPU day) to test if the direction exists pre-RLHF

**ALL TIER 1 EXPERIMENTS ARE COMPLETE. Ready for paper writing.**

### Tier 2: High Value, Do If Time Permits

5. **Graduated ambiguity experiment** (2-3 days, mostly dataset curation)
   - Clear/soft/no ground truth spectrum
   - Addresses deployment relevance

6. **i1b full run on Qwen** (cheap, fills dose-response gap)

### Tier 3: Future Work / Follow-Up Paper

7. **Agentic model behavioral test** (complex experimental design)
8. **Multi-turn instruction sequences** (novel paradigm)
9. **30B mechanistic extraction** (we have 30B behavioral results; running probe/patching on 30B activations would test if the direction scales -- feasible since Qwen3-30B-A3B is MoE with ~3B active params)
10. **SAE decomposition of correction-gating direction**
11. **Different post-training recipes comparison** (DPO vs RLHF vs GRPO)
12. **Soft/conversational endorsement variants**

### Paper Writing Timeline

- ~~April 3-4: PIQA cross-domain transfer~~ **DONE**
- ~~April 3-4: Re-analyze base vs instruct, 30B behavioral~~ **DONE**
- April 4-6: **Known direction comparison** (mandatory, blocks paper positioning)
- April 4-6: **Diagnostic instruction experiment** (behavioral, cheap, can run in parallel with known direction comparison)
- April 7-9: Follow-up mechanistic extraction if diagnostic results warrant (base model, diagnostic prompts)
- April 10-16: Paper outline and first draft (argument: behavioral paradox → mechanistic specificity → causal leverage → distinction from adjacent work)
- April 17-25: Revision, figures, additional experiments from Tier 2
- April 26-May 3: Polish, get feedback, finalize abstract
- May 4: Abstract submission
- May 5-6: Final paper polish
- May 6: Full paper submission

---

## 11. Raw Ideas & Parking Lot

These are unfiltered ideas that came up during brainstorming. Not evaluated, just captured.

1. **Instruction taxonomy**: Create a systematic taxonomy of instruction types (position-oriented, evidence-oriented, source-oriented, certainty-oriented) and map each type's effect on evidence integration.

2. **The "silent endorsement" test**: What if the endorsement is IMPLICIT? E.g., the question itself contains information that should update the model's belief, but it's not framed as an endorsement. Does the gating direction still fire?

3. **Cross-lingual test**: Does the effect replicate when instructions are in one language and questions in another? This would test whether the gating is linguistic or representational.

4. **Temperature dependence**: Does higher sampling temperature bypass the gating? If the model is more "exploratory," does it integrate evidence better?

5. **Fine-tuning TO the direction**: Can you fine-tune a model to always be in "integrate" mode by training toward the correction-gating direction? This would be a direct safety intervention.

6. **The "overconfidence tax"**: Is there a relationship between how overconfident the model is (high |m_N0| but wrong) and how susceptible it is to the gating effect?

7. **Real user conversations**: Mine conversation datasets for moments where users provide corrections. Analyze whether models in deployment show similar instruction-dependent gating patterns.

8. **Reasoning models**: Do o1-style models with internal chain-of-thought show different gating behavior? The extended thinking might provide more opportunities for the model to either entrench or correct.

9. **The information bottleneck perspective**: Frame the correction-gating direction as an information bottleneck that determines how much information from the endorsement flows into the answer. This connects to information-theoretic interpretability.

10. **Multi-endorsement**: What happens when the model receives TWO endorsements for different answers? Does the gating direction determine which endorsement wins?

11. **Temporal dynamics**: Does the gating direction change during generation? If the model starts defending its prior but then generates reasoning that should make it reconsider, does the direction shift mid-output?

12. **The "I don't know" escape**: Under what conditions does the model simply refuse to answer (say "I don't know") instead of either defending or integrating? Is there a third mode beyond defend/integrate?

---

## Appendix: Key Files Reference

| File | Contents |
|------|----------|
| `findings.md` (v3) | Comprehensive findings document |
| `plan.md` (v5) | NeurIPS execution plan with decision gates |
| `paper-examples.md` | Curated examples for factual QA |
| `paper-examples-piqa.md` | Curated examples for PIQA |
| `paper-traces.md` | Reasoning trace analysis |
| `new-phase-results/mechanism/` | Activation extraction data |
| `new-phase-results/qwen3-4b-results/` | Qwen behavioral results |
| `new-phase-results/piqa/` | PIQA behavioral results |
| `new-phase-results/qwen3-30b-results/` | 30B scale results |
| `data/exp7_mc_dataset.jsonl` | MC dataset for confound checks |

---

*This document was generated on April 3, 2026 as a comprehensive dump of all brainstorming, framework analysis, and strategic thinking conducted during the NeurIPS 2026 submission preparation.*

*Update 1 (April 3): Corrections from Codex literature review -- refined novelty claim, calibrated positioning against 5 neighboring papers (Wang et al. 2508.02087, Mammen et al. 2601.13433, BASIL 2508.16846, Beacon 2510.16727, Lu et al. 2601.10387), elevated known direction comparison to mandatory Tier 1, softened overclaims on confidence independence and position localization, switched primary cognitive science anchor from backfire effect to motivated reasoning.*

*Update 2 (April 4): PIQA cross-domain transfer results added (factual<->PIQA vector transfer ~75% accuracy, AUROC >0.82). Base vs instruct and 30B analyses completed. Four new papers added (Epistemic Fragility 2511.22746, CK-PLUG/Parameters vs Context 2503.15888, Task Matters 2506.06485, Inside-Out 2503.15299). Emerging "functional epistemic modes" framing added as soft hypothesis. Diagnostic instruction experiment designed (Section 6.7) to directly test memory-first vs evidence-first modes. Priority ordering and timeline updated.*
