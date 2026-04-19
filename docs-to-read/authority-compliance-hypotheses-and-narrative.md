# Authority-Compliance Hypotheses and Narrative Map

This document is meant to keep the project pointed at the real question, not a weaker proxy question.

The project is **not mainly**:
- "current sycophancy evals are insufficient"

That is one implication. The stronger question is:

> Do some LLMs contain a distinct internal authority-conditioned compliance mechanism that can override truth, and is that mechanism causal, portable, and not well explained by generic sycophancy/persona/affect drift?

## Core research question

The real target is not just whether models defer to authority in behavior.

The stronger target is whether:
- authority framing recruits a **specific internal state/process**
- that state can push a model from a baseline-correct answer to a wrong answer
- and that state is meaningfully different from more generic alternatives such as:
  - user-agreement / generic sycophancy
  - assistant/persona drift
  - affective drift
  - shallow prompt-format artifacts

## Main hypotheses

## H1. Behavioral authority-conditioned truth override

**Claim:** Strong authority framing can systematically push models from correct to wrong answers, even in free generation.

**Why it matters:**  
If this fails, there is no interesting mechanism story to tell. We need the behavioral phenomenon first.

**Would support H1:**
- Under `W1`, baseline-correct `N0` items flip to wrong at nontrivial rates.
- The effect grows with endorsement strength.
- The result is visible in free generation, not only forced choice.

**Would weaken H1:**
- Effects are tiny, unstable, or mostly disappear under better parsing / prompt cleanup.
- Wrong-answer increases are matched by similar wrong-answer increases under non-authoritative prompt variants.
- The effect is mostly a formatting or parser artifact.

**Narrative impact if supported:**  
There is a real authority-conditioned truth-override phenomenon worth explaining mechanistically.

**Narrative impact if weakened:**  
The story becomes "some prompts can nudge answers," which is much less interesting and much less publishable.

## H2. Internal causal mechanism hypothesis

**Claim:** This is not just a surface prompt effect; it corresponds to a causal internal state/process that we can extract and intervene on.

**Why it matters:**  
This is the core upgrade from a benchmark paper to a mechanism paper.

**Would support H2:**
- Activation-space signals predict flips above confounds.
- Forward activation patching reproduces the behavioral shift.
- The effect shows a dose response and layer/position specificity.
- The intervention acts on prompt-side hidden states before generation unfolds.

**Would weaken H2:**
- We can read out behavior, but causal interventions do nothing.
- Effects only appear under extremely blunt interventions that destroy capability.
- The result is inconsistent across seeds / subsets / parsing choices.

**Narrative impact if supported:**  
We can say there is a causal internal compliance state/process, not merely a correlational marker.

**Narrative impact if weakened:**  
The paper becomes "authority-following is decodable" rather than "authority-following is causally mediated by an identifiable mechanism."

## H3. Transfer / portability hypothesis

**Claim:** If this is a real mechanism, it should transfer beyond the source dataset rather than being a benchmark-specific artifact.

**Why it matters:**  
Transfer is one of the strongest signs that we found something broader than a dataset quirk.

**Would support H3:**
- A mechanism extracted on factual QA shifts behavior on PIQA or another out-of-domain task.
- The shift has the expected sign: more wrong answers, fewer correct answers, or stronger compliance-like behavior.

**Important scope note:**  
There are two distinct tests here, and they should not be conflated:
- **Mechanism transfer / out-of-domain generalization:** extract on one benchmark, intervene with that same mechanism on another benchmark, and measure whether behavior shifts in the predicted direction.
- **Sycophancy-specific transfer:** show that the intervention increases or decreases a benchmark-native sycophancy metric on a dataset with explicit endorsement-vs-truth conflict.

PIQA currently supports the first claim, not the second. It is an out-of-domain causal transfer test, not by itself a sycophancy benchmark.

**Would weaken H3:**
- The intervention only works on the extraction benchmark.
- Out-of-domain effects vanish or reverse arbitrarily.
- Transfer only appears under very weak parsing / scoring assumptions.

**Narrative impact if supported:**  
We can argue the mechanism is portable and likely lives on a broader behavioral manifold.

**Narrative impact if weakened:**  
The safer story becomes "benchmark-local internal signature" rather than "general compliance mechanism."

## H3b. Replication on an explicit sycophancy benchmark

**Claim:** If this is genuinely about authority-following / sycophancy-like deference, we should be able to show the effect on a benchmark with explicit endorsement-vs-truth conflict using a sycophancy-native metric.

**Why it matters:**  
This is the cleanest bridge between the mechanism story and the existing sycophancy-evaluation literature.

**Would support H3b:**
- On a sycophancy-style benchmark, the intervention increases endorsement-following relative to baseline.
- The effect is stronger for authoritative endorsement than for matched non-authoritative prompt variants.
- The same direction that transfers to PIQA also modulates sycophancy-native behavior in the expected direction.

**Would weaken H3b:**
- The intervention raises generic error but does not consistently increase endorsement-following.
- The effect only appears on benchmarks that can be reinterpreted as generic accuracy degradation.

**Narrative impact if supported:**  
We can connect the portable mechanism claim directly to a sycophancy-native behavioral metric, not only to generic correctness shifts.

**Narrative impact if weakened:**  
We should frame PIQA-style results as out-of-domain mechanism transfer, while being more cautious about calling them sycophancy transfer.

## H4. Deconfound hypothesis

**Claim:** The mechanism is not well explained by generic assistant/persona drift, cheap affective drift, or simple prompt-format artifacts.

**Why it matters:**  
Without this, the work can be dismissed as "just assistantness," "just emotional tone," or "just a fancy prompt confound."

**Would support H4:**
- Low overlap with assistant-axis / persona directions.
- Low overlap with lexical valence-arousal directions.
- Similar behavior does not appear for matched but non-authoritative prompt variants.
- The authority-conditioned signal is more predictive than style/strength alone.

**Would weaken H4:**
- Strong overlap with assistant/persona directions.
- Strong overlap with valence/arousal directions.
- Prompt-only baselines explain most of the observed variance.

**Narrative impact if supported:**  
We can defend a distinct authority-compliance interpretation.

**Narrative impact if weakened:**  
The story becomes "authority prompts ride generic social/affective drift," which is still useful but notably weaker.

## H5. Evidence-weighting rather than late output flip

**Claim:** The mechanism is not only a last-token readout effect; authority framing perturbs internal evidence weighting before answer generation.

**Why it matters:**  
This is the deeper mechanistic interpretation and the part most likely to matter to senior interpretability / alignment researchers.

**Would support H5:**
- Prompt-side hidden state changes propagate to downstream answer positions before generation starts.
- Teacher-forced first-token effects are weaker than the hidden-state shift, suggesting a distributed generation-time rollout.
- Reasoning/trajectory diagnostics show the effect emerges during internal processing rather than only at final selection.

**Would weaken H5:**
- All measurable effect concentrates in a final answer readout with little upstream movement.
- Prompt-side intervention changes final choice but not intermediate hidden-state structure.

**Narrative impact if supported:**  
We can frame the mechanism as a socially conditioned evidence-weighting process.

**Narrative impact if weakened:**  
We should frame it more conservatively as answer-selection or output-preference override.

## H6. Cross-model generality hypothesis

**Claim:** The authority-compliance mechanism, or something recognizably analogous, should appear in more than one model family.

**Why it matters:**  
A single strong model result can be valuable, but a second clear replication upgrades the claim a lot.

**Would support H6:**
- A second model shows a similarly causal internal authority-compliance signature.
- The exact layer/implementation can differ while preserving the same basic pattern.

**Would weaken H6:**
- GPT-OSS remains the only clean case.
- Other models show only weak, noisy, or contradictory patch effects.

**Narrative impact if supported:**  
We can argue for a broader class of authority-conditioned internal compliance mechanisms across model families.

**Narrative impact if weakened:**  
We should present GPT-OSS as the main case study and treat other models as contrast cases, not full replications.

## Narrative ladder

This is the ranking of stories from strongest to weakest.

### Strongest narrative

At least one model contains a **causal, portable, authority-conditioned internal compliance mechanism** that can override truth and is not well explained by generic assistant or affective drift.

This is the current best-case paper story.

### Medium-strength narrative

Authority framing induces truth override behavior, and the effect is partially mechanistic, but portability or distinctness from confounds is weaker than hoped.

This is still publishable, but the framing should be more cautious.

### Conservative narrative

Authority-conditioned truth override is behaviorally real, but the internal signal is mainly correlational, benchmark-local, or entangled with broader social drift.

This is weaker and should be framed as early mechanistic evidence rather than a clean mechanism claim.

### Negative-result narrative

Authority prompts matter behaviorally, but there is no robust evidence for a distinct portable internal mechanism once confounds and parser artifacts are controlled.

Even this could still matter, but it becomes a paper about limits of current mechanism interpretations.

## What currently looks most important to validate next

1. **Gemma no-thinking rerun**
   This is the cleanest immediate follow-up for separating real weakness from reasoning-trace contamination.

2. **Separate transfer from replication**
   Keep PIQA framed as out-of-domain causal transfer, and add one explicit endorsement benchmark for a sycophancy-native metric.

3. **A second clean model replication**
   GPT-OSS is the strongest case so far. A second model with the same basic pattern would strengthen the claim a lot.

4. **Random-direction / null behavioral baselines**
   These make the intervention story much harder to dismiss.

5. **Sharper late-output vs evidence-weighting diagnostics**
   Useful if the goal is to defend the stronger interpretability claim rather than only the causal patch result.

## Current provisional read

Given the evidence so far, the project currently seems closest to:

- **H1:** supported
- **H2:** supported in GPT-OSS, weaker in Gemma
- **H3:** supported in GPT-OSS as out-of-domain mechanism transfer
- **H3b:** not yet established
- **H4:** supported so far for the tested deconfounds
- **H5:** partially supported, but still worth tightening
- **H6:** not yet established

So the most defensible current claim is:

> In GPT-OSS, strong authority framing appears to recruit a causal internal compliance state/process that can override truth, transfers out of domain, and is not well explained by generic assistant drift or cheap affective drift.

The more cautious companion statement is:

> PIQA currently supports out-of-domain causal transfer of the extracted mechanism, but not yet a sycophancy-native transfer claim. That stronger claim still needs an explicit endorsement benchmark and metric.

And the main open question is:

> Is this a broader model-family phenomenon, or mainly a particularly clean GPT-OSS case?

## Paper-framing reminder

If the work lands well, the main contribution should be framed as:

- not merely "models defer to authority"
- not merely "sycophancy evals miss some cases"
- but rather:

> some models may contain a causal, portable internal mechanism for authority-conditioned truth override, and this failure mode may not be well captured by current surface-level truthfulness or sycophancy evaluations.
