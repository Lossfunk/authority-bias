# Mock NeurIPS Review

## Summary

This paper argues that authority-conditioned compliance in LLMs is not merely a behavioral surface effect, but is mediated by a low-dimensional internal direction that is causal, portable across tasks, and distinguishable from persona or affective confounds. The paper introduces an `N0/C1/W1` behavioral protocol, extracts a compliance direction from activations, applies steering interventions, evaluates transfer to `PIQA` and `SYCON`, and presents a `Qwen3.5` thinking-on vs thinking-off comparison suggesting that chain-of-thought amplifies rather than mitigates this failure mode.

The paper is ambitious and unusually well-shaped around a mechanistic question rather than a benchmark-only phenomenon. The central empirical observation is interesting, and the claimed combination of behavioral override, intervention, transfer, and deconfounding is potentially publication-worthy. However, the current version overstates uniformity of evidence across models, leaves some evaluation-selection details ambiguous, and relies on incomplete per-model support for several broad claims.

## Strengths

- The paper asks a strong question. Moving from "models are sycophantic" to "is there a causal internal authority-compliance state?" is a meaningful step up in ambition.
- The behavioral effect sizes appear large and cross-family, which makes the problem feel real rather than benchmark-fragile.
- The paper has a coherent mechanistic arc: behavior, direction extraction, causal intervention, transfer, deconfounds.
- The `C1-null` placebo is a good control and one of the stronger aspects of the design.
- The transfer story to `PIQA` and `SYCON` is potentially valuable because it pushes beyond in-domain behavioral relabeling.
- The thinking-mode result on Qwen is provocative and, if robust, interesting in its own right.

## Weaknesses

- The paper sometimes overclaims relative to the actual per-model evidence. Several statements read as if all open-weight models have the full suite of nulls, deconfounds, and transfer checks, but the current writeup suggests support is uneven.
- The selection protocol is not yet fully reviewer-proof. It is not consistently clear how `(layer, position, alpha, mode)` are chosen, frozen, and reused across main eval, `PIQA`, and `SYCON`.
- The transfer story appears inconsistent with the claimed frozen-config discipline. Reported layer/alpha choices differ across core steering, `PIQA`, and `SYCON`, which raises concern about post hoc tuning.
- Important deconfound checks are incomplete. In particular, assistant-axis coverage appears missing for some models, and valence/arousal results are stated broadly without enough per-model detail.
- The frontier/API model evidence is behavioral only. Including them in broad framing risks misleading readers into thinking the mechanistic story generalizes further than shown.
- The thinking-mode claim is based on a single model family and should be scoped more narrowly unless replicated.

## Main Concerns

1. **Evaluation hygiene / model selection**
   The main issue is not whether the effect exists, but whether the paper cleanly separates fitting, selection, and final reporting. A mechanistic paper making causal claims has to be especially disciplined here. If transfer and native-benchmark results use different layer/alpha configurations than the main steering result, the paper must explicitly state that these are separate frozen configs or exploratory analyses.

2. **Uneven support across models**
   The paper's best evidence seems concentrated in a subset of open-weight models. That is acceptable, but the framing should match. Claims such as "the direction is orthogonal to the assistant axis" or "a deconfounded internal direction exists across models" need per-model completeness or narrower wording.

3. **Transfer interpretation**
   Transfer to `PIQA` and `SYCON` is interesting, but if it depends on model-specific retuning or benchmark-specific layer choices, it is weaker evidence for a single portable mechanism than the current framing suggests.

4. **Chain-of-thought claim scope**
   The `Qwen3.5` v3-vs-v4 result is interesting, but the paper should avoid presenting this as a general statement about reasoning unless it is replicated on at least one additional thinking-capable model.

5. **Mechanistic depth**
   The paper successfully argues for a causal activation-space effect, but it does not yet trace the mechanism to heads, subcircuits, or token-position dynamics. That is not fatal, but it limits how strongly "mechanism" can be read.

## Questions For The Authors

- What exact split discipline prevents leakage between direction fitting, layer/position selection, alpha/mode selection, and final evaluation?
- Are `PIQA` and `SYCON` run with the exact same frozen config selected on the fitting split, or are they allowed benchmark-specific configuration choices?
- Which open-weight models have all of the following: `C1-null`, shuffled-label null, assistant-axis deconfound, valence/arousal deconfound, `PIQA`, `SYCON`?
- Why is `OLMo-3.1` absent from the main `PIQA` transfer list despite being central elsewhere?
- How sensitive are the main matched-flip claims to denominator size in models with weak baseline `N0` accuracy on the selected slice?
- How stable are the extracted directions across random seeds / resampled fitting subsets?
- For `Qwen3.5`, can the authors exclude alternative explanations tied to model-version differences beyond thinking mode?

## Recommendation

Weak Accept to Borderline

## Scores

- Originality: 8/10
- Technical Quality: 6/10
- Clarity: 7/10
- Significance: 7/10
- Confidence: 0.76

If mapped to an old-style overall score, **6/10 (Weak Accept / Borderline)** in the current state.

## Acceptance Probability

At current stage, based on the red-thread and the issues above, around **35% to 50%**.

Internal breakdown:
- **~15% reject-risk** from "interesting but too selection-fragile / not fully deconfounded."
- **~35% borderline bucket** where reviewer mix determines outcome.
- **~35% solid accept path** if the paper is polished and the missing checks are filled.
- **~15% upside** if the final version includes a clean per-model audit table plus one or two strong strengthening experiments.

Honest summary: **currently plausible but not safe**. The idea is good enough for NeurIPS; the evidence packaging is not yet tight enough to rely on goodwill.
