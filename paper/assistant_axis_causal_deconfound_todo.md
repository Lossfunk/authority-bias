# Assistant-axis causal deconfound — experiment plan

## Why we're doing this

Paras' comment on the assistant-axis deconfound:
> cos ≈ 0 is weak evidence in high dimension. Try adding the assistant axis
> to the previous causal interventions and see what happens.

He's right. The current story leans on cosine alone, and in ~4096-d almost any
two vectors look nearly orthogonal. The stronger version of the deconfound is
causal, not geometric: if the authority direction's effect survives when we
remove the assistant-axis component, the persona story doesn't explain our
results.

## What we already have

- Assistant axis per model, built as `mean(default helpful-assistant) − mean(15 roles)`
  at the mechanism layer. Saved alongside the overlap summaries:
  - GPT-OSS: `external/assistant-axis/hardened/overlap_summary.json`
  - Gemma-4: `neurips-results/gemma4/mechanism/assistant_axis_hardened/`
  - OLMo-2:  `neurips-results/olmo2/mechanism/assistant_axis_hardened/`
  - OLMo-3.1: `neurips-results/olmo31/mechanism/assistant_axis_hardened/`
  - Qwen3.5: `neurips-results/qwen35/mechanism/assistant_axis_hardened/`
- Authority direction per model at the mechanism layer (already used in forward-patch).
- Forward-patch pipeline:
  - Trivia: `src/exp16/run_steering_test.py` (or whatever the current runner is)
  - PIQA: `src/exp16/run_piqa_interventions.py`

## The two experiments

### A. Assistant axis alone as a control

**Question:** does patching the assistant axis by itself flip wrong answers
the way the authority direction does?

**How:** run forward-patch on the trivia eval set and on PIQA, but use the
assistant axis in place of the authority direction. Match residual-stream
norm to our authority patch. Same alpha sweep as the main steering runs.

**What each outcome means:**
- If the assistant axis produces comparable wrong-answer flips → persona is
  doing the work, authority direction is persona in disguise.
- If the assistant axis barely moves behavior → the authority effect isn't
  reproducible from the persona direction alone.

### B. Residualize authority against assistant, then patch

**Question:** does the authority direction's causal effect survive after we
remove its projection onto the assistant axis?

**How:**
1. For each model at the mechanism layer, compute
   `authority_residualized = authority − (authority · assistant) * assistant`
   (with assistant unit-normalized).
2. Rerun forward-patch on trivia and PIQA using `authority_residualized`.
   Match norm to the original authority patch. Same alpha sweep.

**What each outcome means:**
- If matched-flip rates are comparable to the original → persona component
  wasn't what was driving the flips. Authority direction carries non-persona
  content that is causally sufficient.
- If matched-flip rates collapse → persona was carrying most of the causal
  signal. Bad news for the deconfound story.

## Models and scope

Run A and B on all five models: GPT-OSS, OLMo-2, OLMo-3.1, Qwen3.5, Gemma-4.

(Gemma will probably be noisy given the probe weakness we already know about,
but we should still include it for completeness.)

## Skipping

- "Authority + assistant combined" (C in the earlier discussion): hard to
  interpret cleanly, low value. Skip.

## Cost

- Vectors are already saved. No re-extraction.
- Per model: one script run (load one extra vector, one subtraction or
  substitution), same prompts, same alpha sweep as existing forward-patch.
- Wall time: roughly an hour of GPU per model for the full trivia + PIQA
  alpha sweep. Parallelizable across boxes.
- Figure + bookkeeping: a couple of hours.
- Total: one working day end-to-end if we parallelize.

## What we'll report

Small table per model:

| Model | Baseline matched-flip (authority) | Matched-flip (assistant axis only) | Matched-flip (authority residualized) |
|-------|-----------------------------------|------------------------------------|---------------------------------------|

Two rows per task (trivia, PIQA). Pick the best (layer, α) per model from
the original forward-patch runs and reuse that setting for A and B.

## The claim this lets us make

Current (geometric-only):
> Our direction is not a trivial rescaling of the assistant axis.

After A + B:
> Our direction's causal effect on trivia and PIQA survives removing its
> persona-axis component, and the persona axis alone does not reproduce the
> effect. The authority mechanism is not well explained by the known
> assistant/role persona direction.

That's the upgrade.
