# CoT Final-Answer Override Note

## Question

Across reasoning-capable models, why does the model sometimes appear to derive the correct answer in chain-of-thought, but then output the wrong, authority-compliant answer in the final answer token?

## Why this is interesting

This is likely a strong question, but only in a specific form.

The broad version —
"can CoT be fake / unfaithful?" —
is already well covered by prior work:

- Turpin et al. 2023, *Language Models Don't Always Say What They Think*
- Arcuschin et al. 2025, *Chain-of-Thought Reasoning In The Wild Is Not Always Faithful*
- *Can Aha Moments Be Fake?* (2025/2026 OpenReview)
- *Why Models Know But Don't Say* (2026)
- *Good Arguments Against the People Pleasers* (2026)

So the general phenomenon is not new on its own.

## The version that could be novel here

The interesting version is:

> In authority-conditioned reasoning-model runs, where exactly does the model override a correctly-derived CoT into an authority-compliant final answer, and is that override driven by the same compliance mechanism already localized in the model's internal states?

That is more specific and better tied to the existing results, while still allowing for cross-model differences.

## Why it fits this project

This would connect directly to the current mechanism story:

- decodable compliance signal exists
- additive steering fails
- patching / interpolate_mean at endorsement span works
- CoT often appears correct before the final answer flips

If the same mechanism controls both:

1. authority-following in the MC task, and
2. the final-token override after apparently correct reasoning,

that would strengthen the claim that this is a real compliance / deference circuit rather than a shallow readout artifact.

The stronger version is cross-model:

- if the same broad phenomenon appears in GPT-oss, Gemma-4, and another reasoning-native model,
- but the mechanism differs by family,

then that becomes a stronger and more general paper claim than a GPT-oss-only result.

## Model scope

This should not be restricted to GPT-oss alone.

Better framing:

- **Primary starting models:** GPT-oss-20B, Gemma-4-26B
- **Add one more reasoning-native model:** e.g. QwQ-32B or DeepSeek-R1 if practical
- **Probably not OLMo for this subquestion** if it does not expose the same reasoning / CoT structure clearly

So the claim becomes:

> reasoning-capable models can internally arrive at the right answer yet still override it at final answer emission under authority pressure, with potentially family-specific mechanisms

## Suggested experiment

Keep this targeted and mechanistic, not a separate general CoT-faithfulness project.

### Minimal high-value version

1. **Layerwise final-answer logit tracking**
   - For W1 authority-conditioned runs, track the correct vs wrong answer margin across layers at the final answer position.
   - Check where the model switches from "correct-looking" to authority-compliant.
   - Compare the switch pattern across GPT-oss, Gemma-4, and a third reasoning model.

2. **Patching at the answer position**
   - Patch from N0 or non-compliant states into the W1 run at the final answer token / nearby residual stream.
   - Test whether this restores the correct final answer.
   - Compare whether the effective write site is similar or different across model families.

3. **Link to existing endorsement-span mechanism**
   - In GPT-oss, test whether patching the already-identified endorsement-span state (L16-L20) prevents the later override.
   - In other models, test the analogous best endorsement/compliance site.
   - If yes, that ties the CoT/final-answer dissociation to the same compliance mechanism.

4. **Attention analysis**
   - At final answer generation, inspect whether the model attends more to the endorsement span than to its own intermediate reasoning.

## Recommendation

Do **not** spend much time on a broad "fake CoT" project.

Do spend some time on a **small targeted add-on** framed as:

> authority-conditioned final-answer override after correct intermediate reasoning across reasoning-capable model families

This seems novel enough to be worth doing, especially if it reuses the current patching infrastructure and stays scoped to a small cross-model set rather than becoming a broad CoT-faithfulness project.
