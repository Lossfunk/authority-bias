# Next Experiments — Strengthening Authority-Compliance Paper

These are the experiments that most directly answer the likely NeurIPS reviews (see `mock-review.md`).

## Current alignment to feedback

- We are now aligned to the stronger spotlight framing:
  - the paper is **not** "we found an authority vector"
  - it is **"we resolve a live contradiction by showing that verified-source authority is a distinct, causally active, persona-orthogonal feature, and that this changes how sycophancy interventions should be designed"**
- Concretely, this means:
  - center the **Wang et al. vs citation-framed authority** contradiction in the intro
  - treat **Qwen + GPT-OSS** as the cleanest core evidence
  - treat **Gemma** as a boundary case unless the follow-up closes cleanly
  - avoid over-claiming universality, affect, or transfer unless we tighten those sections further

## Immediate experiments to run next

These are the shortest-path experiments that most improve the paper right now.

1. **Finish the missing assistant-axis deconfounds**
   - Gemma-4
   - OLMo-2
   - any missing Qwen / OLMo-3.1 pieces that still block the completeness table
2. **Negative steering / mitigation check**
   - steer against the authority-compliance direction
   - measure reduction in authority-conditioned override
   - measure whether baseline capability is preserved
3. **Per-model completeness table**
   - one row per model
   - explicitly mark core vs extension evidence
4. **Frozen-config audit**
   - separate primary frozen-config numbers from exploratory sweeps
5. **Gemma minimal boundary-case package**
   - shallow-layer transfer test
   - margin check
6. **Seed / resample stability**
   - show the main direction is not a fragile split artifact

## Full List

### 1. Per-model completeness table

Not glamorous, but high value. Add a table with one row per model and columns for:

`behavior`, `steering`, `C1-null`, `shuffled-label`, `assistant-axis`, `valence/arousal`, `PIQA`, `SYCON`, `thinking-mode`.

This directly addresses overclaim and evidence asymmetry.

### 2. Finish missing deconfounds

Run the missing assistant-axis checks, especially for `Qwen3.5` and `OLMo-3.1`.

Also report explicit valence/arousal cosine values per model, not just a family-wide statement.

### 3. Config-freezing audit

Run transfer and `SYCON` with the exact frozen config chosen on the fit split and report those as primary.

If you also want best-case exploratory sweeps, put them in appendix and label them clearly.

This single fix would remove a major reviewer attack.

### 4. Seed / resample stability

Refit the direction across multiple random fit/eval splits or bootstrap subsets and report variance in:

`AUROC`, matched flip, assistant-axis cosine, transfer score.

This directly addresses "is the direction real or a fragile fitting artifact?"

### 5. Replication of thinking-mode result

Replicate the thinking-on vs thinking-off comparison on at least one additional reasoning-capable model.

Even one more matched result would materially strengthen that claim.

### 6. `OLMo-3.1` PIQA transfer

Either run it or explicitly explain why it is excluded.

Right now the omission is too visible.

### 7. Negative steering / mitigation check

Test whether steering in the opposite direction reduces authority-compliance on held-out prompts without severely harming baseline helpfulness or normal instruction following.

This would strengthen both significance and practical relevance.

### 8. Position / token-phase specificity

Show whether the effect depends on applying the patch at authority-note positions versus later prompt tokens.

That would help support the "upstream processing" claim more directly.

### 9. Direction specificity against nearby alternatives

Add one or two extra deconfounds:

- instruction-following strength,
- general refusal tendency,
- or confidence/hedging style.

Reviewers will otherwise say persona and valence are not the only plausible alternatives.

### 10. Circuit follow-up, if feasible

Even a lightweight head-level attribution or patching analysis in one strong model would substantially upgrade the "mechanism" claim.

Not mandatory, but high upside.

### 11. Gemma caveat: boundary-case follow-up, not a paper blocker

Purpose: explain why `Gemma-4` shows a clear in-domain authority/compliance mechanism on trivia, but much weaker out-of-domain transfer on `PIQA`.

Our current view is that this is valuable as a **boundary-case explanation**, not as a make-or-break pillar for the paper. The core claim does not require every model to transfer strongly. A compact Gemma follow-up helps us turn an awkward caveat into an interpretable exception.

What the Gemma follow-up is actually trying to distinguish:

- **margin explanation** — maybe `PIQA` answers are too confident to flip;
- **MoE routing-locality explanation** — maybe the intervention lands too deep inside Gemma's MoE stack to transfer cleanly OOD;
- **wrong-summary explanation** — maybe a single linear `w1 - c1` direction is a poor summary of Gemma's authority signal;
- **style contamination explanation** — maybe the learned direction mixes note-style intensity with authority content;
- **multi-feature / nonlinear explanation** — maybe Gemma needs sparse or nonlinear interventions rather than one residual direction.

Why this is principled rather than ad hoc:

- recent work suggests that a single linear direction can miss the real feature structure (`Shafran et al., 2025`, SNMF);
- recent steering papers argue that linear steering can fail for geometric reasons even when the phenomenon is real (`Curveball Steering`, 2026; `Steered LLM Activations are Non-Surjective`, 2026);
- recent MoE work suggests routing behavior depends on representation organization, so a deep-model exception is plausible (`The Myth of Expert Specialization in MoEs`, 2026).

So the paper-level question is **not** "can we make Gemma look good?" It is: **can we explain why Gemma is the exception?**

Recommended paper stance:

- Treat Gemma as a **boundary case / caveat**.
- Do only the **smallest high-value follow-up** needed to interpret the caveat.
- Do **not** let a full Gemma suite delay submission.

What is worth doing before submission:

1. **E2 margin check** — cheap, already directly informative.
2. **E1 shallow-layer transfer test** — highest value per GPU-hour; if shallow layers transfer better, that strongly supports the MoE-locality story.

What can be deferred to curiosity-driven follow-up work:

- `E3` per-layer refit,
- `E4` style-residualized refit,
- `E5` SNMF decomposition,
- `E6/E7` router drift / router bias.

These are scientifically interesting, but they are not necessary to defend the paper's core claim before submission.

---

## Spotlight Push: revised priorities

If the goal is not just a clean accept but a plausible **spotlight-level** submission, the priority order changes a bit. The paper then has to optimize not only for rigor, but also for a memorable reviewer hook.

### Working spotlight framing

This is the version of the paper we should currently optimize toward:

> The spotlight version of this paper is not "we found an authority vector." It is: **we resolve a live contradiction by showing that verified-source authority is a distinct, causally active, persona-orthogonal feature, and that this distinction changes how sycophancy interventions should be designed.**

### Core spotlight thesis

The strongest version of the paper is likely not:

> there exists another steering vector for authority compliance.

It is closer to:

> we isolate a deconfounded, causally testable endorsement/compliance feature, show cross-task transfer in multiple model families, and identify meaningful boundary cases and training-regime differences in when that feature exists or transfers.

### Where we are currently aligned

- `Qwen assistant-axis A/B` is still the highest-value specificity check.
- `OLMo-3.1` framing needs cleanup if it remains attached to the headline `58.8%` number.
- a per-model audit/completeness table remains necessary.
- thinking-mode should either be replicated on one more reasoning-capable model or demoted from any prominent claim.
- Gemma should stay a **boundary-case mini-story**, not become the main experimental sink.

### Revised spotlight-order priorities

#### Must do

1. **Qwen assistant-axis causal deconfound**
   - assistant-axis-alone control
   - authority residualized against assistant-axis

2. **Negative steering / mitigation demo**
   - show that steering against the direction reduces authority-conditioned compliance on held-out prompts;
   - ideally also show it does not destroy baseline capability (`IFEval`, `MMLU`, or equivalent).

3. **Primary-vs-extension framing cleanup**
   - especially the `OLMo-3.1` issue if its `58.8% matched flip` remains the intro-facing number;
   - either promote `OLMo-3.1` to primary status, or move the headline to a core model such as `GPT-OSS`.

4. **Per-model audit/completeness table**
   - make the primary model set explicit;
   - distinguish core models from extension/OOD models.

#### Strongly consider

5. **Wang et al. reconciliation via cue-class test**
   - self-claimed expertise vs verified-source vs hybrid cue framing;
   - this converts a literature tension into a testable resolution rather than a verbal distinction.

6. **Behavioral comparison against a public prior sycophancy direction**
   - ideally a head-to-head against a public CAA-style direction on one shared model/protocol;
   - this is higher-value than cosine-only comparisons because it tests behavioral distinctness directly.

7. **One more frontier model at matched protocol**
   - useful if we want to promote the frontier-vs-open-weight training-regime angle rather than leaving Gemini as a one-off curiosity.

8. **Thinking-mode replication or demotion**
   - if replicated, it can become a stronger secondary headline;
   - if not, move it clearly into case-study territory.

#### Safe to defer

9. **Gemma deep follow-up beyond the minimal boundary-case package**
10. **More open-weight model expansion**
11. **Larger deconfound families beyond the core reviewer-facing ones**

### Why this differs from the clean-accept plan

For a clean accept, the main job is to close obvious rigor holes. For a spotlight push, we also need at least one of the following to land cleanly:

- a compelling **mitigation/intervention** story,
- a sharp **literature-resolution** story (especially around `Wang et al.`),
- or a memorable **frontier-vs-open-weight divergence** story.

Without at least one of those, the paper can still be strong, but it is more likely to read as a well-executed mechanism paper than a spotlight-shaped one.

---

## Highest-ROI Shortlist

If time is limited, do these first:

1. Finish missing assistant-axis and valence/arousal checks.
2. Clean up frozen-config discipline and rerun/report primary `PIQA`/`SYCON` under that protocol.
3. Add seed/resample stability.
4. Replicate thinking-mode on one more model.
5. Add a per-model audit table.
6. For Gemma, if time permits, do only the minimal caveat package: `E2` margins + `E1` shallow-layer test.

Those five would directly neutralize most of the review.
