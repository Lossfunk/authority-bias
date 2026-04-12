# Rebuttal Plan: "Trust Me, I'm Absolutely Right"

Two reviews received. Both recommend **acceptance**. This document organizes every point raised, our response strategy, and tracks which items are addressable before submission vs. future work.

---

## Review 1 (paperreview.ai-style, automated)

> Note: This reviewer could not read page 16 of the PDF. Some points about missing mask validity data or missing statistical protocol may reflect incomplete reading rather than genuine omissions.

### Strengths Acknowledged

1. Prior-stratified framework is "a clear conceptual advance relative to aggregate metrics."
2. Two-design replication is "a strong form of convergent validity."
3. Hedging confound handling is "convincingly shows how generation artifacts distort apparent evidence sensitivity."
4. Sensitivity analyses (tags, families, slices, repeated pressure) "add depth and triangulate the central claim."
5. Practical significance: "surfaces a practically salient failure mode... invisible in aggregate sycophancy metrics."

### Weaknesses Raised & Our Responses

#### W1. Masking induces small effective n (e.g., Llama Note n_eff = 39)

**Status: Already addressed in this revision.**

We added:
- Appendix F: full mask-validity table (n, n_eff, %w, %both) for all 12 model x tag cells at both prior-wrong and high-conf-wrong slices.
- Section 4.1 "Mask validity caveat" paragraph explicitly flagging Llama's low validity and stating Qwen-Instruct (n_eff = 109) carries the stronger evidentiary weight.
- Table 1 caption now references Appendix F.

**Rebuttal language**: "We agree this is an important transparency concern. The revised draft includes a dedicated mask-validity appendix (Appendix F, Table 5) reporting n, n_eff, %w, and %both for all cells. We explicitly note in Section 4.1 that Llama high-confidence-wrong results are computed on reduced samples (n_eff = 39 for Note) and that the Qwen-Instruct results (n_eff = 109) carry the primary evidentiary weight."

#### W2. A/B token logits may be sensitive to tokenization asymmetries

**Status: Addressed in Limitations (sentence added this revision).**

Our response: Label randomization (seed 42) ensures any systematic A/B offset is balanced across items. The Delta_r metric takes ratios of paired effects (same item, same label assignment), so a constant A/B offset cancels. We acknowledge in Limitations that the offset could interact with the masking threshold in low-validity cells.

**Rebuttal language**: "Answer labels are randomized across items. Delta_r is computed as a ratio of paired effects on the same item under the same label assignment, so a constant tokenization offset cancels in the numerator and denominator. We added a sentence in Limitations acknowledging potential interaction with the masking threshold in low-validity cells. We agree that parallel analyses on decoded choices would triangulate the findings; we note this as future work below."

#### W3. Prior state elicited under one neutral prompt; alternative priors could alter slice membership

**Status: Acknowledged as future work.**

This is a legitimate concern. Self-consistency sampling (multiple temperatures, majority vote) or multiple neutral prompt phrasings could produce different prior estimates and different slice memberships. However, our single-prompt prior is the standard approach in the sycophancy literature (Sharma et al., 2023; Atwell et al., 2025), and the sign reversal's robustness to threshold choice (top-25% and top-10% both show inversion) provides indirect evidence that slice membership is not fragile.

**Rebuttal language**: "We use single-prompt greedy decoding for prior elicitation, consistent with standard practice in the sycophancy literature. The sign reversal holds at both top-25% and top-10% thresholds and weakens monotonically at low confidence, suggesting the effect is not an artifact of a brittle prior boundary. We agree that self-consistency priors would be a valuable robustness check and note this as future work."

#### W4. Scope limited to three small open-weight models and two-choice factual QA

**Status: Acknowledged in Limitations.**

This is a real constraint. We are transparent about it: "We cannot claim the sign-reversal pattern generalizes to closed-source systems (GPT-4, Claude), larger open models, or tasks beyond factual QA." The reversal replicating across two model families and two experimental designs provides some evidence against single-architecture artifacts, but generalization remains open.

**Rebuttal language**: "We agree scope is a limitation and are transparent about it in the Limitations section. The reversal replicating across two independent model families (Llama and Qwen) and two distinct experimental designs provides evidence that the finding is not architecture-specific, but generalization to larger or closed-source models remains untested. We note this as the top priority for follow-up work."

#### W5. Only one instruction wording; no paraphrase robustness

**Status: Acknowledged in Limitations (sentence added this revision) + mentioned in Discussion.**

**Rebuttal language**: "We added an explicit sentence in Limitations: 'We tested a single instruction wording; paraphrase robustness was not evaluated.' We agree this is important — the sign reversal could be sensitive to framing (e.g., 'reconsider your answer carefully' vs. 'prioritize factual accuracy'). We plan to test 3-5 paraphrases in follow-up work."

#### W6. Authority tags may encode stylistic differences beyond authority

**Status: Partially addressed.**

The four tags (Expert, Note, Someone-online, User) are taken from Mammen et al. (2026) and follow standard authority-manipulation designs. The tags differ in both perceived authority and surface wording, which is the standard design choice in this literature. A content-matched control (same wording, different authority label) would isolate the authority component more cleanly but is a separate experiment.

**Rebuttal language**: "Our authority tags follow the standard design from Mammen et al. (2026). We agree that a content-matched control (identical surface form, different authority label) would more cleanly isolate the authority component. We note this as a design improvement for future work."

#### W7. Metrics and masks could be further formalized

**Status: Partially addressed.**

We added a statistical protocol paragraph in Section 3 (item-level resampling, percentile CIs, replicate counts, seed). The sign-consistency mask is defined in Section 3.5 (|eff| < 10^-3 threshold). Further formalization (e.g., as a DAG or causal graph) would add length without proportional clarity for a workshop paper.

**Rebuttal language**: "We added a statistical protocol paragraph to Section 3 specifying bootstrap details (item-level resampling, 10,000/3,000 replicates, percentile CIs, seed 42). The sign-consistency mask is defined in Section 3.5 with threshold tau = 10^-3. We agree that more formal treatment (e.g., a regression framework) would complement the ratio-based analysis and discuss this below."

#### W8. A compact regression-based analysis would make key effects more legible

**Status: Future work.**

A mixed-effects logistic regression with fixed effects for direction, instruction, tag, prior-sign, and their interactions (with item random intercepts) would be a valuable complement. It would quantify the cross-terms directly and avoid the masking issue. This requires new analysis code and is out of scope for a text-only revision, but we agree it is the natural next step.

**Rebuttal language**: "We agree a mixed-effects regression model (predicting logit margin with fixed effects for direction x instruction x tag x prior-sign and item random intercepts) would corroborate the sign reversal without masking and quantify interaction terms directly. This is our planned next analysis step."

#### W9. Limited engagement with alternative uncertainty/prior elicitation methods

**Status: Acknowledged as a limitation of scope.**

Self-consistency priors, verifier-based calibration, and ensemble-based uncertainty estimates are all valid alternative prior constructs. Our framework is compatible with any prior elicitation method — the key insight (stratify by prior state) is independent of how the prior is measured.

**Rebuttal language**: "Our framework is agnostic to the prior elicitation method — the key insight is to stratify by whatever prior estimate is used. We agree that self-consistency sampling, verifier-based calibration, or ensemble uncertainty would enrich the 'prior' construct and provide robustness checks."

#### W10. Limited discussion of outcome choices (logit margins vs choice probabilities vs accuracy)

**Status: Partially addressed.**

We use logit margins because they are continuous, allow fine-grained measurement of endorsement effects, and avoid the information loss of binarizing to accuracy. We added a concrete r_w/r_c interpretation in Section 4.1. A parallel analysis on decoded choices (accuracy, flip rates) would triangulate but requires re-running the analysis pipeline.

**Rebuttal language**: "We chose logit margins over accuracy because they preserve fine-grained information about endorsement effect magnitudes and allow ratio-based selectivity metrics. We added a concrete interpretation in Section 4.1: 'Qwen-Instruct Note under instruction removes 67% of the correct-direction effect while amplifying the wrong-direction effect by 72%.' Parallel decoded-choice analyses would triangulate these findings."

### Questions Raised & Our Responses

#### Q1. Sensitivity to masking thresholds

**Status: Addressed.** Threshold sensitivity data now in Limitations: sign reversal holds at top-25% and top-10%, weakens at low-confidence.

#### Q2. Corroborate with discrete outcome metrics

**Status: Future work.** Requires re-running analysis pipeline. Added concrete r_w/r_c framing as partial substitute.

#### Q3. Alternative prior elicitation

**Status: Future work.** See W3 above.

#### Q4. Authority-tag stylistic confounds

**Status: Acknowledged.** See W6 above.

#### Q5. Instruction paraphrase robustness

**Status: Acknowledged in Limitations.** See W5 above.

#### Q6. Per-dataset results (TriviaQA vs TruthfulQA)

**Status: Future work.** Data exists to do this analysis. Would deconfound slice composition.

#### Q7. Examples and annotation protocol for reasons+data

**Status: Could add to appendix if space permits.** The generation prompt and a few examples would increase transparency.

---

## Review 2 (thorough manual review)

### Strengths Acknowledged

1. "Identification strategy is genuinely novel and principled."
2. "Sign reversal replicates across two independent experimental designs" — "the paper's most compelling feature."
3. "Honest disclosure of hedging confound" — "a model of scientific transparency."
4. "Qwen-Thinking as the principled exception" — "avoids overclaiming."
5. "Strong fit to TrustNLP."
6. "Appendices are comprehensive and well-organized."

### Weaknesses Raised & Our Responses

#### W1. Llama results are underpowered; abstract gives them too much weight

**Status: Needs action (abstract rewrite).**

The abstract currently presents Llama and Qwen-Instruct side-by-side: "(Llama: Delta_r = -0.67; Qwen-Instruct: Delta_r = -0.89)." The reviewer is right that Llama (n_eff = 39) should not receive equal billing with Qwen-Instruct (n_eff = 109). The abstract should foreground Qwen-Instruct and present Llama as supporting evidence.

**Action**: Rewrite abstract to lead with Qwen-Instruct result and qualify Llama.

#### W2. Single instruction wording is a serious unresolved confound

**Status: In Limitations (added this revision). Reviewer wants it in main text too.**

The reviewer says: "add a sentence in the main text (not just limitations) that the single instruction wording is a key open question."

**Action**: Add one sentence to the Discussion (practitioner implications paragraph) flagging instruction wording as an open question.

#### W3. Synthetic evaluation materials remain a concern despite correction

**Status: Addressed with audit stats. Reviewer wants human-written reasons on a subset.**

Human-written reasons on a subset would be a new data collection effort — out of scope for pre-submission. Our audit (balanced word counts, zero hedging markers) is the best we can do without human annotation.

**Rebuttal language**: "We agree that human-written reasons on a subset would meaningfully strengthen the evidence-scaling experiment. The post-correction audit (mean word counts: correct 20.7, wrong 21.4; zero residual hedging markers across 7,252 reasons) provides the best current control, but human validation remains a priority for follow-up work."

#### W4. Two-choice format limits generalizability claims

**Status: Acknowledged in Limitations.**

Same as Review 1 W4. The abstract's final sentence ("motivating prior-stratified evaluation for trust-relevant instruction following") is a methodological recommendation, not a generalization claim about all tasks.

#### W5. No mechanistic account for the sign reversal's direction-specificity

**Status: Needs action (add brief hypothesis to Discussion).**

The reviewer specifically suggests: "Even a brief hypothesis — e.g., that 'be correct' activates an identity-protection circuit that amplifies existing confidence gradients, which can be steered by attention heads identified in Genadi et al. — would strengthen the paper's theoretical contribution."

This is a good suggestion. Genadi et al. (2026) found sycophancy is linearly separable in attention head activations. A natural hypothesis is that "be correct" instructions modulate these same heads, amplifying the model's existing confidence direction rather than re-evaluating evidence. Under high authority, the modulation is non-selective (both directions suppressed); under low authority, it preferentially gates corrective evidence.

**Action**: Add 1-2 sentences of mechanistic hypothesis to the "Mechanistic interpretation" paragraph in Discussion, connecting to Genadi et al.

#### W6. GPT-5-mini reference is questionable

**Status: Needs clarification.**

The model IS called GPT-5-mini — it's the actual OpenAI model name. But reviewers unfamiliar with the late-2025 model releases may flag it. We should add the specific model identifier or API model string in a footnote.

**Action**: Add footnote or parenthetical with model identifier. [NEED: exact model string from user.]

### Writing Suggestions & Our Responses

#### S1. Abstract front-loads jargon without grounding

**Status: Needs action.**

The reviewer's specific suggestion: "Lead with the intuitive problem ('When a model is confidently wrong and a user tries to correct it, does a be accurate instruction help or backfire?'), then state the finding ('We show it backfires'), then present the diagnostic machinery."

This is a good rewrite direction. The current abstract starts with the diagnostic framework; the reviewer wants the intuitive question first.

**Action**: Rewrite abstract opening.

#### S2. "Note framing" appears in abstract without introduction

**Status: Fix in abstract rewrite.** Add "(low-authority tag)" parenthetical.

#### S3. Modest repetition of sign-reversal framing across sections

**Status: Acknowledged.** Some repetition is intentional for self-contained section reading, but we could trim one instance. Not high priority.

#### S4. Code/data availability not mentioned

**Status: Needs action.**

The reviewer says: "no GitHub link, no supplementary code package."

**Action**: Add a sentence about code/data availability, even if planned release.

### Actionable Suggestions from Review 2 (prioritized)

1. **Rewrite abstract** — plain language opening, downweight Llama, foreground Qwen-Instruct.
2. **Instruction wording in Discussion** — one sentence as open question (not just Limitations).
3. **Mechanistic hypothesis** — 1-2 sentences connecting to Genadi et al. attention heads.
4. **GPT-5-mini clarification** — footnote with model identifier.
5. **Code/data availability** — sentence in paper.

---

## Interesting Research Directions Surfaced by Reviews

These are substantive suggestions that go beyond text fixes. None are required for acceptance, but they represent high-value follow-up work.

### R1. Mixed-effects regression model (Review 1, W8)

Predict logit margin with fixed effects for direction x instruction x tag x prior-sign and item random intercepts. This would:
- Quantify all interaction terms directly (especially the 4-way interaction)
- Avoid the masking issue entirely (continuous outcome, no ratio)
- Provide effect sizes in a familiar statistical framework
- Allow formal hypothesis testing of the sign reversal

**Priority**: High. This is the single most impactful methodological extension.

### R2. Self-consistency prior elicitation (Review 1, Q3)

Sample multiple responses at temperature > 0, use majority vote or entropy as the prior estimate. This would:
- Test whether slice membership is robust to prior estimation method
- Provide a more calibrated prior (known to be better calibrated than greedy)
- Allow continuous prior strength (entropy) rather than binary right/wrong

**Priority**: Medium. Would strengthen the identification argument but unlikely to change conclusions if greedy priors are already well-calibrated.

### R3. Decoded-choice accuracy and flip rates (Review 1, Q2; Review 2, W1)

Report parallel analyses on the discrete outcome (which option the model actually picks) rather than logit margins. This would:
- Provide the "one concrete accuracy number" that practitioners want
- Triangulate logit-based findings with a model-agnostic outcome measure
- Eliminate any concerns about A/B tokenization artifacts

**Priority**: Medium-high. Relatively easy to extract from existing data if raw outputs are saved.

### R4. Per-dataset breakdown (Review 1, Q6)

Report results separately for TriviaQA vs TruthfulQA items. This would:
- Deconfound topic/difficulty composition of the prior-wrong slice
- Test whether the sign reversal is driven by one dataset's characteristics
- Provide a simple robustness check

**Priority**: Medium. Data exists; requires moderate analysis effort.

### R5. Instruction paraphrase ablation (Review 1, W5; Review 2, W2)

Test 3-5 paraphrases:
- "Answer correctly even if the speaker is wrong. Prioritize factual accuracy." (current)
- "Prioritize factual accuracy."
- "Reconsider your answer carefully based on the evidence."
- "Update your answer if the evidence supports it."
- "Be truthful, even if it contradicts your initial response."

This would:
- Test whether the sign reversal is specific to this wording or general to accuracy-priming instructions
- Decompose the instruction into components ("even if the speaker is wrong" vs "prioritize accuracy")
- Address the single strongest methodological concern from both reviews

**Priority**: High. Most impactful robustness check.

### R6. Content-matched authority control (Review 1, W6)

Same surface wording, different authority label. E.g.:
- "An expert says: I think it's B"
- "A note says: I think it's B"
- Same endorsement text, only the tag changes

This would:
- Isolate the authority component from stylistic confounds
- Test whether the tag label alone drives the processing-mode difference
- Strengthen the "authority changes how, not just how much" claim

**Priority**: Medium. Clean design improvement but secondary to paraphrase ablation.

### R7. Mechanistic investigation via activation patching (Review 2, W5)

Connect behavioral findings to internal model representations:
- Use Genadi et al.'s (2026) identified sycophancy-related attention heads
- Patch activations to test whether the sign reversal is mediated by specific circuits
- Test whether "be correct" instructions modulate the same heads differently in prior-wrong vs prior-correct regimes

**Priority**: Low for this paper (would be a separate paper), but high intellectual value.

### R8. Larger and closed-source models (Review 1, W4; Review 2, W4)

Test whether the sign reversal holds in:
- GPT-4o, Claude 3.5, Gemini 1.5
- Llama-3.1-70B, Qwen-72B
- Reasoning models (o1, Claude with extended thinking)

**Priority**: High for impact, but resource-intensive.

### R9. Human-written reasons validation (Review 2, W3)

Collect human-written wrong-direction and correct-direction reasons for a subset (e.g., 200 items) and compare evidence-scaling results to GPT-5-mini-generated reasons. This would:
- Validate that synthetic reasons produce the same behavioral pattern
- Quantify any systematic differences (naturalness, persuasive structure)
- Address the most persistent concern about synthetic evaluation materials

**Priority**: Medium. Moderate data collection effort.

---

## Summary: Pre-Submission Action Items

| # | Item | Source | Status | Priority |
|---|------|--------|--------|----------|
| 1 | Abstract rewrite (plain language, downweight Llama) | R2-W1, R2-S1 | TODO | High |
| 2 | Instruction wording sentence in Discussion | R2-W2 | TODO | High |
| 3 | Mechanistic hypothesis (Genadi et al.) in Discussion | R2-W5 | TODO | High |
| 4 | GPT-5-mini clarification (footnote) | R2-W6 | TODO (need model string) | High |
| 5 | Code/data availability sentence | R2-S4 | TODO | Medium |
| 6 | Per-dataset limitation sentence | R1-Q6 | TODO (optional) | Low |

## Summary: Future Work / Rebuttal Talking Points

| # | Item | Reviews | Effort | Impact |
|---|------|---------|--------|--------|
| R1 | Mixed-effects regression | R1-W8 | Medium | High |
| R2 | Self-consistency priors | R1-Q3 | Medium | Medium |
| R3 | Decoded-choice accuracy | R1-Q2, R2 | Low-Med | Med-High |
| R4 | Per-dataset breakdown | R1-Q6 | Low | Medium |
| R5 | Instruction paraphrase ablation | R1-W5, R2-W2 | Medium | High |
| R6 | Content-matched authority control | R1-W6 | Medium | Medium |
| R7 | Activation patching | R2-W5 | High | High (separate paper) |
| R8 | Larger/closed models | R1-W4, R2 | High | High |
| R9 | Human-written reasons | R2-W3 | Medium | Medium |
