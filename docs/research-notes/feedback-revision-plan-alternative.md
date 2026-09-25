# Feedback revision plan

Status: running notes from me, the author. Paras's feedback is in its own section, separated from the Claude/GPT feedback. I will keep adding new comments here as they come in.

## Core diagnosis

The feedback is consistent across reviewers. The paper is promising, but the current framing is too broad and the writing is too defensive and too AI-sounding. The safest claim is not "compliance is not one knob." The safer and stronger claim is:

> A verified-source endorsement and direct user pressure can produce the same wrong answer, but in the two models where I run the full intervention, the two pressures can be attenuated by removing different directions in the residual stream. Mitigation should therefore depend on who is doing the pushing.

My next revision should make this narrower claim obvious from the title, the abstract, the intro, and the order of the main results.

---

# Paras feedback

This section combines Paras's Overleaf comments and his longer Claude red-team feedback, including the follow-up explanations about the affect deconfound and residualization.

## Paras: framing and writing

### 1. The title `Compliance Is Not One Knob` overclaims. Paras suggested something like "Compliance in LLMs is mediated through a cluster of knobs," but the deeper point is that the current title is too broad.

What I think:
Fair point, and I agree. The current title sounds like I am claiming priority over the entire "not one thing" space. I will rename the paper around the source-vs-user split. My current candidates:
- "Authority Is Not User Sycophancy"
- "Wrong-Answer Compliance Routes by Cue Family"
- "Authority and User Pressure Drive Wrong Answers Through Different Directions"
I am leaning toward the third one because it is the most concrete.

### 2. The abstract has many em dashes and sounds AI-generated.

What I think:
Guilty. I will rewrite the abstract from scratch. Short sentences. No em dashes. I will start with what happens and the concrete causal test, not with broad framing language.

### 3. The paper often sounds like an AI-generated report and is hard to keep reading.

What I think:
This stings, but it is right. After the technical revisions are in, I will do a full human-writing pass on the whole paper. I will cut phrases like "honest disclosure," "not cosmetic," "we disclose rather than hide," "the cleanest single piece of evidence," and "rules out." I want the numbers and the scope to do the work, not the rhetoric.

### 4. Technical terms appear too early: `assistant-persona`, `user-sycophancy`, `assistant-axis`, `lexical-affect`.

What I think:
This was me using shorthand for myself. In the abstract and the intro, I will switch to plain language:
- "the model's default assistant role" instead of assistant-persona
- "pressure from what the user said they believe" instead of user-sycophancy
- "confident or emotional wording" instead of lexical-affect
The technical names can stay in Methods, but only after I define them.

### 5. The sentence about source-authority and user-sycophancy doubly dissociating is too complicated.

What I think:
This is one of the worst sentences in the paper. I will break it into two. I will skip "doubly dissociate" in the abstract. I will say: in Qwen3.5 and GPT-OSS, removing the source direction mainly reduces wrong answers caused by source endorsement, and removing the user direction mainly reduces wrong answers caused by user pressure.

### 6. `Compliance mitigation is therefore a portfolio problem, not a single-knob problem` should be the headline and much earlier in the abstract.

What I think:
I agree. The mitigation implication should not be buried at the end. The abstract should pivot on it: mitigation has to depend on who is pushing the wrong answer, because a vector trained against user pressure does not necessarily fix source-endorsement failures.

### 7. `Pareto-competitive mitigation, with an honest boundary case` sounds AI-generated.

What I think:
Yes, that phrase has to go. I will replace it with the actual numbers. Source-authority removal reduces wrong-source compliance with small MMLU-Pro and GSM8K changes in four of the five families. Gemma-4 is largely unresponsive to the linear intervention I tested.

### 8. `This rules out a binary any-note flips the model...` sounds AI-generated.

What I think:
Same problem. I will rewrite it simply: stronger authority wording produces larger flips, so the effect is not just from adding a note.

### 9. `Assistant-axis and affect controls do not explain the split` is too jargony.

What I think:
I will say plainly: controls for the model's default assistant role and for confident or emotional wording do not account for the pattern. If I cannot run the causal wording control, I will say the wording control is geometric only and call out that limitation directly.

## Paras: source/user and mechanism claims

### 10. The paper should not infer fully separable mechanisms from a double dissociation.

What I think:
Agreed. I over-reached. I will stop saying "mechanistically separable." I will say what I actually have: removing one direction attenuates one kind of pressure more than the other, but neither removal is clean. I will add one sentence acknowledging that this kind of result, in a network where everything is mixed together, does not prove there are two separate circuits.

### 11. The off-diagonal effects are not small.

What I think:
I hid these and I should not have. I will report them in the body. In Qwen, removing the source direction also drops user-wrong by 17 pp, and removing the user direction also drops source-wrong by 10.8 pp. The result is a partial split, not a clean one.

### 12. `cos(source,user) ≈ 0.20` is not near-orthogonal in high dimensions.

What I think:
I knew this and used "near-orthogonal" anyway because it sounded better. I will stop. I will call 0.20 moderate overlap. If I keep cosine numbers in the paper, I will compare them to what two random vectors of the same dimension would give.

### 13. Qwen is behaviorally ambiguous in the Wang source/user probe but used for the mechanism split.

What I think:
This is a strength if I frame it right. I will not claim that the behavior pattern predicts what the intervention will show. Qwen is the example: similar behavior across the two pressures does not mean a shared internal cause. Even where the behavior looks mixed, the linear intervention can still pull the two apart. I should make this argument out loud instead of looking defensive.

### 14. OLMo-2 has a source/user ratio of 0.99 and is skipped, but that is the interesting hard case.

What I think:
He is right. If I can run the OLMo-2 source/user split in time, I will, even if it ends up messy or null. If I cannot, I will explicitly say the source/user causal test is scoped to Qwen3.5 and GPT-OSS and does not establish a universal pattern.

## Paras: Gemma-4

### 15. Gemma-4 as a `controllability boundary` sounds handwavy unless we can say what Gemma does differently.

What I think:
It is handwavy. I will cut the phrase. I will say what I actually saw: Gemma-4 has strong behavioral source-authority compliance, but forward steering, PIQA transfer, SYCON transfer, and mitigation are all weak at the slots I tested. I will not speculate about why unless I run new diagnostics.

### 16. Gemma fails on every causal probe, so "boundary case" rhetoric is doing too much work.

What I think:
Agreed. I will leave Gemma out of the source/user separability claim. It is an unresponsive model under the single-direction linear intervention I tested, and that is all I should say.

## Paras: CAA baseline

### 17. The CAA comparison is structurally unfair and oversold.

What I think:
He is right. Saying I "outperform CAA" is a stretch, because I am running their vector against a kind of pressure it was not built for. The honest version: a CAA vector trained for user agreement does not carry over to source-authority compliance. If I want a real head-to-head, I would have to fit a source-matched CAA vector. Otherwise, I should just drop the comparison.

## Paras: authority-vector construction and spans

### 18. The authority-vector paragraph needs to explain why we add both C and W components.

What I think:
I never explained this well, and a fresh reader is right to be confused. Using both correct-source and wrong-source cues targets source attribution in general, not wrongness alone. That is exactly why the correct-source preservation check matters. After authority removal, I need to ask: does the model still benefit from a correct source note, or did I damage source processing in general? I should put this in Methods.

### 19. `Semantically aligned positions` is unclear.

What I think:
This phrase only makes sense to me because I wrote the code. I will replace it with plain language. These are token spans that play the same role across the three prompt versions: the answer span in the neutral prompt, and the endorsed-answer or endorsement span in the correct-source and wrong-source prompts.

### 20. The answer span / endorsement span details should go in the appendix.

What I think:
Easy fix. I will add an appendix table or box with one concrete prompt example, showing the answer span, the endorsement start, the endorsement end, the endorsed-answer token, and the endorsement mean side by side.

### 21. The assistant-persona vector should be explained for readers unfamiliar with Assistant Axis.

What I think:
I will add one sentence in Methods and a short appendix note. It is a direction contrasting the model's default assistant behavior with role-play or persona behavior, following the Assistant Axis paper.

## Paras: multiple comparisons and statistics

### 22. Nested stratified cross-validation over many layers and positions raises multiple-comparisons concerns.

What I think:
I was scanning a lot of cells and reporting the best ones, and a reviewer was always going to flag this. I will add a random baseline. I will shuffle the authority vector coordinates while keeping the norm, or use random unit vectors, then run the same scan and compare. I will report where my actual best cell falls relative to that baseline.

### 23. Paras specifically suggested shuffling authority-vector values while keeping the norm and repeating the process.

What I think:
This is the version I will implement first because it directly answers the Overleaf comment. I will describe the procedure in the appendix.

### 24. Many experiments are N=1 runs. Ideally run multiple times and report CIs.

What I think:
I cannot afford full reruns. My compromise: add Wilson intervals for the binomial rates I already have, rerun a small number of headline cells if time allows, and for everything else say in the caption that the number is from a single greedy decode. I will stop reading meaning into small differences.

### 25. Authority-gradient plot needs CIs / repeated runs.

What I think:
I will add Wilson intervals at every point. If I can sample a few seeds, I will, but the intervals are the floor. The caption should give n and say whether each point is from one greedy decode.

### 26. SYCON results are non-monotone and best-cell selected.

What I think:
I was leaning too hard on small numbers. I will show the full sweep and stop highlighting +3 / +7 / +8 pp shifts as if they meant something on their own. I want to keep GPT-OSS +53 pp as the headline because it is striking, but I will flag that I picked the cell after looking, unless I add a random-direction comparison on the same grid.

### 27. Transfer uses swept layer and alpha.

What I think:
I will say everywhere in the paper that I tried many settings and reported the best, rather than predicted one. I did not freeze a layer and a strength on a validation set and then test on a held-out set, so I should not write as if my extraction predicted the best intervention setting.

## Paras: causal affect deconfound

### 28. The missing causal affect control is the highest-leverage fix.

What I think:
This is the one that worries me the most. The cosine numbers are descriptive only, so emotional or confident wording is still a possible explanation for the effect. If compute allows, I will run the full set of removals.

### 29. Concrete affect experiment.

What I think:
The plan: run on W prompts at alpha=1 for Qwen3.5, GPT-OSS, OLMo-2, and OLMo-3.1. The conditions: no intervention, remove valence, remove arousal, remove the valence+arousal subspace, remove raw authority, and remove authority after first taking out the valence+arousal part. If removing wording on its own does very little and authority-minus-wording still works, the wording explanation gets a lot weaker.

### 30. Why cosine is not enough.

What I think:
Right. Cosine just says the directions point in different geometric places. It does not say that taking out the wording component leaves behavior unchanged. Only the removal experiment can answer that.

## Paras: residualization explanation

### 31. The residualization formula needs a clear walkthrough.

What I think:
I can fix this in an afternoon. I will add a short appendix explanation. For a unit confound c, the residualized target is normalize(v - (v·c)c). For non-unit c, use v - ((v·c)/(c·c))c. For valence and arousal together, subtract the projection onto the 2D wording subspace. I will keep this concise.

## Paras: other methodological gaps

### 32. Correct-source behavior after projection removal is missing.

What I think:
This is the experiment that gates the mitigation story. I have to run the correct-source condition after authority removal, and I have to be honest about what I find. If correct-source accuracy drops, the intervention is removing source processing in general, and I will say so. If correct-source accuracy is preserved, the intervention is more specifically attenuating wrong-source compliance while keeping the benefit of a correct source. Either result is informative; what is not okay is leaving this unmeasured.

### 33. Parser-marker recoveries could weaken confidence.

What I think:
I caught some of these myself and now I want to be public about it. I will split the recovery examples into two buckets: ones where the model wrote the correct answer in plain text, and ones where the parser only picked it up from a tag-like marker. Only the plain-text recoveries belong in headline boxes. I will add a small audit table showing how often each kind shows up.

### 34. SYCON judge uses Gemini-3.1-flash-lite while Gemini-3.1-Pro is evaluated behaviorally.

What I think:
Awkward, but fixable. If I can audit a sample of SYCON judgments by hand in the time I have, I will, and I will report agreement and Cohen's kappa. If I cannot, I will list the judge mismatch as a limitation and move on.

### 35. OLMo-3.1 PIQA parse rate varies with alpha.

What I think:
Reasonable concern. I will report parse rate as a function of alpha, look at a few unparsed outputs to see what the model is doing, and make clear in the caption that the OLMo-3.1 PIQA matched flips are computed only on the items where the answer could be parsed.

### 36. Extraction N is small for OLMo-2 and OLMo-3.1.

What I think:
True, and I should not pretend otherwise. I will report split-half stability of the extracted directions, or bootstrap cosine between halves. Where N is small enough that the direction itself may not be stable, I will flag it instead of glossing over it.

### 37. Item independence between extraction and evaluation is unclear.

What I think:
I am not sure of every case myself, which is its own problem. I will go through and say which item sets are shared between extraction and evaluation, and which are held out. Where there is overlap, I will not write the result as if it generalized to held-out items.

## Paras: final writing guidance

### 38. Remove defensive / honest-disclosure rhetoric.

What I think:
This is fair. I lean on phrases like "honest, not cosmetic," "we disclose rather than hide," and "not introduced as a convenient escape hatch" when I am nervous about a number. I will cut them. If a number is honest, the number itself shows that.

### 39. Appendix should not repeat the main-paper story.

What I think:
The appendix grew a story tail because I kept rewriting the main text and the older paragraphs ended up in the back. I will use the appendix for what it is for: full tables, ablations, prompts, examples, nulls, and grids. The story paragraphs that already appear in the main text get cut.

---

# Claude and GPT feedback

This section combines `claude-feedback.md` and `gpt-feedback.md`, excluding Paras's separate red-team feedback above.

## Claude/GPT: central framing

### 40. The paper is NeurIPS-main plausible and maybe spotlight-potential, but the current framing is too broad.

What I think:
I will take the encouragement and tighten the scope. The paper is about source-authority pressure as one specific kind of compliance pressure, with a causal test in two models. I will not sell it as a broad theory of all compliance.

### 41. The surprising claim is not that sycophancy is plural. The surprising claim is that the same wrong answer can route through different cue-family directions.

What I think:
This is the cleanest framing I have heard for the paper, and I am going to commit to it. The story I want to tell is: same wrong answer, different person endorsing it, different removable direction. I will reorganize tables and examples around that.

### 42. Move center of gravity away from behavioral flips and toward source/user causal split.

What I think:
I have been thinking about this for a while and I agree. The behavioral flip plot motivates the problem, but the punchline is the source/user removal table. If the page budget allows it, I will move the removal table earlier and make it the headline figure.

### 43. The best rewritten research question asks whether wrong-answer compliance cues share a residual-stream compliance direction or instantiate cue-family-specific control directions.

What I think:
I will use this as the intro question, but I will say it in plainer language. Something like: when the model gives a wrong answer because a verified source endorsed it, does the same direction in the residual stream drive the wrong answer when a user endorses it instead, or do those two pressures live on different directions? Then I include the causal test and the mitigation implication in one paragraph.

## Claude/GPT: prior work

### 44. Vennemeyer already shows sycophancy is not one thing.

What I think:
Yes, and I should stop hiding from them. I will use Vennemeyer as the closest prior work and say their axis is the kind of behavior the model produces, while my axis is who is making the wrong claim.

### 45. Joad should be engaged as a foil.

What I think:
Same as the Paras section. I will treat Joad as a complementary finding, not someone I am beating. They show that vectors that look different in geometry can still drive the same behavior under linear intervention. I show one case in two models where the linear intervention does pull the two pressures apart. That is a different setting, not a contradiction.

### 46. Cross-model asymmetry is underused: Gemini-3.1-Pro near 1% vs Grok-4.20 near 88%.

What I think:
This is an interesting observation I am sitting on. I will add a short paragraph in Discussion or Future Work, but I will keep it speculative because I do not have an internal-states view of the closed APIs.

## Claude/GPT: evidence scope

### 47. The decisive source/user split is only in Qwen3.5 and GPT-OSS.

What I think:
Same as the Paras section. I will scope every separability claim to those two models. I will add a third model if I can.

### 48. The affect control is geometric-only.

What I think:
Same as the Paras section. I will run the causal wording control if compute allows, or keep the wording explanation as an open limitation.

### 49. Transfer uses swept layer/scale.

What I think:
I will call it a swept transfer diagnostic everywhere, unless I add a validation/test split for the intervention settings.

### 50. Parser/judge audit is needed.

What I think:
I will separate parser-marker recoveries from plain-text recoveries, and human-audit a sample of SYCON judgments if I can.

### 51. Generic sycophancy baseline should be stronger.

What I think:
I will avoid broad CAA claims unless I either fit a source-matched CAA vector or use multiple CAA contrast sets. Otherwise the comparison is not fair.

## Claude/GPT: fruitfulness

### 52. The paper needs forward predictions about other cue families.

What I think:
I am tempted to add a new kind of pressure to the paper. I am going to resist, because anything I add now will be half-baked. Instead, I will add a short future-work paragraph about applying the same recipe (extract a direction, take out the part shared with a confound, project it out) to system prompts, retrieved documents, expert-persona cues, or numeric-confidence cues.

### 53. RAG / authority-citation jailbreaks are natural downstream applications.

What I think:
If I already have a clean small experiment on disk, I will add it. Otherwise it is future work, and I will not imply I have demonstrated a defense.

## Claude/GPT: possible upgrades

### 54. Add a third model to source/user 2x2.

What I think:
Worth doing if compute and time allow. I will prioritize OLMo-3.1 or OLMo-2.

### 55. Freeze layer/alpha selection on validation, then report held-out test.

What I think:
This would be the clean thing to do, but I am not sure I can fit it in. If I cannot, I will label the best-cell results honestly as best-of-sweep and add the random-baseline comparison.

### 56. Make generic-sycophancy baseline harder to beat.

What I think:
Only if I have time. Otherwise I will reframe the baseline as a non-transfer result for a user-agreement vector and stop framing it as a head-to-head.

---

# Immediate work plan

The order I am going to do this in:

1. Rewrite the title and the abstract around the source-vs-user split.
2. Rewrite the intro so the source/user removal result is the central contribution.
3. Rewrite Related Work around Vennemeyer (behavioral split) and Joad (linear intervention can collapse onto the same behavior).
4. Cut AI-sounding and defensive prose across the paper.
5. Replace strong mechanism language with partial-split language and partially selective linear control.
6. Report the source/user off-diagonal effects in the body.
7. Reframe CAA as a non-transfer result, not method superiority.
8. Cut the Gemma boundary rhetoric and leave Gemma out of the separability claim.
9. Clarify the authority-vector construction, residualization, and matched token spans.
10. Add the layer-position scan random baseline, or soften the best-cell claims if I cannot run it.
11. Add Wilson intervals and single-run caveats where I cannot afford reruns.
12. Run the correct-source preservation check and the causal wording control if possible.

# Experiment shortlist

Things I have to do for the current claims to hold up:
1. Correct-source preservation after authority removal. This decides whether the mitigation claim is about wrong-authority specifically or about source processing in general.
2. Causal wording (valence/arousal) deconfound on W prompts at alpha=1 for Qwen3.5, GPT-OSS, OLMo-2, and OLMo-3.1.
3. Coordinate-shuffled or random-vector layer-position scan baseline.
4. Wilson intervals for the authority-gradient plot and the headline behavioral rates.
5. Plain-text vs parser-marker recovery audit.

Things I would like to do, but that go in the appendix at most:
1. OLMo-2 or OLMo-3.1 source/user 2x2.
2. SYCON random-direction comparison on the same grid.
3. SYCON human audit of a sample with agreement and Cohen's kappa.
4. Split-half direction stability for the extracted directions.

Things I am leaving for the next paper unless I already have a clean version on disk:
1. RAG / authority-citation jailbreak defense.
2. System-prompt or retrieved-document pressure experiments.
