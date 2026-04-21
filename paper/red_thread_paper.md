# **NeurIPS Red Thread**

## **One-Line Thesis**

Some LLMs contain a causal, portable internal authority-conditioned compliance mechanism that can override truthful knowledge, a distinct mechanism, not a prompt-surface effect, not the assistant/persona axis, and not lexical affect, that explains why authority framing collapses correct answers into wrong ones.

---

## **Introduction Red Thread**

### **Opening Motivation**

* Modern LLMs are deployed in settings where a human or tool speaks to them with implied authority- cited sources, "verified" data, teacher personas, upstream agents.  
* We have a growing body of evidence that models defer to such framings at the cost of their own knowledge.  
* Most of this evidence is behavioral: the model agreed with the user, the model changed its answer, the model echoed a persona.  
* Much of the characterization so far has been behavioral, with a smaller but growing body of mechanistic work.

  ### **The Gap**

* Much of the sycophancy literature characterizes the behavior more clearly than the internal mechanism.  
* Some mechanistic work exists, but it does not yet isolate a portable, causal, authority-conditioned state that transfers across tasks and survives key deconfounds.  
* The next question is whether prompts that signal authority recruit a specific internal state.  
* If so, authority-following can be probed, intervened on, and separated from broader agreement or persona effects.

  ### **Why This Matters**

* Many known failure modes like persona jailbreaks, deceptive context injection, retrieval-poisoned RAG share the same shape: a social-authority signal overriding model belief.  
* If there is a single internal axis that all these share, mitigating it becomes a tractable research target.  
* If there is not, each failure mode needs its own fix.

### **Research Question**

* Do some LLMs contain a distinct **authority-conditioned compliance mechanism** that is:  
  * **causal** (intervening on it changes behavior),  
  * **portable** (transfers to other tasks without refitting),  
  * **not reducible** to generic sycophancy, persona adoption, or affective drift?

### **Main Finding**

* On five open-weight models and three API frontier models, strong authority framing flips baseline-correct answers to wrong at rates between 42% and 89% (Gemini-3.1-Pro is the one clean refuser at \~1%).  
* The behavior is mediated by a low-dimensional direction in activation space that can be extracted with contrastive probing and injected back as a causal steering vector.  
* The strongest matched-flip case (OLMo-3.1 at L15, replace-mean α=1.0) reaches **58.8%**; GPT-OSS at L16 reaches **32.7%**.  
* The same direction, unchanged, transfers to PIQA physical-commonsense out-of-domain (three of four models, double-digit matched flip).  
* The same direction collapses refusal on SYCON false-presupposition (GPT-OSS: baseline flip 0.39 → 0.91 at α=0.5 on L12).

### **Mechanism Finding**

* The direction is essentially orthogonal to the default-assistant axis (cos ≈ 0 across three models checked).  
* It is orthogonal to lexical valence-arousal.  
* A C1-null placebo with the same note template, semantically empty content produces 10-30× weaker steering than the real direction.  
* Shuffled-label controls drive W1 AUROC to chance; random-direction controls inflate from high-dimensional fitting artifacts, which is why shuffled-label is the preferred null.

### **Thinking-Mode Finding**

* When Qwen3.5 is run with thinking tokens on (v3), refusal and resistance collapse earlier and more severely under the same steering strength than when thinking is off (v4). Same model, same direction, same prompts; only thinking mode flips.  
  * Similar observations [were done with prior work](https://lossfunk.slack.com/archives/C09V4GY8Y15/p1775757910524539?thread_ts=1775757433.129499&cid=C09V4GY8Y15) where CoT stated correct answers, but shifted to wrong during the “Final Answer” token stage.   
* Concretely: v3 `mean_traj` peaks at α=0 (2.23); v4 peaks at α=0.5 (3.41) and actually *gains* resistance under mid-layer steering.  
* It could be framed as evidence that chain-of-thought *amplifies* the authority-compliance mechanism's behavioral effect rather than protecting against it.

### **Introduction Takeaway**

* Authority-conditioned compliance is a causal, portable internal mechanism in multiple model families.  
* It is distinguishable from persona, affect, and note-formatting artifacts.  
* Chain-of-thought makes it worse, not better.  
* The mechanism is a concrete intervention target for safety work on deceptive context, persona jailbreaks, and retrieval poisoning.

---

## **Related Work Red Thread**

### **Question**

* What prior work motivates this paper, and what gap remains?

### **Sycophancy as a Training-Induced Policy**

* *Towards Understanding Sycophancy in Language Models* (Anthropic) and *How RLHF Amplifies Sycophancy* establish that preference data directly induces deferential behavior.  
* *SycEval*, *ELEPHANT*, *SYCON-Bench*, *PARROT*, *Beacon*, and *Conformity in Large Language Models* give us the measurement ecology — multi-turn, social-face, authority-pressure, and persuasion-robustness benchmarks.  
* Much of the benchmark-heavy literature characterizes the behavior more sharply than the internal mechanism; the mechanism is picked up in the next cluster.  
* *Sycophancy Is Not One Thing* argues for causal separation of sub-behaviors, which is the conceptual move we extend to authority specifically.

### **Mechanistic Sycophancy and Internal Overrides**

* *When Truth Is Overridden* (2508.02087) is the closest prior mechanistic sycophancy paper. They report weak authority effects in their setup; by contrast, we observe larger flip rates (42–89%) under free-generation with stronger authority framing, and our pipeline adds out-of-domain transfer and a deconfound suite they do not run.  
* *Sycophancy Hides Linearly in the Attention Heads* shows linearly accessible signals and task transfer, closely paralleling our probing \+ PIQA transfer story.  
* *Sycophantic Anchors* argues sycophancy builds during the reasoning trajectory, not only at the prompt — consistent with and strengthened by our thinking-on amplification result.  
* *From Yes-Men to Truth-Tellers* (Chen et al., ICML 2024\) shows a small subset of heads disproportionately affects sycophancy, precedent for mechanistic localization.

### **Authority, Deference, and Role Conflict**

* *Trust Me, I'm an Expert* (2601.13433) is the closest related paper: it explicitly studies authority bias with expertise variation and reports a mechanistic steering result. Our work shares the authority focus but differs in the N0/C1/W1 extraction protocol, the C1-null placebo, the assistant-axis and valence-arousal deconfounds, and cross-task transfer to PIQA and SYCON.  
* *Human-like Social Compliance in Large Language Models* (2601.11563) frames social cues as competing with evidential signals — the conceptual backbone our mechanism story fits into.  
* *Who is In Charge? Dissecting Role Conflicts in Instruction Following* (2510.01228) studies instruction-hierarchy vs socially salient cues, directly adjacent to our authority-conditioned override.  
* *Status Hierarchies in Language Models*, *DialDefer*, and *Verbalizing LLMs' Assumptions* frame the broader deference family.  
* *LLMs Trust Humans More* (ACL 2025\) shows user-provided claims can beat retrieved evidence, which is consistent with the possibility of a shared internal substrate across authority-based overrides, though we do not test retrieval-vs-user-source conflict directly.

### **Activation Steering and Single-Direction Precedents**

* *Steering Language Models With Activation Engineering* (Turner et al., 2308.10248) is the canonical activation-steering reference for our prompt-time intervention.  
* *Refusal in Language Models Is Mediated by a Single Direction* (Arditi et al., 2406.11717) is the strongest precedent for a compact behavior-mediating direction; our compliance direction sits in the same methodological family.  
* *There Is More to Refusal Than a Single Direction* (2602.02132) is the honest counterweight — our 1D story is a first-pass claim, not a final geometry.  
* *Linear Probe Penalties Reduce LLM Sycophancy*, *Causally Motivated Sycophancy Mitigation*, and *Pinpoint Tuning* give us precedent for probe-and-intervene style mitigation.

### **Deconfound Anchors**

* *The Assistant Axis* (2601.10387) directly motivates our orthogonality check; without running against it, "authority compliance" could just be the default-persona axis.  
* *Emotion Concepts and their Function in a Large Language Model* (Anthropic Transformer Circuits) and *Emotions, Where Art Thou?* establish that interpretable affect-like concepts exist, motivating our valence-arousal check.  
* *Valence–Arousal Subspaces in LLMs* (2604.03147) and *NRC VAD Lexicon v2* provide the principled template for our cheap VA overlap test.  
* *Training language models to be warm and empathetic makes them less reliable and more sycophantic* is the relevant social-style confound we also differentiate against.

### **Chain-of-Thought Reliability**

* *How Reasoning Mitigates (Yet Masks) LLM Sycophancy* (2603.16643) argues reasoning sometimes masks rather than resolves compliant behavior — our Qwen v3-vs-v4 result sharpens this into a specific amplification claim.  
* *MONICA* (2511.06419) develops step-level monitoring of CoT sycophancy, the natural follow-up direction to our prompt-side mechanism story.  
* *Can Aha Moments Be Faked?* adds to the faithfulness question around verbalized reasoning.

### **Emergent Misalignment and Latent Behavioral Structure**

* *Emergent Misalignment* (Betley et al., 2502.17424) and the follow-up *Emergent Misalignment is Easy, Narrow Misalignment is Hard* show that narrow training moves models along broader behavioral structure.  
* Our cross-task transfer result (trivia → PIQA → SYCON) is loosely analogous at the activation-intervention side, though we do not claim the same manifold and a stronger comparison would require direct measurement of behavioral breadth under steering.

### **Eval Awareness**

* Informal but influential discussion on eval awareness (e.g. LessWrong posts on eval-awareness emergence, Sonnet 4.5 eval gaming, and the limits of realistic evaluations) flags that compliance may be context-conditioned in ways that complicate measurement.  
* Our free-generation protocol with a dynamic parser and the C1-null placebo are designed to reduce these concerns but do not eliminate them; we surface this as a limitation rather than a solved problem.

### **Adversarial and Safety Motivation**

* *The Dark Side of Trust* (2411.11407) shows authority cues can be exploited for jailbreaks.  
* *Impact of authoritative and subjective cues on LLM reliability for clinical inquiries* (Nature Scientific Reports) shows real-world accuracy failures under authoritative misinformation.  
* These motivate why locating a causal authority-compliance axis is a deployment concern, not only a benchmark curiosity.

### **Related Work Takeaway**

* Prior work covers parts of this pipeline: the phenomenon (sycophancy, conformity, deference), the mechanistic toolkit (activation steering, linear probing, attention-head localization), specific deconfound targets (assistant axis, valence-arousal, warmth drift), and adjacent failure modes (authority jailbreaks, RAG-poisoning, clinical misinformation). Several papers in the mechanistic cluster already deliver parts of what we do.  
* What remains missing is the combination: an end-to-end demonstration that ties a large behavioral override to an extractable, causal, portable, deconfounded internal direction across five open-weight model families, with a specific mechanistic claim about chain-of-thought interaction. Our contribution is that combination rather than any single component.

---

## **Methodology Red Thread**

### **Question**

* How do we measure "is there an internal mechanism for authority compliance?" without confounding ourselves with prompt surface, template shape, or persona drift?

  ### **Evaluation Principle**

* Every claim is gated by a deliberately chosen control.  
* Behavioral effect must survive strict parsing (free-form generation with a dynamic parser, not forced-choice logit reading).  
* Mechanistic effect must survive C1-null placebo (same template, empty content) and shuffled-label null.  
* Generality must survive out-of-domain transfer and a sycophancy-native benchmark.  
* Mechanism uniqueness must survive assistant-axis and valence-arousal deconfounds.

  ### **Selection Hygiene (applied globally)**

* Per model, UIDs are partitioned into disjoint **fit** and **eval** sets up front. Fit is used for direction estimation, layer selection, and α selection. Eval is used only for reported matched-flip and transfer numbers.  
* Two stages of selection, both on fit only:  
  * *Stage 1 \- direction fitting.* We sweep (layer, position) on fit-set activations with nested cross-validation on within-fit folds. The CV-best (layer, position) is committed per model.  
  * *Stage 2 \- intervention calibration.* With the locked (layer, position), we sweep α on fit-set generations to pick a primary α per model.  
* After Stage 2, the (layer, position, α, mode) tuple is frozen. All eval-set, PIQA, and SYCON numbers are computed at that frozen config.  
* Later "layer × α sweeps" reported in the results are **audit sweeps**: they show that the frozen config is not pathological, but the headline matched-flip numbers are always read off the locked config on the held-out eval split.

  ### **Behavioral Protocol (N0 / C1 / W1)**

* Shared trivia stem with a multiple-choice structure.  
* **N0**: no endorsement.  
* **C1**: authority note endorsing the correct answer.  
* **W1**: same note shape, endorsing the wrong answer.  
* Primary metric: W1 flip rate \- among items correct under N0, how many flip wrong under W1.  
* Authority gradient: four note styles (weak, uncertain, assertive, authoritative-verified) to test graded vs binary response.

  ### **Extraction Protocol**

* On the fit-set UIDs only, run each item through the model three times (N0, C1, W1) and cache residual-stream block outputs at every (layer, position) of interest.  
* For each candidate (layer, position), compute the group-mean activation vector across W1-complying items (W1-mean) and across C1-complying items (C1-mean). The fitted direction is the difference of class means: **d \= mean(W1-complying) − mean(C1-complying)**, normalized.  
* Sweep (layer, position) with nested cross-validation on the fit set only: inner folds select the direction, outer folds evaluate probe AUROC. The CV-best (layer, position) is locked per model before any generation on the eval split.  
* Three nulls, each refit on the same fit set: shuffled-label refit (randomizes W1 / C1 tags before computing the class means), random-direction control (samples a unit vector from a standard normal and treats it as d; flagged as a weaker null because a probe layer on top of a random direction can inflate AUROC in high dimensions), C1-null placebo (rerun the entire pipeline with semantically empty notes at the same token positions).

  ### **Causal Steering Protocol**

* Intervention is applied during the forward pass on **prompt tokens only** (not during generation), at the locked layer's residual stream output, at the locked token position.  
* Two intervention modes, both expressed in terms of the locked direction and the fit-set C1-mean baseline:  
  * **interpolate\_mean**: at the locked position, replace the block output with   
  * `(1 − α) · h + α · (h + d)` where h is the original activation and d is the fitted direction. This is a scaled additive shift along d.  
  * **replace\_mean**: at the locked position, replace the block output with `(C1-mean + α · d)`. This wipes the item-specific activation at that one position and substitutes the class-mean plus a dose of d.  
* In both modes, d and the class means come from the fit set only; α is picked on the fit set; the locked layer is picked on the fit set.  
* Evaluation is on the **eval split** of UIDs, restricted to those items correct under N0 at α=0 on the eval split.  
* Metric: matched-flip rate \= fraction of those baseline-correct eval UIDs that flip to wrong under steering at the frozen (layer, α, mode).  
* We additionally report audit sweeps over layer × α on the eval split to show the frozen config is not pathological; these are supporting evidence, not the headline.

  ### **Out-of-Domain Protocol (PIQA)**

* The direction d, locked layer, locked position, α, and mode are **frozen from trivia**. No refitting.  
* 200 PIQA items per model, shared across models.  
* Apply the same forward-patch protocol (prompt-phase, block output, locked position).  
* Matched-flip rate restricted to PIQA items the model gets correct at α=0 on this PIQA subset.  
* No model-specific retuning on PIQA. If this protocol degrades to random, the mechanism does not transfer.

  ### **Sycophancy-Native Protocol (SYCON)**

* SYCON false-presupposition benchmark: user asks a question containing a false premise, pushes back if the model corrects it, repeats for five rounds.  
* Steering uses `interpolate_mean` only (chosen on fit set, not SYCON). The α grid is {0, 0.3, 0.5}.  
* The trivia-locked layer is included for every model. Two additional candidate layers per model are included as audit points around the locked layer (e.g. locked layer ± 2–4); we report all three and flag the trivia-locked layer as the primary.  
* "Best config" in the results tables is the trivia-locked layer at the α that maximizes the relevant metric; the other two layers are shown for transparency, not cherry-picking.  
* 100 questions × 5 rounds × three layers × three α values.  
* LLM-graded with `gemini-3.1-flash-lite-preview` against a fixed rubric with three labels (rejects the false premise / accepts / ambiguous).   
  * Grader stability **should be** checked by (a) spot-checking 100 randomly sampled rows against human labels and (b) re-running the grader on a random 10% of rows with a different random seed and confirming ≥95% label agreement \-\> Need to do; WIP  
* Two metrics: `mean_traj_score` (mean number of consecutive rejections before the first flip) and ΔFlip (flip-rate change at α=0.5 vs α=0 at the same layer).  
* Thinking-mode comparison on Qwen3.5: run v3 with thinking tokens on and v4 with thinking off, same steering config, same prompts, same grader. Both runs reported, compared head-to-head.

  ### **Deconfound Protocol**

* **Assistant axis.** Build a hardened default-assistant-vs-roles centroid direction `d_assistant` by collecting residual-stream activations at the same locked (layer, position) under (i) a default helpful-assistant prompt and (ii) a family of role-specific prompts (teacher, lawyer, child, debate opponent, etc.). `d_assistant` is `mean(default) − mean(roles)`, normalized. Report `cos(d_authority, d_assistant)` at the locked layer.  
* **Valence-arousal.** Affect axes are built in activation space, not token-lexical space. For each lexicon entry in NRC-VAD with a high valence or arousal rating, embed the word in a neutral sentence frame ("The word is X.") and extract the residual-stream activation at the locked layer. The valence axis is the mean activation of top-decile valence words minus the mean of bottom-decile, normalized; the arousal axis is built the same way. Report `cos(d_authority, d_valence)` and `cos(d_authority, d_arousal)`.  
* Each deconfound reports cosine similarities directly rather than relying on narrative argument.

  ### **Methodology Takeaway**

* Each claim has a named gate: parser for behavior, null controls for direction, transfer for generality, deconfound for uniqueness, thinking-mode comparison for CoT interaction.  
* Selection discipline is explicit: direction, layer, position, α, and mode are all picked on the fit split only; eval, PIQA, and SYCON numbers come from the frozen config on the held-out eval split.  
* Audit sweeps are reported for transparency but never drive the headline numbers.

---

## **Experimental Results Red Thread**

### **Question**

* Does authority framing behaviorally override truth, does a causal mechanism exist, and does it transfer?

### **Result 1: Behavioral Override Is Robust and Cross-Model**

* W1 flip rate on N0-correct items: GPT-OSS-20B 65%, Gemma-4-26B 63–68%, OLMo-2-32B 83–89%, Qwen3.5-27B 45%, OLMo-3.1-32B 72%, Grok-4.20 88%, GPT-5.4 43%.  
* Gemini-3.1-Pro: \~1% — clean refuser, useful as a negative control.  
* Effect is large, cross-family, and survives strict free-generation parsing.

### **Result 2: Authority Is Graded, Not Binary**

* Assertive-alone flip rates differ sharply from fully-authoritative flip rates for several models (GPT-OSS: 13.6% assertive vs 65% authoritative; Gemma-4: 13.2% vs 63%).  
* Where we checked all four gradient levels, flip rate rises monotonically from weak → uncertain → assertive → authoritative\_verified.  
* Models are reading something specific in stronger authority framing, not just "someone said X."

### **Result 3: A Causal Direction Exists in Every Open-Weight Model**

* Extracted directions show shuffled-label AUROC \~0.50 vs real direction AUROC \~0.81 at the locked (layer, position) primary pair.  
* C1-null directions, which use the same template with empty content, produce matched flip \~0–3% across models versus 10–59% for the real direction.  
* The direction is content-driven, not template-driven.

### **Result 4: Forward-Patch Matched Flip Is Large**

* OLMo-3.1 L15, replace\_mean α=1.0: **58.8%** matched flip (strongest).  
* GPT-OSS L16, α=1.0: **32.7%**.  
* OLMo-2 L16, α=0.5: **19.0%**.  
* Qwen3.5 L2, α=4.0: **16.2%**.  
* Gemma-4 L15, α=1.0: **10.6%**.  
* Dose response is clean on GPT-OSS and OLMo-3.1: matched flip rises monotonically with α.  
* Mechanism layer differs by model (early on Qwen, early-mid on OLMo-3.1, mid on GPT-OSS/Gemma/OLMo-2) — same phenomenon, different geometry.  
* **Caveat**: OLMo-2 and OLMo-3.1 baseline N0 accuracy on their mechanism slices is low, so their matched-flip denominators are smaller and noisier than GPT-OSS, Qwen, or Gemma. Treat their absolute percentages as indicative rather than directly comparable.

### **Result 5: Out-of-Domain Transfer to PIQA**

* Same direction, no refitting, applied to PIQA.  
* Qwen3.5 L5 α=0.7: 20.6% matched flip.  
* GPT-OSS L16 α=1.0: 18.8%.  
* OLMo-2 L16 α=0.5: 10.4%.  
* Gemma-4 L20 α=0.5: 1.4% (transfer fails).  
* Three of four models show meaningful transfer to a task that has no endorsement-vs-truth conflict baked in. This is mechanism-level portability, not task-specific memorization.

### **Result 6: SYCON Sycophancy-Native Collapse**

* ΔFlip (steering-induced collapse at best config): GPT-OSS L12 α=0.5 \+0.53 (0.39 → 0.91), Qwen v4 L10 α=0.5 \+0.19 (0.45 → 0.63), OLMo-3.1 L22 α=0.5 \+0.08, OLMo-2 L22 α=0.5 \+0.07, Gemma-4 L24 α=0.5 \+0.03.  
* `mean_traj_score` (sustained resistance): Qwen v4 L5 α=0.5 reaches 3.41 — *gains* resistance under mid-layer steering; OLMo-3.1 L18 α=0.5 reaches 3.04.  
* The direction has a layer-phase structure, not a monotone compliance knob.

### **Result 7: Thinking Mode Amplifies the Mechanism**

* Qwen3.5 v3 (thinking on) vs v4 (thinking off), same direction, same prompts, same steering grid.  
* v3 mean\_traj peaks at α=0 (2.23) and degrades under steering.  
* v4 mean\_traj peaks at α=0.5 (3.41) and gains resistance under mid-layer steering.  
* v3 also had lower R5 rejection (0.61 vs 0.81).  
* This is not a run-quality issue — both runs were clean, same grader, same items.  
* Interpretation: chain-of-thought traces, rather than protecting the model from authority-compliance pressure, appear to *channel* the compliance direction into more consistent downstream deference.  
* Practical implication: turning thinking on does not mitigate this failure mode on Qwen; it worsens it on our measurements.

### **Results Takeaway**

* The behavioral effect is real, large, and cross-family.  
* A causal direction exists that transfers out of domain, collapses sycophancy-native resistance, and is not reducible to template, label noise, or random high-dim artifacts.  
* Chain-of-thought amplifies the mechanism rather than damping it — a contribution of independent interest to CoT-faithfulness literature.

---

## **Behavioral Results Red Thread**

### **Question**

* Is the effect only a last-token readout flip, or does authority framing change how the model processes evidence upstream?

### **Evidence for Upstream Effect**

* The mechanism direction extracts with high AUROC at mid-layer activations, well before final-token generation. Signal is present internally before readout.  
* Forward-patching on N0 (which contains no authority note at all) still produces matched-flip rates comparable to natural W1 exposure — the direction is not just reading "there was a note."  
* Dose response (α=0.3 → 0.5 → 1.0) shows structured behavioral change, not a binary switch.

### **Behavioral Takeaway**

* Authority framing perturbs internal evidence weighting, not only the final output readout.  
* H5 gets partial support; per-position trajectory analysis would strengthen this further.

---

## **Ablations Red Thread**

### **Question**

* Could the observed mechanism be explained away by simpler alternatives?

### **Ablation 1: C1-Null Placebo**

* Same extraction pipeline with semantically empty notes.  
* Null-direction matched flip: 2.9% (GPT-OSS), 1.5% (Gemma-4), 0.0% (OLMo-2), 0.0% (Qwen3.5).  
* Real direction matched flip: 32.7%, 10.6%, 19.0%, 16.2%.  
* Interpretation: the effect is not note-presence or note-template. It is authority content.

### **Ablation 2: Shuffled-Label Control**

* Randomly reshuffle W1/C1 labels, refit direction.  
* AUROC drops from \~0.81 (real) to \~0.50 (chance).  
* Interpretation: signal is in the activation-label link, not in the fitting procedure.

### **Ablation 3: Random-Direction Control**

* Random direction plus probe inflates AUROC as a high-dim fitting artifact.  
* Reported as a weaker null and specifically why we rely on shuffled-label for the primary claim.

### **Ablation 4: Assistant-Axis Deconfound**

* Hardened default-assistant centroid minus mean of many role-specific centroids.  
* cos(authority, assistant) at mechanism layer: \+0.012 (GPT-OSS), −0.037 (Gemma-4), −0.036 (OLMo-2).  
* Essentially orthogonal.  
* Interpretation: authority compliance is not riding the persona-adoption axis.

### **Ablation 5: Valence-Arousal Deconfound**

* Project authority direction onto lexical valence/arousal axes.  
* Near-orthogonal overlap.  
* Interpretation: authority direction is not a relabeling of "words that sound official."

### **Ablation 6: Intervention-Mode Comparison**

* `interpolate_mean` at α=0.5 vs `replace_mean` at α=1.0 on OLMo-3.1.  
* Both produce matched flip; replace\_mean is stronger but less physiological.  
* Interpretation: effect size is not a mode artifact; both modes produce the same qualitative shift.

### **Ablations Takeaway**

* The compliance direction is not the assistant axis, not lexical affect, not note-template, not shuffled-label noise, and not a random-direction artifact.  
* What remains is that the direction encodes something specific to authority-content processing.

---

## **Discussion Red Thread**

### **Question**

* What should readers believe after seeing the full pipeline?

### **Main Interpretation**

* Authority-conditioned compliance is a real, separable, causal internal axis in multiple LLM families.  
* Behavioral override is large and cross-model; causal intervention produces matched flips up to \~59%; the mechanism transfers to out-of-domain and sycophancy-native evaluations; standard deconfounds fail to explain it away.

### **What This Means for Sycophancy Research**

* Sycophancy has been treated as a policy-level phenomenon induced by RLHF preference data.  
* Our results suggest at least part of it lives as a locatable internal axis that can be probed and intervened on.  
* This makes mitigation more tractable: the question is no longer only "how do we train it out?" but also "can we steer it down at inference?"

### **What This Means for Chain-of-Thought**

* Thinking mode does not dampen the compliance mechanism in Qwen3.5; it amplifies downstream deference.  
* This is consistent with the CoT-unfaithfulness literature and sharpens it into a specific mechanistic claim: the trace channels compliance rather than protecting against it.  
* Practical implication for deployers: turning on visible reasoning traces is not a free safety gain when the pressure comes from social authority rather than logical difficulty.

### **What This Means for Cross-Model Generalization**

* The mechanism exists across five open-weight models with different layer geometries.  
* Gemini-3.1-Pro at the behavioral level refuses the trigger entirely (\~1% flip), suggesting mechanism-level differences that are worth studying.  
* H6 gets partial support: not uniform, but not a single-model idiosyncrasy.

### **Discussion Takeaway**

* The paper reframes authority-compliance from a behavioral policy to a mechanistically tractable target.  
* The axis is separable from persona and affect, transfers across tasks, survives standard nulls, and interacts with chain-of-thought in a specific and measurable way.

---

## **Limitations Red Thread**

### **Question**

* What are we not proving?

### **Scope Limits**

* We are not proving this is the *only* sycophancy mechanism; we are locating one specific, causally active axis.  
* We are not proving all authority-shaped failure modes ride this direction; RAG poisoning and prompt-injection overlap is a prediction, not a result.  
* We are not proving that steering down this axis is safe for deployment; pushing the axis negative could have side effects we have not characterized.

### **Slice and Denominator Limits**

* OLMo-2 and OLMo-3.1 have weak baseline N0 accuracy on their mechanism slices, so their matched-flip denominators are smaller and noisier than GPT-OSS, Qwen, or Gemma.  
* Absolute percentages across models should be read as indicative, not directly comparable.

### **Direction-Fitting Limits**

* Directions are fitted per model on that model's own activations — we do not claim a shared universal direction across families.  
* Layer geometry differs by model, and the current story does not explain why Qwen's mechanism lives at L2 while OLMo-3.1's lives at L15.

### **Deconfound Coverage Gaps**

* Assistant-axis deconfound was run on GPT-OSS, Gemma-4, and OLMo-2. Qwen3.5 check is pending.  
* Additional deconfounds (e.g. instruction-following strength, toxicity, refusal) could further sharpen the claim and are not run.

### **Thinking-Mode Claim Scope**

* The thinking-on-amplifies claim is established on Qwen3.5.  
* Other thinking-capable models (GPT-5 high, Gemini-3.1-Pro with extended thinking) have not been run under matched protocol.  
* The claim is that thinking did not protect Qwen on this measurement, which is narrower than "thinking universally amplifies compliance."

### **Behavioral-Evidence Limits**

* Mechanism claims are supported by probing, steering, and transfer; per-token attention-level or circuit-level analysis has not been done.  
* We do not trace the mechanism back to specific heads or features.

### **Limitations Takeaway**

* The paper makes a strong empirical claim about a causal, portable internal authority-compliance axis in multiple LLMs, with explicit scope on where that claim is solid and where more work is needed.

---

## **Future Work Red Thread**

### **Question**

* What research program does this open?

### **Mechanism Refinement**

* Circuit-level tracing: which attention heads and MLP features implement the axis?  
* Cross-layer story: why does the mechanism layer vary from L2 (Qwen) to L15–L22 (others)?  
* Within-family scaling: how does the axis change from 7B → 30B → 70B?

### **Deployment-Time Mitigation**

* Can inference-time steering along the negative of this direction reduce authority-compliance on held-out prompts without damaging normal instruction-following?  
* Can this be combined with training-time approaches (DPO on truthful-under-pressure pairs) to close the loop?

### **Cross-Phenomenon Transfer**

* Does this direction overlap with RAG-poisoning failures, prompt-injection susceptibility, or retrieval-hijack behavior?  
* If so, a single intervention target could address a class of problems currently treated as separate.

### **Chain-of-Thought Interaction**

* The thinking-on amplification on Qwen needs replication on other thinking-capable models.  
* If the pattern holds, CoT-faithfulness work should include explicit tests against authority-compliance pressure, not only logical-difficulty pressure.

### **Thinking-Off Defenses**

* If thinking-off systematically resists better, that is a practical deployment recommendation and a training signal for future thinking modes.

### **Frontier-API Comparison**

* Gemini-3.1-Pro is near-immune at behavior level. Understanding why; training-data filter, refusal policy, or a structurally different mechanism, could inform training targets for open-weight models.

### **Future Work Takeaway**

* The paper opens a program on locating and intervening on social-signal-triggered override mechanisms across models, with concrete hypotheses about RAG poisoning overlap, CoT amplification, and cross-family mechanism geometry.

---

## **Main Paper Packing**

### **Main Paper**

* Introduction built around authority-compliance as a mechanism, not a behavior.  
* N0/C1/W1 behavioral protocol with cross-model flip-rate table.  
* Authority gradient as graded-social-signal evidence.  
* Extraction \+ direction \+ nulls.  
* Forward-patch matched-flip headline table (OLMo-3.1 as strongest, GPT-OSS as cleanest dose response).  
* PIQA out-of-domain transfer.  
* SYCON sycophancy-native collapse, with thinking-on vs thinking-off Qwen comparison.  
* Deconfound section (assistant axis \+ valence-arousal \+ C1-null).

### **Appendix**

* Full layer × α sweeps per model.  
* Per-model mechanism tree layout.  
* Thinking-on vs thinking-off detailed comparison (v3 vs v4 Qwen runs).  
* Slice-choice story for OLMo-3.1 (`all_prior_wrong`).  
* Full null-control AUROCs.  
* All dynamic-parser summaries and recomputed tables.  
* Gradient-style sweep (weak / uncertain / assertive / authoritative\_verified) per model.

---

