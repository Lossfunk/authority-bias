**One-line question:** *Do the prompt cues that cause LLMs to follow incorrect answers share an internal mechanism, or are verified-source authority, user opinion, and assistant-role expectation separable directions in the residual stream that can be independently controlled?*

---

### Before the experiments: what the terms mean

- **Three prompt variants.** In the prose below, use the reader-friendly names:
  - **neutral** = no note in front of the question
  - **correct endorsement** = note endorsing the correct option
  - **wrong endorsement** = note endorsing a wrong option
- **Mapping to artifact names.** In code, file names, and earlier experiment tables, these same three conditions still appear as:
  - `N0` = neutral
  - `C1` = correct endorsement
  - `W1` = wrong endorsement
- **Authority note.** A short prefix prepended to the stem. In the strongest setting we use wording of the form *"Note: According to the verified source, it is A, not B."*
- **Residual stream.** The per-token activation vector at the output of a transformer layer. This is what we read out during extraction and edit during steering.
- **Direction `d`.** A unit vector in residual-stream space that is supposed to track movement toward authority-conditioned compliance.
- **Authority-minus-assistant vector.** The authority vector after subtracting its overlap with the generic assistant-following vector.
  - Use this phrase for Wang / source-user / Pareto / mitigation experiments.
  - Keep **residualized** for the causal-deconfound section, where the statistical language is clearer.
- **α.** The intervention strength.
  - α=0 → no-op
  - α=1 → one unit of effect
  - larger α → stronger push
- **Matched flip rate.** Our main causal readout.
  - Start from items the model got right without intervention.
  - Count the fraction that become wrong under intervention.
  - This prevents "already wrong" items from inflating the effect.

---

### What we should say carefully in the paper

- **Safe claims now:**
  - "Authority-conditioned compliance is a behavioral and mechanistic phenomenon in multiple open-weight model families."
  - "The authority feature is causal in several models, not just linearly decodable."
  - "The effect is not reducible to generic assistant-role expectation."
  - "Authority-minus-assistant removal reduces wrong-authority compliance with small MMLU-Pro/GSM8K costs in Qwen, OLMo-2, OLMo-3.1, and GPT-OSS."
  - "Generic CAA sycophancy steering misses or weakly captures this effect compared with authority-minus-assistant."
  - "In Qwen and GPT-OSS, verified-source authority and user-sycophancy are separable non-assistant directions: cos(source, user) ≈ 0.20 in both, removing source selectively collapses source-wrong, removing user selectively collapses user-wrong, removing the assistant axis is flat."
  - "Gemma-4 is an important boundary case: the source-authority behavior exists, but the current residual-stream intervention does not control it well."

- **Claims that still need caveats:**
  - "Source authority and user sycophancy are generally separable" → locked for Qwen and GPT-OSS; not yet shown for OLMo-2 / OLMo-3.1 / Gemma-4.
  - "Affect is fully ruled out" → current NRC-VAD / lexical-affect check is useful but not yet as strong as the assistant-axis deconfound.
  - "Mitigation works across all models" → no; Gemma is a clear mitigation-null boundary case.
  - "Transfer is universal" → no; Gemma PIQA transfer and Pareto are weak/null.
  - "The same exact mechanism exists in every family" → too strong; better say there is a recurring authority-compliance feature with meaningful boundary cases.
  - "CAA never works" → too strong; safer: CAA is much weaker / not Pareto-competitive in the current setup.

- **Best current framing:**
  - The main paper should not be "we found one direction in one model."
  - The main paper should be:
    - authority cues create truth-override behavior;
    - this behavior has a residual-stream correlate;
    - the correlate is not generic assistantness;
    - authority-minus-assistant control reduces the behavior with small capability cost in most models;
    - verified-source authority can separate from user-sycophancy mechanistically: locked in two model families (Qwen, GPT-OSS).
  - Qwen is the cleanest explanatory case.
  - GPT-OSS is the key replication case.
  - OLMo-3.1 is the strongest additional mitigation case.
  - Gemma is the boundary case that prevents overclaiming and motivates future work.

---

### What would make this feel spotlight-level rather than just solid

- **Two-family source/user separability — locked.**
  - Qwen and GPT-OSS both show source authority and user-sycophancy split mechanistically: cos(source, user) ≈ 0.20 in both, source-removal selectively collapses source-wrong (−70 pp Qwen, −77 pp GPT-OSS), user-removal selectively collapses user-wrong (−43 pp Qwen, −16 pp GPT-OSS), assistant removal is flat in both, residualizing source/user against the assistant axis preserves the pattern.

- **A strong affect deconfound.**
  - The obvious reviewer objection is: "is this just emotional / confidence / authority-word valence?"
  - A causal affect deconfound would close that gap much more directly than another broad model run.

- **A memorable intervention story.**
  - Not just "we found a direction."
  - The clean version is:
    - **authority-minus-assistant** reduces verified-source truth override;
    - **assistant removal** is flat;
    - **CAA sycophancy removal** is weak;
    - **general capability mostly stays intact.**

- **A clear boundary-case story.**
  - Gemma should not be hidden.
  - It should be used to say the mechanism/control story has limits:
    - strong behavioral source authority;
    - weak current residual-stream control;
    - capability preserved;
    - therefore likely a routing/position/nonlinear-control issue rather than simple absence of source sensitivity.

- **Best spotlight-shaped thesis (GPT-OSS source/user split now locked; affect deconfound still pending):**
  - LLMs often over-weight verified-source authority even when it is wrong.
  - This behavior is carried by an authority-compliance feature that is not generic assistantness.
  - In two open-weights model families (Qwen and GPT-OSS), verified-source authority separates mechanistically from user-sycophancy and can be independently suppressed at α=1.
  - Authority-minus-assistant removal reduces truth override with small capability cost, while CAA-style sycophancy steering largely misses the effect.
  - Boundary cases such as Gemma show that behavioral source sensitivity and controllability can come apart.

---

### Step 1 — Behavioral override on trivia

- **Question this step answers:** do authority notes change answers at all?
- **Setup:**
  - Same trivia stem under:
    - neutral
    - correct endorsement
    - wrong endorsement
  - Same decoding settings across all three.
- **Why this matters:** if models do not flip under wrong endorsement, there is no behavior to explain mechanistically.
- **How to read the result:**
  - The main number here is wrong-answer rate under the **wrong endorsement** condition.
  - Neutral gives the baseline without any authority cue.
- **Result (wrong endorsement flip rate):**
  - GPT-OSS: 65%
  - Gemma-4: 63–68%
  - OLMo-2: 83–89%
  - OLMo-3.1: 83%
  - Qwen3.5: 45%
  - Grok-4.20: 88%
  - Gemini-3.1-Pro: ~1% (outlier)
- **Takeaway:** there is a large cross-model behavioral effect. This makes the mechanism question worth asking.

---

### Step 2 — Authority gradient

- **Question this step answers:** is the effect specifically about authority, or just about someone saying something?
- **Setup:**
  - Hold the stem and endorsed option fixed.
  - Vary only the framing of the note:
    - casual assertion
    - confident assertion
    - explicit authority claim
    - explicit authority claim + "verified / confirmed"
- **Why this matters:** a flat gradient would make the "authority" interpretation much weaker.
- **How to read the result:**
  - We compare wrong-answer rate under wrong endorsement across the four note strengths.
  - If stronger social authority framing produces stronger override, that is evidence for graded authority sensitivity rather than a binary "note present / absent" effect.
- **Result:**
  - Monotone gradient within model where we checked all four.
  - Assertive alone only flips:
    - OLMo-2: 76%
    - OLMo-3.1: 55%
    - Qwen: 42%
  - `authoritative_verified` is the big jump for GPT-OSS and Gemma.
- **Takeaway:** social authority is graded, not binary.

---

### Step 3 — Activation extraction and authority direction fitting

- **Question this step answers:** can we find a consistent internal direction that separates authority-driven wrong compliance from authority-driven correct compliance?
- **Setup:**
  - Run labeled neutral / correct-endorsement / wrong-endorsement items.
  - Cache residual-stream activations across many `(layer, token position)` pairs.
- **What we fit:**
  - At each `(layer, position)`, compute:
    - `d = unit(mean_wrong-complying − mean_correct-complying)`
  - This gives one candidate authority direction per `(layer, position)`.
- **How we score it:**
  - Evaluate each candidate by AUROC on held-out items.
  - Keep the cleanest `(layer, position)` as the model's canonical extraction slot.
- **Null controls in this step:**
  - shuffled labels
  - random unit directions
  - correct-null notes with the same token shape but no authority content
- **Result:**
  - Best extraction slot per model:
    - GPT-OSS: L18
    - OLMo-3.1: L15
    - OLMo-2: L22
    - Gemma-4: L16
    - Qwen3.5: L2
  - Shuffled-label AUROC ~0.50 vs real 0.81+.
  - Correct-null directions do not probe the same thing.
- **Takeaway:** there is a real, separable internal direction. But this is still probe evidence, not yet causal evidence.

---

### Step 4 — Causal steering on trivia

- **Question this step answers:** if we push the model along the fitted direction, can we make a clean prompt behave more like an authority-conditioned one?
- **Setup:**
  - Start from a **neutral** prompt.
  - At a chosen layer and token position, add `α * d` to the residual stream.
  - Nothing else about generation changes.
- **Primary readout:**
  - **matched flip rate** = fraction of baseline-correct items that become wrong under steering.
- **Why this matters:** this is the first actual causal test. A good probe alone is not enough.
- **Result (best config per model):**
  - OLMo-3.1 L15, α=1.0 → **58.8%**
  - GPT-OSS L16, α=1.0 → **32.7%**
  - OLMo-2 L16, α=0.5 → **19.0%**
  - Qwen3.5 L2, α=4.0 → **16.2%**
  - Gemma-4 L15, α=1.0 → **10.6%**
- **Takeaway:** the direction is causal, not just descriptive. The effect size differs by model.

---

### Step 5 — Transfer to PIQA

- **Question this step answers:** is this just a trivia-specific pattern, or does the same internal direction transfer to another task?
- **Setup:**
  - Freeze the trivia-fit direction and its `(layer, position)`.
  - Apply it to PIQA with no refitting.
- **Why this matters:** transfer distinguishes a generic compliance knob from a trivia-specific feature.
- **Result (matched flip on PIQA):**
  - Qwen: 20.6%
  - GPT-OSS: 18.8%
  - OLMo-2: 10.4%
  - Gemma-4: 1.4%
- **Takeaway:** transfer works for 3/4 models. Gemma is the clear caveat.

---

### Step 6 — Transfer to SYCON, a sycophancy-native benchmark

- **Question this step answers:** does the same direction matter in a setting where endorsement-vs-truth conflict is the whole task, not just a transfer side effect?
- **Setup:**
  - SYCON false-presupposition benchmark
  - 100 questions × 5 conversational rounds
  - steer the trivia-fit direction at α ∈ {0, 0.3, 0.5}
  - check three layers per model
  - score with gemini-3.1-flash-lite as judge
- **Two readouts:**
  - **ΔFlip** = how much more often the model caves under steering than at α=0
  - **mean_traj** = average stance across the five turns; higher means it resists longer
- **Important note:** Qwen v3 had thinking enabled and did worse than v4; v4 is the reported version.
- **Result — ΔFlip (steering-induced collapse):**
  - GPT-OSS L12 α=0.5: 0.39 → **0.91** (Δ +0.53) ← cleanest collapse
  - Qwen v4 L10 α=0.5: 0.45 → **0.63** (Δ +0.19)
  - OLMo-3.1 L22 α=0.5: 0.25 → 0.33 (Δ +0.08)
  - OLMo-2 L22 α=0.5: 0.46 → 0.53 (Δ +0.07)
  - Gemma-4 L24 α=0.5: 0.58 → 0.62 (Δ +0.03)
- **Result — mean_traj (sustained resistance):**
  - Qwen v4 L5 α=0.5: **3.41**
  - OLMo-3.1 L18 α=0.5: **3.04**
  - Gemma L22 α=0.3: 2.96
  - OLMo-2 L10 α=0.3: 2.91
- **Takeaway:** the effect is not a simple monotone knob. Some layers collapse resistance, while others reinforce it.

---

### Step 7 — Is the authority direction just the assistant persona axis?

- **Question this step answers:** are we rediscovering generic assistantness, or is this a distinct compliance mechanism?

#### 7a. Geometric check

- **Setup:**
  - Build a contrast set:
    - default-assistant transcripts
    - matched role-play transcripts where the model is asked to be some non-assistant persona
  - Fit an **assistant axis** from that contrast at the mechanism layer.
- **Readout:**
  - `cos(authority_direction, assistant_axis)`
- **Result:**
  - Qwen3.5: +0.06
  - GPT-OSS: +0.01
  - Gemma: −0.04
  - OLMo-2: −0.04
- **Takeaway:** the authority direction is close to orthogonal to the assistant axis.
- **Caution:** low cosine is not enough on its own. Two directions can still produce similar behavior even if they are nearly orthogonal.

#### 7b. Causal deconfound

- **Setup:**
  - Work at one specific layer and token position.
  - Take the residual-stream activation.
  - Measure how much of it points along a chosen direction `d`.
  - Subtract exactly that component.
- **Plain-language interpretation:** we zero out the part of the activation that points along `d`, while leaving everything perpendicular to `d` untouched.
- **We repeat this for three choices of `d`:**
  - the **authority** direction
  - the **assistant** axis
  - the **authority direction after removing its assistant-axis component**
- **α semantics:**
  - α=1 removes the whole projection
  - α=0.5 removes half
  - α=0 is a no-op
- **What a clean result looks like:**
  - removing authority → large behavioral drop
  - removing assistant alone → little or no drop
  - removing residualized → nearly matches removing authority
- **Result (`endorsement_span`, α=1):**

  **Factual QA, sorted by authority-removal magnitude**

  | Rank | Model | Baseline W1 wrong | Remove authority | Remove assistant | Remove residualized authority |
  |---:|---|---:|---:|---:|---:|
  | 1 | Qwen3.5 | 39.4% | **−33.1 pp** (→6.3%) | +1.0 pp | **−33.6 pp** |
  | 2 | OLMo-2 | 83.9% | **−20.8 pp** (→63.1%) | +1.1 pp | **−21.1 pp** |
  | 3 | OLMo-3.1 | 81.6% | **−19.8 pp** (→61.8%) | +0.1 pp | **−20.1 pp** |
  | 4 | GPT-OSS | 56.9% | **−16.7 pp** (→40.2%) | +0.5 pp | **−16.4 pp** |
  | 5 | Gemma-4 | 57.8% | −2.7 pp | −2.9 pp | −3.4 pp |

  **PIQA, sorted by authority-removal magnitude**

  | Rank | Model | Baseline W1 wrong | Remove authority | Remove assistant | Remove residualized authority |
  |---:|---|---:|---:|---:|---:|
  | 1 | Qwen3.5 | 58.7% | **−53.3 pp** (→5.3%) | +0.4 pp | **−54.2 pp** |
  | 2 | OLMo-2 | 89.0% | **−20.1 pp** (→68.9%) | ~0 pp | **−20.2 pp** |
  | 3 | OLMo-3.1 | 96.1% | **−18.7 pp** (→77.4%) | +1.4 pp | **−19.5 pp** |
  | 4 | GPT-OSS | 86.5% | **−14.8 pp** (→71.7%) | +0.3 pp | **−14.6 pp** |
  | 5 | Gemma-4 | 97.1% | ~0 pp | +0.6 pp | ~0 pp |

- **Takeaway:** the causal deconfound supports the same conclusion as the geometric check: the authority effect is not carried by generic assistantness. Authority removal is consistently much larger than assistant-axis removal, and residualizing against assistant largely preserves the authority effect. Qwen is the cleanest case; Gemma is the clearest weak/boundary case.

---

### Step 8 — Is source authority the same mechanism as user sycophancy?

- **Question this step answers:** when the model follows a wrong answer, is it using the same internal feature for a verified external source and for the user's own stated preference?
- **Why this matters:** this is the cleanest bridge to the Wang / sycophancy literature. Prior sycophancy work often studies agreement with the user; our strongest behavioral effect is agreement with a verified source. If those are separable internally, then "source authority" is not just a relabeled user-sycophancy effect.

#### 8a. Behavioral Wang-style source-vs-user probe

- **Setup:**
  - Use the same item set under three conditions:
    - neutral
    - wrong answer attributed to a verified source
    - wrong answer attributed to the user / user's expertise
  - Compare the wrong-answer increase from neutral under source authority vs user expertise.
- **Result:**

  | Model | n | Neutral wrong | Source wrong | Δsource | User wrong | Δuser | User/source ratio |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | Gemma-4 | 512 | 2.1% | 80.2% | +78.1 pp | 17.3% | +15.2 pp | 0.195 |
  | GPT-OSS | 512 | 9.4% | 88.7% | +79.3 pp | 30.2% | +20.8 pp | 0.262 |
  | OLMo-3.1 | 158 | 14.3% | 85.1% | +70.8 pp | 67.5% | +53.3 pp | 0.753 |
  | Qwen3.5 | 512 | 7.5% | 71.8% | +64.3 pp | 62.1% | +54.5 pp | 0.849 |
  | OLMo-2 | 93 | 18.2% | 88.8% | +70.5 pp | 87.8% | +69.5 pp | 0.986 |

- **Takeaway:** the Wang-style behavioral probe does not give one universal answer. Gemma-4 and GPT-OSS are clearly more source-authority-sensitive than user-sycophancy-sensitive. Qwen and OLMo-3.1 show large effects for both, and OLMo-2 is almost fully ambiguous behaviorally. That is why the mechanistic split below matters: behavioral asymmetry alone cannot tell us whether source authority and user agreement share a direction.

#### 8b. Mechanistic source-vs-user split

- **Setup:**
  - Build five matched prompt conditions:
    - neutral
    - source correct endorsement
    - source wrong endorsement
    - user correct endorsement
    - user wrong endorsement
  - Fit separate directions for:
    - **source authority**
    - **user authority / sycophancy**
    - the generic **assistant** axis
    - source and user directions after removing any assistant-axis component
  - Evaluate projection-removal at α=1 on source-wrong and user-wrong endorsement prompts.
- **What a clean source/user split looks like:**
  - source and user directions should have low cosine
  - removing the source direction should mostly fix source-wrong prompts
  - removing the user direction should mostly fix user-wrong prompts
  - removing the assistant axis should do little
- **Result — Qwen3.5, L5:**
  - Geometry:
    - `cos(source, user) ≈ 0.20`
    - `cos(source, assistant) ≈ 0.00`
    - `cos(user, assistant) ≈ −0.003`
    - `cos(source_resid, user_resid) ≈ 0.20`
  - Wrong-answer rate, lower is better:
    - baseline source-wrong: **87.3%**
    - baseline user-wrong: **59.6%**
    - remove source direction:
      - source-wrong: **87.3% → 16.8%** (−70.5 pp)
      - user-wrong: 59.6% → 42.7% (−17.0 pp)
    - remove user direction:
      - source-wrong: 87.3% → 76.5% (−10.8 pp)
      - user-wrong: **59.6% → 16.9%** (−42.7 pp)
    - remove assistant direction:
      - source-wrong: 87.3% → 87.2%
      - user-wrong: 59.6% → 59.3%
    - residualizing source/user against assistant preserves the same pattern.
- **Takeaway:** in Qwen, verified-source authority and user-sycophancy are mechanistically separable non-assistant directions. Source authority is also behaviorally stronger: the same wrong answer is followed much more often when attributed to a verified source than when attributed to the user.
- **Result — GPT-OSS-20B, L16 (parse rate ≥ 99% on every cell):**
  - Geometry:
    - `cos(source, user) ≈ 0.20` at L16; `0.22` at L18; `0.21` at L20
    - `cos(source, assistant) ≈ 0.10` at L16
    - `cos(user, assistant) ≈ 0.05` at L16
    - `cos(source_resid, user_resid) ≈ 0.19` (residualizing against assistant barely changes the source/user separability)
  - Wrong-answer rate (denominator = all 512 prompts), lower is better:
    - baseline source-wrong: **95.5%**
    - baseline user-wrong: **33.9%**
    - remove source direction:
      - source-wrong: **95.5% → 18.0%** (−77.3 pp)
      - user-wrong: 33.9% → 42.2% (+8.6 pp)
    - remove user direction:
      - source-wrong: 95.5% → 86.1% (−9.4 pp)
      - user-wrong: **33.9% → 17.4%** (−16.0 pp)
    - remove assistant direction:
      - source-wrong: 95.5% → 95.7% (+0.2 pp, control flat)
      - user-wrong: 33.9% → 35.4% (+1.6 pp, control flat)
    - residualized variants at L16 are flat (`source_resid_assistant` = +0.8 / −2.9 pp; `user_resid_assistant` = +0.8 / +2.7 pp), consistent with the cos≈0.10 / 0.05 overlap with the assistant axis being too small to drive the source/user behavior on its own.
  - At L18 α=1, parse rate drops to ~26% on the source/user variants (the patch destabilizes generation), while the assistant control still parses at 100%. Among the items that do parse, source-removal at L18 reproduces the L16 source-selective pattern (source-wrong → 16.4%, user-wrong → 10.2%, denominator = all 512 prompts).
- **Takeaway:** GPT-OSS replicates Qwen's geometric separability (cos(source, user) ≈ 0.20 in both models). The causal story is also cross-model:
  - Removing the source direction selectively collapses source-wrong (−77 pp GPT-OSS, −70 pp Qwen) without collapsing user-wrong.
  - Removing the user direction selectively collapses user-wrong (−16 pp GPT-OSS, −43 pp Qwen) without collapsing source-wrong.
  - Removing the assistant direction is flat in both models.
  - Residualizing source / user against the assistant axis preserves the separability and the selective collapse pattern in both models.
  - In two open-weights families, verified-source authority and user-sycophancy are separable directions in residual-stream space and can be **independently** suppressed at α=1.
- **Current status:** locked for Qwen and GPT-OSS at the layer where parse rate is intact (Qwen L5, GPT-OSS L16). Stronger-α layer behaviour (GPT-OSS L18 / L20) destabilises generation; we report it as a secondary observation, not as the primary readout.

---

### Step 9 — Is the direction just affective vocabulary?

- **Question this step answers:** are we really measuring authority, or just emotionally charged words like "verified" and "confirmed"?
- **Setup:**
  - Use the NRC-VAD lexicon.
  - Build:
    - a **valence axis**
    - an **arousal axis**
  - Each is defined as a high-score minus low-score mean activation contrast at the mechanism layer.
- **Readout:**
  - project the authority direction onto each axis
- **Result:** near-orthogonal.
- **Takeaway:** the authority direction is not just a relabeling of lexical affect.

---

### Step 10 — Null controls

- **Question this step answers:** is the pipeline measuring authority content, or could it just be reacting to note format or fit-time noise?

#### 10a. Correct-null placebo

- **Setup:**
  - Replace authority notes with notes of the same token length and grammatical shape, but without authority content.
  - Refit the direction.
  - Re-run the steering test.

#### 10b. Shuffled-label control

- **Setup:**
  - Keep the real notes.
  - Randomly permute the wrong/correct labels before fitting.
  - Refit and rerun.

- **Result (correct-null matched flip):**
  - GPT-OSS: 2.9%
  - Gemma: 1.5%
  - OLMo-2: 0.0%
  - Qwen: 0.0%
- **Result (shuffled labels):**
  - AUROC goes to chance.
- **Takeaway:** the real authority direction is 10–30× stronger than the matched-shape placebo, and shuffled labels kill the signal. That makes a simple pipeline-artifact story unlikely.

---

### Step 11 — Projection-removal Pareto: can we reduce authority compliance without wrecking general capability?

- **Question this step answers:** can the same projection-removal intervention turn down wrong-authority compliance, and is that just a disguised capability hit?
- **Why this matters:** a reviewer can always object that "removing the direction" merely makes the model worse. The relevant test is whether authority-conditioned wrong compliance falls while general capability on unrelated tasks is mostly preserved.
- **Setup:**
  - Use projection removal on the authority direction, especially the **authority-minus-assistant vector**.
  - Evaluate:
    - wrong-answer compliance under the wrong endorsement condition
    - MMLU-Pro accuracy
    - GSM8K accuracy
  - Report MMLU-Pro and GSM8K as error rate when comparing to wrong-answer compliance, so all three readouts are "lower is better."
- **Result — Qwen3.5:**
  - Wrong-authority compliance:
    - authority-minus-assistant removal on trivia: **42.7% → 5.2%** wrong-rate at α=1 (−37.5 pp)
    - authority-minus-assistant removal on PIQA: **57.3% → 4.5%** wrong-rate at α=1 (−52.8 pp)
    - assistant-axis control is flat:
      - trivia: 42.7% → 43.3% (+0.6 pp)
      - PIQA: 57.3% → 60.0% (+2.7 pp)
    - CAA / sycophancy-axis control is also flat:
      - trivia: 42.7% → 42.2% (−0.5 pp)
      - PIQA: 57.3% → 57.1% (−0.1 pp)
  - General capability:
    - MMLU-Pro: 60.7% → 59.6% accuracy at α=1 (error +1.1 pp)
    - GSM8K: 46.5% → 42.5% accuracy at α=1 (error +4.0 pp)
    - assistant-axis and CAA controls are flat on these capability checks.
- **Result — OLMo-2:**
  - Wrong-authority compliance:
    - authority-minus-assistant removal on trivia:
      - best at L10: **82.8% → 53.3%** wrong-rate at α=1 (−29.5 pp)
      - L16: −18.7 pp
      - L22: −8.9 pp
    - authority-minus-assistant removal on PIQA:
      - best at L16: **89.0% → 66.5%** wrong-rate at α=1 (−22.5 pp)
      - L10: −18.8 pp
      - L22: −16.1 pp
    - assistant-axis control is flat:
      - trivia: about −1.1 pp across L10/L16/L22
      - PIQA: about 0 pp across L10/L16/L22
    - CAA / sycophancy-axis control is weaker and less consistent:
      - trivia: best −15.4 pp at L10
      - PIQA: best −12.6 pp at L16
  - General capability is also largely stable in the same MMLU-Pro / GSM8K check.
  - Largest observed absolute changes are small:
    - MMLU-Pro: about 0.0 to −0.7 pp accuracy depending on layer/control
    - GSM8K: about −0.5 to +0.5 pp accuracy depending on layer/control
- **Result — GPT-OSS:**
  - Wrong-authority compliance:
    - authority-minus-assistant removal on trivia: **59.9% → 44.0%** wrong-rate at α=1 (−15.9 pp)
    - authority-minus-assistant removal on PIQA: **87.6% → 73.0%** wrong-rate at α=1 (−14.6 pp)
    - assistant-axis control is flat:
      - trivia: 59.9% → 60.0% (+0.1 pp)
      - PIQA: 87.6% → 88.0% (+0.4 pp)
    - CAA / sycophancy-axis control is much weaker:
      - trivia: 59.9% → 58.0% (−1.9 pp)
      - PIQA: 87.6% → 86.0% (−1.6 pp)
  - General capability:
    - MMLU-Pro: α=0 baseline 52.4%; authority-minus-assistant at α=1 ranges 50.5–54.2% across layers
    - GSM8K: α=0 baseline 88.5%; all variants stay within about ±1 pp
- **Result — OLMo-3.1:**
  - Wrong-authority compliance:
    - authority-minus-assistant removal on trivia: **83.4% → 64.0%** wrong-rate at α=1 (−19.4 pp)
    - authority-minus-assistant removal on PIQA: **96.2% → 78.0%** wrong-rate at α=1 (−18.2 pp)
  - General capability:
    - MMLU-Pro: α=0 baseline 44.3%; authority-minus-assistant at α=1 is 44.3–44.7%
    - GSM8K: α=0 baseline 26.5%; authority-minus-assistant at α=1 is 26.0–26.5%
  - Caution:
    - PIQA parse rate is low for OLMo-3.1 (~39%), so PIQA numbers should be framed as parsed-subset results.
- **Result — Gemma-4:**
  - Wrong-authority compliance:
    - trivia: weak / null; best authority-minus-assistant reduction is about **−2.9 pp**
    - PIQA: flat; wrong-rate stays around **97–98%**
  - General capability:
    - MMLU-Pro: α=0 baseline 59.3%; authority-minus-assistant at α=1 is 59.0–59.7%
    - GSM8K: α=0 baseline 52.0%; authority-minus-assistant at α=1 is 50.5–52.0%
  - Interpretation:
    - Gemma is not a capability-cost failure. It is a **mechanistic control failure / boundary case**: the intervention mostly preserves capability but does not meaningfully reduce wrong-authority compliance.
- **Cross-model takeaway:**
  - The mitigation story is not "make the model generally worse until it stops complying."
  - Qwen, OLMo-2, OLMo-3.1, and GPT-OSS all show meaningful authority-compliance reductions with small MMLU-Pro/GSM8K costs.
  - Assistant-axis removal is consistently flat.
  - CAA / generic sycophancy steering is much weaker than the authority-minus-assistant vector, especially in Qwen and GPT-OSS.
  - Gemma-4 is the main boundary case: capability is preserved, but mitigation is weak.

---

### What the current evidence already supports

- **Primary claim we can already defend:**
  - Verified-source authority is a real behavioral pressure across the open-weight model set.
    - Wrong endorsements strongly increase wrong-answer rates in GPT-OSS, Qwen3.5, OLMo-2, OLMo-3.1, and Gemma-4.
  - Multiple model families contain a residual-stream direction that tracks authority-conditioned compliance.
    - Probe/null evidence says this is not just fit-time noise.
    - Forward steering says the direction is causal in several models, not merely descriptive.
  - The authority direction is not well explained by generic assistant-role expectation.
    - Geometric assistant-axis cosines are near zero.
    - Causal deconfound: removing authority changes behavior much more than removing assistantness.
  - The **authority-minus-assistant vector** is the strongest current intervention object.
    - Qwen, OLMo-2, OLMo-3.1, and GPT-OSS all show meaningful wrong-authority reductions.
    - MMLU-Pro and GSM8K are mostly preserved in the same runs.
  - CAA / generic sycophancy steering is not enough.
    - It is flat or much weaker than authority-minus-assistant on the Pareto package.
  - In Qwen, verified-source authority and user-sycophancy are mechanistically separable.
    - Source removal mostly fixes source-wrong prompts.
    - User removal mostly fixes user-wrong prompts.
    - Assistant removal is flat.
  - Gemma-4 is the useful boundary case.
    - It has strong source-authority behavioral effects.
    - But authority-minus-assistant removal barely moves trivia and is flat on PIQA.
    - This looks like a control/steering failure rather than a capability-damage story.

- **Strongest current evidence chain:**
  - Step 1: behavioral override exists across models.
  - Step 2: stronger authority wording strengthens override; the effect is not just "some note exists."
  - Step 3: a separable internal direction can be fit with strong null controls.
  - Step 4: steering along that direction causally changes neutral trivia behavior.
  - Step 5: transfer to PIQA works for Qwen, GPT-OSS, and OLMo-2; Gemma is weak.
  - Step 6: SYCON shows the direction also matters in a sycophancy-native setting, but layer effects are non-monotone.
  - Step 7: assistant-axis geometry and causal deconfound show the authority effect is not generic assistantness.
  - Step 8: Qwen source-vs-user split shows verified-source authority is not just user agreement.
  - Step 9: affect/valence evidence currently argues against a simple lexical-affect story, but needs a stronger causal deconfound.
  - Step 10: correct-null and shuffled-label controls argue against prompt-format and label-noise artifacts.
  - Step 11: Pareto/capability package shows authority compliance can be reduced without broad capability collapse in 4/5 models.

- **Model status after the latest runs:**
  - **Qwen3.5** — cleanest complete model.
    - Strong behavioral override.
    - Strong mechanism / steering / PIQA transfer.
    - Assistant-axis deconfound passes.
    - Source-vs-user split is locked.
    - Authority-minus-assistant Pareto gives the largest mitigation with small capability cost.
  - **GPT-OSS** — second core model, source/user split now locked.
    - Strong behavioral override.
    - Mechanism and PIQA transfer work.
    - Assistant-axis deconfound passes.
    - Authority-minus-assistant Pareto reduces trivia and PIQA wrong-rate with flat capability.
    - CAA is nearly flat.
    - Source-vs-user split (L16, parse rate ≥ 99%): cos(source, user) ≈ 0.20; remove source → source-wrong −77 pp / user-wrong +9 pp; remove user → user-wrong −16 pp / source-wrong −9 pp; remove assistant flat. Replicates Qwen's two-direction selectivity.
  - **OLMo-3.1** — strongest OLMo headline for mitigation.
    - Strong behavioral override.
    - Strong causal deconfound and Pareto mitigation.
    - Capability is flat.
    - PIQA mitigation is strong but parse-rate-limited, so phrase PIQA as parsed-subset.
    - Source-vs-user split is optional given Qwen + GPT-OSS already cover the two-family claim.
  - **OLMo-2** — strong but less clean for the source/user story.
    - Strong behavioral override.
    - Good Pareto/capability package.
    - Behavioral source-vs-user probe is nearly ambiguous: user and source both move the model strongly.
    - Mechanistic source-vs-user split is not done.
  - **Gemma-4** — boundary case / failure-analysis target.
    - Strong source-authority behavior.
    - Mechanism exists in extraction/probe artifacts.
    - Authority-minus-assistant mitigation is weak/null.
    - Capability stays stable, so the issue is not simply that steering damages the model.

### What still needs to be done before submission

- **Highest-priority remaining experiment:**
  - **Run the affect deconfound properly.**
    - Current NRC-VAD / valence-arousal evidence is not enough for the strongest reviewer version.
    - We want an affect-control result analogous to the assistant-axis deconfound.
    - Minimal target:
      - fit valence/arousal or affect directions;
      - measure cosines with authority;
      - remove affect alone;
      - remove authority-minus-affect;
      - show authority effect survives while affect removal is flat or much smaller.
    - This is higher priority than Gemma failure analysis because it blocks a broad reviewer objection.

- **Start paper drafting now:**
  - Build the outline while affect deconfound finishes.
  - Draft around claims already locked (including the Qwen + GPT-OSS source-vs-user split).
  - Leave slots for:
    - affect deconfound table;
    - final per-model completeness table.

- **Synthesis artifacts to make next:**
  - **Per-model completeness table**
    - rows: Qwen3.5, GPT-OSS, OLMo-2, OLMo-3.1, Gemma-4
    - columns: behavior, steering, PIQA, SYCON, assistant-axis deconfound, source-vs-user split, CAA control, Pareto/capability, affect deconfound, boundary notes
  - **Pareto summary table**
    - authority-minus-assistant vs assistant vs CAA
    - trivia wrong-rate, PIQA wrong-rate, MMLU-Pro, GSM8K
  - **Source-vs-user table**
    - Qwen and GPT-OSS locked
  - **Boundary-case table**
    - Gemma behavioral effect vs weak control effect

- **Lower-priority / optional experiments:**
  - **OLMo-3.1 source-vs-user split**
    - would extend the separability story to a third family
    - not blocking; Qwen + GPT-OSS already give the two-family result
  - **Gemma failure analysis**
    - useful, but should come after affect deconfound and initial paper draft
    - likely checks:
      - wrong layer / position;
      - no-thinking mismatch;
      - nonlinear or distributed feature;
      - MoE / routing issue;
      - margins too saturated for projection removal.
  - **Seed / resample stability**
    - still valuable, especially for final appendix / reviewer defense.
  - **Thinking-mode replication**
    - keep demoted unless we replicate it cleanly.
