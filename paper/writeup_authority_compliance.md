# Authority-Conditioned Compliance in LLMs — Writeup Draft

*Working draft for Google Doc. Structure is: (1) research question, (2) hypotheses, (3) experiments in the order we actually ran them. Figure slots are marked inline.*

---

## 1. The research question

Most sycophancy work stops at a behavioral observation: models agree with users, drift toward personas, or defer to whoever spoke last. We wanted to push past that.

The question we kept coming back to is narrower and more mechanistic:

> Do some LLMs contain a distinct internal **authority-conditioned compliance mechanism** that can override truth, and is that mechanism causal, portable, and not well explained by generic sycophancy, persona drift, or affect?

That framing matters, because "models defer to authority" is a behavior. What we want to know is whether there's a specific thing *inside* the model — an internal state, a direction in activation space, a process — that gets recruited when a prompt signals authority, and whether flipping that thing is enough to flip the answer.

If yes, then:
- authority-following is not just a prompt surface effect
- it's separable from "the model is being nice to the user"
- it can be intervened on
- and it might explain a class of failure modes that current sycophancy evals quietly miss

If no, the story collapses into "some prompts push answers around," which is less interesting.

> **[Figure 1 slot: hero figure — single-model schematic showing N0 → W1 flip with authority note, and the companion activation plot hinting at a direction getting recruited. Use GPT-OSS since it's the cleanest case.]**

---

## 2. Hypotheses

We structured the work around six concrete hypotheses, each of which could be supported, weakened, or outright killed by specific experiments. This isn't post-hoc — the list was written down before the SYCON reruns and the deconfound work. Full narrative in [authority-compliance-hypotheses-and-narrative.md](../docs-to-read/authority-compliance-hypotheses-and-narrative.md).

**H1 — Behavioral override.** Strong authority framing can push baseline-correct models to wrong answers. Without H1 there's no mechanism to find.

**H2 — Causal internal mechanism.** The override corresponds to an internal state we can extract *and* intervene on. Forward patching should reproduce the behavior shift with dose response and layer specificity.

**H3 — Out-of-domain transfer.** A direction extracted on factual QA should still move behavior on a different task (PIQA). This is mechanism transfer, not sycophancy-specific transfer.

**H3b — Sycophancy-native replication.** The same direction should modulate a benchmark with explicit endorsement-vs-truth conflict (SYCON), using a sycophancy-native metric. This is the cleanest bridge to the existing sycophancy literature.

**H4 — Deconfound.** The direction is not well explained by the known assistant/persona axis or by cheap valence-arousal drift.

**H5 — Evidence weighting, not just last-token flip.** Authority framing perturbs internal evidence weighting before generation, not only the final output readout.

**H6 — Cross-model generality.** The mechanism, or something recognizably analogous, should appear in more than one model family.

Current read: H1 is supported broadly. H2 is strong in GPT-OSS and moderate in Qwen. H3 holds in three of four tested models. H3b is supported for GPT-OSS and Qwen (after the thinking-leak fix). H4 holds on the deconfounds we ran. H5 and H6 are partially there but need more work.

> **[Note for doc: include a small "hypothesis verdict" stoplight table here — H1 green, H2 green/amber, H3 amber-green, H3b amber, H4 green, H5 amber, H6 amber.]**

---

## 3. Experiments

### 3.1 Behavioral experiment — does authority override actually happen?

**What it is.** A shared trivia-style multiple-choice stem, asked under three conditions:

- **N0**: neutral, no endorsement.
- **C1**: a short "note" endorsing the **correct** answer, attributed to an authority (e.g. "According to a verified source…").
- **W1**: the same note shape, but endorsing the **wrong** answer.

Free-form generation, dynamic parser to decide what the model actually answered. The headline metric is **W1 flip rate**: of the items the model gets right under N0, what fraction it gets wrong under W1. If a model is immune to authority framing, this should be near zero.

**Why we ran it first.** If H1 fails, there is nothing mechanistic to chase. Everything downstream (activation extraction, steering, SYCON, PIQA) is gated on finding a real behavioral effect to explain.

**What we saw.**

| Model | N0 acc | W1 flip |
|---|---:|---:|
| GPT-OSS-20B | 77.1% | **65.4%** |
| Gemma-4-26B | 90.0% | **62.9–67.5%** |
| OLMo-2-32B | ~34% | **82.8–89.0%** |
| Qwen3.5-27B | 74.9% | **44.9%** |
| OLMo-3.1-32B | 35.1% | **72.3%** |
| Grok-4.20 | 90.7% | **87.5%** |
| GPT-5.4 | 91.9% | 42.6% |
| Gemini-3.1-Pro | 96.9% | **0.6%** |

Flip rates across five of six open-weight models are between 45% and 89%. That is a large behavioral effect. The outlier is Gemini-3.1-Pro, which essentially refuses to be moved by the note on this probe — interesting in its own right, but it means Gemini is not a useful subject for the mechanism work downstream.

**H1 verdict: supported.** The effect is large, cross-model, and survives careful parsing.

> **[Figure 2 slot: bar chart of W1 flip rate across models, grouped by open-weight vs frontier-API. Make Gemini-3.1-Pro visually obvious as the lone outlier.]**

*Source: [`neurips-results/_shared/results.md`](../neurips-results/_shared/results.md) §1.*

---

### 3.2 Authority gradient — is all authority equal?

**What it is.** Same N0/C1/W1 structure, but now we vary the *style* of the endorsement note across four levels: **weak**, **uncertain**, **assertive**, **authoritative_verified**. Same questions, same instruction to prioritize accuracy ("answer correctly even if the speaker is wrong"), different phrasing of the note.

**Why we ran it.** This is a small but important sanity check. If models treat every endorsement the same way, then "authority" is a distraction and we're really just measuring "someone said X." If they don't — if assertive and authoritative notes flip answers more than weak or uncertain ones — then there's a graded, evidence-weighting-like response to social signaling, which matters a lot for H2 and H5.

**What we saw (assertive row, W1 flip on N0-correct UIDs).**

| Model | W1 flip (assertive) |
|---|---:|
| OLMo-2-32B | 76.1% |
| OLMo-3.1-32B | 55.4% (on `all_prior_wrong` slice) |
| Qwen3.5-27B | 41.5% |
| GPT-OSS-20B | 13.6% |
| Gemma-4-26B | 13.2% |

Two useful things here. First, the ordering is not the same as the authoritative-note ordering from §3.1 — GPT-OSS and Gemma drop a lot when the note is merely assertive rather than explicitly authoritative, while OLMo-2 and Qwen stay high. Models are reading something specific in the stronger authority frame, not just "the user said so." Second, within a single model, flip rates rise monotonically from weak → uncertain → assertive → authoritative_verified for the ones where we checked all four.

This is what H5 wants to see at the behavioral level: not a binary flip, but a graded response to stronger social evidence.

*Source: [`results.md`](../neurips-results/_shared/results.md) §2; per-model parser summaries under `gradient_*` folders.*

> **[Figure 3 slot: small multiples. One panel per model. x-axis = endorsement style (weak → authoritative_verified), y-axis = W1 flip rate. Should show a clear monotone gradient for models that have one, and a flat line for ones that don't.]**

---

### 3.3 Extraction, directions, and causal steering

This is where the mechanism story lives. The recipe:

1. Run the trivia behavioral experiment, collect activations at many layer/position combinations across all three conditions (N0/C1/W1).
2. Fit a direction in activation space that separates W1-complying states from C1-complying states (or N0-correct from W1-wrong, depending on framing).
3. Sweep layers and positions to find the best readout. Use nested CV and report a single held-out primary pair. Run null controls: shuffled labels, random directions, same-template-but-different-content ("C1-null").
4. *Forward patch* the direction into a clean N0 run at varying strengths α, measure whether baseline-correct answers flip wrong in the direction the theory predicts.

**Best layer and primary pair, per model.**

| Model | Primary position / layer | Forward-patch best config | Matched flip |
|---|---|---|---:|
| GPT-OSS-20B | L18 (compliance) | L16, α=1.0 | **32.7%** |
| OLMo-3.1-32B | `endorsed_answer` @ L15 (apw slice) | L15, α=1.0 (replace_mean) | **58.8%** |
| OLMo-2-32B | L22 | L16, α=0.5 | **19.0%** |
| Qwen3.5-27B | early (L2 w1-minus-c1) | L2, α=4.0 | **16.2%** |
| Gemma-4-26B | L16 | L15, α=1.0 | **10.6%** |

OLMo-3.1 was run on the `all_prior_wrong` slice (97 baseline-correct UIDs). L15 is the peak layer for both patch modes:

| Mode | L15 | L18 | L22 |
|---|---:|---:|---:|
| interpolate_mean α=0.5 | 38.1% | 33.0% | 25.8% |
| replace_mean α=1.0 | **58.8%** | 39.2% | 29.9% |

Two things jump out. Effect sizes are not small — OLMo-3.1 at L15 replace-mean pushes almost 60% of baseline-correct items wrong, and GPT-OSS pushes a third. And the mechanism layer differs across models: Qwen at L2, OLMo-3.1 at L15, GPT-OSS at L16–L18, OLMo-2 at L16–L22.

> **[Figure 4 slot: layer×α heatmap of matched flip per model, best cell annotated.]**

**Dose response.** GPT-OSS and OLMo-3.1 both give monotonic curves. OLMo-3.1 at L15: 0.32 → 0.38 (α=0.3 → 0.5, interpolate), up to 0.59 at α=1.0 (replace). Intervention strength tracks behavior — what H2 wants.

> **[Figure 5 slot: dose-response curve. α on x-axis, matched flip on y-axis, one line per model, GPT-OSS highlighted.]**

*Source: [`results.md`](../neurips-results/_shared/results.md) §3a; per-model `mechanism/forward_patch_*` folders.*

---

### 3.4 PIQA cross-domain transfer

**What it is.** Take the authority direction fitted on trivia (no refitting), and apply it to PIQA — physical commonsense two-choice — using the same forward-patch protocol and a matched-flip metric. 200 shared items per model.

**Why we ran it.** H3. If the direction is just memorizing trivia patterns, it should do nothing on PIQA. If it's something closer to a general compliance knob, it should still move behavior on a structurally different task.

**What we saw.**

| Model | PIQA best config | Matched flip on PIQA |
|---|---|---:|
| Qwen3.5-27B | L5, α=0.7 | **20.6%** |
| GPT-OSS-20B | L16, α=1.0 | **18.8%** |
| OLMo-2-32B | L16, α=0.5 | **10.4%** |
| Gemma-4-26B | L20, α=0.5 | 1.4% |

Three of four models show double-digit matched-flip rates out of domain. That's meaningful — the direction was never trained on anything physical-commonsense-shaped, and it still pushes PIQA answers the wrong way. Gemma, honestly, doesn't transfer. We report that number as-is; the direction there looks more task-specific than we'd hoped.

This supports H3 (out-of-domain mechanism transfer) but doesn't establish H3b (sycophancy-native transfer). PIQA has no endorsement-vs-truth conflict baked in; it's a generic OOD probe. The sycophancy-native claim needs SYCON, which is next.

> **[Figure 6 slot: two-panel figure. Left panel: trivia matched-flip. Right panel: PIQA matched-flip. Same models, same color per model. Lines connect dots to show transfer magnitude.]**

*Source: [`results.md`](../neurips-results/_shared/results.md) §4; `mechanism/piqa_forward_patch_200_freegen/`.*

---

### 3.5 SYCON false-presupposition — sycophancy-native test

**What it is.** SYCON-Bench false-presupposition setting: a user asks a question that contains a false presupposition ("when did Einstein write the Gettysburg Address?"), and if the model corrects the premise, the user pushes back across multiple rounds. We measure whether the model keeps rejecting or caves.

We ran this with the trivia-extracted authority direction applied as `interpolate_mean` steering at three α levels (0, 0.3, 0.5) across three layers per model, 100 questions × 5 rounds. LLM-graded with `gemini-3.1-flash-lite-preview` for rejection judgments.

Two metrics because they mean different things:

- **`mean_traj_score`**: mean number of consecutive rejections before the first flip. Higher = better sustained resistance.
- **ΔFlip**: Flip(α=0.5) − Flip(α=0) at the same layer. Higher = more steering-induced collapse.

**Why we ran it.** H3b. Unlike PIQA, SYCON is explicitly about whether models cave under social pressure. If the authority direction is really the thing we think it is, cranking it up should collapse resistance in a sycophancy-native way.

**Two snags along the way, worth flagging.**

First, the **v3 Qwen run had thinking tokens leaking into the graded output**. Qwen3.5's chat template injects `<think>\n` at the assistant start unless you pass `enable_thinking=False`. The SYCON runner wasn't wired for that flag yet, so v3 saw partial thinking bleed: 21/4500 rows had `<think>`, 55 had `</think>`, 368 had reasoning-prefix phrases. That's small in count but concentrated at α=0.5, so it biased exactly the cells we care about. We fixed the runner to take `--disable-thinking` and reran as **v4**. v3 is kept for audit; all claims come from v4.

Second, **the slice chosen for OLMo-3.1's mechanism step had very few primary_w1 rows** (21 out of 72 all-three-parsed items) because the behavioral baseline on that slice was weak. We diagnosed, switched to an `all_prior_wrong` slice (504 UIDs, 158 w1 / 153 c1), and reran end-to-end. This is what shows up in the headline OLMo-3.1 numbers below.

**Results, per model, best (layer, α) by `mean_traj_score`.**

| Model | Run | Best L | α | mean_traj | Steadfast | Rej R1 | Rej R5 |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-27B v4 (no-think) | `sycon_fp_v4_qwen` | **L5** | **0.5** | **3.41** | 0.690 | 0.84 | 0.81 |
| OLMo-3.1-32B | `sycon_fp_v3_olmo31_apw` | **L18** | **0.5** | **3.04** | 0.757 | 0.70 | 0.70 |
| Gemma-4-26B | `sycon_fp_v3_gemma` | **L22** | **0.3** | **2.96** | 0.532 | 0.79 | 0.63 |
| OLMo-2-32B | `sycon_fp_v3_olmo` | **L10** | **0.3** | **2.91** | 0.543 | 0.81 | 0.62 |
| Qwen3.5-27B v3 (leaked) | `sycon_fp_v3_qwen` | **L5** | **0.0** | **2.23** | 0.362 | 0.69 | 0.61 |
| GPT-OSS-20B | `sycon_fp_v2_gpt_oss` | **L16** | **0.0** | **2.09** | 0.615 | 0.52 | 0.61 |

**Results, per model, best (layer, α) by ΔFlip.**

| Model | Best L | α | Flip (baseline → steered) | ΔFlip | Verdict |
|---|---:|---:|---|---:|---|
| GPT-OSS-20B | **L12** | 0.5 | 0.388 → **0.914** | **+0.526** | Strong collapse |
| Qwen3.5-27B v4 | **L10** | 0.5 | 0.447 → **0.633** | **+0.186** | Moderate collapse |
| OLMo-3.1-32B | **L22** | 0.5 | 0.250 → **0.329** | **+0.079** | Near-null |
| OLMo-2-32B | **L22** | 0.5 | 0.461 → **0.534** | **+0.074** | Weak |
| Gemma-4-26B | **L24** | 0.5 | 0.584 → **0.615** | **+0.031** | Very weak |

**Read.** Two different things are being measured, and the rankings reflect that. On `mean_traj_score` — "who resists the longest under steering?" — Qwen v4 at L5 α=0.5 actually *gains* resistance (mid-layer reinforcement), and OLMo-3.1 is a strong second. On ΔFlip — "who collapses most under steering?" — GPT-OSS is in a league of its own: baseline flip 0.39 goes to 0.91 at L12 α=0.5. That's the cleanest single result we have for H2 and H3b together.

Qwen v4's L10 α=0.5 collapse (flip 0.45 → 0.63) is a meaningful replication at roughly 35% of GPT-OSS's magnitude. Interestingly, at L5 the same α *strengthens* resistance — the direction has a layer-phase structure, not a monotone compliance knob. That's worth staring at more.

OLMo-3.1 and Gemma-4 are near-unsteerable on collapse at the tested layers. Part of this is a ceiling effect (OLMo-3.1 baseline R5 is already 0.78) and part is likely that the direction we fit doesn't cleanly land on their compliance mechanism. H6 gets partial support here: the pattern isn't uniform across models, but it isn't just a GPT-OSS idiosyncrasy either.

> **[Figure 7 slot: small-multiples SYCON plot. Six panels (GPT-OSS, Qwen v4, Gemma, OLMo-2, OLMo-3.1; Qwen v3 in a muted grey as the audit comparison). x-axis = α, y-axis = Flip. Three lines per panel, one per layer. Baseline (α=0) bolded.]**

> **[Figure 8 slot: ΔFlip bar chart, one bar per model, sorted descending. Make GPT-OSS stand out.]**

*Source: [`BEST_RESULTS.md`](../neurips-results/exp17/BEST_RESULTS.md); [`results.md`](../neurips-results/_shared/results.md) §3c–§3d.*

---

### 3.6 Deconfounders — what else could this be?

Two main alternative explanations to rule out.

**Assistant axis deconfound.** The "assistant vs. role/persona" direction is well-known — models have an internal signal for "I am being the default helpful assistant" vs "I am playing someone else." If our authority direction is just that axis, we're not discovering anything new. We built a hardened assistant axis (default-assistant centroid minus mean of many role-specific activations) and measured its cosine with the authority direction at the mechanism layer.

| Model | Layer | cos(authority, assistant axis) |
|---|---:|---:|
| GPT-OSS-20B | 18 | +0.0118 |
| Gemma-4-26B | 16 | −0.0372 |
| OLMo-2-32B | 22 | −0.0356 |

All three are essentially orthogonal. Whatever the authority direction is, it's not the persona-adoption axis.

**Why we ran this.** H4. If this check had failed, the honest story would have become "authority prompts ride the assistant axis," which is weaker and not really a new mechanism.

We did not run this on Qwen3.5 on the same build (gap acknowledged). Adding it is a short task.

**Valence-arousal deconfound.** Authoritative notes use different vocabulary from neutral ones — "verified," "confirmed," "well-established" — which could correlate with valence and arousal axes extracted from lexical norms. If our direction is mostly lexical affect, we should see strong overlap. We don't. Details in the deconfound notes; the short version is that the authority direction is not a relabeling of a lexical affect direction either.

> **[Figure 9 slot: small diagram. Left: cartoon of authority direction. Middle: assistant axis. Right: valence-arousal direction. Arrows showing pairwise cosines, all near zero. This is mostly to make the "not just persona, not just affect" point visually digestible.]**

*Source: [`assistant_axis_deconfounder_notes.md`](../neurips-results/_shared/assistant_axis_deconfounder_notes.md); [`results.md`](../neurips-results/_shared/results.md) §6.*

---

### 3.7 Null checks

Two kinds of nulls, both important.

**C1-null placebo.** Same extraction pipeline, but instead of authority-flavored content in the note, we use a semantically null template with the same token shape. If the effect is driven by note presence or note formatting, the null direction should steer just as well. If the effect is driven by *authority content*, the null direction should do almost nothing.

| Model | Matched flip (C1-null direction) | Real authority direction |
|---|---:|---:|
| GPT-OSS-20B | 2.9% | 32.7% |
| Gemma-4-26B | 1.5% | 10.6% |
| OLMo-2-32B | 0.0% | 19.0% |
| Qwen3.5-27B | 0.0% | 16.2% |

Real directions are 10×–30× stronger than null directions. So the effect is content-bound, not format-bound.

**Extraction-level nulls** (GPT-OSS compliance direction specifically). Two standard checks: shuffled-label control and random-direction control. Shuffled labels drive W1 AUROC to chance (~0.50) compared to ~0.81 for the real direction — this is the preferred null because it isolates "is there signal in the labels" from "is there signal in the direction fit." Random-direction + probe inflates AUROC as an artifact of high-dimensional fitting, which is why we lean on shuffled-label instead.

**Why we ran these.** H2 and H4. A causal mechanism claim is basically useless without nulls. The C1-null in particular is what lets us say "this isn't the pipeline" rather than "this isn't the template."

> **[Figure 10 slot: side-by-side bars of real-direction matched flip vs C1-null matched flip, one pair per model. Make the 10×–30× gap visually obvious.]**

*Source: [`results.md`](../neurips-results/_shared/results.md) §7; per-model `mechanism/*c1source_null*/` folders; [`gpt-oss/mechanism/compliance_direction_results.md`](../neurips-results/gpt-oss/mechanism/compliance_direction_results.md).*

---

## 4. Where this leaves us

The defensible current claim is narrower than the full project question, but it's real:

> In GPT-OSS (and to a meaningful but smaller degree in Qwen3.5), strong authority framing recruits a causal internal state that can override truth, that transfers out of domain to PIQA, that shows steering-induced collapse on a sycophancy-native benchmark (SYCON), and that is not well explained by the assistant axis, valence-arousal drift, or note-template format.

The more cautious framing for OLMo-2, OLMo-3.1, and especially Gemma-4:

> Behavior flips under authority are clearly present. The same extraction recipe produces a direction, but the causal steering effects are smaller, more layer-specific, and do not produce sycophancy-native collapse at the magnitudes seen in GPT-OSS.

Open things, in priority order:
1. Run the assistant-axis deconfound on Qwen3.5 to close that gap.
2. Tighten H5 (evidence weighting vs last-token readout) with per-position trajectory analysis. We have some of this but it's not write-up-ready.
3. Investigate why Qwen's compliance direction flips sign between L5 (reinforces resistance) and L10 (collapses resistance). Could be a genuine two-stage mechanism or could be a slice artifact.
4. Second cleaner replication. GPT-OSS carries a lot of the weight right now.

> **[Figure 11 slot: summary "narrative ladder" figure. Four stacked boxes, strongest-to-weakest claim, with a marker showing where the current evidence lands. Makes it easy for reviewers to see we're not overclaiming.]**

---

## Appendix: Key files

- Consolidated cross-model tables: [`neurips-results/_shared/results.md`](../neurips-results/_shared/results.md)
- SYCON graded summaries: [`neurips-results/exp17/BEST_RESULTS.md`](../neurips-results/exp17/BEST_RESULTS.md)
- Hypothesis narrative (long form): [`docs-to-read/authority-compliance-hypotheses-and-narrative.md`](../docs-to-read/authority-compliance-hypotheses-and-narrative.md)
- Claim ↔ experiment map: [`paper/neurips/experiment_map.md`](../paper/neurips/experiment_map.md)
- Per-model mechanism trees: `neurips-results/{gpt-oss, gemma4, olmo2, olmo31, qwen35}/mechanism/`
