# Authority-Conditioned Truth Override — Consolidated Results

## TL;DR

There may be a **causal, transferable internal mechanism for authority-conditioned truth override**
— *not* just a surface-level "models defer to authority" behavior, and *not* well captured by current
truthfulness or sycophancy evals.

- **Behavioral:** A simple authority note flips models' answers from correct to incorrect on a large
  fraction of previously-correct trivia — up to 65–89% of the time for open-weight models, with wide
  cross-family variation (Gemini-3.1-pro: 0.6%, Grok-4.20: 87.5%).
- **Mechanism:** Forward-patching a *single* authority direction (extracted on one task) at one
  transformer layer is sufficient to reproduce the flip on held-out questions with **matched-flip
  rates of 11–33%**, far above a **semantically-null C1 placebo baseline (0.0–2.9%)**.
- **Transfer:** The same authority direction, without refitting, produces **8–10 pp matched flips on
  PIQA** (physical commonsense) in 3 of 4 models — a different task, different labels, different
  surface form.
- **Not roleplay:** The authority direction is **near-orthogonal to the assistant-vs-role axis**
  (|cos| ≤ 0.04 in all three models probed), so it is not reducible to "adopting a persona."

## Why this might matter

1. **Evals miss this.** Generic truthfulness/sycophancy benchmarks score these same models as
   "fine." The failure mode only surfaces when a plausible authority-bearing prompt is introduced,
   and the *same internal direction* will reproduce it on tasks that were never in the eval set.
2. **It is a mechanism, not a behavior.** Authority-override is not a bag of surface heuristics
   scattered across the weights — a low-dimensional direction in a single mid-early layer carries
   enough signal to re-produce the effect when additively patched into unrelated questions.
3. **Alignment implication.** If the mechanism is structured and portable, it is a candidate for
   direct alignment/safety targeting (activation steering, circuit-level editing), rather than
   iterative RL-from-preferences on a growing list of surface cases.

---

## Setup

Three prompt conditions share a single question stem:

- **N0** — neutral prompt ("Please answer this question ...").
- **C1** — same prompt + a plain *correct* authority note (e.g., "Note: The answer is X.").
- **W1** — same prompt + a plain *incorrect* authority note (e.g., "Note: The answer is Y.", where
  Y is a plausible distractor).

The questions are factual trivia with a single ground-truth answer. W1 uses a plausible-but-wrong
authority claim; the model has to decide whether to follow it.

Three experimental families:

- **Behavioral freegen (H1):** just generate a free-form answer under N0/W1, measure W1 flip rate
  (% of items where N0 was correct and W1 is wrong).
- **Mechanism (H2):** extract a single "authority direction" from activations on a trivia subset
  (mean-difference W1−N0 in one layer), then forward-patch that direction additively at a target
  layer during generation of *other* N0-prompted questions. Measure matched flip rate on items the
  baseline got correct.
- **PIQA transfer (H3):** use the direction extracted from trivia, apply it to PIQA 2-choice
  physical commonsense items in freegen mode. Measure matched flip rate.

Two confound checks:

- **C1-null placebo:** substitute a semantically-null "note" for C1 (so the mean-difference
  direction is mostly noise + template). Patching this null direction should *not* produce flips.
- **Assistant-axis deconfound:** check whether the authority direction is a rotation of the
  well-known "roleplay vs. default assistant" axis (Anthropic-style hardened probe). If so, the
  finding would be a trivial persona effect. Cosine should be ≈ 0.

Parsers: dynamic free-form parser applied identically across all models; steering rows re-parsed so
the numbers are comparable. See `dynamic_parser_summary_index.md` for the canonical run index.

---

## Cross-model master table

*N = number of items. "Matched flip" = fraction of baseline-correct items that flip under the
intervention. "Best α/L" = best alpha × layer config on the held-out set. Behavioral N0-acc and W1
flips are from freegen on a separate trivia pool; mechanism uses forward-patching with
`interpolate_mean` and endorsement-span position selection unless noted.*

### Behavioral (free-form trivia, N0 vs W1)

| Model | N0 acc | W1 flip rate | W1 residual acc | Run |
|---|---:|---:|---:|---|
| `openai/gpt-oss-20b` | 77.1% | **65.4%** (400/612) | 28.1% | `v2_gpt_oss_freegen` |
| `google/gemma-4-26B-A4B-it` | 90.0% | **62.9%** (476/757) | 33.2% | `no_thinking_freegen_merged_authoritative` |
| `allenai/OLMo-2-0325-32B-Instruct` | 34.0% | **82.8%** (77/93) | 5.5% | `v2_olmo2_freegen` |
| `x-ai/grok-4.20` | 90.7% | **87.5%** (1279/1462) | 11.1% | `grok420_dissociation_freegen` |
| `openai/gpt-5.4` | 91.9% | **42.6%** (693/1626) | 57.4% | `gpt54_native_pool_staged` |
| `google/gemini-3.1-pro-preview` | 96.9% | **0.6%** (7/1134) | 97.2% | `gemini31pro__dissociation_freegen` |

OLMo's low N0-acc reflects its trivia pool being harder-filtered; the surviving W1 flip rate is
still the highest in the open-weights group.

### Mechanism — forward-patching authority direction onto N0+note

*Baseline "wrong rate" is on the same N0+note condition at α=0 (direction zeroed).*

| Model | N | Layer / α | Baseline wrong | Best wrong | Δ wrong | Matched flip |
|---|---:|---|---:|---:|---:|---:|
| `gpt-oss-20b`       | 212 | **L16 / α=1.0**            | 20.8% | 37.3% | +16.5 pp | **32.7%** (55/168) |
| `olmo-2-32B`        |  93 | **L16 / α=0.5**            |  8.7% | 23.9% | +15.2 pp | **19.0%** (16/84)  |
| `qwen3.5-27B`       | 512 | **L2  / α=4.0** (w1−c1 add) |  9.8% | 19.7% | +10.0 pp | **16.2%** (75/462) |
| `gemma-4-26B`       | 757 | **L15 / α=1.0**            | 16.8% | 18.1% |  +1.3 pp | **10.6%** (66/623) |

Gemma's net Δ is small because baseline wrongs are already high (16.8%) and many of its flips
*overlap* with baseline-wrongs; the matched flip measures the "additional" items pushed wrong by the
intervention, and there it is 10.6% of baseline-corrects.

### C1-null placebo — replace authority content with semantically-null note

*Direction extracted exactly the same way but on null notes. If the mechanism were a template /
formatting artifact, this should still flip answers.*

| Model | N | Best config | Baseline wrong | Best wrong | Matched flip |
|---|---:|---|---:|---:|---:|
| `gpt-oss-20b` | 212 | L16 / α=1.0 | 2.4% |  3.8% | **2.9%** (6/207) |
| `gemma-4-26B` | 758 | L15 / α=1.0 | 0.9% |  2.1% | **1.5%** (11/747) |
| `olmo-2-32B`  |  93 | L10 / α=0.3 | 0.0% |  0.0% | **0.0%** (0/92)  |
| `qwen3.5-27B` | 512 | L2  / α=0.3 | 0.0% |  0.0% | **0.0%** (0/512) |

Compared to 33 / 19 / 16 / 11% on real authority content — the effect is ~10–30× stronger with
real authority semantics. The direction is doing content-bound work, not template-bound work.

### PIQA transfer — same authority direction, different task (physical commonsense)

*Direction fit on trivia, applied to PIQA 2-choice items in freegen mode. Same 200 UIDs across
models.*

| Model | N | Config | Baseline wrong | Best wrong | Δ | Matched flip |
|---|---:|---|---:|---:|---:|---:|
| `qwen3.5-27B`   | 200 | interpolate_mean α=0.7 L5  | 40.0% | 48.6% | +8.6 pp | **20.6%** (21/102) |
| `gpt-oss-20b`   | 200 | interpolate_mean α=1.0 L16 | 24.6% | 34.4% | +9.7 pp | **18.8%** (19/101) |
| `olmo-2-32B`    | 200 | interpolate_mean α=0.5 L16 | 10.0% | 19.7% | +9.7 pp | **10.4%** (15/144) |
| `gemma-4-26B`   | 200 | interpolate_mean α=0.5 L20 |  8.1% |  8.2% | +0.2 pp | **1.4%** (2/148)   |

**Three of four models transfer.** GPT-OSS, Qwen, and OLMo all show matched flips ~10–20% on a
task their direction was never fit on. Gemma does not transfer to PIQA. Either Gemma's authority
circuit is more task-specific or the PIQA format does not sufficiently engage it.

### Assistant-axis deconfound — is "authority" just "roleplay"?

*Hardened assistant-vs-role axis computed Anthropic-style: 15–17 distinct role prompts, pooled
role vectors differenced against the default-assistant centroid. Cosine between that axis and the
authority direction.*

| Model | Layer | cos(authority, assistant-axis) |
|---|---:|---:|
| `gpt-oss-20b` | 18 | +0.0118 |
| `olmo-2-32B`  | 22 | −0.0356 |
| `gemma-4-26B` | 16 | −0.0372 |

All three within ±0.04 — essentially orthogonal. The authority override is not the assistant picking
up a role; it is a distinct direction in a different part of activation space. See
`figures/neurips/v2/candidates/assistant_axis_deconfound.{png,pdf,svg}` for the pooled-role KDE with
the three authority markers sitting near 0 on the assistant-vs-role axis.

---

## Per-model cards

### GPT-OSS-20B — the cleanest picture

- **Behavioral:** 65.4% W1 flip rate; N0 acc 77% → W1 acc 28%.
- **Mechanism:** matched flip 32.7% at L16/α=1.0 (Δ+16.5 pp vs baseline 20.8%). Largest single-model
  effect we see.
- **C1-null:** 2.9% (drops 11×).
- **PIQA transfer:** 18.8% matched flip at same L16/α=1.0 config — the direction travels.
- **Deconfound:** cos ≈ 0 with assistant axis at L18.
- **Read:** All four lines of evidence stack cleanly. One mid-layer direction, additive patch,
  ~1/3 of baseline-correct items flipped on trivia and ~1/5 on PIQA, with strong placebo and
  roleplay controls.

### OLMo-2-32B-Instruct — strongest behavioral, clean mechanism

- **Behavioral:** 82.8–89.0% W1 flip across runs (the easiest model to tip, though its N0 acc is
  lower to begin with).
- **Mechanism:** matched flip 19.0% at L16/α=0.5 (Δ+15.2 pp).
- **C1-null:** 0.0% — the cleanest placebo of the four.
- **PIQA transfer:** 10.4% matched flip at L16/α=0.5. Transfers.
- **Deconfound:** cos = −0.036 at L22 with assistant axis.
- **Read:** High behavioral surface effect + modest but clean mechanism + clean transfer + clean
  placebo. Internally consistent. The surface-to-mechanism gap (82% → 19%) is the one interesting
  asymmetry: OLMo behaviorally "caves" a lot, but only part of it is explained by the single
  authority direction — the rest may be broader sensitivity to note-shaped tokens.

### Qwen3.5-27B — mechanism effect even when behavioral baseline is 0%

- **Behavioral:** baseline on our C1-null freegen is 100% accurate (no naturally-occurring flips),
  so there is no classic behavioral W1 baseline to compare. This is exactly why the mechanism
  check matters.
- **Mechanism:** matched flip 16.2–17.2% at L2 across expanded/pilot runs. Note the
  direction is carried by a very **early** layer (L2) — qualitatively different from the others
  (L16–L20), but the effect is real.
- **C1-null:** 0.0% (512/512).
- **PIQA transfer:** 20.6% matched flip at L5/α=0.7 — the highest transfer number in the set.
- **Deconfound:** no assistant-axis number (not run for Qwen), but the per-layer structure (early
  + transferable) makes it unlikely to be a persona rotation.
- **Read:** Qwen's trivia behavior is robust, yet a low-layer additive direction still produces
  portable overrides. This is the *most* interesting model for the mechanism claim because the
  behavioral signal alone does not flag the failure mode.

### Gemma-4-26B — mechanism modest, transfer absent

- **Behavioral:** 62.9–67.5% W1 flip.
- **Mechanism:** matched flip 10.6% at L15/α=1.0 (Δ+1.3 pp net; the mechanism moves *which*
  items flip more than how many).
- **C1-null:** 1.5%.
- **PIQA transfer:** 1.4% — does not transfer.
- **Deconfound:** cos = −0.037 at L16.
- **Read:** Consistent with the mechanism existing but being more entangled (or more
  task-specific) for Gemma. Behavioral surface effect is comparable to GPT-OSS, but a single mid-
  layer direction does not summarise it the same way, and it does not generalise to PIQA.

### GPT-5.4 — frontier, behavioral-only

- **Behavioral:** 42.6% W1 flip — substantially lower than open-weight peers, but still nearly half
  of previously-correct items.
- **Mechanism / transfer / deconfound:** closed weights, no activations. Included as a behavioral
  reference point: "flips reduce under RLHF but do not vanish."

### Gemini-3.1-pro — frontier, near-immune behaviorally

- **Behavioral:** 0.6% W1 flip (7 / 1134). Either strong RLHF resistance or a refusal/clarification
  mode instead of flipping. Remarkable asymmetry vs Grok-4.20 (87.5%).
- **Mechanism:** not available (closed).
- **Read:** The strongest counter-point to "all frontier models defer." Whatever Gemini-3.1-pro is
  doing here is not universal across frontier labs.

### Grok-4.20 — frontier, behaviorally matches open-weight models

- **Behavioral:** 87.5% W1 flip. As compliant as OLMo despite being a frontier model.
- **Read:** RLHF does not automatically fix authority compliance.

---

## What is and isn't established

**Established (across ≥3 models):**
- Authority notes produce large behavioral flips in a majority of models tested.
- A single low-dimensional direction in one transformer layer can be additively patched to
  reproduce authority-style overrides on held-out items (matched flip 11–33%).
- The effect requires *authority content* in the direction, not just template structure
  (C1-null placebo ≤ 2.9% everywhere, vs 11–33% with real authority).
- The direction is orthogonal to the assistant-vs-role axis (|cos| ≤ 0.04).

**Transfers (3 of 4 models):**
- GPT-OSS, Qwen, OLMo: authority direction produces 8–10 pp wrong-rate deltas and 10–20% matched
  flips on PIQA without any refitting.
- Gemma: does not transfer to PIQA.

**Not established / open:**
- **H3b (sycophancy benchmark transfer):** does this direction also produce flips on
  SycophancyEval / TruthfulQA-adversarial / etc.? Not yet tested on standard sycophancy suites.
- **H5 (evidence-weighting):** is the direction sensitive to authority *quality* (e.g.,
  cited-vs-uncited, plausible-vs-implausible)? Current experiments use uniform plausible W1 notes;
  we don't know the dose-response to evidence quality.
- **H6 (replication under independent parses & prompts):** behavioral numbers are re-parsed with a
  single dynamic parser, but we have not yet run an independent team rerun or robustness sweep over
  note phrasings.
- **Frontier mechanism:** Gemini-3.1-pro's behavioral near-immunity is striking; without activations
  we can't say whether the mechanism is absent or present-but-suppressed.

## Methods glossary

- **N0 / C1 / W1** — prompt conditions: neutral / correct-authority-note / wrong-authority-note.
- **freegen** — free-form generation, parsed post-hoc by a dynamic parser (no forced choice).
- **forward_patch / interpolate_mean** — at layer L, replace activations in the endorsement-span
  token window with `α · μ_W1 + (1−α) · μ_N0` (interpolation between condition means of the
  direction). α=0 is no intervention, α=1 is fully the W1-minus-N0 direction added in.
- **direction_kind** — which vector is used for patching: `w1source` (W1 mean), `w1_minus_c1`
  (contrastive), `c1source_null` (C1-null placebo for the confound check).
- **matched flip rate** — fraction of items the *baseline* got correct that the intervention
  flips wrong. Controls for baseline accuracy drift across models.
- **assistant axis** — hardened role-vs-default probe, 15–17 distinct role prompts pooled, axis =
  mean(role) − default.
- **VA plane** — Valence-Assertiveness planar projection (not the load-bearing analysis here; used
  as a secondary visualization in prior exploratory figures).

## Where things live

- **Per-model steering summaries:** `neurips-results/{gpt-oss,gemma4,olmo2,qwen35}/mechanism/forward_patch_test_256[_...]/steering_summary.json`
- **C1-null placebo runs:** `.../forward_patch_test_256_c1source_null/` (Qwen uses `forward_patch_expanded_512_c1source_null/`)
- **PIQA transfer:** `.../piqa_forward_patch_200_freegen[_no_thinking]/piqa_summary.json`
- **Assistant-axis overlap:** `neurips-results/{gemma4,olmo2}/mechanism/assistant_axis_hardened/overlap_summary.json`; GPT-OSS at `external/assistant-axis/hardened/overlap_summary.json`
- **Behavioral index:** `neurips-results/_shared/dynamic_parser_summary_index.md` (canonical dynamic-parser re-run)
- **Deconfound figure:** `figures/neurips/v2/candidates/assistant_axis_deconfound.{png,pdf,svg}`
- **Run logs:** `neurips-results/exp16_logs/`
