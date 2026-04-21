# Results summary — behavior, gradients, steering, transfer, and controls

This document pulls together the main empirical threads in this repo: **behavioral** authority override on trivia, the **assertive-gradient** dissociation line, **compliance / activation steering** (trivia mechanism + SYCON false-presupposition), **PIQA** cross-task transfer, **SYCON** graded runs (**`mean_traj_score`** for resistance trajectories vs. **ΔFlip** for steering collapse), **assistant-axis deconfounding**, and **null** checks (C1-null placebo + extraction-level nulls). Numbers below match the consolidated tables and linked artifacts unless noted.

---

## 1. Behavioral results — open-weight and API models

Setup: shared factual-trivia stem with conditions **N0** (neutral), **C1** (correct authority note), **W1** (incorrect authority note). **W1 flip rate** = among items with **N0 correct**, fraction that become **wrong under W1** (dynamic parser on free-form answers, or stored `final_is_correct` flags in the row files where noted).

| Model | Behavioral run | N0 acc | W1 flip | Notes |
| --- | --- | ---: | ---: | --- |
| `openai/gpt-oss-20b` | `v2_gpt_oss_freegen` | 77.1% | **65.4%** | Shared pool; strong surface override. |
| `google/gemma-4-26B-A4B-it` | `no_thinking_freegen_merged_authoritative` / `v2_gemma4_freegen` | 90.0% | **62.9–67.5%** | Shared pool; high N0 acc; still large flips. |
| `allenai/OLMo-2-0325-32B-Instruct` | `v2_olmo2_freegen` | ~34% | **82.8–89.0%** | Shared pool; hard filter → low N0 acc, very high flip rates. |
| **`Qwen/Qwen3.5-27B`** | **`authoritative_verified_shared_h100`** | **74.9%** | **44.9%** (608/1355) | **1813-item** pool; **1808** UIDs with all three conditions parsed (`neurips-results/qwen35/authoritative_verified_shared_h100/dynamic_parser_recomputed_summary.md`). |
| **`allenai/OLMo-3.1-32B-Instruct`** | **`authoritative_verified_all_prior_wrong_shared_h100`** | **35.1%** (177/504) | **72.3%** (128/177) | **`all_prior_wrong`** slice (504 UIDs); flip = N0-correct → W1-wrong on complete triples (`authoritative_verified_all_prior_wrong_shared_h100/…_dissociation_rows.jsonl`). See `neurips-results/olmo31/README.md` for why this slice replaced top-k seeds. |
| `x-ai/grok-4.20` | `grok420_dissociation_freegen` | 90.7% | **87.5%** | Frontier API. |
| `openai/gpt-5.4` | `gpt54_native_pool_staged` | 91.9% | **42.6%** | Frontier API. |
| `google/gemini-3.1-pro-preview` | `gemini31pro__dissociation_freegen` | 96.9% | **0.6%** | Frontier API; near-immune on this probe. |

Canonical cross-model table (shared-pool rows): `neurips-results/_shared/results_consolidated.md`. **Qwen** and **OLMo-3.1** rows above use the **authoritative_verified** pipeline and (for OLMo-3.1) the mechanism slice **`all_prior_wrong`** — not identical item sets as the classic `v2_*_freegen` rows, so compare headline numbers only with that caveat.

---

## 2. Assertive gradient (open-weight models)

These are **dissociation** runs with an **assertive** endorsement style and instruction to prioritize accuracy (“answer correctly even if the speaker is wrong …”). They isolate how often a model that was correct under N0 **flips wrong** under W1 when the note is phrased assertively — the “gradient” from neutral → wrong-authority under assertive pressure.

Headline **W1 flip (N0 correct → W1 wrong)** from dynamic-parser re-runs:

| Model | Run (folder pattern) | W1 flip rate (parsed diagnostics) | W1 acc (parsed) |
| --- | --- | --- | ---: |
| **OLMo-2-32B** | `gradient_olmo2_assertive` | **76.1%** (70/92) | 8.6% |
| **Qwen3.5-27B** | `gradient_assertive_shared_h100` | **41.5%** (567/1366 N0-correct UIDs) | 52.4% (W1 row) |
| **GPT-OSS-20B** | `gradient_gpt_oss_assertive` | **13.6%** (83/610) | 75.4% |
| **Gemma-4-26B** | `gradient_gemma4_assertive` | **13.2%** (100/760) | 79.3% |
| **OLMo-3.1-32B** | `gradient_assertive_all_prior_wrong_shared_h100` | **55.4%** (82/148) | 23.5% (W1 row parsed) |

OLMo-3.1 assertive row uses the same **`all_prior_wrong`** UID slice as the mechanism tree (`neurips-results/olmo31/gradient_assertive_all_prior_wrong_shared_h100/dynamic_parser_recomputed_summary.md`). N0 accuracy on **parsed-all-three** UIDs is **46.9%** in that file.

**Read:** OLMo-2 is the most **compliant** under assertive wrong authority on its split; **OLMo-3.1** sits between OLMo-2 and Qwen on flip rate but on a **harder** subset. Gemma and GPT-OSS show **much lower** flip rates under the same-style probe on their pools. Qwen sits in between at the pooled scale (`neurips-results/qwen35/gradient_assertive_shared_h100/dynamic_parser_recomputed_summary.md`).

For **weak / uncertain** gradient variants (same dataset family, different endorsement style), see per-model `gradient_*_weak`, `gradient_*_uncertain` under `neurips-results/{gpt-oss,gemma4,olmo2,olmo31,qwen35}/` and the index `neurips-results/_shared/dynamic_parser_summary_index.md`.

---

## 3. Compliance and steering — how steerable, how much flip?

### 3a. Trivia mechanism (authority direction, forward-patching)

Extract a **single-layer** mean-difference **authority direction** from trivia activations; **forward-patch** into N0 runs; report **matched flip** (baseline-correct items pushed wrong). Best configs from consolidated results:

| Model | Layer / α | Matched flip | Baseline wrong @ α=0 |
| --- | --- | ---: | ---: |
| GPT-OSS-20B | L16, α=1.0 | **32.7%** | 20.8% |
| OLMo-2-32B | L16, α=0.5 | **19.0%** | 8.7% |
| Qwen3.5-27B | L2, α=4.0 (W1−C1 add) | **16.2%** | 9.8% |
| Gemma-4-26B | L15, α=1.0 | **10.6%** | 16.8% |

**Steerability story:** GPT-OSS shows the largest **causal** matched-flip rate; Qwen’s effect lives in a **very early** layer (L2) vs mid/late layers for the others — same qualitative “one direction moves behavior” claim with different geometry. Gemma has a non-trivial matched flip but **small** net Δ wrong vs baseline because of overlap with items already wrong.

### 3b. OLMo-3.1 — compliance pipeline (tiered mechanism track)

End-to-end layout for OLMo-3.1 matches the other open-weight models: **activations → directions → probes → nulls → figures**, on the **`all_prior_wrong`** slice (`neurips-results/olmo31/README.md`).

| Stage | Role | Artifact |
| --- | --- | --- |
| **1. Behavioral / extraction seed** | 504 UIDs, `authoritative_verified` + gradient arms | `exp10_n0_seed/`, `authoritative_verified_all_prior_wrong_shared_h100/`, `gradient_*_all_prior_wrong_shared_h100/` |
| **2. Authority activations** | Cached hidden states + label masks | `mechanism/authority_activations_all_prior_wrong_shared_h100/` |
| **3. Compliance analysis** | Layer×position sweep, nested CV, **primary direction** | `mechanism/compliance_analysis_all_prior_wrong_shared_h100/` |

Headline quantities from `compliance_analysis_all_prior_wrong_shared_h100/probe_results.json`: **primary readout** selected as **`endorsement_start` @ layer 5** (nested-CV folds split across positions/layers). On the **held-out primary pair**, oriented **W1** AUROC ≈ **0.649** (95% CI in JSON); **C1** AUROC on the same geometry is near chance (~0.50). **Layer×position sweep** matrices (`layer_position_sweep.json`) and heatmaps (`figures/heatmap_auroc_w1.png`, `heatmap_auroc_c1.png`) give the full tiering of readout sites. **Nulls** (`null_controls.json`) compare the fitted direction to random directions and label shuffles; **authority-level-only** baselines in `probe_results.json` mark behavioral-confound ceilings (cf. GPT-OSS compliance doc).

This is **not** the same as trivia **forward-patching** in §3a (OLMo-3.1 steering there is evaluated mainly on **SYCON** / false-presupposition runs in §3c).

### 3c. SYCON false-presupposition — best graded configs (`mean_traj_score`)

Same **compliance-style** direction applied during **SYCON-Bench–style** false-presupposition dialogs. All numbers come from **`grader_gemini/sycon_grader_summary.json`** under each run in `neurips-results/exp17/`.

**Primary metric — `mean_traj_score`:** mean number of **consecutive rounds** the model keeps **rejecting** the false presupposition **before its first flip** (higher = stronger multi-turn resistance trajectory).

For **each model**, the row below is the single **best** `(layer, α)` in that run’s grid by **maximum `mean_traj_score`** (not by ΔFlip).

| Model | Run | Best L | α | mean_traj | Steadfast | Rej R1 | Rej R5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-27B **v3**\* | `sycon_fp_v3_qwen_100q_l2_l5_l10_a0_a03_a05` | **L5** | **0.0** | **2.23** | 0.362 | 0.69 | 0.61 |
| Qwen3.5-27B **v4** (no-think) | `sycon_fp_v4_qwen_100q_l2_l5_l10_a0_a03_a05_nothink` | **L5** | **0.5** | **3.41** | 0.690 | 0.84 | 0.81 |
| Gemma-4-26B | `sycon_fp_v3_gemma_100q_l20_l22_l24_a0_a03_a05` | **L22** | **0.3** | **2.96** | 0.532 | 0.79 | 0.63 |
| OLMo-2-32B | `sycon_fp_v3_olmo_100q_l10_l16_l22_a0_a03_a05` | **L10** | **0.3** | **2.91** | 0.543 | 0.81 | 0.62 |
| OLMo-3.1-32B | `sycon_fp_v3_olmo31_100q_l15_l18_l22_a0_a03_a05_apw` | **L18** | **0.5** | **3.04** | 0.757 | 0.70 | 0.70 |
| GPT-OSS-20B | `sycon_fp_v2_gpt_oss_100q` | **L16** | **0.0** | **2.09** | 0.615 | 0.52 | 0.61 |

\* **v3:** thinking tokens leaked into evaluated output → not comparable to v4 for primary claims; row kept for audit.

**Best of the best (by `mean_traj_score`):** (1) **Qwen v4** no-think — **3.41** @ L5, α=0.5. (2) **OLMo-3.1** — **3.04** @ L18, α=0.5. (3) **Gemma-4** — **2.96** @ L22, α=0.3. (4) **OLMo-2** — **2.91** @ L10, α=0.3.

**α sweep:** α=0.5 is best for **Qwen v4** and **OLMo-3.1**, but **not** universally — **Gemma** and **OLMo-2** peak at **α=0.3**; **GPT-OSS** and **Qwen v3** peak at **α=0.0** (no steering) on this metric.

### 3d. SYCON — complementary metric: steering **collapse** (ΔFlip)

Same grader JSONs; here **best** = largest **ΔFlip = Flip(α, L) − Flip(α=0, same layer)** (how much steering *collapses* resistance). This ranks models differently from §3c — e.g. **GPT-OSS** shows the largest collapse at L12 / α=0.5 even though **L16 / α=0.0** wins on **`mean_traj_score`**.

| Model | Best layer | Best α | Flip (baseline α=0 → steered) | ΔFlip | Grader JSON |
| --- | ---: | ---: | --- | ---: | --- |
| GPT-OSS-20B | **12** | **0.5** | 0.388 → **0.914** | **+0.526** | `exp17/sycon_fp_v2_gpt_oss_100q/grader_gemini/sycon_grader_summary.json` |
| Qwen3.5-27B **v4** (no-think) | **10** | **0.5** | 0.447 → **0.633** | **+0.186** | `exp17/sycon_fp_v4_qwen_100q_l2_l5_l10_a0_a03_a05_nothink/grader_gemini/sycon_grader_summary.json` |
| Qwen3.5-27B v3\* | **10** | **0.5** | 0.676 → **0.831** | **+0.154** | `exp17/sycon_fp_v3_qwen_100q_l2_l5_l10_a0_a03_a05/grader_gemini/sycon_grader_summary.json` |
| OLMo-3.1-32B | **22** | **0.5** | 0.250 → **0.329** | **+0.079** | `exp17/sycon_fp_v3_olmo31_100q_l15_l18_l22_a0_a03_a05_apw/grader_gemini/sycon_grader_summary.json` |
| OLMo-2-32B | **22** | **0.5** | 0.461 → **0.534** | **+0.074** | `exp17/sycon_fp_v3_olmo_100q_l10_l16_l22_a0_a03_a05/grader_gemini/sycon_grader_summary.json` |
| Gemma-4-26B | **24** | **0.5** | 0.584 → **0.615** | **+0.031** | `exp17/sycon_fp_v3_gemma_100q_l20_l22_l24_a0_a03_a05/grader_gemini/sycon_grader_summary.json` |

\* **v3 row:** same caveat as §3c.

Together, **3a** shows trivia-level **matched flips** from patching; **3c** ranks **graded** multi-turn **resistance** trajectories; **3d** isolates **steering-induced collapse** (flip) separately.

---

## 4. PIQA domain cross-transfer

Same **trivia-fitted** authority direction (no refit) applied to **PIQA** two-choice physical commonsense, freegen + matched-flip metric. **200** shared items across models.

| Model | Best config | Matched flip on PIQA |
| --- | --- | ---: |
| Qwen3.5-27B | L5, α=0.7 | **20.6%** |
| GPT-OSS-20B | L16, α=1.0 | **18.8%** |
| OLMo-2-32B | L16, α=0.5 | **10.4%** |
| Gemma-4-26B | L20, α=0.5 | **1.4%** (honest non-transfer) |

**Read:** Three of four models show **out-of-domain** effect sizes on the order of **10–21%** matched flips — evidence the direction is not trivia-memorization only. Gemma is the exception (task-specific or format engagement).

Artifacts: `neurips-results/{gpt-oss,gemma4,olmo2,qwen35}/mechanism/piqa_forward_patch_200_freegen/piqa_summary.json`.

---

## 5. SYCON transfer (false presupposition / multi-turn pressure)

- **Headline ranking (graded):** **§3c** — per-model max **`mean_traj_score`** and **best of the best**; **§3d** — **ΔFlip** collapse (complementary).
- **Run roots:** `neurips-results/exp17/sycon_fp_v2_gpt_oss_100q`, `sycon_fp_v4_qwen_*`, `sycon_fp_v3_*` (Gemma, OLMo-2, OLMo-3.1, Qwen v3).
- **Early Qwen pilot** (`sycon_false_presupp_qwen_24q_256tok/`): smaller **n**, generic pushback — suggestive only.

**Claim framing:** **`mean_traj_score`** measures sustained **rejection** before first flip (higher = better resistance trajectory). **ΔFlip** measures how much steering **breaks** that resistance — use the metric that matches the claim (trajectory quality vs. steerability / collapse).

Summary tables: `neurips-results/exp17/BEST_RESULTS.md`.

---

## 6. Deconfounder results — assistant axis vs authority direction

**Question:** Is the extracted “authority / compliance” direction just the known **assistant vs role** axis (persona adoption)?

**Method (headline):** hardened **assistant axis** = default-assistant centroid minus mean of many role-specific activations; cosine with the **authority direction** at the mechanism layer.

| Model | Layer | cos(authority, assistant axis) |
| --- | ---: | ---: |
| GPT-OSS-20B | 18 | **+0.0118** |
| Gemma-4-26B | 16 | **−0.0372** |
| OLMo-2-32B | 22 | **−0.0356** |

All **|cos| ≤ ~0.04** — **near orthogonal**; the effect is not reducible to “roleplay.”

- Artifacts: `external/assistant-axis/hardened/overlap_summary.json` (GPT-OSS), `neurips-results/{gemma4,olmo2}/mechanism/assistant_axis_hardened/overlap_summary.json`
- Method detail: `neurips-results/_shared/assistant_axis_deconfounder_notes.md`
- Figure: `figures/neurips/v2/candidates/assistant_axis_deconfound.{png,pdf,svg}`

**Qwen3.5:** assistant-axis overlap **not run** on the same build (gap noted in `paper/neurips/experiment_map.md`).

---

## 7. Null checks

### 7a. C1-null placebo (steering / mechanism null)

**Procedure:** Build the same mean-difference **direction** but with **semantically null** “note” content (template/format control). Forward-patch like the real authority direction. **Matched flip** should stay near zero if the effect is **content-bound**.

| Model | Matched flip (C1-null) | vs real authority direction |
| --- | ---: | ---: |
| GPT-OSS-20B | **2.9%** | vs 32.7% |
| Gemma-4-26B | **1.5%** | vs 10.6% |
| OLMo-2-32B | **0.0%** | vs 19.0% |
| Qwen3.5-27B | **0.0%** | vs 16.2% |

Real directions are on the order of **~10×–30×** stronger than null-template directions — consistent with **authority semantics**, not note-shaped tokens alone.

Paths: `neurips-results/{gpt-oss,gemma4,olmo2,qwen35}/mechanism/*c1source_null*/steering_summary.json` (see `results_consolidated.md` for exact folder names).

### 7b. Extraction-level nulls (GPT-OSS compliance direction)

For the **learned compliance direction** on GPT-OSS, additional checks appear in `neurips-results/gpt-oss/mechanism/compliance_direction_results.md`: e.g. **shuffled-label** control drives W1 AUROC to chance (~0.50) vs **~0.81** for the real direction; **random direction + probe** inflates AUROC (high-dimensional fitting artifact), so the shuffled-label test is the preferred null.

---

## 8. Where to look next

| Topic | Primary paths |
| --- | --- |
| Master behavioral / mechanism / PIQA / C1-null tables | `neurips-results/_shared/results_consolidated.md` |
| Per-run parser index | `neurips-results/_shared/dynamic_parser_summary_index.md` |
| Claim ↔ experiment map | `paper/neurips/experiment_map.md` |
| SYCON graded summaries | `neurips-results/exp17/BEST_RESULTS.md`; **§3c–§3d** |
