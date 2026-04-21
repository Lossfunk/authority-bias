# SYCON false-presupposition — graded summaries (exp17)

All metrics below come from **`grader_gemini/sycon_grader_summary.json`** in each run directory under `neurips-results/exp17/`.

---

## 1. Best per model — `mean_traj_score` (primary)

**`mean_traj_score`** = mean number of **consecutive rounds** the model keeps **rejecting** the false presupposition **before its first flip**. **Higher is better** (stronger multi-turn resistance trajectory).

For each model/run, the row is the **single best** `(layer, α)` in that grid by **maximum `mean_traj_score`**.

| Model | Run | Best L | α | mean_traj | Steadfast | Rej R1 | Rej R5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-27B **v3**\* | `sycon_fp_v3_qwen_100q_l2_l5_l10_a0_a03_a05` | **L5** | **0.0** | **2.23** | 0.362 | 0.69 | 0.61 |
| Qwen3.5-27B **v4** (no-think) | `sycon_fp_v4_qwen_100q_l2_l5_l10_a0_a03_a05_nothink` | **L5** | **0.5** | **3.41** | 0.690 | 0.84 | 0.81 |
| Gemma-4-26B | `sycon_fp_v3_gemma_100q_l20_l22_l24_a0_a03_a05` | **L22** | **0.3** | **2.96** | 0.532 | 0.79 | 0.63 |
| OLMo-2-32B | `sycon_fp_v3_olmo_100q_l10_l16_l22_a0_a03_a05` | **L10** | **0.3** | **2.91** | 0.543 | 0.81 | 0.62 |
| OLMo-3.1-32B | `sycon_fp_v3_olmo31_100q_l15_l18_l22_a0_a03_a05_apw` | **L18** | **0.5** | **3.04** | 0.757 | 0.70 | 0.70 |
| GPT-OSS-20B | `sycon_fp_v2_gpt_oss_100q` | **L16** | **0.0** | **2.09** | 0.615 | 0.52 | 0.61 |

\* **v3:** thinking tokens leaked into evaluated output — weaker baselines than v4; use **v4** for primary science.

### Best of the best

Overall strongest graded result: **Qwen v4** no-think, **L5**, **α=0.5**, **`mean_traj_score` = 3.41**.

Then: **OLMo-3.1** — **3.04** @ L18, α=0.5 → **Gemma-4** — **2.96** @ L22, α=0.3 → **OLMo-2** — **2.91** @ L10, α=0.3.

### α sweep (on `mean_traj_score`)

- **Qwen v4:** **α=0.5** is clearly best.
- **OLMo-3.1:** **α=0.5** is also best.
- **Gemma-4:** **α=0.3** is best (not 0.5).
- **OLMo-2:** **α=0.3** is best (not 0.5).
- **GPT-OSS:** **α=0.0** (baseline, no steering) is best.
- **Qwen v3:** **α=0.0** is best.

So α=0.5 is a good default for some stacks but **not** universal.

---

## 2. Complementary metric — steering collapse (ΔFlip)

**Best** = largest **ΔFlip = Flip(α, L) − Flip(α=0, same layer)** — how much steering *collapses* resistance (higher ΔFlip = more collapse). This ranks models **differently** from §1: e.g. **GPT-OSS** shows the largest collapse at **L12 / α=0.5** even though **L16 / α=0.0** wins **`mean_traj_score`**.

Metric key: **Traj** = trajectory context in notes; **Rej R1/R5** = rejection at round 1 / 5; **Steadfast** = rejection at final round; **Flip** = 1 − Steadfast.

| Model | Run | Best L | α | Traj (base → steer) | Rej R1 | Rej R5 (base → steer) | Steadfast (base → steer) | Flip (base → steer) | ΔFlip | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **GPT-OSS-20B** | `sycon_fp_v2_gpt_oss_100q` | **L12** | 0.5 | 1.95 → **0.73** | 0.49 → 0.35 | 0.61 → **0.04** | 0.612 → **0.086** | 0.388 → **0.914** | **+0.526** | Strong collapse |
| **Qwen3.5-27B (v4)** | `sycon_fp_v4_qwen_100q_l2_l5_l10_a0_a03_a05_nothink` | **L10** | 0.5 | 3.18 → **2.48** | 0.85 → 0.79 | 0.65 → **0.41** | 0.553 → **0.367** | 0.447 → **0.633** | **+0.186** | Moderate collapse |
| **OLMo-3.1-32B** | `sycon_fp_v3_olmo31_100q_l15_l18_l22_a0_a03_a05_apw` | **L22** | 0.5 | 2.93 → 3.02 | 0.68 → 0.76 | 0.78 → 0.73 | 0.750 → **0.671** | 0.250 → **0.329** | **+0.079** | Near-null |
| **OLMo-2-32B** | `sycon_fp_v3_olmo_100q_l10_l16_l22_a0_a03_a05` | **L22** | 0.5 | 2.82 → **2.44** | 0.76 → 0.73 | 0.61 → 0.60 | 0.539 → **0.466** | 0.461 → **0.534** | **+0.074** | Weak collapse |
| **Gemma-4-26B** | `sycon_fp_v3_gemma_100q_l20_l22_l24_a0_a03_a05` | **L24** | 0.5 | 2.56 → 2.56 | 0.77 → 0.78 | 0.65 → **0.57** | 0.416 → **0.385** | 0.584 → **0.615** | **+0.031** | Very weak / possibly null |

### Deprecated (ΔFlip audit)

| Model | Run | Best L | α | Traj | Rej R5 | Steadfast | Flip | ΔFlip | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-27B **v3**\* | `sycon_fp_v3_qwen_100q_l2_l5_l10_a0_a03_a05` | L10 | 0.5 | 2.11 → 1.24 | 0.61 → 0.19 | 0.324 → 0.169 | 0.676 → 0.831 | +0.154 | Thinking leakage; superseded by v4 for claims. |

## Takeaways

- **§1 (`mean_traj_score`):** **Qwen v4** leads graded resistance trajectories (**3.41**); **OLMo-3.1** is second (**3.04**). **GPT-OSS** and **Qwen v3** look best at **α=0** on this metric (no steering).
- **§2 (ΔFlip):** **GPT-OSS** is the headline **collapse** case (~**+0.53** ΔFlip at L12/α=0.5); **OLMo-3.1** / **Gemma** show small ΔFlip — “unsteerable” on collapse even when trajectories differ.
- **v3** must not be cited as primary — use **v4**.
