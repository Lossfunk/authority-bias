## Persona Vectors: Authority Bias and Endorsement Effects in LLMs

This repository implements the experimental pipeline for studying **endorsement effects** and **authority-weighted bias** in language models. We ask: when someone endorses an answer, does the model track truth or merely defer? Do "be correct" instructions induce truth-tracking, or do they amplify prior-consistency?

**Core finding**: "Be correct" instructions do not induce truth-tracking. They induce confidence-conditioned prior-consistency control — the model trusts its own beliefs more, which can entrench errors when the model is confidently wrong. Evidence-quality sensitivity exists but is conditional on prior state, authority tag, and model family.

---

## 1. Provenance and How to Get the Scripts

- **Dataset**: TriviaQA + TruthfulQA (n = 1,813 items), prepared as multiple-choice prompts with correct/wrong answer pairs.
- **Models**: Llama-3.1-8B (Base, Instruct), Qwen3-4B-Instruct-2507, Qwen3-4B-Thinking-2507.
- **Sycophancy-eval reference**: We use conventions from `sycophancy-eval` (Tong et al.). A vendored copy lives in `external/sycophancy-eval/datasets/`.

### How to recreate the codebase

```bash
git clone <this-repo-url>
cd persona-vectors
```

---

## 2. Directory & File Overview

### 2.1 Data

- **`data/exp7_mc_dataset.jsonl`**: Multi-choice dataset (1,813 items) used by Exp7 onward. Created by `src.exp7` from TriviaQA + TruthfulQA.
- **`data/exp14_reasons.jsonl`** / **`data/exp14_reasons_assertive_v2.jsonl`**: Generated reasons for Exp14 evidence-quality experiments (bare → reason1 → reason2 → reason_data).
- **`external/sycophancy-eval/`**: Vendored sycophancy-eval datasets and utils.

### 2.2 Results

- **`new-phase-results/`**: Outputs from the endorsement/authority pipeline (Exp6–Exp14).
  - `new-phase-results/llama-3.1-8b-results/`, `new-phase-results/qwen3-4b-results/`, `new-phase-results/qwen/`: per-model experiment results.
  - `new-phase-results/figures/`, `new-phase-results/figures/paper/`: figures.


### 2.3 Source Code (`src/`)

**Shared infrastructure**

- **`src/models/llama_loader.py`**: Load Llama/Qwen models and tokenizers.
- **`src/metrics/syc_metric.py`**: Sycophancy metric (\(D_{syc} = \log P_\text{wrong} - \log P_\text{right}\)).
- **`src/hooks/`**: Attention and steering hooks (used by legacy exp0–exp3; kept for compatibility).

**Experiment pipeline (Exp6–Exp14)**

| Experiment | Purpose |
|------------|---------|
| **Exp6** | Phase 1/2 sycophancy gut check (prompt-induced shift, forced-choice). |
| **Exp7** | Lexical-fixed endorsement measurement — endorsement survives when both options appear. |
| **Exp8** | Order vs endorsement decomposition; **Exp8_Speakers**: authority hierarchy (Expert > Note > User > Online). |
| **Exp9** | Instruction override test — "be correct" reduces endorsement effects. |
| **Exp10** | Correct-endorsement test — Instruct shows apparent selectivity (truth-tracking vs gating). |
| **Exp11** | Inverted-prior test — selectivity collapses (Expert) or inverts (Note) on confidently-wrong items → **prior-consistency**. |
| **Exp12** | Temporal dynamics — context persistence, instruction timing (`t0` vs `t1`/`t2`), repeated pressure. |
| **Exp13** | Signal strength — certainty phrasing and formatting effects. |
| **Exp14** | Evidence-quality scaling — bare → reasons → data; prior-wrong vs all-items stratification. |
| **Exp14D** | Authority × evidence factorial — crossover interaction in Qwen-Instruct (Expert vs Note). |

**Key modules**

- **`src/exp7/`**: `dataset_mc.py`, `run_lexical_fixed.py`, `scoring.py` — dataset creation and forced-choice scoring.
- **`src/exp8/`**: `run_speaker_tags.py`, `speaker_conditions.py` — authority-tagged endorsement conditions.
- **`src/exp9/`**: `run_instruction_override.py` — instruction vs no-instruction comparison.
- **`src/exp10/`**: `run_correct_endorse.py` — neutral / wrong / correct endorsement conditions.
- **`src/exp11/`**: `analyze_inverted_prior.py`, `logit_metrics.py` — inverted-prior test and selectivity metrics.
- **`src/exp12/`**: `run_persistence_washout.py`, `run_repeated_endorsement.py` — temporal dynamics.
- **`src/exp13/`**: `run_signal_strength.py` — certainty sweep and formatting variants.
- **`src/exp14/`**: `run_evidence_quality.py`, `generate_reasons.py`, `analyze_evidence_quality.py` — evidence-quality scaling and prior stratification.

**Paper figures**

- **`src/paper_figures/`**: Publication-style figures (fig_endorsement, fig_authority, fig_selectivity, fig_evidence_quality, fig_signal_strength, etc.).

---

## 3. Experimental Progression (Summary)

The pipeline follows a confound-driven progression:

```
Exp6/7  → Endorsement effect survives lexical control
Exp8    → Endorsement >> order; authority hierarchy (Expert top)
Exp9    → Instructions reduce endorsement effects
Exp10   → Instruct shows apparent selectivity (wrong > correct suppressed)
Exp11   → On confidently-wrong items: Expert → non-selective; Note → prior-consistency (dr = -0.67)
Exp12   → Temporal persistence; t0 > delayed correction; fresh restarts wash out
Exp13   → Certainty phrasing = gain knob; formatting model-dependent
Exp14   → Evidence quality: conditional truth-sensitivity; prior-wrong vs all-items sign reversal
Exp14D  → Authority × evidence crossover in Qwen-Instruct
```


---

## 4. How to Reproduce

### 4.1 Environment

```bash
uv sync
```

Python 3.12, PyTorch, transformers, etc. are managed via `pyproject.toml`.

### 4.2 Data setup

The MC dataset is built from TriviaQA + TruthfulQA. If `data/exp7_mc_dataset.jsonl` does not exist, Exp7 will create it when run. For Exp14, you need `data/exp14_reasons.jsonl` or `data/exp14_reasons_assertive_v2.jsonl` (see `src.exp14.generate_reasons`).

### 4.3 Running experiments

Commands assume repo root and `uv run`:

**Exp7 — Lexical-fixed endorsement**

```bash
uv run python -m src.exp7.run_lexical_fixed --model meta-llama/Llama-3.1-8B-Instruct --output-dir results/exp7
```

**Exp8 — Speaker tags (authority hierarchy)**

```bash
uv run python -m src.exp8.run_speaker_tags --mc-dataset-path data/exp7_mc_dataset.jsonl --output-dir results/exp8_speakers
```

**Exp9 — Instruction override**

```bash
uv run python -m src.exp9.run_instruction_override --mc-dataset-path data/exp7_mc_dataset.jsonl --output-dir results/exp9
```

**Exp10 — Correct endorsement (selectivity)**

```bash
uv run python -m src.exp10.run_correct_endorse --mc-dataset-path data/exp7_mc_dataset.jsonl --output-dir results/exp10
```

**Exp11 — Inverted-prior test** (requires Exp10 results)

```bash
uv run python -m src.exp11.analyze_inverted_prior  # uses exp10 results
```

**Exp12 — Temporal dynamics**

```bash
uv run python -m src.exp12.run_persistence_washout --mc-dataset-path data/exp7_mc_dataset.jsonl --output-dir results/exp12
uv run python -m src.exp12.run_repeated_endorsement --mc-dataset-path data/exp7_mc_dataset.jsonl --output-dir results/exp12_k
```

**Exp13 — Signal strength**

```bash
uv run python -m src.exp13.run_signal_strength
```

**Exp14 — Evidence quality**

```bash
uv run python -m src.exp14.run_evidence_quality --mc-dataset-path data/exp7_mc_dataset.jsonl --reasons-path data/exp14_reasons_assertive_v2.jsonl --output-dir new-phase-results/exp14
```

### 4.4 Paper figures

```bash
uv run python -m src.paper_figures.fig_endorsement
uv run python -m src.paper_figures.fig_authority
uv run python -m src.paper_figures.fig_selectivity
# etc.
```

Output paths are typically under `new-phase-results/figures/` or `new-phase-results/figures/paper/`.

---

## 5. Quickstart

Minimal run to see endorsement effects:

```bash
uv sync
uv run python -m src.exp7.run_lexical_fixed --model meta-llama/Llama-3.1-8B-Instruct --num-questions 20 --output-dir results/exp7
uv run python -m src.exp7.analyze_results --results-dir results/exp7
```

This will create the MC dataset (if missing), run a small lexical-fixed experiment, and produce summary outputs.

---

## 6. Key Definitions

- **effect_wrong_I0**: Wrong-endorsement effect with no instruction.
- **r_w, r_c**: Relative suppression of wrong vs correct endorsement under instruction.
- **dr = r_w − r_c**: Differential suppression; positive = truth-tracking, negative = prior-consistency.
- **m_N0 < 0**: Model's neutral prior is wrong (confidently wrong slice).
- **tau_gap**: Kendall tau(correct) − tau(wrong); positive = truth-sensitive evidence scaling.

