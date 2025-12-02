## Persona Vectors Experiments

This repository implements the experimental pipeline for evaluating whether persona / anti-sycophancy steering vectors mechanistically rely on sycophancy attention circuits.

The code follows the spec in `experiment.md` and runs four main experiments (Exp0–Exp3) on a sycophancy dataset using Llama-family models.

---

## 1. Provenance and How to Get the Scripts

- **Dataset source**:  
  - The sycophancy data comes from the `sycophancy-eval` project (Tong et al., Anthropic).  
  - A vendored copy of the core dataset lives in `external/sycophancy-eval/datasets/answer.jsonl`.  
  - For a fresh copy, search for `meg-tong/sycophancy-eval` on Hugging Face or GitHub and download `answer.jsonl` (and related JSONL files if desired).

- **Where these scripts came from**:  
  - The overall experimental design is documented in `experiment.md`.    
  - The Python scripts in `src/` were implemented to match these specs, using standard PyTorch + Hugging Face `transformers` patterns (attention hooks, residual steering, etc.).

### **How someone else can recreate the codebase** 

Most users should instead **clone this repository directly**, which already contains the implemented scripts and configs:

```bash
git clone <this-repo-url>
cd persona-vectors
```

---

## 2. Directory & File Overview (What the Files Do)


### 2.1 Configuration

- **`config/experiment_config.yaml`**: Global config used by Exp0/baseline; points to dataset paths, model name, and seeds.  
- **`config/exp1_caa.yaml`**: Config for Exp1 (CAA anti-sycophancy vector extraction and tuning).  
- **`config/exp2_path_patching.yaml`**: Config for Exp2 (per-head path patching to identify sycophancy heads).  
- **`config/exp3_mediation.yaml`**: Config for Exp3 (mediation grid with head ablations + steering).

These YAML files control model names, device, split paths, layer ranges, and output directories.

### 2.2 Data

- **`data/answer.jsonl`**: Raw sycophancy dataset (copied from `sycophancy-eval`).  
- **`data/split_A_user_wrong.jsonl`**: User-wrong subset for Exp1 (CAA vector).  
- **`data/split_B_user_wrong.jsonl`**: User-wrong subset for Exp2 (path patching).  
- **`data/split_C_user_wrong.jsonl`**: User-wrong subset for Exp3 (mediation grid).

The splits are created deterministically (seeded) by Exp0 / dataset utilities and then reused across experiments.

### 2.3 External dataset copy

- **`external/sycophancy-eval/`**: Vendored copy of the original sycophancy-eval repo.  
  - `datasets/answer.jsonl`, `datasets/are_you_sure.jsonl`, `datasets/feedback.jsonl` – original JSONL files.  
  - `utils.py`, `example.ipynb`, `README.md` – helpers and examples from that project.

You generally do **not** need to modify this; it is here for reference and provenance.

### 2.4 Source code (`src/`)

- **`src/models/llama_loader.py`**  
  - Utility to load Llama-family models and tokenizers from Hugging Face (`LlamaForCausalLM`-style).  
  - Handles device placement, dtype, and other model-loading knobs.

- **`src/data/syc_dataset.py`**  
  - Dataset utilities around the sycophancy data.  
  - Filtering for user-wrong examples, constructing neutral prompts, and handling static A/B/C splits.  
  - Provides helper functions for the experiment scripts to get prompts and answers in the right shape.

- **`src/metrics/syc_metric.py`**  
  - Implements the sycophancy metric \(D_{syc} = \log P_\text{wrong} - \log P_\text{right}\).  
  - Exposes functions to compute \(D_{syc}\) and derived quantities (e.g., binary syc label) under teacher forcing.

- **`src/hooks/head_patch_hooks.py`**  
  - Attention-hook utilities for per-head path patching and mean ablation.  
  - Provides abstractions to cache per-head context vectors at a chosen token position and then patch or ablate them in subsequent runs.

- **`src/hooks/steering_hooks.py`**  
  - Residual-stream hook utilities for injecting steering vectors (like the CAA anti-sycophancy vector).  
  - Used by Exp1 and Exp3 to apply \( \alpha \cdot v_{\text{anti}} \) at a particular layer/position.

- **`src/exp0/run_baseline.py`**  
  - Implements **Exp0**: baseline sycophancy and dataset prep.  
  - Loads the model + dataset, computes baseline \(D_{syc}\) on user-wrong prompts, and saves:  
    - `results/exp0/baseline_records.jsonl` – per-example \(D_{syc}\) records.  
    - `results/exp0/baseline_metrics.json` – summary stats (syc rate, histograms, etc.).  
  - Also responsible for generating and saving the A/B/C user-wrong splits if they don’t already exist.

- **`src/exp1/run_caa_vector.py`**  
  - Implements **Exp1**: anti-sycophancy CAA vector.  
  - On Split A, labels sycophantic vs truthful cases using margins on \(D_{syc}\), collects residual activations, and computes per-layer CAA vectors.  
  - Runs a grid search over layers and steering strengths to find a good `(L*, α*)`.  
  - Outputs:  
    - `results/exp1/layer_vectors.pt` – per-layer steering vectors.  
    - `results/exp1/vector_selection.json` – metadata (chosen layer/α, margin, counts).

- **`src/exp2/run_path_patching.py`**  
  - Implements **Exp2**: per-head path patching to identify sycophancy heads.  
  - On Split B, constructs pairs of prompts `(P_wrong, P_neutral)` and:  
    - Computes baseline \(D_{syc}\) on `P_wrong`.  
    - Caches per-head activations for `P_neutral`.  
    - For each layer/head, patches from neutral into wrong and measures the change in \(D_{syc}\).  
  - Outputs raw per-head per-example records to `results/exp2/head_results_raw.jsonl`.

- **`src/exp2/aggregate_head_scores.py`**  
  - Aggregates `head_results_raw.jsonl` by `(layer, head)` to produce head-level scores.  
  - Selects a minimal set of sycophancy heads `H_syc` that explain a target fraction of total positive effect, plus matched random heads `H_rand`.  
  - Outputs:  
    - `results/exp2/head_scores.jsonl` – per-head aggregates.  
    - `results/exp2/syc_heads.json`, `results/exp2/rand_heads.json`.  
    - Optionally a summary JSON (e.g., layer bounds) for use by other experiments.

- **`src/exp3/run_mediation_grid.py`**  
  - Implements **Exp3**: mediation grid on Split C.  
  - Uses:  
    - `H_syc`, `H_rand` from Exp2.  
    - `(L*, α*)` and `v_anti` from Exp1.  
  - Runs six conditions per example (no steering vs steering, with/without syc/random head ablation) and saves:  
    - `results/exp3/mediation_results.jsonl` – per-example \(D_{syc}\) under all conditions.  
    - `results/exp3/mediation_summary.json` – aggregate RAS-style statistics.

- **`src/analysis/plot_results.py`**  
  - Convenience plotting script that turns numeric results into publication-style figures.  
  - Reads `results/exp0`, `results/exp1`, `results/exp2`, `results/exp3` and writes PNGs to `results/figures/`:  
    - `baseline_dsyc_hist.png`, `exp1_caa_grid.png`, `exp2_head_deltas.png`, `exp3_mediation_grid.png`.

- **`src/persona/`**  
  - Placeholder package for the optional **persona bridge** experiments described in `experiment.md`.  
  - Intended to host scripts like `run_persona_vector.py` and `compare_vectors.py` (cosine similarity between instruction-based and behavioral vectors).  
  - Current status: **WIP** – behavior may not be fully implemented yet.

### 2.5 Notebooks and tests

- **`notebooks/prototype_logic.ipynb`**: Early scratchpad for exploring dataset structure and experiment logic.  
- **`test.ipynb`, `test.py`**: Ad hoc test harnesses / sanity checks; not part of the formal pipeline.

### 2.6 Results

- **`results/exp0/`**: Baseline metrics and per-example records.  
- **`results/exp1/`**: Learned CAA vectors and selection metadata.  
- **`results/exp2/`**: Per-head path patching results and selected head sets.  
- **`results/exp3/`**: Mediation-grid results and summary stats.  
- **`results/figures/`**: Generated figures for quick inspection and paper inclusion.

---

## 3. How to Reproduce the Results (WIP)

This section describes the intended end-to-end pipeline. Some pieces (especially persona-vector scripts) are still **work in progress**, but the core Exp0–Exp3 path is implemented.

### 3.1 Environment setup

This project is configured for **Python 3.12** and uses `uv` for dependency management.

```bash
# In the repo root
uv sync  # creates .venv and installs dependencies from pyproject.toml

# (Optional) if you haven’t already initialized uv metadata here:
# uv init
```

Activate the environment using your preferred method (e.g., `uv run` or by activating `.venv` directly).

### 3.2 Data setup

There are two common paths:

- **Using the vendored data (simplest)**:
  - `data/answer.jsonl` is already populated from `external/sycophancy-eval/datasets/answer.jsonl`.  
  - You can run Exp0–Exp3 directly without additional downloads.

- **Regenerating from the canonical source**:
  1. Download `answer.jsonl` from the `sycophancy-eval` project (e.g., `meg-tong/sycophancy-eval` on Hugging Face).  
  2. Place it at `data/answer.jsonl`.  
  3. (Optional) Update the path in `config/experiment_config.yaml` if you use a custom location.

The user-wrong splits (`split_A/B/C_user_wrong.jsonl`) will be created deterministically by Exp0 if they don’t already exist.

### 3.3 Running the experiments

All commands below assume you’re in the repo root and using `uv run` to execute Python within the project environment.

- **Exp0 – Baseline + Split Creation**

  ```bash
  uv run python -m src.exp0.run_baseline --config config/experiment_config.yaml
  ```

  This will:
  - Load the model and dataset.  
  - Compute baseline \(D_{syc}\) on user-wrong prompts.  
  - Create `data/split_A_user_wrong.jsonl`, `data/split_B_user_wrong.jsonl`, `data/split_C_user_wrong.jsonl` if missing.  
  - Write metrics to `results/exp0/`.

- **Exp1 – CAA Anti-Sycophancy Vector**

  ```bash
  uv run python -m src.exp1.run_caa_vector --config config/exp1_caa.yaml
  ```

  This will:
  - Use Split A and `metrics/syc_metric.py` to label sycophantic vs truthful examples.  
  - Collect residual activations and compute per-layer CAA vectors.  
  - Run a grid search over `(layer, α)` and pick `(L*, α*)`.  
  - Save vectors and selection metadata in `results/exp1/`.

- **Exp2 – Sycophancy Heads via Path Patching**

  ```bash
  uv run python -m src.exp2.run_path_patching --config config/exp2_path_patching.yaml
  uv run python -m src.exp2.aggregate_head_scores --config config/exp2_path_patching.yaml
  ```

  This will:
  - For examples in Split B, run per-head path patching between `P_wrong` and `P_neutral`.  
  - Save raw per-head results to `results/exp2/head_results_raw.jsonl`.  
  - Aggregate and select `H_syc` and `H_rand`, writing JSON/JSONL outputs in `results/exp2/`.

- **Exp3 – Mediation Grid**

  ```bash
  uv run python -m src.exp3.run_mediation_grid --config config/exp3_mediation.yaml
  ```

  This will:
  - Use Split C, the CAA vector from Exp1, and head sets from Exp2.  
  - Run the 6-condition mediation grid for each example.  
  - Save detailed results and a summary in `results/exp3/`.

### 3.4 Generating figures

Once Exp0–Exp3 have run, you can regenerate the main diagnostic figures with:

```bash
uv run python -m src.analysis.plot_results
```

This will write PNGs into `results/figures/`, matching the figures currently checked into the repo.

### 3.5 Persona-vector bridge (WIP)

The optional persona bridge described in `experiment.md` (instruction-based persona vectors and cosine similarity to the CAA vector) is **not yet fully wired up**:

- `src/persona/` is intended to contain:
  - `run_persona_vector.py` – extract sycophantic vs truthful persona vectors from system prompts.  
  - `compare_vectors.py` – compute cosine similarity between persona vectors and the behavioral CAA vector.
- To reproduce those results, you would need to implement these scripts following Section 9 of `experiment.md`, then run them after Exp1 so they can reuse `v_anti` and `(L*, α*)`.

---

## 4. Quickstart (Minimal)

If you just want to see something run and produce outputs:

```bash
# In the repo root
uv sync

# Baseline (Exp0)
uv run python -m src.exp0.run_baseline --config config/experiment_config.yaml

# Plots
uv run python -m src.analysis.plot_results
```

This should populate `results/exp0/` and `results/figures/` with baseline metrics and figures. From there, you can proceed to Exp1–Exp3 as described above.
