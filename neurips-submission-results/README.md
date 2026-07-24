# NeurIPS Submission Results

This folder contains the curated JSON result artifacts that back the current submission claims, the experiment code that produced them, and a single shell script that re-runs the full pipeline on a fresh GPU box.

It is intentionally lightweight: no `.pt` files, activation caches, direction tensors, row dumps, smoke tests, sanity runs, or archive folders are included.

## Top-level layout

```
neurips-submission-results/
  README.md                  this file
  run_all.sh                 single-command end-to-end reproduction script
  manifest.json              per-result-file source/dest/category/model/size index
  code_manifest.json         per-code-file source/dest index for codes/
  behavioral_override_and_gradient/<model>/ ...
  mechanism_and_transfer/<model>/ ...
  sycon_transfer/<model>/ ...
  wang_source_user_pareto_capability/<model>/ ...
  codes/
    behavioral_override_and_gradient/
    mechanism_and_transfer/
    sycon_transfer/
    wang_source_user_pareto_capability/
    selection_holdout_50_50/
    authority_transfer_contexts/
```

### New rebuttal-driven categories

- `codes/selection_holdout_50_50/` — exp22 clean 50/50 held-out validation harness (`src.exp22.run_selection_holdout`, `src.exp22.summarize_selection_holdout`). Reuses the source/user split runner and per-model exp20 sweeps to fit directions only on a deterministic train fold and evaluate headline cells only on the held-out fold.
- `codes/authority_transfer_contexts/` — exp23 transfer of the source/user directions to system-prompt and RAG authority cues (`src.exp23.run_authority_transfer_contexts`). Held-out 50/50 fold reuse exp22 directions when present.

## Result categories

Results are grouped first by result type, then by model:

- `behavioral_override_and_gradient/<model>/` — behavioral override and authority-gradient summaries used for the headline behavior/gradient figures.
- `mechanism_and_transfer/<model>/` — activation-analysis summaries, probe/null-control summaries, steering summaries, PIQA transfer summaries, assistant-axis overlap summaries, and lexical VAD summaries.
- `sycon_transfer/<model>/` — SYCON transfer summaries and grader summaries.
- `wang_source_user_pareto_capability/<model>/` — Wang-style source/user probes, source-vs-user split summaries and geometry, authority-minus-assistant/assistant/CAA Pareto runs, CAA direction metadata, and MMLU-Pro/GSM8K capability summaries.

The main open-weight models are:

- `qwen35`   — Qwen/Qwen3.5-27B
- `gpt_oss`  — openai/gpt-oss-20b
- `olmo2`    — allenai/OLMo-2-0325-32B-Instruct
- `olmo31`   — allenai/OLMo-3.1-32B-Instruct
- `gemma4`   — google/gemma-4-26B-A4B-it

The behavioral folder also includes API/closed-model comparison summaries used in the headline behavior figure:

- `gpt54`        — openai/gpt-5.4
- `grok`         — x-ai/grok-4.20
- `gemini31pro`  — google/gemini-3.1-pro-preview

## Codes

`codes/<category>/` contains only the Python files needed to reproduce that
category's results, preserving the original `src/expNN/...` layout so each
file can be invoked directly with `python -m src.exp16.run_steering_test ...`
(etc.) from the repository root.

There is no `pyproject.toml` or virtual-env config inside `codes/` — those
live in the repository root and are installed by `run_all.sh` via `uv`.

## Reproducing the results

`run_all.sh` is the single entry point that reproduces every JSON in this
folder from a fresh GPU box (Linux, CUDA, ≥1× A100/H100 80 GB).

```bash
# From the repository root, on a Linux GPU box:
HF_TOKEN=hf_xxx bash neurips-submission-results/run_all.sh
```

The script runs the following phases in order. Each phase has a sentinel
file the next run checks before re-doing work, so the script is idempotent;
delete the corresponding per-cell summary JSON to force a re-run.

| # | Phase           | What runs                                                                                                |
|--:|:----------------|:----------------------------------------------------------------------------------------------------------|
| 1 | `setup`         | Install `uv`, `uv sync` (base env), then `uv sync --extra gpu` with `--no-build-isolation-package` for `flash-attn`, `flash-linear-attention`, `causal-conv1d` so they compile against the in-venv torch. Detects compute capability from `nvidia-smi` and chooses `MAX_JOBS` based on cgroup memory. Verifies torch + flash-attn imports. |
| 2 | `exp10_priors`  | Per-model Exp10 forced-choice baselines (`src.exp10.run_correct_endorse`). The `*_results.jsonl` priors are needed by the dissociation runner's `all_prior_wrong` UID selection. |
| 3 | `behavioral`    | Four-style sweep (`authoritative_verified`, `weak`, `uncertain`, `assertive`) using `src.exp16.run_dissociation_test` with `subset_mode=all_prior_wrong`. Produces the dissociation summary + rows per (model, style). |
| 4 | `mechanism`     | Per model: `src.exp16.extract_authority_activations` → `src.exp16.analyze_compliance_direction` → `src.exp16.run_steering_test` (W1 endorsement_span, layers from the per-model sweep) → `src.exp16.run_piqa_interventions`. |
| 5 | `sycon`         | `src.exp17.run_sycon_false_presupp_freegen` at α ∈ {0, 0.3, 0.5}, 100 questions per model. |
| 6 | `wang`          | `src.exp20.run_wang_probe` across all 5 models. Produces the behavioral source-vs-user probe table. |
| 7 | `source_user`   | `src.exp20.run_source_user_authority_split` (Qwen + GPT-OSS by default; configurable via `SOURCE_USER_MODELS`). Fits per-layer source / user / assistant directions and runs the causal split. |
| 8 | `pareto`        | Clones [`nrimsky/CAA`](https://github.com/nrimsky/CAA) into `/tmp/caa-readonly`, runs `src.exp20.fit_caa_sycophancy`, then `src.exp20.run_mitigation_pareto` for `residualized,assistant,caa` on trivia + PIQA (W1_note). |
| 9 | `capability`    | `src.exp20.run_general_capability_pareto` for MMLU-Pro + GSM8K with the same three vectors. |
|10 | `selection_holdout` | `src.exp22.run_selection_holdout` 50/50 fold0 across `qwen35,gpt_oss,olmo2,olmo31` for forward steering, source/user, and mitigation tasks. Re-fits directions only on the train fold; evaluates only on the held-out fold. |
|11 | `authority_transfer` | `src.exp23.run_authority_transfer_contexts` reuses the exp22 train-fold directions and evaluates them on `note`, `system`, and `rag` source/user authority prompts on the held-out fold. |

Phase / model gating is supported via env vars:

```bash
# only set up the env, then exit:
PHASES=setup bash neurips-submission-results/run_all.sh

# only run Qwen mechanism + Pareto + capability:
MODELS=qwen35 PHASES=mechanism,pareto,capability bash neurips-submission-results/run_all.sh

# add OLMo-3.1 to the source-vs-user split:
SOURCE_USER_MODELS=qwen35,gpt_oss,olmo31 PHASES=source_user bash neurips-submission-results/run_all.sh
```

### Configuration knobs (env vars)

| Variable                | Default                                  | Purpose                                                                 |
|:------------------------|:------------------------------------------|:------------------------------------------------------------------------|
| `HF_TOKEN`              | unset (warns)                             | HuggingFace token. Required for gated models (Gemma / OLMo / Llama).    |
| `MODELS`                | `qwen35,gpt_oss,olmo2,olmo31,gemma4`      | Subset of the five open-weight models to run.                           |
| `SOURCE_USER_MODELS`    | `qwen35,gpt_oss`                          | Subset for which to fit source-vs-user causal splits.                   |
| `PHASES`                | all 9 phases                              | Comma-separated subset of `setup,exp10_priors,behavioral,mechanism,sycon,wang,source_user,pareto,capability`. |
| `LOG_DIR`               | `<repo>/run_all_logs`                     | Where to tee per-phase logs.                                            |
| `BEHAVIORAL_BATCH`      | `8`                                       | Per-GPU batch size for behavioral / Exp10 baselines.                    |
| `TRIVIA_MAX_ITEMS`      | `512`                                     | Trivia items per Pareto cell.                                            |
| `PIQA_MAX_ITEMS`        | `200`                                     | PIQA items per Pareto cell.                                              |
| `SOURCE_USER_MAX_ITEMS` | `512`                                     | Source-vs-user split items per cell.                                     |
| `ALPHAS`                | `0,0.25,0.5,0.75,1`                       | Steering alphas swept by the Pareto pipeline.                            |
| `MAX_JOBS`              | auto from cgroup mem                      | Parallel `nvcc` invocations during GPU extras compile.                   |
| `TORCH_CUDA_ARCH_LIST`  | auto from `nvidia-smi`                    | Torch compute-cap target. Override for older GPUs (e.g. `8.0` for A100). |
| `USE_HUB_KERNELS`       | `NO`                                      | Set so transformers uses our locally-built flash-attn-2 instead of fetching the Hopper-only flash-attn-3 from HF Hub. |

### Required answer suffix

The Pareto / source-vs-user / capability pipelines all pass a fixed suffix to
the steering and PIQA runners so dynamic-parser markers line up across runs:

```
Answer the question in one short sentence using the full answer text, not option letters.
```

This is set as `ANSWER_SUFFIX` near the top of `run_all.sh` and threaded
through every relevant subcommand.

### Cost / time

End-to-end the full pipeline (5 models × phases 3–9) takes roughly **1 week
of wall-clock time on a single A100-80GB**, possibly longer.

Speed varies sharply by model:

- `qwen35`, `olmo2`, `olmo31` benefit from the flash-attn-2 / flash-linear-attention
  fast paths and finish individual cells in tens of minutes.
- `gpt_oss` and `gemma4` do **not** benefit from the same speedups (the
  Hopper-only flash-attn-3 path is disabled via `USE_HUB_KERNELS=NO`, MoE
  routing in gpt-oss adds overhead, and Gemma's no-thinking decoding path is
  serialized). Per-experiment cells on these two models commonly take
  4–5 hours each, so a single model can take 1.5–2.5 days end-to-end.

If you have multiple GPUs, the cheapest parallelisation is to run
`MODELS=<one_model> bash neurips-submission-results/run_all.sh` per GPU in
separate shells (one model per process, with `CUDA_VISIBLE_DEVICES` set).
Phases are sequential within a single invocation; the script does not
auto-parallelise across models.

Plan for at least:

- 2–3 days each on `gpt_oss` and `gemma4`,
- 1 day each on `qwen35`, `olmo2`, `olmo31`,
- another 12–24 h on the cross-model phases (`wang`, `source_user`, `pareto`
  CAA fits, capability eval).

Total: roughly **5–8 days** on one A100-80GB, less with multi-GPU
parallelism across models.

## Provenance

`manifest.json` lists every copied result file with:

- source path in the repository
- destination path in this curated folder
- category
- model
- file size

`code_manifest.json` does the same for the files under `codes/`.

Shared aggregate files were omitted when model-specific summaries were
available, to keep this directory strictly model-wise.
