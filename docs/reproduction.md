# Experiment guide

The repository contains the original experiments and a later, separately packaged rebuttal pipeline. Start with the experiment you want to inspect. Running every numbered module is neither required nor a supported one-command reproduction procedure.

## Code-to-experiment map

| Experiment | Entry points |
| --- | --- |
| Matched prompt variants and baseline behavior | `src/exp7/`, `src/exp10/` |
| Authority activations, direction fitting, and steering | `src/exp16/extract_authority_activations.py`, `analyze_compliance_direction.py`, `run_steering_test.py` |
| PIQA transfer | `src/exp16/run_piqa_interventions.py`, `scripts/run_piqa_mechanism_transfer.sh` |
| Multi-turn SYCON evaluation and judging | `src/exp17/` |
| Assistant and affect controls; Gemma follow-ups | `src/exp19/` |
| Source/user removal comparison | `src/exp20/run_source_user_authority_split.py` |
| Original CAA baseline and mitigation sweeps | `src/exp20/fit_caa_sycophancy.py`, `run_mitigation_pareto.py` |
| Capability checks | `src/exp20/run_capability_eval.py`, `run_general_capability_pareto.py` |
| Held-out selection checks | `src/exp22/` |
| Retrieved-document and system-message transfer | `src/exp23/run_authority_transfer_contexts.py` |
| Later attribution patching and optimized CAA comparisons | `rebuttal/` |
| Original paper figure scripts | `src/paper_figures/` |

Run original modules from the repository root with `uv run python -m`, followed by the dotted module name. Use `--help` to inspect their arguments before launching a model. Many defaults refer to historical artifact paths, so pass the dataset, direction, layer, and output paths for the intended run explicitly.

## Fixed multiple-choice data

`scripts/prepare_trivia_data.py` reconstructs the input used by the frozen rebuttal configs. The upstream SycophancyEval answer file contains TriviaQA and TruthfulQA entries; the prepared file preserves all 1,813 deduplicated items. Eligibility masks and split lists select the evaluated items.

Preparation uses `src.exp7.dataset_mc.create_mc_examples` with seed 7, preserving question order, wrong answers, UIDs, and option assignment. It checks the raw download and the generated file against the archived SHA-256 values. Changing the seed or rebuilding against a different upstream file will not reproduce the frozen input.

```bash
python3 scripts/prepare_trivia_data.py --download
```

An existing `data/exp7_mc_dataset.jsonl` is validated, not overwritten. Model weights and API keys are separate prerequisites. The CAA pipeline fetches its evaluation dataset from the commit recorded in the model config. SYCON is a Git submodule at a fixed revision:

```bash
git submodule update --init external/SYCON-Bench
```

## Original experiment artifacts

The original pipelines commonly read activations, extracted directions, and per-item generation files from an earlier stage. The Git repository keeps compact summaries but omits the large dumps. Recreate missing artifacts with the corresponding extraction or evaluation module before running dependent analyses. A saved summary alone cannot reconstruct a per-item bootstrap interval.

PIQA preparation is supported by `scripts/build_commonsense_mc_dataset.py` and by the explicit rebuild options in `src.exp16.run_piqa_interventions`. For a historical run, preserve its recorded item IDs and option assignments instead of assuming a newly downloaded split has the same ordering.

The `scripts/run_*` and `experiments/run_*` files record experiment launches. Some older scripts contain machine-specific working directories or environment paths. Read and adapt those paths before using them on a new host; the later `rebuttal/run_qwen_h200.sh` resolves its repository root automatically.

## Reading results

Keep the model revision, item subset, layer, intervention strength, prompt wording, and scoring rule with each result. In particular:

- Matched flip rates condition on a baseline-correct subset; wrong-answer rates need not use that denominator.
- Probability-margin scores are not percentages of generated answers.
- A direction fitted at one position is not interchangeable with a cue-span direction.
- Tuning scores, selected sweep cells, and held-out evaluations answer different questions.
- Historical directories contain intermediate runs as well as final analyses. Their names alone do not identify the corresponding preprint table.

For a new run, retain the manifest, split IDs, full configuration, and per-item outputs outside Git. These are needed to verify a table or recompute uncertainty later.
