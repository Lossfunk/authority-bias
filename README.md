# Authority Bias in Language Models

Research code for **Authority Bias in Language Models: Source Deference and User Agreement Are Not Interchangeable**, by Abhinav Rajeev Kumar and Paras Chopra, Lossfunk.

Language models can abandon a correct answer when a prompt attributes a conflicting claim to a "verified" source. We study whether this source deference differs from agreement with a user, and whether activation interventions can change it. The experiments combine matched prompt comparisons, direction removal, attribution patching, and transfer to other tasks.

The source and user directions can overlap strongly while having different behavioral effects. These effects vary by model; the code includes the assistant-direction controls and the Gemma failure analysis, not just the successful interventions.

## Getting started

Clone the repository with its pinned evaluation dependencies. Use Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/Lossfunk/persona-vectors.git
cd persona-vectors
```

There are two Python environments. The root project contains the original experiments; `rebuttal/` contains the later attribution-patching and CAA comparisons with a separate lockfile. Install the environment for the experiment you intend to run.

For the rebuttal pipeline and CPU tests:

```bash
uv sync --project rebuttal --extra test --frozen
uv run --project rebuttal --extra test --frozen pytest rebuttal/tests
uv run --project rebuttal --extra test --frozen ruff check rebuttal
```

For the original experiments:

```bash
uv sync --frozen
uv run python -m src.exp16.run_steering_test --help
```

Model experiments require a suitable NVIDIA GPU and model access. The rebuttal runner targets a single H200 with a CUDA 12.8-compatible driver. CPU tests do not download model weights or reproduce the paper's numerical results.

## Data and reproducibility

Prepare the fixed multiple-choice dataset before running experiments:

```bash
python3 scripts/prepare_trivia_data.py --download
```

This downloads a pinned version of the public [SycophancyEval](https://github.com/meg-tong/sycophancy-eval) answer dataset and rebuilds `data/exp7_mc_dataset.jsonl` with the original label-randomization seed. Both files are checksum-verified. To use an existing download instead:

```bash
python3 scripts/prepare_trivia_data.py --raw-path /path/to/answer.jsonl
```

The repository includes compact result summaries, recorded split IDs, configurations, and the small assistant vectors required by the rebuttal configs. Activation dumps, model weights, raw generation logs, and local credentials are excluded. Upstream datasets and models remain subject to their own terms.

See the [experiment guide](docs/reproduction.md) for the code-to-experiment map, required artifacts, and the distinction between probability-margin and generated-answer evaluations. Older result folders record different runs and should not be treated as interchangeable estimates.

## Attribution patching and CAA

After preparing the data, validate the Qwen configuration without loading a model:

```bash
uv run --project rebuttal --frozen qwen-rebuttal prepare \
  --repo-root . --output-root /tmp/persona-vectors-prepare
```

On the GPU host, export `HF_TOKEN` through your shell or secret manager, then run:

```bash
REBUTTAL_OUTPUT_ROOT=/path/outside/the/repo/results \
  bash rebuttal/run_qwen_h200.sh
```

The runner executes tests before the experiments. Its default `reviewer-run` command covers attribution patching, controls, robustness checks, CAA tuning, and a generated-answer check. The separate `head-to-head` command compares authority removal with tuned CAA on matched generated-answer evaluations. Configurations for Qwen3.5, GPT-OSS, and OLMo-3.1 are under `rebuttal/configs/`; commands and hardware requirements are documented in the [pipeline README](rebuttal/README.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | Original behavioral, intervention, transfer, and analysis code |
| `rebuttal/` | Later experiments, pinned model configs, and unit tests |
| `scripts/` | Data preparation, run scripts, and analysis helpers |
| `config/` | Original model and experiment configurations |
| `neurips-results/`, `causal-deconfound/`, `wang-pareto-results/` | Archived compact results and run metadata |
| `external/` | Third-party evaluation code, including the SYCON and SycophancyEval submodules |
| `docs/` | Reproduction guide and historical research notes |
| `paper/`, `neurips_2026.tex` | Historical manuscript sources, not the current preprint |

The numbered `src/exp*` directories retain their original names because scripts and saved manifests refer to them. The experiment guide identifies the relevant modules without changing those paths.

## Citation and contact

Until the preprint has a public identifier, this entry cites the repository:

```bibtex
@misc{kumar2026authoritybiascode,
  author = {Kumar, Abhinav Rajeev and Chopra, Paras},
  title = {Authority Bias in Language Models: Source Deference and User Agreement Are Not Interchangeable},
  year = {2026},
  howpublished = {Research code},
  url = {https://github.com/Lossfunk/persona-vectors}
}
```

For questions about the experiments, open an issue or contact [Abhinav Rajeev Kumar](mailto:abhinav.kumar@lossfunk.com). The repository uses evaluation resources from SycophancyEval, SYCON-Bench, CAA, and other upstream projects; their references and notices are retained in the corresponding code and directories.
