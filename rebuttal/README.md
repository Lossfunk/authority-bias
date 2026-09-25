# Attribution patching and CAA comparisons

This package contains the later source/user experiments for Qwen3.5, GPT-OSS, and OLMo-3.1. It has its own environment and does not import the original `src/` experiment modules. Historical datasets, splits, and saved assistant vectors are declared inputs.

The package retains the `qwen_rebuttal` module and `qwen-rebuttal` command names for compatibility with recorded runs.

## Setup and tests

Run these commands from the repository root:

```bash
uv sync --project rebuttal --extra test --frozen
uv run --project rebuttal --extra test --frozen pytest rebuttal/tests
uv run --project rebuttal --extra test --frozen ruff check rebuttal
python3 scripts/prepare_trivia_data.py --download
```

The CPU tests exercise prompt construction, direction arithmetic, hook placement, scoring, data partitions, and artifact contracts using small or mock inputs. They do not validate full-model results.

## Configurations and inputs

| Config | Model |
| --- | --- |
| `configs/qwen_h200.json` | Qwen/Qwen3.5-27B |
| `configs/gptoss_h200.json` | openai/gpt-oss-20b |
| `configs/olmo31_h200.json` | allenai/OLMo-3.1-32B-Instruct |

Each config pins the model revision, attention backend, dataset hash, split hashes, and CAA source revision. The assistant-vector files are under `configs/assistant_axis/` and are also checksum-verified. The three configs share the recorded Qwen eligibility mask and split lists; their paths do not imply model-specific baseline filtering.

The configs preserve the archived experimental settings. Check the selected command and the run manifest when comparing an output with a paper table; not every command reproduces every table.

You can validate inputs and write the run manifest without loading a model:

```bash
uv run --project rebuttal --frozen qwen-rebuttal prepare \
  --repo-root . \
  --config rebuttal/configs/qwen_h200.json \
  --output-root /tmp/authority-bias-prepare
```

Outputs must be outside the repository. Commands other than `prepare` require a clean code checkout so the manifest identifies the code used.

## GPU runs

The runner expects Linux, one visible H200, a CUDA 12.8-compatible driver, `git`, `uv`, at least 180 GiB free under the output root, and access to the pinned checkpoint. Export `HF_TOKEN` before running; do not put it in a config or commit it.

```bash
REBUTTAL_OUTPUT_ROOT=/path/outside/the/repo/results \
  bash rebuttal/run_qwen_h200.sh
```

This installs the GPU extras, runs tests, and executes `reviewer-run` with the Qwen config. To select another configuration or experiment:

```bash
REBUTTAL_CONFIG=rebuttal/configs/gptoss_h200.json \
REBUTTAL_COMMAND=head-to-head \
REBUTTAL_OUTPUT_ROOT=/path/outside/the/repo/gptoss-results \
  bash rebuttal/run_qwen_h200.sh
```

The supported experiment commands are:

| Command | Evaluation |
| --- | --- |
| `smoke` | Hardware, model, tokenizer, and intervention checks |
| `reviewer-run` | Bidirectional attribution patching, controls, nearby-layer and prompt robustness, tuned CAA margins, and a generated-answer check |
| `caa-only` | CAA fitting and tuning followed by probability-margin evaluation |
| `head-to-head` | Authority removal versus tuned CAA on matched generated-answer items |
| `analyze` | Summarize an existing run with a saved CAA selection |
| `bundle` | Bundle an existing run and report its checksum |

Probability margins and generated-answer compliance are different readouts. A CAA tuning score is not a behavioral mitigation percentage.

Set `CUDA_VISIBLE_DEVICES` to choose the GPU. Caches default to `/home/.cache/uv` and `/home/.cache/huggingface`; override `UV_CACHE_DIR` and `HF_HOME` if needed. The runner checks model and input revisions rather than silently substituting them. Changed configurations create different run IDs, and resume checks require the saved manifest to match.
