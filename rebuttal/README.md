# Clean Qwen rebuttal pipeline

This package is an isolated implementation of the Qwen3.5 source/user
experiments. It deliberately does not import anything from the repository's
legacy `src/` tree. Historical code and results are read only as declared input
data.

## Run on one H100

Requirements: Linux, one visible H100 80GB, CUDA 12.8-compatible driver,
`git`, `uv`, at least 180 GiB free under the output root, and access to the
pinned Qwen checkpoint.

```bash
HF_TOKEN=... bash rebuttal/run_qwen_h100.sh
```

Override the result location with `REBUTTAL_OUTPUT_ROOT`. The default is
`/home/rebuttal-results`. Result rows, tensors, generated text, and bundles are
never written into Git.

The script uses physical GPU 0 by default. Set `CUDA_VISIBLE_DEVICES` explicitly
before the command to select a different H100.

The runner fails rather than silently changing a model revision, dependency,
attention backend, dtype, device, tokenizer behavior, or prompt span. A run is
resumed only when its complete content-addressed manifest matches. Any relevant
configuration change creates a different run ID.

Useful development commands:

```bash
uv sync --project rebuttal --extra test
uv run --project rebuttal pytest
uv run --project rebuttal ruff check rebuttal
uv run --project rebuttal qwen-rebuttal prepare --repo-root .
```

The final bundle contains immutable manifests, split files, direction geometry,
raw rows and checksums, statistics, plots, a parser audit, a reviewer-oriented
plain-English summary, and a neutral claim decision table.
