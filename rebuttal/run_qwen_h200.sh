#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${REBUTTAL_OUTPUT_ROOT:-/home/rebuttal-results}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN must be set; anonymous model access is not permitted." >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required and was not found on PATH." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required; this command must run on the H200 host." >&2
  exit 2
fi

export UV_PROJECT_ENVIRONMENT="${SCRIPT_DIR}/.venv"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS="${MAX_JOBS:-16}"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

mkdir -p "${OUTPUT_ROOT}"

uv sync \
  --project "${SCRIPT_DIR}" \
  --extra gpu \
  --extra test \
  --frozen \
  --python 3.12

uv run --project "${SCRIPT_DIR}" --extra gpu --extra test --frozen \
  pytest "${SCRIPT_DIR}/tests"

uv run --project "${SCRIPT_DIR}" --extra gpu --extra test --frozen \
  qwen-rebuttal all \
  --repo-root "${REPO_ROOT}" \
  --config "${SCRIPT_DIR}/configs/qwen_h200.json" \
  --output-root "${OUTPUT_ROOT}"
