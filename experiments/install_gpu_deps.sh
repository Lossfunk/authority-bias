#!/usr/bin/env bash
# Install GPU accelerators (flash-attn, flash-linear-attention, causal-conv1d) via uv.
#
# Usage on a fresh box:
#   cd /home/persona-vectors   # or wherever this repo is
#   bash experiments/install_gpu_deps.sh
#
# Why two uv sync passes:
#   flash-attn / flash-linear-attention / causal-conv1d must build against the torch
#   that ends up in the venv. With [tool.uv] no-build-isolation-package set for those
#   three packages, uv needs torch (+ ninja + packaging + setuptools + wheel) already
#   present in the venv before it starts their compile step.
#
# Why not `--no-build-isolation` as a flag:
#   That disables isolation for every package, which is heavier than we want.
#   The pyproject already pins only those three via [tool.uv].
#
# Tuning knobs:
#   MAX_JOBS                  number of parallel nvcc invocations. 4 is safe on a
#                             box with a 120 GB cgroup limit; 16-32 on boxes with
#                             more headroom. Override with env.
#   TORCH_CUDA_ARCH_LIST      torch/causal-conv1d/fla honor this. '8.0' for A100.
#   FLASH_ATTN_CUDA_ARCHS     flash-attn's own gencode env var (does not honor the
#                             one above). Use '80' for A100 only.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# Detect GPU arch if not pinned.
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
        if [[ -n "$CAP" ]]; then
            TORCH_CUDA_ARCH_LIST="$CAP"
            FLASH_ATTN_CUDA_ARCHS="${CAP//./}"
        fi
    fi
fi
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-${TORCH_CUDA_ARCH_LIST//./}}"

# Memory-aware default MAX_JOBS. Each nvcc can spike ~6-8 GB; cap on cgroup headroom.
if [[ -z "${MAX_JOBS:-}" ]]; then
    if [[ -r /sys/fs/cgroup/memory.max ]]; then
        LIMIT="$(cat /sys/fs/cgroup/memory.max)"
    elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
        LIMIT="$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"
    else
        LIMIT="max"
    fi
    if [[ "$LIMIT" == "max" || "$LIMIT" -gt $((400 * 1024 * 1024 * 1024)) ]]; then
        MAX_JOBS=16
    elif [[ "$LIMIT" -gt $((200 * 1024 * 1024 * 1024)) ]]; then
        MAX_JOBS=8
    else
        MAX_JOBS=4
    fi
fi

export MAX_JOBS TORCH_CUDA_ARCH_LIST FLASH_ATTN_CUDA_ARCHS
export NVCC_THREADS="${NVCC_THREADS:-1}"

echo "[install_gpu_deps] REPO_ROOT=$REPO_ROOT"
echo "[install_gpu_deps] MAX_JOBS=$MAX_JOBS TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST FLASH_ATTN_CUDA_ARCHS=$FLASH_ATTN_CUDA_ARCHS NVCC_THREADS=$NVCC_THREADS"

# Pass 1: base env (torch + ninja + packaging + setuptools + wheel + psutil).
echo "[install_gpu_deps] Step 1/2: uv sync (base)"
uv sync

# Pass 2: accelerators. no-build-isolation-package is configured in pyproject.
echo "[install_gpu_deps] Step 2/2: uv sync --extra gpu (builds flash-attn, fla, causal-conv1d)"
uv sync --extra gpu

echo "[install_gpu_deps] Verifying imports..."
uv run python -c "
import importlib, sys
for mod in ('flash_attn', 'flash_attn_2_cuda', 'fla', 'causal_conv1d'):
    try:
        m = importlib.import_module(mod)
        print(f'  ok  {mod:30s} {getattr(m, \"__version__\", \"\")}')
    except Exception as e:
        print(f'  ERR {mod:30s} {e}')
        sys.exit(1)
"
echo "[install_gpu_deps] Done."
