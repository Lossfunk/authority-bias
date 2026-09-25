#!/usr/bin/env bash
# gpt-oss project_out_direction A/B causal deconfound.
# Runs on 2× A100-80GB, splitting variants across GPUs.
#   GPU0: authority
#   GPU1: assistant,residualized
# Single-condition N0_note,W1_note,C1_note sweep at alpha=0,1.0; max-items matches paper.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_ROOT="results/authority/exp19/gpt_oss_ab_project_out_allcond"
LOGS_DIR="results/controls/logs"
mkdir -p "$LOGS_DIR"

COMMON_ARGS=(
    --repo-root "$(pwd)"
    --models gpt_oss
    --run-trivia --run-piqa
    --execute
    --output-root "$OUT_ROOT"
    --trivia-conditions "N0_note,W1_note,C1_note"
    --piqa-conditions "N0_note,W1_note,C1_note"
    --trivia-intervention-mode project_out_direction
    --piqa-intervention-mode project_out_direction
    --trivia-norm-scaling none
    --piqa-norm-scaling none
    --trivia-alpha-override "1.0"
    --piqa-alpha-override "1.0"
    --trivia-position-mode auto
    --piqa-position-mode auto
    --trivia-max-items 512
    --piqa-max-items 200
    --trivia-batch-size 256
    --piqa-batch-size 256
    --piqa-max-new-tokens 256
    --piqa-max-batch-tokens 262144
    --no-compile
    --dedupe-alpha-zero
    --skip-missing-axis
    --allow-missing-frozen-settings
)

GPU0_LOG="$LOGS_DIR/exp19_gpt_oss_ab_gpu0_authority.log"
GPU1_LOG="$LOGS_DIR/exp19_gpt_oss_ab_gpu1_assistant_residualized.log"

# Disable HuggingFace hub kernel dispatch so transformers uses the locally-installed
# flash-attn-2 (which works on A100/Ampere) instead of pulling flash-attn-3 from the
# hub cache (which errors with "S aux is currently only supported for Hopper GPUs"
# on gpt-oss because of its attention-sinks term).
export USE_HUB_KERNELS=NO

echo "[gpt_oss] Launching authority on GPU0 -> $GPU0_LOG"
CUDA_VISIBLE_DEVICES=0 USE_HUB_KERNELS=NO nohup uv run python -m src.exp19.run_assistant_axis_causal_deconfound \
    "${COMMON_ARGS[@]}" \
    --variants authority \
    --checkpoint-path "$OUT_ROOT/manifest_gpu0.json" \
    > "$GPU0_LOG" 2>&1 &
GPU0_PID=$!
echo "[gpt_oss] GPU0 pid=$GPU0_PID"

echo "[gpt_oss] Launching assistant+residualized on GPU1 -> $GPU1_LOG"
CUDA_VISIBLE_DEVICES=1 USE_HUB_KERNELS=NO nohup uv run python -m src.exp19.run_assistant_axis_causal_deconfound \
    "${COMMON_ARGS[@]}" \
    --variants assistant,residualized \
    --checkpoint-path "$OUT_ROOT/manifest_gpu1.json" \
    > "$GPU1_LOG" 2>&1 &
GPU1_PID=$!
echo "[gpt_oss] GPU1 pid=$GPU1_PID"

echo "PID_GPU0=$GPU0_PID PID_GPU1=$GPU1_PID"
