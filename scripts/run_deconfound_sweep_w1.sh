#!/usr/bin/env bash
# W1-only α-sweep of project_out_direction for one of {gpt_oss, gemma4, olmo2}.
# Task-split across GPUs for a balanced 2-way parallelism:
#   GPU0: trivia, all 3 variants (authority/assistant/residualized)
#   GPU1: PIQA,  all 3 variants
# One python invocation per GPU sweeps α ∈ {0, 0.5, 1, 2, 4, 8} (α=0 only runs
# for authority due to --dedupe-alpha-zero).
#
# Usage: bash scripts/run_deconfound_sweep_w1.sh <gpt_oss|gemma4|olmo2>
set -euo pipefail

MODEL="${1:?usage: $0 <gpt_oss|gemma4|olmo2>}"

cd "$(dirname "$0")/.."

case "$MODEL" in
    gpt_oss) TRIVIA_BS=256; PIQA_BS=256; PIQA_MAX_TOK=262144 ;;
    gemma4)  TRIVIA_BS=128; PIQA_BS=128; PIQA_MAX_TOK=131072 ;;
    olmo2)   TRIVIA_BS=96;  PIQA_BS=96;  PIQA_MAX_TOK=98304 ;;
    *) echo "unknown model $MODEL"; exit 1 ;;
esac

ALPHAS="0.5,1,2,4"
OUT_ROOT="results/authority/exp19/${MODEL}_sweep_w1"
LOGS_DIR="results/controls/logs"
mkdir -p "$LOGS_DIR"

COMMON_ARGS=(
    --repo-root "$(pwd)"
    --models "$MODEL"
    --variants authority,assistant,residualized
    --execute
    --output-root "$OUT_ROOT"
    --trivia-conditions W1_note
    --piqa-conditions W1_note
    --trivia-intervention-mode project_out_direction
    --piqa-intervention-mode project_out_direction
    --trivia-norm-scaling none
    --piqa-norm-scaling none
    --trivia-alpha-override "$ALPHAS"
    --piqa-alpha-override "$ALPHAS"
    --trivia-position-mode auto
    --piqa-position-mode auto
    --trivia-max-items 512
    --piqa-max-items 200
    --trivia-batch-size "$TRIVIA_BS"
    --piqa-batch-size "$PIQA_BS"
    --piqa-max-new-tokens 256
    --piqa-max-batch-tokens "$PIQA_MAX_TOK"
    --no-compile
    --dedupe-alpha-zero
    --skip-missing-axis
    --allow-missing-frozen-settings
)

GPU0_LOG="$LOGS_DIR/exp19_${MODEL}_sweep_w1_gpu0_trivia.log"
GPU1_LOG="$LOGS_DIR/exp19_${MODEL}_sweep_w1_gpu1_piqa.log"

echo "[$MODEL sweep] GPU0 (trivia, all variants) -> $GPU0_LOG"
CUDA_VISIBLE_DEVICES=0 USE_HUB_KERNELS=NO nohup uv run python -m src.exp19.run_assistant_axis_causal_deconfound \
    "${COMMON_ARGS[@]}" \
    --run-trivia \
    --checkpoint-path "$OUT_ROOT/manifest_gpu0_trivia.json" \
    > "$GPU0_LOG" 2>&1 &
GPU0_PID=$!

echo "[$MODEL sweep] GPU1 (PIQA, all variants) -> $GPU1_LOG"
CUDA_VISIBLE_DEVICES=1 USE_HUB_KERNELS=NO nohup uv run python -m src.exp19.run_assistant_axis_causal_deconfound \
    "${COMMON_ARGS[@]}" \
    --run-piqa \
    --checkpoint-path "$OUT_ROOT/manifest_gpu1_piqa.json" \
    > "$GPU1_LOG" 2>&1 &
GPU1_PID=$!

echo "PID_GPU0=$GPU0_PID PID_GPU1=$GPU1_PID MODEL=$MODEL"
