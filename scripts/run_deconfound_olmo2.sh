#!/usr/bin/env bash
# olmo2 project_out_direction A/B causal deconfound. Same protocol as gpt_oss.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_ROOT="results/authority/exp19/olmo2_ab_project_out_allcond"
LOGS_DIR="results/controls/logs"
mkdir -p "$LOGS_DIR"

COMMON_ARGS=(
    --repo-root "$(pwd)"
    --models olmo2
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
    --trivia-batch-size 96
    --piqa-batch-size 96
    --piqa-max-new-tokens 256
    --piqa-max-batch-tokens 98304
    --no-compile
    --dedupe-alpha-zero
    --skip-missing-axis
    --allow-missing-frozen-settings
)

GPU0_LOG="$LOGS_DIR/exp19_olmo2_ab_gpu0_authority.log"
GPU1_LOG="$LOGS_DIR/exp19_olmo2_ab_gpu1_assistant_residualized.log"

CUDA_VISIBLE_DEVICES=0 nohup uv run python -m src.exp19.run_assistant_axis_causal_deconfound \
    "${COMMON_ARGS[@]}" --variants authority \
    --checkpoint-path "$OUT_ROOT/manifest_gpu0.json" \
    > "$GPU0_LOG" 2>&1 &
GPU0_PID=$!

CUDA_VISIBLE_DEVICES=1 nohup uv run python -m src.exp19.run_assistant_axis_causal_deconfound \
    "${COMMON_ARGS[@]}" --variants assistant,residualized \
    --checkpoint-path "$OUT_ROOT/manifest_gpu1.json" \
    > "$GPU1_LOG" 2>&1 &
GPU1_PID=$!

echo "PID_GPU0=$GPU0_PID PID_GPU1=$GPU1_PID"
