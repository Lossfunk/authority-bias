#!/usr/bin/env bash
# Launch a W1-only α sweep for a single (model, task) combo on a specific GPU.
# Used when we want fine-grained scheduling (e.g. overlapping gpt_oss-trivia on GPU0
# with gemma4-piqa on GPU1).
#
# Usage: bash experiments/run_sweep_single_task.sh <gpt_oss|gemma4|olmo2> <trivia|piqa> <gpu_id>
set -euo pipefail

MODEL="${1:?usage: $0 <model> <task> <gpu_id>}"
TASK="${2:?usage: $0 <model> <task> <gpu_id>}"
GPU_ID="${3:?usage: $0 <model> <task> <gpu_id>}"

cd "$(dirname "$0")/.."

case "$MODEL" in
    gpt_oss) TRIVIA_BS=256; PIQA_BS=256; PIQA_MAX_TOK=262144 ;;
    gemma4)  TRIVIA_BS=128; PIQA_BS=128; PIQA_MAX_TOK=131072 ;;
    olmo2)   TRIVIA_BS=96;  PIQA_BS=96;  PIQA_MAX_TOK=98304 ;;
    *) echo "unknown model $MODEL"; exit 1 ;;
esac

ALPHAS="0.5,1,2,4"
OUT_ROOT="neurips-results/exp19/${MODEL}_sweep_w1"
LOGS_DIR="causal-deconfound/logs"
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

case "$TASK" in
    trivia) RUN_FLAG="--run-trivia" ;;
    piqa)   RUN_FLAG="--run-piqa" ;;
    *) echo "unknown task $TASK"; exit 1 ;;
esac

LOG="$LOGS_DIR/exp19_${MODEL}_sweep_w1_gpu${GPU_ID}_${TASK}.log"
CKPT="$OUT_ROOT/manifest_gpu${GPU_ID}_${TASK}.json"

echo "[$MODEL sweep] GPU$GPU_ID ($TASK) -> $LOG"
CUDA_VISIBLE_DEVICES="$GPU_ID" USE_HUB_KERNELS=NO nohup uv run python -m src.exp19.run_assistant_axis_causal_deconfound \
    "${COMMON_ARGS[@]}" \
    "$RUN_FLAG" \
    --checkpoint-path "$CKPT" \
    > "$LOG" 2>&1 &
PID=$!

echo "PID=$PID MODEL=$MODEL TASK=$TASK GPU=$GPU_ID"
