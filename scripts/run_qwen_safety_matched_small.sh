#!/usr/bin/env bash
set -euo pipefail

# Small, structure-matched safety pilot:
# - N0_safety: neutral source note
# - C1_safety: source says refuse
# - W1_safety: source says comply
#
# Usage examples:
#   scripts/run_qwen_safety_matched_small.sh
#   scripts/run_qwen_safety_matched_small.sh --prompts-jsonl data/redbench_sampled_prompts.jsonl
#   scripts/run_qwen_safety_matched_small.sh --max-samples 60 --model Qwen/Qwen3.5-27B-Instruct

MODEL="${MODEL:-Qwen/Qwen3.5-27B-Instruct}"
BACKEND="${BACKEND:-api}"
OUTPUT_DIR="${OUTPUT_DIR:-results/authority/authority_steering/safety_matched_qwen_small}"
MAX_SAMPLES="${MAX_SAMPLES:-80}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-50}"
SEED="${SEED:-42}"

python3 -m src.authority_steering.run_safety_matched_conditions \
  --model "${MODEL}" \
  --backend "${BACKEND}" \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
