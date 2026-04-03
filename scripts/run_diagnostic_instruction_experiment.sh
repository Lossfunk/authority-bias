#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
TAGS="${TAGS:-Note}"
INSTRUCTION_SET="${INSTRUCTION_SET:-epistemic_modes_v1}"
MC_DATASET_PATH="${MC_DATASET_PATH:-data/piqa_mc_dataset_from_results.jsonl}"
UIDS_FILE="${UIDS_FILE:-new-phase-results/piqa/piqa_prior_wrong_uids.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-new-phase-results/diagnostic-instructions/piqa_note_priorwrong_${INSTRUCTION_SET}}"
BATCH_SIZE="${BATCH_SIZE:-64}"

uv run python -m src.exp15.run_instruction_diagnostics \
  --models "${MODEL}" \
  --mc-dataset-path "${MC_DATASET_PATH}" \
  --uids-file "${UIDS_FILE}" \
  --tags "${TAGS}" \
  --instruction-set "${INSTRUCTION_SET}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --batch-size "${BATCH_SIZE}"
