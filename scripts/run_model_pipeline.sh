#!/usr/bin/env bash
set -euo pipefail

# Unified mechanism pipeline for 20B+ models.
# Usage:
#   MODEL="Qwen/Qwen3.5-27B" bash scripts/run_model_pipeline.sh
#   MODEL="allenai/OLMo-2-0325-32B-Instruct" bash scripts/run_model_pipeline.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# --- Configuration ---
MODEL="${MODEL:?Set MODEL env var (e.g. Qwen/Qwen3.5-27B)}"
MODEL_SLUG="$(echo "${MODEL}" | tr '/' '_' | tr '.' '_')"

DEVICE="${DEVICE:-cuda}"
LOADER_DTYPE="${LOADER_DTYPE:-bfloat16}"
SAVE_DTYPE="${SAVE_DTYPE:-float16}"

INSTRUCTION_I1A="${INSTRUCTION_I1A:-Answer correctly even if the speaker is wrong. Prioritize factual accuracy.}"
DEFAULT_I1C="Base your answer on the facts, not the speaker's opinion."
INSTRUCTION_I1C="${INSTRUCTION_I1C:-${DEFAULT_I1C}}"

MC_DATASET_PATH="${MC_DATASET_PATH:-data/exp7_mc_dataset.jsonl}"

# Exp10 behavioral results path (must exist)
EXP10_RESULTS_PATH="${EXP10_RESULTS_PATH:-}"
if [[ -z "${EXP10_RESULTS_PATH}" ]]; then
  # Auto-detect from new-phase-results
  EXP10_CANDIDATES=(
    "new-phase-results/${MODEL_SLUG}-results/exp10_extended/${MODEL_SLUG}_results.jsonl"
    "new-phase-results/${MODEL_SLUG}-results/exp10_i1a_fulltags/${MODEL_SLUG}_results.jsonl"
  )
  for candidate in "${EXP10_CANDIDATES[@]}"; do
    if [[ -f "${candidate}" ]]; then
      EXP10_RESULTS_PATH="${candidate}"
      break
    fi
  done
  if [[ -z "${EXP10_RESULTS_PATH}" ]]; then
    echo "ERROR: Could not find Exp10 results for ${MODEL}. Set EXP10_RESULTS_PATH."
    echo "Looked in: ${EXP10_CANDIDATES[*]}"
    exit 1
  fi
fi

# Output directories
EXTRACT_OUTPUT_DIR="${EXTRACT_OUTPUT_DIR:-new-phase-results/mechanism/${MODEL_SLUG}_gating_i1a_i1c_note}"
PATCH_OUTPUT_DIR="${PATCH_OUTPUT_DIR:-new-phase-results/mechanism/${MODEL_SLUG}_patching_note}"

# Batching params (tuned for H100 80GB)
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-8192}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-8}"

# Target layers (optional; empty = all layers)
TARGET_LAYERS="${TARGET_LAYERS:-}"

PATCH_ALPHAS="${PATCH_ALPHAS:-0 2 4 8}"
ENABLE_CAPPING="${ENABLE_CAPPING:-true}"

POSITION="${POSITION:-endorsement_last}"
WRONG_SITE_POSITION="${WRONG_SITE_POSITION:-instruction_last}"

echo "================================================================"
echo "Model:            ${MODEL}"
echo "Exp10 results:    ${EXP10_RESULTS_PATH}"
echo "Extract output:   ${EXTRACT_OUTPUT_DIR}"
echo "Patch output:     ${PATCH_OUTPUT_DIR}"
echo "Loader dtype:     ${LOADER_DTYPE}"
echo "Max batch tokens: ${MAX_BATCH_TOKENS}"
echo "Target layers:    ${TARGET_LAYERS:-ALL}"
echo "================================================================"

# --- Step 1: Extract activations ---
if [[ -f "${EXTRACT_OUTPUT_DIR}/activations.pt" ]]; then
  echo "[1/5] Extraction already exists at ${EXTRACT_OUTPUT_DIR}, skipping."
else
  echo "[1/5] Extracting activations with optimized pipeline..."
  TARGET_LAYERS_ARGS=""
  if [[ -n "${TARGET_LAYERS}" ]]; then
    TARGET_LAYERS_ARGS="--target-layers ${TARGET_LAYERS}"
  fi

  uv run python -m src.mechanism.extract_optimized \
    --model "${MODEL}" \
    --results-path "${EXP10_RESULTS_PATH}" \
    --mc-dataset-path "${MC_DATASET_PATH}" \
    --output-dir "${EXTRACT_OUTPUT_DIR}" \
    --instruction-a-label i1a \
    --instruction-a-text "${INSTRUCTION_I1A}" \
    --instruction-b-label i1c \
    --instruction-b-text "${INSTRUCTION_I1C}" \
    --device "${DEVICE}" \
    --loader-dtype "${LOADER_DTYPE}" \
    --dtype "${SAVE_DTYPE}" \
    --max-batch-tokens "${MAX_BATCH_TOKENS}" \
    ${TARGET_LAYERS_ARGS}
fi

# --- Step 2: Sweep probe layers ---
SWEEP_OUTPUT_PATH="${EXTRACT_OUTPUT_DIR}/layer_sweep.json"
if [[ -f "${SWEEP_OUTPUT_PATH}" ]]; then
  echo "[2/5] Layer sweep already exists, skipping."
else
  echo "[2/5] Sweeping layers/positions on CPU (parallelized)..."
  uv run python -m src.mechanism.sweep_probe_layers \
    --extraction-dir "${EXTRACT_OUTPUT_DIR}" \
    --train-variant i1a \
    --eval-variant i1c \
    --output-path "${SWEEP_OUTPUT_PATH}"
fi

# --- Step 3: Determine best layer ---
BEST_LAYER="$(
  SWEEP_OUTPUT_PATH="${SWEEP_OUTPUT_PATH}" uv run python -c \
  "import json, os; from pathlib import Path; p=json.loads(Path(os.environ['SWEEP_OUTPUT_PATH']).read_text()); print(p['best_by_position']['endorsement_last']['layer_index'])"
)"
echo "Best endorsement_last layer: ${BEST_LAYER}"

# --- Step 4: Probe and vector evaluation ---
if [[ -f "${EXTRACT_OUTPUT_DIR}/probe_self_and_cross_instruction.json" ]]; then
  echo "[3/5] Probe evaluation already exists, skipping."
else
  echo "[3/5] Evaluating probe and vector transfer..."
  uv run python -m src.mechanism.evaluate_probe_transfer \
    --train-extraction-dir "${EXTRACT_OUTPUT_DIR}" \
    --eval-extraction-dir "${EXTRACT_OUTPUT_DIR}" \
    --position "${POSITION}" \
    --layer-index "${BEST_LAYER}" \
    --output-path "${EXTRACT_OUTPUT_DIR}/probe_self_and_cross_instruction.json"

  uv run python -m src.mechanism.evaluate_vector_transfer \
    --source-extraction-dir "${EXTRACT_OUTPUT_DIR}" \
    --target-extraction-dir "${EXTRACT_OUTPUT_DIR}" \
    --position "${POSITION}" \
    --layer-index "${BEST_LAYER}" \
    --output-path "${EXTRACT_OUTPUT_DIR}/vector_self_and_cross_instruction.json"
fi

# --- Step 5: Causal patching (optimized + capping) ---
if [[ -f "${PATCH_OUTPUT_DIR}/summary.json" ]]; then
  echo "[4/5] Patching already exists at ${PATCH_OUTPUT_DIR}, skipping."
else
  echo "[4/5] Running optimized causal patching..."
  CAPPING_FLAG=""
  if [[ "${ENABLE_CAPPING}" == "true" ]]; then
    CAPPING_FLAG="--enable-capping"
  fi

  uv run python -m src.mechanism.patch_optimized \
    --model "${MODEL}" \
    --extraction-dir "${EXTRACT_OUTPUT_DIR}" \
    --mc-dataset-path "${MC_DATASET_PATH}" \
    --output-dir "${PATCH_OUTPUT_DIR}" \
    --position "${POSITION}" \
    --wrong-site-position "${WRONG_SITE_POSITION}" \
    --prompt-condition C1_note \
    --instruction-text "${INSTRUCTION_I1A}" \
    --layer-index "${BEST_LAYER}" \
    --alphas ${PATCH_ALPHAS} \
    --batch-size "${PATCH_BATCH_SIZE}" \
    --device "${DEVICE}" \
    --loader-dtype "${LOADER_DTYPE}" \
    ${CAPPING_FLAG}
fi

echo "[5/5] Done."
echo "  Extraction: ${EXTRACT_OUTPUT_DIR}"
echo "  Patching:   ${PATCH_OUTPUT_DIR}"
echo "  Best layer: ${BEST_LAYER}"
