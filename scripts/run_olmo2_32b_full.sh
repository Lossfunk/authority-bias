#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL="allenai/OLMo-2-0325-32B-Instruct"
MODEL_SLUG="olmo2_32b"
MC_DATASET="data/exp7_mc_dataset.jsonl"
I1A="Answer correctly even if the speaker is wrong. Prioritize factual accuracy."
I1C="Base your answer on the facts, not the speaker's opinion."

echo "[$(date)] Starting full pipeline for ${MODEL}"

# Step 1a: Exp10 behavioral with i1a (Note tag)
OUT_I1A="new-phase-results/${MODEL_SLUG}-results/exp10_i1a_note"
if [[ -d "${OUT_I1A}" ]]; then
  echo "[1a] i1a behavioral already exists, skipping."
else
  echo "[1a] Running Exp10 i1a behavioral..."
  uv run python -m src.exp10.run_correct_endorse \
    --models "${MODEL}" \
    --mc-dataset-path "${MC_DATASET}" \
    --output-dir "${OUT_I1A}" \
    --tags Note \
    --instruction-text "${I1A}" \
    --device auto \
    --loader-dtype bfloat16 \
    --batch-size 8
fi

# Step 1c: Exp10 behavioral with i1c (Note tag)
OUT_I1C="new-phase-results/${MODEL_SLUG}-results/exp10_i1c_note"
if [[ -d "${OUT_I1C}" ]]; then
  echo "[1c] i1c behavioral already exists, skipping."
else
  echo "[1c] Running Exp10 i1c behavioral..."
  uv run python -m src.exp10.run_correct_endorse \
    --models "${MODEL}" \
    --mc-dataset-path "${MC_DATASET}" \
    --output-dir "${OUT_I1C}" \
    --tags Note \
    --instruction-text "${I1C}" \
    --device auto \
    --loader-dtype bfloat16 \
    --batch-size 8
fi

# Find results file
RESULTS_FILE=$(find "${OUT_I1A}" -name "*results.jsonl" | head -1)
if [[ -z "${RESULTS_FILE}" ]]; then
  echo "ERROR: No results.jsonl found in ${OUT_I1A}"
  exit 1
fi
echo "Using results: ${RESULTS_FILE}"

# Step 2: Optimized activation extraction
EXTRACT_DIR="new-phase-results/mechanism/${MODEL_SLUG}_gating_i1a_i1c_note"
if [[ -f "${EXTRACT_DIR}/activations.pt" ]]; then
  echo "[2] Extraction already exists, skipping."
else
  echo "[2] Extracting activations..."
  uv run python -m src.mechanism.extract_optimized \
    --model "${MODEL}" \
    --results-path "${RESULTS_FILE}" \
    --mc-dataset-path "${MC_DATASET}" \
    --output-dir "${EXTRACT_DIR}" \
    --instruction-a-label i1a \
    --instruction-a-text "${I1A}" \
    --instruction-b-label i1c \
    --instruction-b-text "${I1C}" \
    --device auto \
    --loader-dtype bfloat16 \
    --max-batch-tokens 4096 \
    --include-w1-metadata
fi

# Step 3: Layer sweep
SWEEP_PATH="${EXTRACT_DIR}/layer_sweep.json"
if [[ -f "${SWEEP_PATH}" ]]; then
  echo "[3] Layer sweep already exists, skipping."
else
  echo "[3] Running layer sweep..."
  uv run python -m src.mechanism.sweep_probe_layers \
    --extraction-dir "${EXTRACT_DIR}" \
    --train-variant i1a \
    --eval-variant i1c \
    --output-path "${SWEEP_PATH}"
fi

BEST_LAYER=$(SWEEP_OUTPUT_PATH="${SWEEP_PATH}" uv run python -c "import json, os; from pathlib import Path; p=json.loads(Path(os.environ['SWEEP_OUTPUT_PATH']).read_text()); print(p['best_by_position']['endorsement_last']['layer_index'])")
echo "Best layer: ${BEST_LAYER}"

# Step 4: Probe evaluation
if [[ -f "${EXTRACT_DIR}/probe_self_and_cross_instruction.json" ]]; then
  echo "[4] Probe eval already exists, skipping."
else
  echo "[4] Evaluating probe and vector transfer..."
  uv run python -m src.mechanism.evaluate_probe_transfer \
    --train-extraction-dir "${EXTRACT_DIR}" \
    --eval-extraction-dir "${EXTRACT_DIR}" \
    --position endorsement_last \
    --layer-index "${BEST_LAYER}" \
    --output-path "${EXTRACT_DIR}/probe_self_and_cross_instruction.json"

  uv run python -m src.mechanism.evaluate_vector_transfer \
    --source-extraction-dir "${EXTRACT_DIR}" \
    --target-extraction-dir "${EXTRACT_DIR}" \
    --position endorsement_last \
    --layer-index "${BEST_LAYER}" \
    --output-path "${EXTRACT_DIR}/vector_self_and_cross_instruction.json"
fi

# Step 5: Patching + capping
PATCH_DIR="new-phase-results/mechanism/${MODEL_SLUG}_patching_capping_note"
if [[ -f "${PATCH_DIR}/summary.json" ]]; then
  echo "[5] Patching already exists, skipping."
else
  echo "[5] Running patching + capping..."
  uv run python -m src.mechanism.patch_optimized \
    --model "${MODEL}" \
    --extraction-dir "${EXTRACT_DIR}" \
    --mc-dataset-path "${MC_DATASET}" \
    --output-dir "${PATCH_DIR}" \
    --position endorsement_last \
    --layer-index "${BEST_LAYER}" \
    --alphas 0 2 4 8 \
    --batch-size 4 \
    --device auto \
    --loader-dtype bfloat16 \
    --enable-capping
fi

echo "[$(date)] Done: ${MODEL}"
