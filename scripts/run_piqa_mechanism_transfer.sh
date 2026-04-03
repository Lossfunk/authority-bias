#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
DEVICE="${DEVICE:-cuda}"
LOADER_DTYPE="${LOADER_DTYPE:-bfloat16}"
SAVE_DTYPE="${SAVE_DTYPE:-float16}"

PIQA_RESULTS="${PIQA_RESULTS:-new-phase-results/piqa/qwen_i1a_exp10/Qwen__Qwen3-4B-Instruct-2507_results.jsonl}"
PIQA_UIDS="${PIQA_UIDS:-new-phase-results/piqa/piqa_prior_wrong_uids.txt}"
PIQA_MC_DATASET="${PIQA_MC_DATASET:-data/piqa_mc_dataset_from_results.jsonl}"
PIQA_EXTRACTION_DIR="${PIQA_EXTRACTION_DIR:-new-phase-results/mechanism/piqa_correction_gating_i1a_i1c_note}"

FACTUAL_EXTRACTION_DIR="${FACTUAL_EXTRACTION_DIR:-new-phase-results/mechanism/qwen_correction_gating_i1a_i1c_note}"
POSITION="${POSITION:-endorsement_last}"
LAYER_INDEX="${LAYER_INDEX:-23}"

echo "[1/6] Rebuilding PIQA MC dataset from existing results..."
uv run python scripts/build_mc_dataset_from_results.py \
  --results-path "${PIQA_RESULTS}" \
  --output-path "${PIQA_MC_DATASET}"

echo "[2/6] Extracting PIQA correction-gating activations on GPU..."
uv run python -m src.mechanism.extract_correction_gating_activations \
  --model "${MODEL}" \
  --results-path "${PIQA_RESULTS}" \
  --mc-dataset-path "${PIQA_MC_DATASET}" \
  --uids-file "${PIQA_UIDS}" \
  --output-dir "${PIQA_EXTRACTION_DIR}" \
  --device "${DEVICE}" \
  --loader-dtype "${LOADER_DTYPE}" \
  --dtype "${SAVE_DTYPE}"

echo "[3/6] PIQA probe self-evaluation and i1a->i1c transfer..."
uv run python -m src.mechanism.evaluate_probe_transfer \
  --train-extraction-dir "${PIQA_EXTRACTION_DIR}" \
  --eval-extraction-dir "${PIQA_EXTRACTION_DIR}" \
  --position "${POSITION}" \
  --layer-index "${LAYER_INDEX}" \
  --output-path "${PIQA_EXTRACTION_DIR}/probe_self_and_cross_instruction.json"

echo "[4/6] Factual -> PIQA probe transfer..."
uv run python -m src.mechanism.evaluate_probe_transfer \
  --train-extraction-dir "${FACTUAL_EXTRACTION_DIR}" \
  --eval-extraction-dir "${PIQA_EXTRACTION_DIR}" \
  --position "${POSITION}" \
  --layer-index "${LAYER_INDEX}" \
  --output-path "${PIQA_EXTRACTION_DIR}/probe_transfer_factual_to_piqa.json"

echo "[5/6] Factual -> PIQA vector transfer..."
uv run python -m src.mechanism.evaluate_vector_transfer \
  --source-extraction-dir "${FACTUAL_EXTRACTION_DIR}" \
  --target-extraction-dir "${PIQA_EXTRACTION_DIR}" \
  --position "${POSITION}" \
  --layer-index "${LAYER_INDEX}" \
  --output-path "${PIQA_EXTRACTION_DIR}/vector_transfer_factual_to_piqa.json"

echo "[6/6] PIQA -> Factual vector transfer..."
uv run python -m src.mechanism.evaluate_vector_transfer \
  --source-extraction-dir "${PIQA_EXTRACTION_DIR}" \
  --target-extraction-dir "${FACTUAL_EXTRACTION_DIR}" \
  --position "${POSITION}" \
  --layer-index "${LAYER_INDEX}" \
  --output-path "${PIQA_EXTRACTION_DIR}/vector_transfer_piqa_to_factual.json"

echo "Done. Outputs are in ${PIQA_EXTRACTION_DIR}"
