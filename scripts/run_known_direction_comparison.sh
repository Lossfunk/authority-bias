#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
DEVICE="${DEVICE:-cuda}"
LOADER_DTYPE="${LOADER_DTYPE:-bfloat16}"
POSITION="${POSITION:-endorsement_last}"
LAYER_INDEX="${LAYER_INDEX:-23}"
RESULTS_PATH="${RESULTS_PATH:-new-phase-results/qwen3-4b-results/exp10_extended/Qwen__Qwen3-4B-Instruct-2507_results.jsonl}"
UIDS_FILE="${UIDS_FILE:-new-phase-results/mechanism/qwen_correction_gating_i1a_i1c_note/selected_uids.txt}"
EXTRACT_DIR="${EXTRACT_DIR:-new-phase-results/mechanism/known_direction_extract_note}"
VECTOR_DIR="${VECTOR_DIR:-new-phase-results/mechanism/known_direction_vectors_note}"
ANALYSIS_DIR="${ANALYSIS_DIR:-new-phase-results/mechanism/known_direction_analysis_note}"
TARGET_EXTRACTION_DIR="${TARGET_EXTRACTION_DIR:-new-phase-results/mechanism/qwen_correction_gating_i1a_i1c_note}"
PIQA_EXTRACTION_DIR="${PIQA_EXTRACTION_DIR:-new-phase-results/mechanism/piqa_correction_gating_i1a_i1c_note}"

echo "[1/8] Extracting condition activations for neighboring directions..."
uv run python -m src.mechanism.extract_condition_activations \
  --model "${MODEL}" \
  --results-path "${RESULTS_PATH}" \
  --condition-codes N0_note W0_note C0_note W0_expert C0_expert \
  --uids-file "${UIDS_FILE}" \
  --output-dir "${EXTRACT_DIR}" \
  --device "${DEVICE}" \
  --loader-dtype "${LOADER_DTYPE}"

echo "[2/8] Building correction-gating reference vector..."
uv run python -m src.mechanism.build_correction_gating_vector \
  --extraction-dir "${TARGET_EXTRACTION_DIR}" \
  --position "${POSITION}" \
  --layer-index "${LAYER_INDEX}" \
  --output-path "${VECTOR_DIR}/correction_gating_i1a.pt"

echo "[3/8] Building opinion-like wrong-vs-neutral Note vector..."
uv run python -m src.mechanism.build_condition_difference_vector \
  --extraction-dir "${EXTRACT_DIR}" \
  --position "${POSITION}" \
  --layer-index "${LAYER_INDEX}" \
  --positive-condition-codes W0_note \
  --negative-condition-codes N0_note \
  --output-path "${VECTOR_DIR}/opinion_like_w0_vs_n0_note.pt"

echo "[4/8] Building authority-like Expert-vs-Note wrong-endorsement vector..."
uv run python -m src.mechanism.build_condition_difference_vector \
  --extraction-dir "${EXTRACT_DIR}" \
  --position "${POSITION}" \
  --layer-index "${LAYER_INDEX}" \
  --positive-condition-codes W0_expert \
  --negative-condition-codes W0_note \
  --output-path "${VECTOR_DIR}/authority_like_w0_expert_vs_w0_note.pt"

echo "[5/8] Building authority-like Expert-vs-Note correct-endorsement vector..."
uv run python -m src.mechanism.build_condition_difference_vector \
  --extraction-dir "${EXTRACT_DIR}" \
  --position "${POSITION}" \
  --layer-index "${LAYER_INDEX}" \
  --positive-condition-codes C0_expert \
  --negative-condition-codes C0_note \
  --output-path "${VECTOR_DIR}/authority_like_c0_expert_vs_c0_note.pt"

echo "[6/8] Computing cosine similarities..."
uv run python -m src.mechanism.compare_direction_vectors \
  --vector-paths \
    "${VECTOR_DIR}/correction_gating_i1a.pt" \
    "${VECTOR_DIR}/opinion_like_w0_vs_n0_note.pt" \
    "${VECTOR_DIR}/authority_like_w0_expert_vs_w0_note.pt" \
    "${VECTOR_DIR}/authority_like_c0_expert_vs_c0_note.pt" \
  --output-path "${ANALYSIS_DIR}/cosine_summary.json"

echo "[7/8] Ablating neighboring directions on factual transfer..."
uv run python -m src.mechanism.evaluate_vector_transfer \
  --source-extraction-dir "${TARGET_EXTRACTION_DIR}" \
  --target-extraction-dir "${TARGET_EXTRACTION_DIR}" \
  --position "${POSITION}" \
  --layer-index "${LAYER_INDEX}" \
  --build-variants i1a \
  --eval-variants i1a i1c \
  --project-out-vector-paths \
    "${VECTOR_DIR}/opinion_like_w0_vs_n0_note.pt" \
    "${VECTOR_DIR}/authority_like_w0_expert_vs_w0_note.pt" \
    "${VECTOR_DIR}/authority_like_c0_expert_vs_c0_note.pt" \
  --output-path "${ANALYSIS_DIR}/factual_ablation_eval.json"

echo "[8/8] Ablating neighboring directions on PIQA transfer..."
uv run python -m src.mechanism.evaluate_vector_transfer \
  --source-extraction-dir "${TARGET_EXTRACTION_DIR}" \
  --target-extraction-dir "${PIQA_EXTRACTION_DIR}" \
  --position "${POSITION}" \
  --layer-index "${LAYER_INDEX}" \
  --build-variants i1a \
  --eval-variants i1a i1c \
  --project-out-vector-paths \
    "${VECTOR_DIR}/opinion_like_w0_vs_n0_note.pt" \
    "${VECTOR_DIR}/authority_like_w0_expert_vs_w0_note.pt" \
    "${VECTOR_DIR}/authority_like_c0_expert_vs_c0_note.pt" \
  --output-path "${ANALYSIS_DIR}/piqa_ablation_eval.json"
