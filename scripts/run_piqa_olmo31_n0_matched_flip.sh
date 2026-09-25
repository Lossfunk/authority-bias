#!/usr/bin/env bash
# OLMo-3.1 PIQA N0-matched-flip (standard protocol).
#
# Matches the protocol used for GPT-OSS, Qwen3.5, Gemma-4, OLMo-2 in
# results/authority/{model}/mechanism/piqa_forward_patch_200_freegen/:
#   - conditions:            N0_note   (evaluate under N0; inject W1 mean)
#   - patch_source_condition: W1_note   (forward-patch *toward* W1 residual state)
#   - direction_position, patch_source_position: each model's primary-pair position
#   - intervention: forward-patch (interpolate_mean and replace_mean)
#   - 200 UIDs, 256 max_new_tokens, matched-flip metric
#
# OLMo-3.1 primary pair (from compliance_analysis_all_prior_wrong_shared_h100):
#   position = endorsed_answer, direction_layer = 15
#   layers swept = L15, L18, L22 (same as SYCON and trivia forward-patch)
#
# H100 box 121, FA2 enabled via updated llama_loader.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUT_DIR_INTERP="results/authority/exp16/olmo31_piqa_n0_matched_flip_interpolate_l15_l18_l22"
OUT_DIR_REPLACE="results/authority/exp16/olmo31_piqa_n0_matched_flip_replace_l15_l18_l22"
mkdir -p "$OUT_DIR_INTERP" "$OUT_DIR_REPLACE" logs

COMMON_ARGS=(
  --model allenai/OLMo-3.1-32B-Instruct
  --mc-dataset-path data/piqa_mc_validation.jsonl
  --extraction-dir results/authority/olmo31/mechanism/authority_activations_all_prior_wrong_shared_h100
  --direction-path results/authority/olmo31/mechanism/compliance_analysis_all_prior_wrong_shared_h100/primary_direction.pt
  --masks-path results/authority/olmo31/mechanism/authority_activations_all_prior_wrong_shared_h100/label_masks.json
  --direction-key vector
  --direction-position endorsed_answer
  --patch-source-position endorsed_answer
  --patch-source-style authoritative_verified
  --patch-source-condition W1_note
  --direction-uids-subset w1
  --patch-uids-subset w1
  --eval-uids-subset all
  --max-samples 200
  --conditions N0_note
  --target-layers 15,18,22
  --single-layer-sweep
  --components block_output
  --position-mode endorsed_answer
  --apply-phase prompt
  --norm-scaling none
  --additive-modes ""
  --direction-kinds ""
  --additive-alphas ""
  --sparse-topk-values ""
  --answer-suffix "Answer the question in one short sentence using the full answer text, not option letters."
  --batch-size 24
  --max-batch-tokens 24576
  --max-new-tokens 256
  --temperature 0.0
  --top-p 1.0
  --top-k 50
  --seed 42
)

echo "[$(date -u +%FT%TZ)] STAGE 1/2: interpolate_mean (alpha=0,0.3,0.5,0.7,1.0)  conds=N0_note"
uv run python -m src.exp16.run_piqa_interventions \
  "${COMMON_ARGS[@]}" \
  --patch-modes interpolate_mean \
  --patch-alphas 0,0.3,0.5,0.7,1.0 \
  --output-dir "$OUT_DIR_INTERP" \
  2>&1 | tee "$OUT_DIR_INTERP/run.log"

echo "[$(date -u +%FT%TZ)] STAGE 2/2: replace_mean (alpha=1.0)  conds=N0_note"
uv run python -m src.exp16.run_piqa_interventions \
  "${COMMON_ARGS[@]}" \
  --patch-modes replace_mean \
  --patch-alphas 1.0 \
  --output-dir "$OUT_DIR_REPLACE" \
  2>&1 | tee "$OUT_DIR_REPLACE/run.log"

echo "[$(date -u +%FT%TZ)] DONE OLMo-3.1 PIQA N0-matched-flip (interpolate + replace)"
