#!/usr/bin/env bash
# PIQA freegen transfer for OLMo-3.1-32B with both interpolate_mean and
# replace_mean interventions. Matches the trivia forward-patch config
# exactly except for intervention mode and eval task.
#
# Reference config (from results/authority/exp16/olmo31_steering_replace_mean_l15_l18_l22_apw/steering_meta.json):
#   - direction_path: primary_direction.pt (shared_within_label, w1, endorsed_answer @ L15)
#   - extraction_dir / masks_path: all_prior_wrong_shared_h100
#   - direction-position: endorsed_answer
#   - patch-source-position: endorsed_answer
#   - patch-source-style: authoritative_verified
#   - patch-source-condition: W1_note
#   - direction/patch-uids-subset: w1
#   - target layers: 15, 18, 22 (single-layer sweep)
#   - components: block_output, position-mode: endorsed_answer, apply-phase: prompt
#
# Freegen knobs (matches exp19 PIQA convention):
#   - max-new-tokens: 256
#   - answer-suffix: "Answer the question in one short sentence using the full answer text, not option letters."
#   - max-samples: 200 (matches all prior PIQA runs: Qwen, GPT-OSS, Gemma, OLMo-2)
#
# Interventions:
#   - interpolate_mean alphas: 0 (baseline), 0.5, 1.0
#   - replace_mean alphas: 1.0 (replace_mean ignores alpha; one config per layer)
#
# Total configs: 3 interpolate + 1 replace = 4 per layer × 3 layers = 12 configs, 512 UIDs each.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUT_DIR_INTERP="results/authority/exp16/olmo31_piqa_interpolate_l15_l18_l22_w1_h100"
OUT_DIR_REPLACE="results/authority/exp16/olmo31_piqa_replace_l15_l18_l22_w1_h100"
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
  --conditions W1_note
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

echo "[$(date -u +%FT%TZ)] STAGE 1/2: interpolate_mean (alpha=0/0.5/1.0)"
uv run python -m src.exp16.run_piqa_interventions \
  "${COMMON_ARGS[@]}" \
  --patch-modes interpolate_mean \
  --patch-alphas 0,0.5,1.0 \
  --output-dir "$OUT_DIR_INTERP" \
  2>&1 | tee "$OUT_DIR_INTERP/run.log"

echo "[$(date -u +%FT%TZ)] STAGE 2/2: replace_mean (alpha=1.0)"
uv run python -m src.exp16.run_piqa_interventions \
  "${COMMON_ARGS[@]}" \
  --patch-modes replace_mean \
  --patch-alphas 1.0 \
  --output-dir "$OUT_DIR_REPLACE" \
  2>&1 | tee "$OUT_DIR_REPLACE/run.log"

echo "[$(date -u +%FT%TZ)] DONE OLMo-3.1 PIQA (interpolate+replace)"
