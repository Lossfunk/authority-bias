#!/usr/bin/env bash
# Run SYCON false-presupposition freegen for GPT-OSS, Gemma-4, and OLMo-2
# sequentially on box 118. 24 questions, 5 rounds, 3 prompt types.
# Layers per model are chosen per their primary_pair.layer in the
# compliance-analysis summary; we sweep (primary, primary-3, primary-6)
# as single-layer configs to get a layer dose response.
set -euo pipefail

cd /home/persona-vectors

OUT_ROOT="neurips-results/exp17"
mkdir -p "$OUT_ROOT"

COMMON_ARGS=(
  --sycon-root external/SYCON-Bench/false-presuppositions-setting
  --intervention-mode interpolate_mean
  --alphas 0,0.3,0.5,0.7,1.0
  --prompt-types base,critical,presupposition
  --num-rounds 5
  --max-questions 24
  --position-mode all_prompt_tokens
  --batch-size 24
  --max-batch-tokens 24576
  --max-new-tokens 256
)

run_one() {
  local tag="$1"; shift
  local out_dir="$OUT_ROOT/sycon_false_presupp_${tag}_24q_256tok"
  mkdir -p "$out_dir"
  echo "[$(date -u +%FT%TZ)] START $tag -> $out_dir"
  uv run python -m src.exp17.run_sycon_false_presupp_freegen \
    "$@" \
    --output-dir "$out_dir" \
    > "$out_dir/run.log" 2>&1 < /dev/null
  echo "[$(date -u +%FT%TZ)] END   $tag"
}

# GPT-OSS (primary layer 18, pos endorsement_end). Sweep L12/L15/L18.
run_one gpt_oss \
  --model openai/gpt-oss-20b \
  --extraction-dir neurips-results/gpt-oss/mechanism/gpt_oss_authority_activations \
  --direction-path neurips-results/gpt-oss/mechanism/gpt_oss_compliance_analysis/primary_direction.pt \
  --masks-path neurips-results/gpt-oss/mechanism/gpt_oss_authority_activations/label_masks.json \
  --direction-position endorsement_end \
  --patch-source-position endorsement_end \
  --target-layers 12,15,18 --single-layer-sweep \
  "${COMMON_ARGS[@]}"

# Gemma-4 no-thinking (primary layer 22, pos endorsement_mean). Sweep L15/L18/L22.
run_one gemma4 \
  --model google/gemma-4-26B-A4B-it \
  --extraction-dir neurips-results/gemma4/mechanism/gemma4_authority_activations_no_thinking \
  --direction-path neurips-results/gemma4/mechanism/gemma4_compliance_analysis_no_thinking/primary_direction.pt \
  --masks-path neurips-results/gemma4/mechanism/gemma4_authority_activations_no_thinking/label_masks.json \
  --direction-position endorsement_mean \
  --patch-source-position endorsement_mean \
  --target-layers 15,18,22 --single-layer-sweep \
  "${COMMON_ARGS[@]}"

# OLMo-2 (primary layer 16, pos answer_position). Sweep L10/L16/L22.
run_one olmo2 \
  --model allenai/OLMo-2-0325-32B-Instruct \
  --extraction-dir neurips-results/olmo2/mechanism/olmo2_authority_activations \
  --direction-path neurips-results/olmo2/mechanism/olmo2_compliance_analysis/primary_direction.pt \
  --masks-path neurips-results/olmo2/mechanism/olmo2_authority_activations/label_masks.json \
  --direction-position answer_position \
  --patch-source-position answer_position \
  --target-layers 10,16,22 --single-layer-sweep \
  "${COMMON_ARGS[@]}"

echo "[$(date -u +%FT%TZ)] ALL DONE"
