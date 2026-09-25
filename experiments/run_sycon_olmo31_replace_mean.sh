#!/usr/bin/env bash
# SYCON false-presupposition rerun for OLMo-3.1 with REPLACE_MEAN (vs the original
# interpolate_mean run) to match the stronger forward-patch recipe that produced
# the 58.8% matched-flip headline on trivia. Keeps every other knob identical to
# neurips-results/exp17/sycon_fp_v3_olmo31_100q_l15_l18_l22_a0_a03_a05_apw so the
# two runs are directly comparable.
#
# Hardware: H100 (80GB), FA2 enabled via the updated llama_loader (prefer_flash_attention
# for "olmo" models). Bumped batch size and token budget to exploit the H100.
#
# Usage:
#   bash experiments/run_sycon_olmo31_replace_mean.sh
set -euo pipefail

cd /home/persona-vectors

OUT_DIR="neurips-results/exp17/sycon_fp_v3_olmo31_100q_l15_l18_l22_replace_mean_apw_h100"
mkdir -p "$OUT_DIR" logs

uv run python -m src.exp17.run_sycon_false_presupp_freegen \
  --model allenai/OLMo-3.1-32B-Instruct \
  --sycon-root external/SYCON-Bench/false-presuppositions-setting \
  --pushback-csv external/SYCON-Bench/false-presuppositions-setting/data/push_back.csv \
  --pushback-mode benchmark \
  --extraction-dir neurips-results/olmo31/mechanism/authority_activations_all_prior_wrong_shared_h100 \
  --direction-path neurips-results/olmo31/mechanism/compliance_analysis_all_prior_wrong_shared_h100/primary_direction.pt \
  --masks-path neurips-results/olmo31/mechanism/authority_activations_all_prior_wrong_shared_h100/label_masks.json \
  --direction-key vector \
  --direction-position endorsed_answer \
  --patch-source-position endorsed_answer \
  --direction-kind shared_within_label \
  --direction-uids-subset w1 \
  --patch-uids-subset w1 \
  --patch-source-style authoritative_verified \
  --patch-source-condition W1_note \
  --intervention-mode replace_mean \
  --alphas 1.0 \
  --target-layers 15,18,22 \
  --single-layer-sweep \
  --components block_output \
  --position-mode all_prompt_tokens \
  --apply-phase prompt \
  --norm-scaling none \
  --sparse-topk 0 \
  --prompt-types presupposition \
  --num-rounds 5 \
  --max-questions 100 \
  --batch-size 24 \
  --max-batch-tokens 24576 \
  --max-new-tokens 256 \
  --temperature 0.0 \
  --top-p 1.0 \
  --top-k 50 \
  --seed 42 \
  --output-dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/run.log"

echo "[$(date -u +%FT%TZ)] DONE OLMo-3.1 SYCON replace_mean -> $OUT_DIR"
