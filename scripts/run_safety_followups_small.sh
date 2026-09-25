#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_safety_followups_small.sh <PROMPTS_JSONL> <OUTPUT_DIR>
#
# Example:
#   scripts/run_safety_followups_small.sh data/redbench_sampled_prompts.jsonl neurips-results/mechanism/redbench_followups

PROMPTS_JSONL="${1:-}"
OUTPUT_DIR="${2:-}"

if [[ -z "${PROMPTS_JSONL}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Usage: $0 <PROMPTS_JSONL> <OUTPUT_DIR>"
  exit 1
fi

uv run python -m src.exp16.run_harmbench_interventions \
  --model openai/gpt-oss-20b \
  --prompts-jsonl "${PROMPTS_JSONL}" \
  --target-layers 16,18,20 \
  --patch-modes interpolate_mean,replace_mean \
  --patch-alphas 0.5,0.7,1.0 \
  --additive-modes "" \
  --direction-kinds "" \
  --additive-alphas 0 \
  --sparse-topk-values 0 \
  --batch-size 8 --max-new-tokens 128 --temperature 0.0 \
  --output-dir "${OUTPUT_DIR}/multilayer"

uv run python -m src.exp16.run_harmbench_interventions \
  --model openai/gpt-oss-20b \
  --prompts-jsonl "${PROMPTS_JSONL}" \
  --target-layers 16,18,20 --single-layer-sweep \
  --apply-phase decode \
  --patch-modes interpolate_mean \
  --patch-alphas 0.5,0.7,1.0 \
  --additive-modes "" \
  --direction-kinds "" \
  --additive-alphas 0 \
  --sparse-topk-values 0 \
  --batch-size 8 --max-new-tokens 128 --temperature 0.0 \
  --output-dir "${OUTPUT_DIR}/decode"

uv run python -m src.exp16.run_harmbench_interventions \
  --model openai/gpt-oss-20b \
  --prompts-jsonl "${PROMPTS_JSONL}" \
  --target-layers 16,18,20 --single-layer-sweep \
  --patch-modes interpolate_mean,replace_mean \
  --patch-alphas 0.7,1.0 \
  --patch-source-condition W1_note \
  --additive-modes "" \
  --direction-kinds "" \
  --additive-alphas 0 \
  --sparse-topk-values 0 \
  --batch-size 8 --max-new-tokens 128 --temperature 0.0 \
  --output-dir "${OUTPUT_DIR}/w1_source"

echo "Done. Outputs under: ${OUTPUT_DIR}"
