#!/usr/bin/env bash
set -euo pipefail

REMOTE="root@217.18.55.79"
SSH_OPTS=(-o StrictHostKeyChecking=no)
ROOT="/Users/majortimberwolf/Projects/lossfunk/persona-vectors"

cd "$ROOT"

pull_dir() {
  local name="$1"
  mkdir -p "new-phase-results/mechanism/$name"
  rsync -avz -e "ssh ${SSH_OPTS[*]}" \
    "$REMOTE:/root/persona-vectors/new-phase-results/mechanism/$name/" \
    "new-phase-results/mechanism/$name/" >/dev/null 2>&1 || true
}

done_flag() {
  local name="$1"
  ssh "${SSH_OPTS[@]}" "$REMOTE" \
    "test -f /root/persona-vectors/new-phase-results/mechanism/$name/summary.json && echo 1 || echo 0"
}

TARGETS=(
  qwen_correction_gating_patching_w1note_entrenching
  qwen_correction_gating_patching_c1expert_entrenching
  qwen_correction_gating_patching_boundary_entrenching
  qwen_correction_gating_patching_boundary_correcting
)

for name in "${TARGETS[@]}"; do
  mkdir -p "new-phase-results/mechanism/$name"
done

while true; do
  for name in "${TARGETS[@]}"; do
    pull_dir "$name"
  done

  w1_done="$(done_flag qwen_correction_gating_patching_w1note_entrenching)"
  exp_done="$(done_flag qwen_correction_gating_patching_c1expert_entrenching)"
  be_done="$(done_flag qwen_correction_gating_patching_boundary_entrenching)"
  bc_done="$(done_flag qwen_correction_gating_patching_boundary_correcting)"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] w1=$w1_done c1expert=$exp_done boundary_ent=$be_done boundary_corr=$bc_done" \
    >> new-phase-results/mechanism/watch_qwen_remote_mech.log

  if [[ "$w1_done" == "1" && "$exp_done" == "1" && "$be_done" == "1" && "$bc_done" == "1" ]]; then
    exit 0
  fi

  sleep 120
done
