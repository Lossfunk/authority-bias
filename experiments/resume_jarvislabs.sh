#!/usr/bin/env bash
# resume_jarvislabs.sh — Restore a JarvisLabs instance after pause/resume.
#
# Usage:
#   ./experiments/resume_jarvislabs.sh ssh -p 11014 root@sshj.jarvislabs.ai
#
# What it does:
#   1. Restores SSH keys and GitHub known_hosts
#   2. Restores repo and HF cache symlinks (/home -> /root)
#   3. Installs rsync and fixes terminal
#   4. Verifies everything works (CUDA, imports, git pull)
#
# Note: The SSH command changes every time you pause/resume.
#       Check the JarvisLabs dashboard for the new one.

set -euo pipefail

SSH_ARGS=()
BRANCH="neurips-submission"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_SYCO_DIR="${LOCAL_REPO_ROOT}/external/sycophancy-eval"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --branch) BRANCH="$2"; shift 2 ;;
        *) SSH_ARGS+=("$1"); shift ;;
    esac
done

if [[ ${#SSH_ARGS[@]} -eq 0 ]]; then
    echo "Usage: $0 <ssh-args...>"
    echo ""
    echo "Example:"
    echo "  $0 ssh -p 11014 root@sshj.jarvislabs.ai"
    exit 1
fi

SSH_CMD=("${SSH_ARGS[@]}" -o StrictHostKeyChecking=no)
SSH_DEST="${SSH_ARGS[$((${#SSH_ARGS[@]} - 1))]}"
SSH_BIN="${SSH_ARGS[0]}"
SSH_RSYNC_ARGS=("${SSH_ARGS[@]:1:$((${#SSH_ARGS[@]} - 2))}" -o StrictHostKeyChecking=no)
SSH_RSYNC_SHELL="${SSH_BIN}"
if [[ ${#SSH_RSYNC_ARGS[@]} -gt 0 ]]; then
    SSH_RSYNC_SHELL+=" ${SSH_RSYNC_ARGS[*]}"
fi

remote() {
    "${SSH_CMD[@]}" "$1"
}

echo ""
echo "============================================================"
echo "  JarvisLabs Post-Resume Restore"
echo "============================================================"
echo "  SSH: ${SSH_CMD[*]}"
echo "  Branch: ${BRANCH}"
echo "============================================================"
echo ""

echo "[1/4] Restoring SSH keys and GitHub known_hosts..."
remote '
cp -r /home/.ssh/* /root/.ssh/ 2>/dev/null || true
ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null
echo "done"
'

echo "[2/4] Restoring symlinks (/home -> /root)..."
remote '
# Repo
ln -sf /home/persona-vectors /root/persona-vectors 2>/dev/null
echo "  /root/persona-vectors -> /home/persona-vectors"

# HF cache
mkdir -p /root/.cache
ln -sf /home/.cache/huggingface /root/.cache/huggingface 2>/dev/null
echo "  /root/.cache/huggingface -> /home/.cache/huggingface"
'

echo "[3/4] Installing rsync, fixing terminal, syncing sycophancy-eval..."
remote '
apt-get update -qq && apt-get install -y -qq rsync >/dev/null 2>&1
grep -q "TERM=xterm-256color" ~/.bashrc 2>/dev/null || echo "export TERM=xterm-256color" >> ~/.bashrc
mkdir -p /root/persona-vectors/external /root/persona-vectors/data
echo "done"
'
if [[ ! -d "${LOCAL_SYCO_DIR}" ]]; then
    echo "Error: local sycophancy-eval not found at ${LOCAL_SYCO_DIR}"
    exit 1
fi
remote 'rm -rf /root/persona-vectors/external/sycophancy-eval'
rsync -az --delete -e "${SSH_RSYNC_SHELL}" "${LOCAL_SYCO_DIR}/" "${SSH_DEST}:/root/persona-vectors/external/sycophancy-eval/"
remote '
cd /root/persona-vectors
if [ -f external/sycophancy-eval/datasets/answer.jsonl ]; then
    ln -sfn ../external/sycophancy-eval/datasets/answer.jsonl data/answer.jsonl
    echo "synced sycophancy-eval"
else
    echo "Error: sycophancy-eval sync missing datasets/answer.jsonl"
    exit 1
fi
'

echo "[4/4] Verifying..."
remote '
cd /root/persona-vectors

# Git pull
echo "  git update..."
git checkout ${BRANCH}
git fetch origin
git pull --ff-only origin ${BRANCH} 2>&1 | head -5

# CUDA check
uv run python -c "
import torch
n = torch.cuda.device_count()
print(f\"  GPUs: {n}\")
for i in range(n):
    print(f\"    GPU {i}: {torch.cuda.get_device_name(i)}\")
print(\"  All good.\")
"
'

echo ""
echo "============================================================"
echo "  Resume complete! Instance is ready."
echo "============================================================"
echo "  SSH in:  ${SSH_CMD[*]}"
echo ""
