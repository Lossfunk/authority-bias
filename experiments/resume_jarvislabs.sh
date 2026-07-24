#!/usr/bin/env bash
# resume_jarvislabs.sh — Rehydrate a paused JarvisLabs instance from persistent /home.
#
# Usage:
#   ./experiments/resume_jarvislabs.sh ssh -p 11014 root@sshj.jarvislabs.ai
#
# What it does:
#   1. Reinstalls required system tools (rsync/git/curl)
#   2. Restores /root symlinks and shell env sourcing
#   3. Restores known_hosts + PAT/SSH git auth behavior
#   4. Updates repo, syncs sycophancy-eval, verifies CUDA + env

set -euo pipefail

GH_TOKEN="${GITHUB_TOKEN:-}"
GH_USER="${GITHUB_USER:-MajorTimberWolf}"
HF_TOKEN="${HF_TOKEN:-${HF_token:-}}"
SSH_ARGS=()
BRANCH="neurips-submission"
REPO_SLUG="lossfunk/persona-vectors"
GITHUB_SSH_KEY_PATH="/home/.ssh/id_ed25519_github_jarvis"
GITHUB_SSH_PUB_PATH="${GITHUB_SSH_KEY_PATH}.pub"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_SYCO_DIR="${LOCAL_REPO_ROOT}/external/sycophancy-eval"

escape_squotes() {
    printf "%s" "$1" | sed "s/'/'\\\\''/g"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --branch) shift 2 ;; # kept for backwards compatibility; branch is fixed
        --gh-token) GH_TOKEN="$2"; shift 2 ;;
        --gh-user) GH_USER="$2"; shift 2 ;;
        --hf-token) HF_TOKEN="$2"; shift 2 ;;
        --repo) REPO_SLUG="$2"; shift 2 ;;
        *) SSH_ARGS+=("$1"); shift ;;
    esac
done

if [[ ${#SSH_ARGS[@]} -eq 0 ]]; then
    echo "Usage: $0 <ssh-args...> [--gh-token TOKEN] [--gh-user USER] [--hf-token TOKEN] [--repo OWNER/REPO]"
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

GH_TOKEN_ESC="$(escape_squotes "${GH_TOKEN}")"
GH_USER_ESC="$(escape_squotes "${GH_USER}")"
HF_TOKEN_ESC="$(escape_squotes "${HF_TOKEN}")"

echo ""
echo "============================================================"
echo "  JarvisLabs Post-Resume Restore"
echo "============================================================"
echo "  SSH:    ${SSH_CMD[*]}"
echo "  Repo:   ${REPO_SLUG}"
echo "  GitHub: ${GH_USER}"
echo "  Branch: ${BRANCH}"
echo "============================================================"
echo ""

echo "[1/5] Reinstalling base tools + shell hooks..."
remote "
set -euo pipefail
NEED_APT=0
for bin in rsync git curl ssh; do
  if ! command -v \"\$bin\" >/dev/null 2>&1; then
    NEED_APT=1
  fi
done
if [ ! -f /etc/ssl/certs/ca-certificates.crt ]; then
  NEED_APT=1
fi
if [ \"\$NEED_APT\" -eq 1 ]; then
  APT_FLAGS='-o Acquire::Retries=1 -o Acquire::http::Timeout=10 -o Acquire::https::Timeout=10'
  apt-get \$APT_FLAGS update -qq || echo 'Warning: apt update timed out; continuing with existing packages.'
  DEBIAN_FRONTEND=noninteractive apt-get \$APT_FLAGS install -y -qq rsync git curl ca-certificates openssh-client >/dev/null || \
    echo 'Warning: apt install failed/timed out; continuing with currently installed tools.'
fi
mkdir -p /home/.ssh /home/.cache/huggingface /home/.config/persona-vectors /root/.ssh /root/.cache
touch /home/.ssh/known_hosts
ssh-keyscan github.com >> /home/.ssh/known_hosts 2>/dev/null || true
chmod 700 /home/.ssh /root/.ssh
chmod 600 /home/.ssh/known_hosts
for rc in /root/.bashrc /home/.bashrc; do
  touch \"\$rc\"
  grep -q 'export TERM=xterm-256color' \"\$rc\" || echo 'export TERM=xterm-256color' >> \"\$rc\"
  grep -q 'source /home/.config/persona-vectors/env.sh' \"\$rc\" || echo '[ -f /home/.config/persona-vectors/env.sh ] && source /home/.config/persona-vectors/env.sh' >> \"\$rc\"
done
"

echo "[2/5] Restoring /home -> /root symlinks and SSH state..."
remote '
set -euo pipefail
ln -sfn /home/persona-vectors /root/persona-vectors
ln -sfn /home/.cache/huggingface /root/.cache/huggingface
rsync -a /home/.ssh/ /root/.ssh/
chmod 700 /root/.ssh
chmod 600 /root/.ssh/known_hosts 2>/dev/null || true
'

echo "[3/5] Restoring PAT-based git auth and pulling latest branch..."
remote "
set -euo pipefail
cat > /home/.git-credentials <<'EOF'
https://${GH_USER_ESC}:${GH_TOKEN_ESC}@github.com
EOF
chmod 600 /home/.git-credentials
cat > /home/.config/persona-vectors/env.sh <<'EOF'
export GITHUB_TOKEN='${GH_TOKEN_ESC}'
export GH_TOKEN='${GH_TOKEN_ESC}'
export GITHUB_USER='${GH_USER_ESC}'
export GH_USER='${GH_USER_ESC}'
export GITHUB_SSH_KEY='${GITHUB_SSH_KEY_PATH}'
export HF_TOKEN='${HF_TOKEN_ESC}'
export HF_token='${HF_TOKEN_ESC}'
export HUGGINGFACEHUB_API_TOKEN='${HF_TOKEN_ESC}'
export HF_HOME='/home/.cache/huggingface'
EOF
chmod 600 /home/.config/persona-vectors/env.sh
if [ -n '${HF_TOKEN_ESC}' ]; then
  printf '%s' '${HF_TOKEN_ESC}' > /home/.cache/huggingface/token
  chmod 600 /home/.cache/huggingface/token
fi
git config --global credential.helper 'store --file /home/.git-credentials'
REPO_URL_HTTPS=\"https://github.com/${REPO_SLUG}.git\"
REPO_URL_HTTPS_AUTH=\"https://${GH_USER_ESC}:${GH_TOKEN_ESC}@github.com/${REPO_SLUG}.git\"
REPO_URL_SSH=\"git@github.com:${REPO_SLUG}.git\"
SSH_KEY='${GITHUB_SSH_KEY_PATH}'
SSH_PUB='${GITHUB_SSH_PUB_PATH}'
SSH_CMD=\"ssh -i \${SSH_KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new\"
AUTH_MODE='pat'
if [ -z '${GH_TOKEN_ESC}' ] || ! git ls-remote \"\${REPO_URL_HTTPS_AUTH}\" HEAD >/dev/null 2>&1; then
  AUTH_MODE='ssh'
fi
if [ \"\${AUTH_MODE}\" = 'ssh' ]; then
  echo 'PAT unavailable for this repo. Trying SSH fallback...'
  mkdir -p /home/.ssh
  chmod 700 /home/.ssh
  if [ ! -f \"\${SSH_KEY}\" ]; then
    ssh-keygen -t ed25519 -C 'jarvislabs-github' -f \"\${SSH_KEY}\" -N '' >/dev/null
  fi
  chmod 600 \"\${SSH_KEY}\"
  chmod 644 \"\${SSH_PUB}\"
  touch /home/.ssh/known_hosts
  ssh-keyscan github.com >> /home/.ssh/known_hosts 2>/dev/null || true
  chmod 600 /home/.ssh/known_hosts
  if ! GIT_SSH_COMMAND=\"\${SSH_CMD}\" git ls-remote \"\${REPO_URL_SSH}\" HEAD >/dev/null 2>&1; then
    echo ''
    echo '============================================================'
    echo 'SSH fallback needs GitHub key authorization.'
    echo 'Add this public key to GitHub and rerun resume:'
    cat \"\${SSH_PUB}\"
    echo 'https://github.com/settings/keys'
    echo '============================================================'
    exit 25
  fi
fi
if [ ! -d /home/persona-vectors/.git ]; then
  if [ \"\${AUTH_MODE}\" = 'ssh' ]; then
    GIT_SSH_COMMAND=\"\${SSH_CMD}\" git clone \"\${REPO_URL_SSH}\" /home/persona-vectors
  else
    git clone \"\${REPO_URL_HTTPS_AUTH}\" /home/persona-vectors
  fi
fi
ln -sfn /home/persona-vectors /root/persona-vectors
cd /root/persona-vectors
if [ \"\${AUTH_MODE}\" = 'ssh' ]; then
  git remote set-url origin \"\${REPO_URL_SSH}\"
  GIT_SSH_COMMAND=\"\${SSH_CMD}\" git fetch origin
  git checkout '${BRANCH}'
  GIT_SSH_COMMAND=\"\${SSH_CMD}\" git pull --ff-only origin '${BRANCH}'
else
  git remote set-url origin \"\${REPO_URL_HTTPS}\"
  git fetch origin
  git checkout '${BRANCH}'
  git pull --ff-only origin '${BRANCH}'
fi
"

echo "[4/5] Syncing external/sycophancy-eval..."
if [[ ! -d "${LOCAL_SYCO_DIR}" ]]; then
    echo "Error: local sycophancy-eval not found at ${LOCAL_SYCO_DIR}"
    exit 1
fi
remote '
set -euo pipefail
mkdir -p /root/persona-vectors/external /root/persona-vectors/data
rm -rf /root/persona-vectors/external/sycophancy-eval
'
rsync -az --delete -e "${SSH_RSYNC_SHELL}" "${LOCAL_SYCO_DIR}/" "${SSH_DEST}:/root/persona-vectors/external/sycophancy-eval/"
remote '
set -euo pipefail
cd /root/persona-vectors
if [ -f external/sycophancy-eval/datasets/answer.jsonl ]; then
  ln -sfn ../external/sycophancy-eval/datasets/answer.jsonl data/answer.jsonl
else
  echo "Error: sycophancy-eval sync missing datasets/answer.jsonl"
  exit 1
fi
'

echo "[5/5] Verifying env + CUDA + imports..."
remote '
set -euo pipefail
[ -f /home/.config/persona-vectors/env.sh ] && source /home/.config/persona-vectors/env.sh
cd /root/persona-vectors
if command -v uv >/dev/null 2>&1; then
  uv sync >/dev/null
  uv run python -c "
import os
import torch
print(f\"GPUs: {torch.cuda.device_count()}\")
print(\"HF_TOKEN set:\", bool(os.environ.get(\"HF_TOKEN\")))
"
else
  echo "Warning: uv is missing; run setup_jarvislabs.sh once."
fi
'

echo ""
echo "============================================================"
echo "  Resume complete. Instance is ready."
echo "============================================================"
echo "  SSH in: ${SSH_CMD[*]}"
echo ""
