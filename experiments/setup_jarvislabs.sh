#!/usr/bin/env bash
# setup_jarvislabs.sh — Idempotent first-time setup for a JarvisLabs GPU instance.
#
# Usage:
#   ./experiments/setup_jarvislabs.sh ssh -p 11014 root@sshk.jarvislabs.ai \
#       --gh-token ghp_xxx --hf-token hf_xxx
#
# Optional:
#   --gh-token TOKEN         (PAT; if unavailable script falls back to SSH key)
#   --gh-user USER           (default: MajorTimberWolf)
#   --repo OWNER/REPO        (default: lossfunk/persona-vectors)
#   --run exp13              (launches the existing exp13 tmux helper)

set -euo pipefail

GH_TOKEN="${GITHUB_TOKEN:-}"
GH_USER="${GITHUB_USER:-MajorTimberWolf}"
HF_TOKEN="${HF_TOKEN:-${HF_token:-}}"
BRANCH="neurips-submission"
REPO_SLUG="lossfunk/persona-vectors"
RUN_EXP=""
SSH_ARGS=()
GITHUB_SSH_KEY_PATH="/home/.ssh/id_ed25519_github_jarvis"
GITHUB_SSH_PUB_PATH="${GITHUB_SSH_KEY_PATH}.pub"
SSH_NO_HOSTKEY_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_SYCO_DIR="${LOCAL_REPO_ROOT}/external/sycophancy-eval"

escape_squotes() {
    printf "%s" "$1" | sed "s/'/'\\\\''/g"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gh-token) GH_TOKEN="$2"; shift 2 ;;
        --gh-user) GH_USER="$2"; shift 2 ;;
        --hf-token) HF_TOKEN="$2"; shift 2 ;;
        --branch) shift 2 ;; # kept for backwards compatibility; branch is fixed
        --repo) REPO_SLUG="$2"; shift 2 ;;
        --run) RUN_EXP="$2"; shift 2 ;;
        *) SSH_ARGS+=("$1"); shift ;;
    esac
done

if [[ ${#SSH_ARGS[@]} -eq 0 ]]; then
    echo "Usage: $0 <ssh-args...> [--gh-token <token>] [--gh-user <username>] [--hf-token <hf_token>] [--repo owner/repo]"
    exit 1
fi

if [[ -z "${GH_TOKEN}" ]]; then
    echo "No GitHub PAT provided; script will use SSH key fallback."
fi

if [[ -z "${HF_TOKEN}" ]]; then
    read -rsp "HF token (optional, press Enter to skip): " HF_TOKEN
    echo ""
fi

SSH_CMD=("${SSH_ARGS[@]}" "${SSH_NO_HOSTKEY_OPTS[@]}")
SSH_DEST="${SSH_ARGS[$((${#SSH_ARGS[@]} - 1))]}"
SSH_BIN="${SSH_ARGS[0]}"
SSH_RSYNC_ARGS=("${SSH_ARGS[@]:1:$((${#SSH_ARGS[@]} - 2))}" "${SSH_NO_HOSTKEY_OPTS[@]}")
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
BRANCH_ESC="$(escape_squotes "${BRANCH}")"
REPO_ESC="$(escape_squotes "${REPO_SLUG}")"

echo ""
echo "============================================================"
echo "  JarvisLabs Setup for persona-vectors"
echo "============================================================"
echo "  SSH:    ${SSH_CMD[*]}"
echo "  Repo:   ${REPO_SLUG}"
echo "  GitHub: ${GH_USER}"
echo "  Branch: ${BRANCH}"
echo "============================================================"
echo ""

echo "[1/7] Installing base packages, persistent dirs, and shell hooks..."
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
mkdir -p /home/.config/persona-vectors /home/.cache/huggingface /home/.ssh /root/.ssh /root/.cache
chmod 700 /home/.ssh /root/.ssh
touch /home/.ssh/known_hosts
ssh-keyscan github.com >> /home/.ssh/known_hosts 2>/dev/null || true
chmod 600 /home/.ssh/known_hosts
for rc in /root/.bashrc /home/.bashrc; do
  touch \"\$rc\"
  grep -q 'export TERM=xterm-256color' \"\$rc\" || echo 'export TERM=xterm-256color' >> \"\$rc\"
  grep -q 'source /home/.config/persona-vectors/env.sh' \"\$rc\" || echo '[ -f /home/.config/persona-vectors/env.sh ] && source /home/.config/persona-vectors/env.sh' >> \"\$rc\"
done
"

echo "[2/7] Saving PAT/HF env + git credentials to persistent /home..."
remote "
set -euo pipefail
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
cat > /home/.git-credentials <<'EOF'
https://${GH_USER_ESC}:${GH_TOKEN_ESC}@github.com
EOF
chmod 600 /home/.git-credentials
git config --global credential.helper 'store --file /home/.git-credentials'
"

echo "[3/7] Cloning/updating repo (PAT first, SSH fallback)..."
remote "
set -euo pipefail
source /home/.config/persona-vectors/env.sh
REPO_URL_HTTPS='https://github.com/${REPO_ESC}.git'
REPO_URL_HTTPS_AUTH=\"https://${GH_USER_ESC}:\${GITHUB_TOKEN}@github.com/${REPO_ESC}.git\"
REPO_URL_SSH='git@github.com:${REPO_ESC}.git'
SSH_KEY='${GITHUB_SSH_KEY_PATH}'
SSH_PUB='${GITHUB_SSH_PUB_PATH}'
SSH_CMD=\"ssh -i \${SSH_KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new\"
AUTH_MODE='pat'
if [ -z \"\${GITHUB_TOKEN}\" ] || ! git ls-remote \"\${REPO_URL_HTTPS_AUTH}\" HEAD >/dev/null 2>&1; then
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
    echo 'Add this public key to GitHub and rerun setup:'
    cat \"\${SSH_PUB}\"
    echo 'https://github.com/settings/keys'
    echo '============================================================'
    exit 24
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
cd /home/persona-vectors
if [ \"\${AUTH_MODE}\" = 'ssh' ]; then
  git remote set-url origin \"\${REPO_URL_SSH}\"
  GIT_SSH_COMMAND=\"\${SSH_CMD}\" git fetch origin
  git checkout '${BRANCH_ESC}'
  GIT_SSH_COMMAND=\"\${SSH_CMD}\" git pull --ff-only origin '${BRANCH_ESC}'
else
  git remote set-url origin \"\${REPO_URL_HTTPS}\"
  git fetch origin
  git checkout '${BRANCH_ESC}'
  git pull --ff-only origin '${BRANCH_ESC}'
fi
"

echo "[4/7] Syncing external/sycophancy-eval..."
if [[ ! -d "${LOCAL_SYCO_DIR}" ]]; then
    echo "Error: local sycophancy-eval not found at ${LOCAL_SYCO_DIR}"
    exit 1
fi
remote "
set -euo pipefail
mkdir -p /home/persona-vectors/external /home/persona-vectors/data
rm -rf /home/persona-vectors/external/sycophancy-eval
"
rsync -az --delete -e "${SSH_RSYNC_SHELL}" "${LOCAL_SYCO_DIR}/" "${SSH_DEST}:/home/persona-vectors/external/sycophancy-eval/"
remote "
set -euo pipefail
cd /home/persona-vectors
if [ -f external/sycophancy-eval/datasets/answer.jsonl ]; then
  ln -sfn ../external/sycophancy-eval/datasets/answer.jsonl data/answer.jsonl
else
  echo 'Error: sycophancy-eval sync missing datasets/answer.jsonl'
  exit 1
fi
"

echo "[5/7] Installing uv (if needed) and running uv sync..."
remote "
set -euo pipefail
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
fi
cd /home/persona-vectors
uv python install 3.12
echo '3.12' > .python-version
uv sync
"

echo "[6/7] Configuring persistent HF cache + /root symlinks..."
remote "
set -euo pipefail
source /home/.config/persona-vectors/env.sh
mkdir -p /home/.cache/huggingface /root/.cache /root/.ssh
if [ -n \"\${HF_TOKEN}\" ]; then
  printf '%s' \"\${HF_TOKEN}\" > /home/.cache/huggingface/token
  chmod 600 /home/.cache/huggingface/token
fi
ln -sfn /home/.cache/huggingface /root/.cache/huggingface
rsync -a /home/.ssh/ /root/.ssh/
chmod 700 /root/.ssh
chmod 600 /root/.ssh/known_hosts 2>/dev/null || true
"

echo "[7/7] Verifying CUDA, imports, and env visibility..."
remote "
set -euo pipefail
source /home/.config/persona-vectors/env.sh
cd /root/persona-vectors
uv run python -c '
import os
import torch
import transformers
n_gpus = torch.cuda.device_count()
print(f\"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {n_gpus}\")
for i in range(n_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f\"  GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1024**3:.1f} GB)\")
print(f\"Transformers {transformers.__version__}\")
print(\"HF_TOKEN set:\", bool(os.environ.get(\"HF_TOKEN\")))
from src.exp7.scoring import score_prompt_forced_choice
print(\"Scoring import OK\")'
"

echo ""
echo "============================================================"
echo "  Setup complete."
echo "============================================================"

if [[ "${RUN_EXP}" == "exp13" ]]; then
    echo ""
    echo "Launching exp13 tmux sessions..."
    remote '
set -euo pipefail
cd /root/persona-vectors
tmux new-session -d -s llama "\
  cd /root/persona-vectors && \
  CUDA_VISIBLE_DEVICES=0 uv run python -m src.exp13.run_signal_strength \
    --models meta-llama/Llama-3.1-8B-Instruct \
    --batch-size 32 \
    --output-dir new-phase-results/exp13 \
    2>&1 | tee /tmp/llama_exp13.log"
tmux new-session -d -s qwen "\
  cd /root/persona-vectors && \
  CUDA_VISIBLE_DEVICES=1 uv run python -m src.exp13.run_signal_strength \
    --models Qwen/Qwen3-4B-Instruct-2507 \
    --batch-size 32 \
    --output-dir new-phase-results/exp13 \
    2>&1 | tee /tmp/qwen_exp13.log"
tmux list-sessions
'
fi

echo ""
echo "SSH command: ${SSH_CMD[*]}"
echo "After reconnect, run: source /home/.config/persona-vectors/env.sh"
echo ""
