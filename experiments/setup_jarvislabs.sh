#!/usr/bin/env bash
# setup_jarvislabs.sh — One-shot JarvisLabs instance provisioning for persona-vectors.
#
# Usage (setup only):
#   ./experiments/setup_jarvislabs.sh ssh -p 11014 root@sshk.jarvislabs.ai
#
# Usage (setup + launch exp13):
#   ./experiments/setup_jarvislabs.sh ssh -p 11014 root@sshk.jarvislabs.ai --run exp13
#
# Usage (override defaults):
#   ./experiments/setup_jarvislabs.sh ssh -p 11014 root@sshk.jarvislabs.ai \
#       --hf-token hf_XXXX --gh-mode pat --gh-token ghp_XXXX
#
# GitHub auth modes:
#   --gh-mode ssh  (default)  Generates an SSH key on the instance, prints it,
#                             and waits for you to add it to GitHub before continuing.
#   --gh-mode pat             Uses a GitHub PAT for HTTPS cloning (no key setup needed).
#
# What it does:
#   1. Fixes terminal for tmux (TERM=xterm-256color)
#   2. Clones the repo and checks out the target branch
#   3. Syncs sycophancy-eval from the local checkout
#   4. Installs Python 3.12 via uv and runs uv sync
#   5. Logs in to HuggingFace
#   6. Sets up persistent storage (/home symlinks)
#   7. Verifies CUDA, imports, and GPU availability
#
# After setup, optionally launches experiments in parallel tmux sessions.
#
# ----- Tokens (quick reference) -----
# HF:     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
# GitHub: SSH key auth (instance generates a key; add to GitHub each time)

set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
HF_TOKEN="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"
GH_MODE="ssh"       # "ssh" or "pat"
GH_TOKEN=""          # only used when GH_MODE=pat
SSH_ARGS=()
BRANCH="neurips-submission"
RUN_EXP=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_SYCO_DIR="${LOCAL_REPO_ROOT}/external/sycophancy-eval"

# --------------------------------------------------------------------------- #
# Parse arguments
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-token)  HF_TOKEN="$2"; shift 2 ;;
        --gh-mode)   GH_MODE="$2"; shift 2 ;;
        --gh-token)  GH_TOKEN="$2"; shift 2 ;;
        --branch)    BRANCH="$2"; shift 2 ;;
        --run)       RUN_EXP="$2"; shift 2 ;;
        *)           SSH_ARGS+=("$1"); shift ;;
    esac
done

if [[ ${#SSH_ARGS[@]} -eq 0 ]]; then
    echo "Usage: $0 <ssh-args...> [options]"
    echo ""
    echo "Options:"
    echo "  --hf-token TOKEN   HuggingFace token (default: built-in)"
    echo "  --gh-mode ssh|pat  GitHub auth mode (default: ssh)"
    echo "  --gh-token TOKEN   GitHub PAT (only for --gh-mode pat)"
    echo "  --branch BRANCH    Git branch to checkout (default: neurips-submission)"
    echo "  --run EXPERIMENT   Launch experiment after setup (e.g. exp13)"
    echo ""
    echo "Examples:"
    echo "  $0 ssh -p 11014 root@sshk.jarvislabs.ai"
    echo "  $0 ssh -p 11014 root@sshk.jarvislabs.ai --run exp13"
    echo "  $0 ssh -p 11014 root@sshk.jarvislabs.ai --gh-mode pat --gh-token ghp_XXX"
    exit 1
fi

# Prompt for PAT if gh-mode is pat and no token given
if [[ "$GH_MODE" == "pat" && -z "$GH_TOKEN" ]]; then
    echo ""
    echo "GitHub PAT (for HTTPS cloning). Create at: https://github.com/settings/tokens"
    read -rsp "GitHub PAT (ghp_...): " GH_TOKEN
    echo ""
    if [[ -z "$GH_TOKEN" ]]; then
        echo "Error: GitHub PAT is required for --gh-mode pat."
        exit 1
    fi
fi

SSH_CMD=("${SSH_ARGS[@]}" -o StrictHostKeyChecking=no)
SSH_DEST="${SSH_ARGS[$((${#SSH_ARGS[@]} - 1))]}"
SSH_BIN="${SSH_ARGS[0]}"
SSH_RSYNC_ARGS=("${SSH_ARGS[@]:1:$((${#SSH_ARGS[@]} - 2))}" -o StrictHostKeyChecking=no)
SSH_RSYNC_SHELL="${SSH_BIN}"
if [[ ${#SSH_RSYNC_ARGS[@]} -gt 0 ]]; then
    SSH_RSYNC_SHELL+=" ${SSH_RSYNC_ARGS[*]}"
fi

echo ""
echo "============================================================"
echo "  JarvisLabs Setup for persona-vectors"
echo "============================================================"
echo "  SSH:       ${SSH_CMD[*]}"
echo "  Branch:    ${BRANCH}"
echo "  GitHub:    ${GH_MODE} auth"
echo "  Run:       ${RUN_EXP:-none (setup only)}"
echo "============================================================"
echo ""

# Helper: run a command on the remote instance.
remote() {
    "${SSH_CMD[@]}" "$1"
}

# --------------------------------------------------------------------------- #
# Step 1: Terminal fix
# --------------------------------------------------------------------------- #
echo "[1/7] Fixing terminal, installing rsync, adding GitHub to known_hosts..."
remote '
grep -q "TERM=xterm-256color" ~/.bashrc 2>/dev/null || echo "export TERM=xterm-256color" >> ~/.bashrc
apt-get update -qq && apt-get install -y -qq rsync >/dev/null 2>&1
ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null
echo "done"
'

# --------------------------------------------------------------------------- #
# Step 2: Clone repo
# --------------------------------------------------------------------------- #
echo "[2/7] Cloning persona-vectors repo..."

if [[ "$GH_MODE" == "ssh" ]]; then
    # --- SSH key flow ---
    # Check if repo already exists (skip key setup)
    REPO_EXISTS=$(remote 'test -d /root/persona-vectors && echo yes || (test -d /home/persona-vectors && echo home || echo no)')

    if [[ "$REPO_EXISTS" == "yes" ]]; then
        echo "  Repo already exists, updating..."
        remote "cd /root/persona-vectors && git checkout ${BRANCH} && git fetch origin && git pull --ff-only origin ${BRANCH}"
    elif [[ "$REPO_EXISTS" == "home" ]]; then
        echo "  Restoring repo from /home (post-resume)..."
        remote "
            ln -sf /home/persona-vectors /root/persona-vectors
            cd /root/persona-vectors && git checkout ${BRANCH} && git fetch origin && git pull --ff-only origin ${BRANCH}
        "
    else
        # Generate SSH key if needed
        echo "  Generating SSH key on instance..."
        PUBKEY=$(remote '
            if [ ! -f /root/.ssh/id_ed25519 ]; then
                ssh-keygen -t ed25519 -C "jarvislabs-persona-vectors" -f /root/.ssh/id_ed25519 -N "" >/dev/null 2>&1
            fi
            cat /root/.ssh/id_ed25519.pub
        ')
        echo ""
        echo "  ┌──────────────────────────────────────────────────────┐"
        echo "  │  Add this SSH key to GitHub, then press Enter:       │"
        echo "  │  https://github.com/settings/keys > New SSH key      │"
        echo "  └──────────────────────────────────────────────────────┘"
        echo ""
        echo "  ${PUBKEY}"
        echo ""
        read -rp "  Press Enter once the key is added to GitHub... "

        remote "
            cd /root
            GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no' git clone git@github.com:Lossfunk/persona-vectors.git
            cd persona-vectors
            git checkout ${BRANCH}
            echo 'Cloned fresh via SSH.'
        "
    fi
else
    # --- HTTPS + PAT flow ---
    remote "
        if [ -d /root/persona-vectors ]; then
            echo 'Repo already exists, updating...'
            cd /root/persona-vectors
            git checkout ${BRANCH}
            git fetch origin
            git pull --ff-only origin ${BRANCH}
        elif [ -d /home/persona-vectors ]; then
            echo 'Restoring repo from /home (post-resume)...'
            ln -sf /home/persona-vectors /root/persona-vectors
            cd /root/persona-vectors
            git checkout ${BRANCH}
            git fetch origin
            git pull --ff-only origin ${BRANCH}
        else
            cd /root
            git clone https://${GH_TOKEN}@github.com/Lossfunk/persona-vectors.git
            cd persona-vectors
            git checkout ${BRANCH}
            echo 'Cloned fresh via HTTPS.'
        fi
    "
fi

# --------------------------------------------------------------------------- #
# Step 3: Sync sycophancy-eval
# --------------------------------------------------------------------------- #
echo "[3/7] Syncing sycophancy-eval from local checkout..."
if [[ ! -d "${LOCAL_SYCO_DIR}" ]]; then
    echo "Error: local sycophancy-eval not found at ${LOCAL_SYCO_DIR}"
    exit 1
fi
remote '
cd /root/persona-vectors
mkdir -p external data
rm -rf external/sycophancy-eval
'
rsync -az --delete -e "${SSH_RSYNC_SHELL}" "${LOCAL_SYCO_DIR}/" "${SSH_DEST}:/root/persona-vectors/external/sycophancy-eval/"
remote '
cd /root/persona-vectors
if [ -f external/sycophancy-eval/datasets/answer.jsonl ]; then
    ln -sfn ../external/sycophancy-eval/datasets/answer.jsonl data/answer.jsonl
    echo "Synced sycophancy-eval and refreshed data/answer.jsonl symlink."
else
    echo "Error: sycophancy-eval sync missing datasets/answer.jsonl"
    exit 1
fi
'

# --------------------------------------------------------------------------- #
# Step 4: Python + dependencies
# --------------------------------------------------------------------------- #
echo "[4/7] Installing Python 3.12 and dependencies..."
remote '
cd /root/persona-vectors
uv python install 3.12 2>&1 | tail -1
echo "3.12.10" > .python-version
uv sync 2>&1 | tail -5
echo "Dependencies installed."
'

# --------------------------------------------------------------------------- #
# Step 5: HuggingFace login
# --------------------------------------------------------------------------- #
echo "[5/7] Logging in to HuggingFace..."
remote "
cd /root/persona-vectors
uv run huggingface-cli login --token ${HF_TOKEN} 2>&1 | grep -E 'Login|Token is valid|Error'
"

# --------------------------------------------------------------------------- #
# Step 6: Persistent storage
# --------------------------------------------------------------------------- #
echo "[6/7] Setting up persistent storage (/home symlinks)..."
remote '
# Copy repo to /home if not already there
if [ ! -d /home/persona-vectors ] && [ -d /root/persona-vectors ] && [ ! -L /root/persona-vectors ]; then
    cp -r /root/persona-vectors /home/persona-vectors
    rm -rf /root/persona-vectors
    ln -sf /home/persona-vectors /root/persona-vectors
    echo "Moved repo to /home."
elif [ -d /home/persona-vectors ] && [ ! -L /root/persona-vectors ]; then
    rm -rf /root/persona-vectors
    ln -sf /home/persona-vectors /root/persona-vectors
    echo "Symlinked existing /home repo."
else
    echo "Already configured."
fi

# HF cache
mkdir -p /home/.cache/huggingface
mkdir -p /root/.cache
ln -sf /home/.cache/huggingface /root/.cache/huggingface 2>/dev/null || true

# SSH keys
cp -r /root/.ssh /home/.ssh 2>/dev/null || true
echo "done"
'

# --------------------------------------------------------------------------- #
# Step 7: Verify
# --------------------------------------------------------------------------- #
echo "[7/7] Verifying installation..."
remote '
cd /root/persona-vectors
uv run python -c "
import torch
n_gpus = torch.cuda.device_count()
print(f\"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {n_gpus}\")
for i in range(n_gpus):
    print(f\"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)\")
import transformers
print(f\"Transformers {transformers.__version__}\")
from src.exp7.scoring import score_prompt_forced_choice
print(\"Scoring import OK\")
print(\"All good.\")
"
'

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"

# --------------------------------------------------------------------------- #
# Optional: launch experiment
# --------------------------------------------------------------------------- #
if [[ "$RUN_EXP" == "exp13" ]]; then
    echo ""
    echo "Launching exp13 in parallel tmux sessions..."
    remote '
cd /root/persona-vectors

# Llama on GPU 0
tmux new-session -d -s llama "\
  cd /root/persona-vectors && \
  CUDA_VISIBLE_DEVICES=0 uv run python -m src.exp13.run_signal_strength \
    --models meta-llama/Llama-3.1-8B-Instruct \
    --batch-size 32 \
    --output-dir new-phase-results/exp13 \
    2>&1 | tee /tmp/llama_exp13.log"

# Qwen on GPU 1
tmux new-session -d -s qwen "\
  cd /root/persona-vectors && \
  CUDA_VISIBLE_DEVICES=1 uv run python -m src.exp13.run_signal_strength \
    --models Qwen/Qwen3-4B-Instruct-2507 \
    --batch-size 32 \
    --output-dir new-phase-results/exp13 \
    2>&1 | tee /tmp/qwen_exp13.log"

echo "tmux sessions:"
tmux list-sessions
echo ""
echo "Monitor:"
echo "  tmux attach -t llama"
echo "  tmux attach -t qwen"
echo "  tail -f /tmp/llama_exp13.log"
'
    echo ""
    echo "Experiments running. SSH in to monitor."

elif [[ -n "$RUN_EXP" ]]; then
    echo ""
    echo "Unknown experiment: ${RUN_EXP}. Only 'exp13' is supported with --run."
    echo "SSH in and run manually."
fi

echo ""
echo "To SSH in:  ${SSH_CMD[*]}"
echo ""
