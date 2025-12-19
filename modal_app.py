"""
Modal GPU runner for persona-vectors experiments.

Usage (locally, with Modal CLI configured):
    modal run modal_app.py --exp exp2
Available exp values: exp0, exp1, exp2, exp3, exp2_agg, validation

Notes:
- Code is mounted read-only from the local repo snapshot.
- Results are synced to a persistent Modal volume named "persona-vectors".
- Place large artifacts (e.g., data/answer.jsonl) in that volume beforehand:
    modal volume put persona-vectors data/answer.jsonl /data/answer.jsonl
- Warm pool defaults: MODAL_MIN_CONTAINERS=1 keeps one GPU container warm to avoid cold starts.
  Override via environment variables if you want to change pool size or scaledown window.
"""

from __future__ import annotations

import shutil
import subprocess
import os
from pathlib import Path

import modal

# Bump BUILD_VERSION to force Modal to rebuild the image when dependencies or hooks change.
BUILD_VERSION: int = 20

# Warm container configuration: keep a small pool alive to avoid cold starts.
# Set env vars to 0 to disable if you don't want to pay for idle GPU time.
# These can be overridden via environment variables when running `modal run`.
WARM_MIN_CONTAINERS = int(os.environ.get("MODAL_MIN_CONTAINERS", "1"))
WARM_BUFFER_CONTAINERS = int(os.environ.get("MODAL_BUFFER_CONTAINERS", "0"))
WARM_SCALEDOWN_WINDOW = int(os.environ.get("MODAL_SCALEDOWN_WINDOW", "300"))


def _make_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .env({"IMAGE_BUILD_VERSION": str(BUILD_VERSION)})
        .pip_install(
            "accelerate>=1.12.0",
            "datasets>=4.4.1",
            "huggingface-hub>=0.36.0",
            "matplotlib>=3.10.7",
            "numpy>=2.3.5",
            "pandas>=2.3.3",
            "pyyaml>=6.0.3",
            "scipy>=1.16.3",
            "seaborn>=0.13.2",
            "sentencepiece>=0.2.1",
            "torch>=2.9.1",
            "tqdm>=4.67.1",
            "transformers>=4.57.3",
            "modal>=1.2.2",
        )
        .add_local_dir(
            ".",
            remote_path="/workspace",
            # Keep noisy/generated artifacts out of the build context so Modal doesn't
            # abort when they change mid-build. Modal passes a PurePath; compare on str.
            ignore=lambda p: (
                (rel := p.as_posix())
                and (
                    rel.startswith(".git/")
                    or rel.startswith(".venv/")
                    or rel.startswith("results/")
                    or rel.startswith("updated-results/")
                    or rel.startswith("llama-results/")
                    or rel.startswith("data/")
                    or "__pycache__" in rel
                )
            ),
        )
    )


image = _make_image()

volume = modal.Volume.from_name("persona-vectors", create_if_missing=True)

app = modal.App("persona-vectors")
hf_secret = modal.Secret.from_name("huggingface-token")


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=12 * 60 * 60,
    volumes={"/volume": volume},
    secrets=[hf_secret],
    min_containers=WARM_MIN_CONTAINERS,
    buffer_containers=WARM_BUFFER_CONTAINERS,
    scaledown_window=WARM_SCALEDOWN_WINDOW,
)
def run_exp(exp: str = "exp2", config: str | None = None):
    """
    Run one of the experiments on a GPU.
    exp: exp0 | exp1 | exp2 | exp3 | exp2_agg | validation
    """
    workdir = Path("/workspace")
    data_dir = Path("/volume/data")
    results_dir = Path("/volume/results")
    llama_results_dir = Path("/volume/llama-results")
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    llama_results_dir.mkdir(parents=True, exist_ok=True)

    # Make volume data visible at the expected repo-relative path.
    repo_data_link = workdir / "data"
    if not repo_data_link.exists():
        repo_data_link.symlink_to(data_dir)

    # Route updated-results/* writes to the persistent results volume.
    repo_updated_results_link = workdir / "updated-results"
    if not repo_updated_results_link.exists():
        repo_updated_results_link.symlink_to(results_dir)

    # Route llama-results/* to the persistent volume (for validation experiments).
    repo_llama_results_link = workdir / "llama-results"
    if not repo_llama_results_link.exists():
        repo_llama_results_link.symlink_to(llama_results_dir)

    env = {
        "DATA_DIR": str(data_dir),
        "RESULTS_DIR": str(results_dir),
        "LLAMA_RESULTS_DIR": str(llama_results_dir),
    }

    cmd_map = {
        "exp0": ["python", "-m", "src.exp0.run_baseline"],
        "exp1": ["python", "-m", "src.exp1.run_caa_vector"],
        "exp2": ["python", "-m", "src.exp2.run_path_patching"],
        "exp2_agg": ["python", "-m", "src.exp2.aggregate_head_scores"],
        "exp3": ["python", "-m", "src.exp3.run_mediation_grid"],
        "validation": ["python", "-m", "src.exp3.run_validation"],
    }
    if exp not in cmd_map:
        raise ValueError(f"Unknown exp '{exp}', choose from {list(cmd_map)}")

    cmd = cmd_map[exp]
    if config:
        cmd = [*cmd, "--config", config]
    merged_env = {**os.environ, **env}
    _run(cmd, cwd=workdir, env=merged_env)

    # Sync results to persistent volume
    src_results = workdir / "results"
    if src_results.exists():
        # shutil.copytree with dirs_exist_ok is available in Py 3.12
        shutil.copytree(src_results, results_dir, dirs_exist_ok=True)


@app.local_entrypoint()
def main(exp: str = "exp2", config: str | None = None):
    """
    Local convenience wrapper. Examples:
        modal run modal_app.py --exp exp2
        modal run modal_app.py --exp exp3
        modal run modal_app.py --exp validation
    """
    run_exp.remote(exp=exp, config=config)
