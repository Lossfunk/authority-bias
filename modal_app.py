"""
Modal GPU runner for persona-vectors experiments.

Usage (locally, with Modal CLI configured):
    modal run modal_app.py --exp exp2
Available exp values: exp0, exp1, exp2, exp3

Notes:
- Code is mounted read-only from the local repo snapshot.
- Results are synced to a persistent Modal volume named "persona-vectors".
- Place large artifacts (e.g., data/answer.jsonl) in that volume beforehand:
    modal volume put persona-vectors data/answer.jsonl /data/answer.jsonl
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import modal

# Bump BUILD_VERSION to force Modal to rebuild the image when dependencies or hooks change.
BUILD_VERSION: int = 10


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
            ignore=lambda p: ".git" in str(p) or "/results/" in str(p) or "/.venv" in str(p) or "__pycache__" in str(p),
        )
    )


image = _make_image()

volume = modal.Volume.from_name("persona-vectors", create_if_missing=True)

app = modal.App("persona-vectors")


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


@app.function(image=image, gpu="L4", timeout=12 * 60 * 60, volumes={"/volume": volume})
def run_exp(exp: str = "exp2"):
    """
    Run one of the experiments on a GPU.
    exp: exp0 | exp1 | exp2 | exp3
    """
    workdir = Path("/workspace")
    data_dir = Path("/volume/data")
    results_dir = Path("/volume/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "DATA_DIR": str(data_dir),
        "RESULTS_DIR": str(results_dir),
    }

    cmd_map = {
        "exp0": ["python", "-m", "src.exp0.run_baseline"],
        "exp1": ["python", "-m", "src.exp1.run_caa_vector"],
        "exp2": ["python", "-m", "src.exp2.run_path_patching"],
        "exp2_agg": ["python", "-m", "src.exp2.aggregate_head_scores"],
        "exp3": ["python", "-m", "src.exp3.run_mediation_grid"],
    }
    if exp not in cmd_map:
        raise ValueError(f"Unknown exp '{exp}', choose from {list(cmd_map)}")

    _run(cmd_map[exp], cwd=workdir)

    # Sync results to persistent volume
    src_results = workdir / "results"
    if src_results.exists():
        # shutil.copytree with dirs_exist_ok is available in Py 3.12
        shutil.copytree(src_results, results_dir, dirs_exist_ok=True)


@app.local_entrypoint()
def main(exp: str = "exp2"):
    """
    Local convenience wrapper. Examples:
        modal run modal_app.py --exp exp2
        modal run modal_app.py --exp exp3
    """
    run_exp.remote(exp=exp)
