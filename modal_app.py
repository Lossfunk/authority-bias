"""
Modal GPU runner for persona-vectors experiments.

Usage (locally, with Modal CLI configured):
    modal run modal_app.py --exp exp2
    modal run modal_app.py --exp exp7 --models "Qwen/Qwen3-4B,Qwen/Qwen3-4B-Instruct-2507" --output-dir results/qwen/exp7
    modal run modal_app.py --exp smoke_qwen --model "Qwen/Qwen3-4B"
    modal run modal_app.py --exp exp11_logit --model "Qwen/Qwen3-4B-Instruct-2507" --results-dir results/qwen/exp10 --output-dir results/qwen/exp11/part_a
    modal run modal_app.py --exp exp12 --models "meta-llama/Llama-3.1-8B-Instruct" --output-dir results/exp12
Available exp values: exp0, exp1, exp2, exp3, exp4, exp5, exp2_agg, phase1, phase2, validation,
                      exp7, exp7_analyze, exp8, exp8_analyze, exp8_speakers, exp8_speakers_analyze,
                      exp9, exp9_analyze, exp10, exp10_analyze,
                      exp11_logit, exp11_inverted, exp11_confidence,
                      exp12, exp12_analyze,
                      smoke_qwen

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
BUILD_VERSION: int = 28

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
def run_exp(
    exp: str = "exp2",
    models: str = "",
    model: str = "",
    output_dir: str = "",
    results_dir: str = "",
    extended_tags: bool = False,
    turns: int = 3,
    batch_size: int = 64,
    probe_modes: str = "",
    probe_styles: str = "",
    no_correct_history: bool = False,
    max_length: int = 0,
):
    """
    Run one of the experiments on a GPU.

    exp: experiment key (see cmd_map below)
    models: comma-separated HF model ids for scripts that accept --models
    model: single HF model id for scripts that accept --model (e.g. exp11_* and smoke_qwen)
    output_dir: override output directory for results (e.g. "results/qwen/exp7")
    results_dir: override input results directory for analysis scripts
    extended_tags: when True and exp in {exp10, exp12}, include User/Someone online tags
    turns/batch_size/probe_modes/probe_styles/no_correct_history/max_length: exp12 runtime knobs
    """
    workdir = Path("/workspace")
    data_dir = Path("/volume/data")
    results_volume_dir = Path("/volume/results")
    llama_results_dir = Path("/volume/llama-results")
    data_dir.mkdir(parents=True, exist_ok=True)
    results_volume_dir.mkdir(parents=True, exist_ok=True)
    llama_results_dir.mkdir(parents=True, exist_ok=True)

    # Make volume data visible at the expected repo-relative path.
    repo_data_link = workdir / "data"
    if not repo_data_link.exists():
        repo_data_link.symlink_to(data_dir)

    # Route updated-results/* writes to the persistent results volume.
    repo_updated_results_link = workdir / "updated-results"
    if not repo_updated_results_link.exists():
        repo_updated_results_link.symlink_to(results_volume_dir)

    # Route results/* writes to the persistent results volume (for exp7, etc.).
    repo_results_link = workdir / "results"
    if not repo_results_link.exists():
        repo_results_link.symlink_to(results_volume_dir)

    # Route llama-results/* to the persistent volume (for validation experiments).
    repo_llama_results_link = workdir / "llama-results"
    if not repo_llama_results_link.exists():
        repo_llama_results_link.symlink_to(llama_results_dir)

    env = {
        "DATA_DIR": str(data_dir),
        "RESULTS_DIR": str(results_volume_dir),
        "LLAMA_RESULTS_DIR": str(llama_results_dir),
        "PYTHONPATH": str(workdir),
    }

    cmd_map = {
        "exp0": ["python", "-m", "src.exp0.run_baseline"],
        "exp1": ["python", "-m", "src.exp1.run_caa_vector"],
        "exp2": ["python", "-m", "src.exp2.run_path_patching"],
        "exp2_agg": ["python", "-m", "src.exp2.aggregate_head_scores"],
        "exp3": ["python", "-m", "src.exp3.run_mediation_grid"],
        "exp4": ["python", "-m", "src.exp4.run_distributed_test"],
        "exp5": ["python", "-m", "src.exp5.run_ccm"],
        "phase1": ["python", "-m", "src.exp6.run_phase1_sycophancy"],
        "phase2": ["python", "-m", "src.exp6.run_phase2_sycophancy"],
        "validation": ["python", "-m", "src.exp3.run_validation", "--output-dir", "llama-results/validation"],
        # Exp7: Clean lexical-fixed sycophancy measurement
        "exp7": ["python", "-m", "src.exp7.run_lexical_fixed"],
        "exp7_analyze": ["python", "-m", "src.exp7.analyze_results"],
        # Exp8: Label-balanced endorsement decomposition
        "exp8": ["python", "-m", "src.exp8.run_label_balanced"],
        "exp8_analyze": ["python", "-m", "src.exp8.analyze_results"],
        # Exp8 Speaker Tags: User-specific deference test
        "exp8_speakers": ["python", "-m", "src.exp8.run_speaker_tags"],
        "exp8_speakers_analyze": ["python", "-m", "src.exp8.analyze_speaker_tags"],
        # Exp9: Instruction override test (compliance vs belief-updating)
        "exp9": ["python", "-m", "src.exp9.run_instruction_override"],
        "exp9_analyze": ["python", "-m", "src.exp9.analyze_instruction_override"],
        # Exp10: Correct-endorsement test (truth-tracking vs gating)
        "exp10": ["python", "-m", "src.exp10.run_correct_endorse"],
        "exp10_analyze": ["python", "-m", "src.exp10.analyze_selectivity"],
        # Exp11: Logit-space analysis + inverted-prior test (truth-tracking vs prior-consistency)
        "exp11_logit": ["python", "-m", "src.exp11.analyze_logit_space"],
        "exp11_inverted": ["python", "-m", "src.exp11.analyze_inverted_prior"],
        "exp11_confidence": ["python", "-m", "src.exp11.analyze_confidence"],
        # Exp12: Persistence/Washout (multi-turn carryover test)
        "exp12": ["python", "-m", "src.exp12.run_persistence_washout"],
        "exp12_analyze": ["python", "-m", "src.exp12.analyze_persistence"],
        # Smoke test for Qwen model compatibility
        "smoke_qwen": ["python", "-m", "src.smoke_test_qwen"],
    }
    if exp not in cmd_map:
        raise ValueError(f"Unknown exp '{exp}', choose from {list(cmd_map)}")

    cmd = list(cmd_map[exp])

    # Experiments that accept --models (run scripts only).
    ACCEPTS_MULTI_MODELS = {"exp7", "exp8", "exp8_speakers", "exp9", "exp10", "exp12"}
    # Experiments that accept a single --model argument.
    ACCEPTS_SINGLE_MODEL = {"smoke_qwen", "exp11_logit", "exp11_inverted", "exp11_confidence"}
    # Experiments that accept --output-dir (run + analyze scripts).
    ACCEPTS_OUTPUT_DIR = ACCEPTS_MULTI_MODELS | {
        "exp7_analyze", "exp8_analyze", "exp8_speakers_analyze",
        "exp9_analyze", "exp10_analyze",
        "exp11_logit", "exp11_inverted", "exp11_confidence",
        "exp12", "exp12_analyze",
    }
    # Analysis scripts that accept --results-dir.
    ACCEPTS_RESULTS_DIR = {
        "exp7_analyze", "exp8_analyze", "exp8_speakers_analyze",
        "exp9_analyze", "exp10_analyze",
        "exp11_logit", "exp11_inverted", "exp11_confidence",
        "exp12_analyze",
    }

    if model and exp in ACCEPTS_SINGLE_MODEL:
        cmd.extend(["--model", model])
    elif models:
        model_list = [m.strip() for m in models.split(",") if m.strip()]
        if exp in ACCEPTS_SINGLE_MODEL and model_list:
            cmd.extend(["--model", model_list[0]])
        elif exp in ACCEPTS_MULTI_MODELS:
            cmd.extend(["--models"] + model_list)

    if results_dir and exp in ACCEPTS_RESULTS_DIR:
        cmd.extend(["--results-dir", results_dir])

    if output_dir and exp in ACCEPTS_OUTPUT_DIR:
        cmd.extend(["--output-dir", output_dir])

    if extended_tags and exp in {"exp10", "exp12"}:
        cmd.append("--extended-tags")

    if exp == "exp12":
        cmd.extend(["--turns", str(turns)])
        cmd.extend(["--batch-size", str(batch_size)])
        if probe_modes:
            cmd.extend(["--probe-modes", probe_modes])
        if probe_styles:
            cmd.extend(["--probe-styles", probe_styles])
        if no_correct_history:
            cmd.append("--no-correct-history")
        if max_length > 0:
            cmd.extend(["--max-length", str(max_length)])

    merged_env = {**os.environ, **env}
    _run(cmd, cwd=workdir, env=merged_env)

    # Sync results to persistent volume
    src_results = workdir / "results"
    if src_results.exists():
        # If results is already a symlink into the volume, no copy is needed.
        try:
            if src_results.is_symlink() or src_results.resolve() == results_volume_dir.resolve():
                return
        except FileNotFoundError:
            # If resolution fails, fall through to copytree.
            pass
        # shutil.copytree with dirs_exist_ok is available in Py 3.12
        shutil.copytree(src_results, results_volume_dir, dirs_exist_ok=True)


@app.local_entrypoint()
def main(
    exp: str = "exp2",
    models: str = "",
    model: str = "",
    output_dir: str = "",
    results_dir: str = "",
    extended_tags: bool = False,
    turns: int = 3,
    batch_size: int = 64,
    probe_modes: str = "",
    probe_styles: str = "",
    no_correct_history: bool = False,
    max_length: int = 0,
):
    """
    Local convenience wrapper. Examples:
        modal run modal_app.py --exp exp2
        modal run modal_app.py --exp exp7                  # Lexical-fixed sycophancy (Llama defaults)
        modal run modal_app.py --exp exp7_analyze          # Analyze exp7 results
        modal run modal_app.py --exp exp8                  # Label-balanced decomposition
        modal run modal_app.py --exp exp8_analyze          # Analyze exp8 results
        modal run modal_app.py --exp exp8_speakers         # Speaker tag test
        modal run modal_app.py --exp exp8_speakers_analyze # Analyze speaker tag results
        modal run modal_app.py --exp exp9                  # Instruction override test
        modal run modal_app.py --exp exp9_analyze          # Analyze instruction override
        modal run modal_app.py --exp exp10                 # Correct-endorsement test
        modal run modal_app.py --exp exp10 --extended-tags # Correct-endorsement with Expert/Note/User/Someone online
        modal run modal_app.py --exp exp10_analyze         # Analyze selectivity
        modal run modal_app.py --exp exp11_logit           # Exp11 Part A: Logit-space replication
        modal run modal_app.py --exp exp11_inverted        # Exp11 Part B: Inverted-prior test (CRITICAL)
        modal run modal_app.py --exp exp11_confidence      # Exp11 Part C: Confidence-binned (descriptive)
        modal run modal_app.py --exp exp12                 # Exp12: persistence/washout
        modal run modal_app.py --exp exp12_analyze         # Analyze Exp12 outputs
        modal run modal_app.py --exp smoke_qwen --model "Qwen/Qwen3-4B"
        modal run modal_app.py --exp exp7 --models "Qwen/Qwen3-4B,Qwen/Qwen3-4B-Instruct-2507" --output-dir results/qwen/exp7
        modal run modal_app.py --exp exp11_logit --model "Qwen/Qwen3-4B-Instruct-2507" --results-dir results/qwen/exp10 --output-dir results/qwen/exp11/part_a
    """
    run_exp.remote(
        exp=exp,
        models=models,
        model=model,
        output_dir=output_dir,
        results_dir=results_dir,
        extended_tags=extended_tags,
        turns=turns,
        batch_size=batch_size,
        probe_modes=probe_modes,
        probe_styles=probe_styles,
        no_correct_history=no_correct_history,
        max_length=max_length,
    )
