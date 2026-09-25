from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from qwen_rebuttal import cli

REPO_ROOT = Path(__file__).resolve().parents[2]
SHELL_SCRIPTS = sorted(
    path
    for directory in ("scripts", "experiments", "rebuttal")
    for path in (REPO_ROOT / directory).glob("*.sh")
)


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_shell_script_syntax(script):
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True, text=True)


def _data_module():
    script = Path(__file__).resolve().parents[2] / "scripts/prepare_trivia_data.py"
    spec = importlib.util.spec_from_file_location("prepare_trivia_data", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_data_setup_does_not_replace_existing_file(tmp_path):
    module = _data_module()
    output = tmp_path / "dataset.jsonl"
    output.write_bytes(b"a different dataset\n")
    with pytest.raises(ValueError, match="SHA-256"):
        module.prepare(tmp_path / "missing.jsonl", output, download=True)
    assert output.read_bytes() == b"a different dataset\n"


def test_data_setup_checks_source_before_writing(tmp_path):
    module = _data_module()
    raw = tmp_path / "raw.jsonl"
    raw.write_bytes(b"not the pinned source\n")
    output = tmp_path / "dataset.jsonl"
    with pytest.raises(ValueError, match="SHA-256"):
        module.prepare(raw, output)
    assert not output.exists()


def test_data_setup_accepts_verified_existing_output(tmp_path, monkeypatch):
    module = _data_module()
    output = tmp_path / "dataset.jsonl"
    payload = b"already prepared\n"
    output.write_bytes(payload)
    monkeypatch.setattr(module, "OUTPUT_SHA256", hashlib.sha256(payload).hexdigest())
    module.prepare(tmp_path / "missing.jsonl", output)
    assert output.read_bytes() == payload


def test_head_to_head_runs_preflight_before_loading_model(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["qwen-rebuttal", "head-to-head"])
    monkeypatch.setenv("HF_TOKEN", "test-only")
    monkeypatch.setattr(cli, "prepare_run", lambda **kwargs: (object(), None, None, None))

    def stop_at_preflight(**kwargs):
        raise RuntimeError("preflight reached")

    monkeypatch.setattr(cli, "preflight", stop_at_preflight)
    with pytest.raises(RuntimeError, match="preflight reached"):
        cli.main()
