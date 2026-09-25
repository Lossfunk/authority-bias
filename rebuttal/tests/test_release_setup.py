from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from configparser import ConfigParser
from pathlib import Path

import pytest

from qwen_rebuttal import cli

REPO_ROOT = Path(__file__).resolve().parents[2]
SHELL_SCRIPTS = sorted(
    path
    for directory in ("scripts", "rebuttal")
    for path in (REPO_ROOT / directory).glob("*.sh")
)


def test_readme_local_links_resolve():
    readme = (REPO_ROOT / "README.md").read_text()
    for target in re.findall(r"\]\(([^)]+)\)", readme):
        if ":" not in target and not target.startswith("#"):
            assert (REPO_ROOT / target.split("#")[0]).exists(), target


def test_numbered_experiment_directories_are_replaced():
    for parent in (REPO_ROOT / "src", REPO_ROOT / "results/authority"):
        assert not any(re.fullmatch(r"exp\d+", path.name) for path in parent.iterdir())


def test_first_party_import_targets_exist():
    # Resolve internal imports without importing GPU-only dependencies.
    for directory in ("src", "scripts", "rebuttal"):
        for path in (REPO_ROOT / directory).rglob("*.py"):
            if ".venv" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                modules = []
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    modules = [node.module]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                for module in modules:
                    if module.startswith("src."):
                        target = REPO_ROOT.joinpath(*module.split("."))
                        assert target.is_dir() or target.with_suffix(".py").is_file(), (
                            path, module
                        )


def test_head_to_head_parser_loads_from_renamed_source():
    path = REPO_ROOT / "rebuttal/score_head_to_head.py"
    spec = importlib.util.spec_from_file_location("score_head_to_head", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.load_paper_parser(REPO_ROOT / "src")
    assert parser is not None


@pytest.mark.parametrize("model", ["qwen", "gptoss", "olmo31"])
def test_frozen_configs_resolve_moved_inputs(model):
    config = json.loads((REPO_ROOT / f"rebuttal/configs/{model}_h200.json").read_text())
    inputs = config["inputs"]
    for key in ("eligible_masks", "fit_uids", "historical_uids", "assistant_axis"):
        path = REPO_ROOT / inputs[key]
        assert path.is_file(), path
        if expected := inputs.get(f"{key}_sha256"):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_both_evaluation_submodules_have_clone_urls():
    config = ConfigParser()
    config.read(REPO_ROOT / ".gitmodules")
    expected = {
        "external/SYCON-Bench": "https://github.com/JiseungHong/SYCON-Bench.git",
        "external/sycophancy-eval": "https://github.com/meg-tong/sycophancy-eval.git",
    }
    actual = {config[s]["path"]: config[s]["url"] for s in config.sections()}
    assert actual == expected


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
