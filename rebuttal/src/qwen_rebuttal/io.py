from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .types import (
    CAASpec,
    ExperimentSpec,
    InputSpec,
    ModelSpec,
    PipelineConfig,
    RunManifest,
    SplitSpec,
)


def canonical_json(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_lines(lines: Iterable[str]) -> str:
    return sha256_text("".join(f"{line}\n" for line in lines))


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_text(path: Path, payload: str) -> None:
    atomic_write_bytes(path, payload.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    if is_dataclass(value):
        value = asdict(value)
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl_atomic(path: Path, rows: Iterable[Any], key_field: str = "row_key") -> None:
    existing: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row[key_field]] = row
    for raw in rows:
        row = asdict(raw) if is_dataclass(raw) else dict(raw)
        key = row[key_field]
        previous = existing.get(key)
        if previous is not None and canonical_json(previous) != canonical_json(row):
            raise RuntimeError(f"Refusing to replace non-identical row {key} in {path}")
        existing[key] = row
    ordered = [canonical_json(existing[key]) for key in sorted(existing)]
    atomic_write_text(path, "\n".join(ordered) + ("\n" if ordered else ""))


def load_config(path: Path) -> PipelineConfig:
    raw = json.loads(path.read_text())
    return PipelineConfig(
        schema_version=int(raw["schema_version"]),
        model=ModelSpec(**raw["model"]),
        inputs=InputSpec(**raw["inputs"]),
        splits=SplitSpec(**raw["splits"]),
        experiment=ExperimentSpec(
            **{
                **raw["experiment"],
                "robustness_layers": tuple(raw["experiment"]["robustness_layers"]),
                "random_seeds": tuple(raw["experiment"]["random_seeds"]),
                "shuffle_seeds": tuple(raw["experiment"]["shuffle_seeds"]),
            }
        ),
        caa=CAASpec(
            **{
                **raw["caa"],
                "multipliers": tuple(raw["caa"]["multipliers"]),
            }
        ),
    )


def git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def assert_git_code_clean(repo_root: Path) -> None:
    tracked = subprocess.run(
        ["git", "diff", "--quiet", "--", "rebuttal", ".gitignore"],
        cwd=repo_root,
        check=False,
    )
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", "rebuttal", ".gitignore"],
        cwd=repo_root,
        check=False,
    )
    if tracked.returncode or staged.returncode:
        raise RuntimeError("Commit the rebuttal code before producing evidence")


def build_manifest(
    *,
    config: PipelineConfig,
    config_path: Path,
    repo_root: Path,
    dependency_lock: Path,
    input_hashes: dict[str, str],
    split_hashes: dict[str, str],
    template_hash: str,
    intervention_catalog_hash: str,
) -> RunManifest:
    provisional = RunManifest(
        schema_version=config.schema_version,
        created_utc=datetime.now(UTC).isoformat(),
        git_commit=git_commit(repo_root),
        dependency_lock_sha256=sha256_file(dependency_lock),
        config_sha256=sha256_file(config_path),
        model=config.model,
        input_hashes=input_hashes,
        split_hashes=split_hashes,
        template_hash=template_hash,
        layers=config.experiment.layers,
        intervention_catalog_hash=intervention_catalog_hash,
        seeds={
            "statistics": config.experiment.statistics_seed,
            "random": config.experiment.random_seeds,
            "shuffle": config.experiment.shuffle_seeds,
            "caa_split": config.caa.split_seed,
        },
    )
    identity = sha256_text(canonical_json(provisional.content_without_identity()))
    return RunManifest(**{**asdict(provisional), "model": config.model, "run_id": identity[:24]})


def ensure_matching_manifest(path: Path, manifest: RunManifest) -> None:
    if not path.exists():
        atomic_write_json(path, manifest)
        return
    current = json.loads(path.read_text())
    expected = json.loads(canonical_json(manifest))
    current_created = current.pop("created_utc", None)
    expected.pop("created_utc", None)
    if current != expected:
        raise RuntimeError(
            f"Manifest mismatch in existing run {manifest.run_id}; refusing unsafe resume"
        )
    if current_created is None:
        raise RuntimeError("Existing manifest lacks creation timestamp")


def checkpoint_done(path: Path, manifest_hash: str, payload: dict[str, Any]) -> None:
    atomic_write_json(path, {"manifest_hash": manifest_hash, "payload": payload})


def read_checkpoint(path: Path, manifest_hash: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("manifest_hash") != manifest_hash:
        raise RuntimeError(f"Checkpoint {path} belongs to a different manifest")
    return value["payload"]
