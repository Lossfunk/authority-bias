from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .io import atomic_write_text, hash_lines, sha256_file
from .types import PipelineConfig


@dataclass(frozen=True)
class MCExample:
    uid: str
    question: str
    option_a: str
    option_b: str
    correct_label: str
    wrong_label: str

    def answer_text(self, label: str) -> str:
        if label == "A":
            return self.option_a
        if label == "B":
            return self.option_b
        raise ValueError(f"Unsupported label: {label}")


@dataclass(frozen=True)
class FrozenSplits:
    fit: tuple[str, ...]
    historical: tuple[str, ...]
    fresh: tuple[str, ...]
    unused: tuple[str, ...]

    def hashes(self) -> dict[str, str]:
        return {
            "fit": hash_lines(self.fit),
            "historical": hash_lines(self.historical),
            "fresh": hash_lines(self.fresh),
            "unused": hash_lines(self.unused),
        }


def _read_uids(path: Path) -> tuple[str, ...]:
    values = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate UIDs in {path}")
    return values


def load_dataset(path: Path) -> dict[str, MCExample]:
    rows: dict[str, MCExample] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        example = MCExample(
            uid=raw["uid"],
            question=raw["question"],
            option_a=raw["option_a"],
            option_b=raw["option_b"],
            correct_label=raw["correct_label"],
            wrong_label=raw["wrong_label"],
        )
        if example.uid in rows:
            raise ValueError(f"Duplicate dataset UID {example.uid} at line {line_number}")
        if {example.correct_label, example.wrong_label} != {"A", "B"}:
            raise ValueError(f"Invalid labels for {example.uid}")
        rows[example.uid] = example
    return rows


def freeze_splits(config: PipelineConfig, repo_root: Path) -> FrozenSplits:
    inputs = config.inputs
    dataset_path = repo_root / inputs.dataset
    fit_path = repo_root / inputs.fit_uids
    historical_path = repo_root / inputs.historical_uids
    for path, expected in (
        (dataset_path, inputs.dataset_sha256),
        (fit_path, inputs.fit_uids_sha256),
        (historical_path, inputs.historical_uids_sha256),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")

    dataset = load_dataset(dataset_path)
    masks = json.loads((repo_root / inputs.eligible_masks).read_text())
    eligible = tuple(masks[inputs.eligible_mask_key])
    fit = _read_uids(fit_path)
    historical = _read_uids(historical_path)
    eligible_set = set(eligible)
    if not set(fit) <= eligible_set or not set(historical) <= eligible_set:
        raise RuntimeError("Recorded fit or historical UID is not eligible")
    if set(fit) & set(historical):
        raise RuntimeError("Fit and historical partitions overlap")
    if not eligible_set <= dataset.keys():
        raise RuntimeError("Eligible masks contain UIDs missing from the dataset")

    remaining = eligible_set - set(fit) - set(historical)
    ordered = sorted(
        remaining,
        key=lambda uid: (
            hashlib.sha256(f"{config.splits.fresh_seed_prefix}{uid}".encode()).hexdigest(),
            uid,
        ),
    )
    count = config.splits.fresh_count
    fresh = tuple(ordered[:count])
    unused = tuple(ordered[count:])
    if len(fresh) != count:
        raise RuntimeError(f"Requested {count} fresh items, found only {len(ordered)}")
    splits = FrozenSplits(fit=fit, historical=historical, fresh=fresh, unused=unused)
    assert_partition_contract(splits, eligible_set)
    return splits


def assert_partition_contract(splits: FrozenSplits, eligible: set[str]) -> None:
    partitions = [set(splits.fit), set(splits.historical), set(splits.fresh), set(splits.unused)]
    for left_index, left in enumerate(partitions):
        for right in partitions[left_index + 1 :]:
            if left & right:
                raise AssertionError("Data partitions overlap")
    if set().union(*partitions) != eligible:
        raise AssertionError("Data partitions do not cover the eligible set")


def write_splits(splits: FrozenSplits, directory: Path) -> None:
    for name in ("fit", "historical", "fresh", "unused"):
        values = getattr(splits, name)
        atomic_write_text(directory / f"{name}_uids.txt", "".join(f"{uid}\n" for uid in values))
