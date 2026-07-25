from __future__ import annotations

from dataclasses import dataclass

import pytest

from qwen_rebuttal.io import (
    append_jsonl_atomic,
    canonical_json,
    ensure_matching_manifest,
    sha256_text,
)


def _row(key: str, value: float) -> dict[str, object]:
    return {"row_key": key, "value": value}


def test_interrupted_and_uninterrupted_rows_are_identical(tmp_path):
    resumed = tmp_path / "resumed.jsonl"
    uninterrupted = tmp_path / "uninterrupted.jsonl"
    rows = [_row(f"k{index}", float(index)) for index in range(8)]
    append_jsonl_atomic(resumed, rows[:3])
    append_jsonl_atomic(resumed, rows[3:])
    append_jsonl_atomic(uninterrupted, rows)
    assert resumed.read_bytes() == uninterrupted.read_bytes()


def test_resume_rejects_changed_row(tmp_path):
    path = tmp_path / "rows.jsonl"
    append_jsonl_atomic(path, [_row("same", 1)])
    with pytest.raises(RuntimeError):
        append_jsonl_atomic(path, [_row("same", 2)])


def test_relevant_changes_invalidate_content_identity():
    base = {
        "prompt": "abc",
        "model_revision": "rev1",
        "layers": [5],
        "direction": "source",
    }
    base_hash = sha256_text(canonical_json(base))
    mutations = [
        {**base, "prompt": "abd"},
        {**base, "model_revision": "rev2"},
        {**base, "layers": [6]},
        {**base, "direction": "user"},
    ]
    assert all(sha256_text(canonical_json(value)) != base_hash for value in mutations)


@dataclass(frozen=True)
class _TupleManifest:
    created_utc: str
    run_id: str
    layers: tuple[int, ...]


def test_manifest_resume_normalizes_json_arrays_and_python_tuples(tmp_path):
    path = tmp_path / "run.json"
    initial = _TupleManifest(
        created_utc="2026-07-25T00:00:00+00:00",
        run_id="same",
        layers=(3, 4, 5),
    )
    resumed = _TupleManifest(
        created_utc="2026-07-25T01:00:00+00:00",
        run_id="same",
        layers=(3, 4, 5),
    )

    ensure_matching_manifest(path, initial)
    ensure_matching_manifest(path, resumed)
