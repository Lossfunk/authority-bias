"""Reason dataset helpers for Exp14 evidence-quality experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple


MissingReasonsPolicy = Literal["skip", "error"]


@dataclass(frozen=True)
class ReasonEntry:
    """One row from data/exp14_reasons.jsonl."""

    uid: str
    question: str
    correct_answer: str
    wrong_answer: str
    reasons_correct: Tuple[str, ...]
    reasons_wrong: Tuple[str, ...]
    data_correct: str
    data_wrong: str
    topic_phrases: Tuple[str, ...]
    metadata: Dict[str, object]

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "ReasonEntry":
        def _as_non_empty_str(value: object, field: str) -> str:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Invalid or missing string field '{field}'")
            return value.strip()

        def _as_str_list(value: object, field: str, min_len: int) -> Tuple[str, ...]:
            if not isinstance(value, list):
                raise ValueError(f"Invalid or missing list field '{field}'")
            out = [str(v).strip() for v in value if str(v).strip()]
            if len(out) < min_len:
                raise ValueError(
                    f"Field '{field}' requires at least {min_len} non-empty entries; got {len(out)}"
                )
            return tuple(out)

        metadata_raw = payload.get("metadata", {})
        metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

        return cls(
            uid=_as_non_empty_str(payload.get("uid"), "uid"),
            question=_as_non_empty_str(payload.get("question"), "question"),
            correct_answer=_as_non_empty_str(payload.get("correct_answer"), "correct_answer"),
            wrong_answer=_as_non_empty_str(payload.get("wrong_answer"), "wrong_answer"),
            reasons_correct=_as_str_list(payload.get("reasons_correct"), "reasons_correct", min_len=2),
            reasons_wrong=_as_str_list(payload.get("reasons_wrong"), "reasons_wrong", min_len=2),
            data_correct=_as_non_empty_str(payload.get("data_correct"), "data_correct"),
            data_wrong=_as_non_empty_str(payload.get("data_wrong"), "data_wrong"),
            topic_phrases=_as_str_list(payload.get("topic_phrases"), "topic_phrases", min_len=2),
            metadata={str(k): v for k, v in metadata.items()},
        )

    def to_prompt_payload(self) -> Dict[str, str]:
        """Flatten this record into fields consumed by prompt formatting."""
        return {
            "reason_correct_1": self.reasons_correct[0],
            "reason_correct_2": self.reasons_correct[1],
            "reason_wrong_1": self.reasons_wrong[0],
            "reason_wrong_2": self.reasons_wrong[1],
            "data_correct": self.data_correct,
            "data_wrong": self.data_wrong,
            "topic_phrase_1": self.topic_phrases[0],
            "topic_phrase_2": self.topic_phrases[1],
        }


def load_reasons_dataset(path: Path) -> Dict[str, ReasonEntry]:
    """Load and validate Exp14 reasons JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Reasons dataset not found: {path}")

    entries: Dict[str, ReasonEntry] = {}
    with path.open("r") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid row type at {path}:{line_no}; expected object")
            entry = ReasonEntry.from_json(payload)
            if entry.uid in entries:
                raise ValueError(f"Duplicate uid '{entry.uid}' in reasons dataset ({path}:{line_no})")
            entries[entry.uid] = entry
    if not entries:
        raise ValueError(f"No valid reason rows loaded from {path}")
    return entries


def match_reason_rows(
    *,
    uids: Iterable[str],
    reasons_by_uid: Dict[str, ReasonEntry],
    policy: MissingReasonsPolicy,
) -> Tuple[List[str], List[str]]:
    """Return (available_uids, missing_uids) in dataset order."""
    available: List[str] = []
    missing: List[str] = []
    for uid in uids:
        if uid in reasons_by_uid:
            available.append(uid)
        else:
            missing.append(uid)
    if missing and policy == "error":
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"Missing reason rows for {len(missing)} items. Example uids: {preview}"
        )
    return available, missing
