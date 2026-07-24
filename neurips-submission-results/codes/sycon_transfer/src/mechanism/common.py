from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CORE_TAGS: Tuple[str, ...] = ("Expert", "Note")
CORE_EVIDENCE_LEVELS: Tuple[str, ...] = ("bare", "reason1", "reason2", "reason_data")
CORE_INSTRUCTION_KEYS: Tuple[str, ...] = ("0", "1")
CORE_SOURCE_TYPES: Tuple[str, ...] = ("base", "source")


def normalize_tag(tag: str) -> str:
    return tag.lower().replace(" ", "_")


def model_tag(model_name: str) -> str:
    return model_name.replace("/", "__")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def stable_run_id(prefix: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:12]}"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(payload)
    return rows


def bootstrap_mean_ci(values: Sequence[float], *, n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample = arr[rng.integers(0, arr.size, arr.size)]
        boot[idx] = float(np.mean(sample))
    return {
        "mean": float(np.mean(arr)),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "n": int(arr.size),
    }


_ARTICLE_RE = re.compile(r"^(a|an|the)\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    lowered = text.strip().lower()
    lowered = _PUNCT_RE.sub(" ", lowered)
    lowered = _ARTICLE_RE.sub("", lowered)
    lowered = _WS_RE.sub(" ", lowered).strip()
    return lowered


def alias_match(prediction: str, aliases: Sequence[str]) -> bool:
    pred = normalize_answer(prediction)
    normalized_aliases = {normalize_answer(alias) for alias in aliases if alias}
    return pred in normalized_aliases


def answer_length_bucket(text: str) -> int:
    n_tokens = len(text.split())
    if n_tokens <= 2:
        return 0
    if n_tokens <= 4:
        return 1
    return 2


def margin_bin(value: float, *, width: float = 0.5, cap: float = 12.0) -> int:
    clipped = min(abs(value), cap)
    return int(math.floor(clipped / width))


def prompt_length_bucket(prompt_text: str, *, width: int = 32) -> int:
    n_chars = len(prompt_text)
    return n_chars // width


def confidence_slice(abs_margin: float, q75: float, q90: Optional[float]) -> str:
    if q90 is not None and abs_margin >= q90:
        return "top10"
    if abs_margin >= q75:
        return "high_conf_top25"
    return "other"


@dataclass(frozen=True)
class PanelRow:
    uid: str
    dataset: str
    topic_domain: str
    question: str
    correct_answer: str
    wrong_answer: str
    answer_format: str
    correct_label: str
    wrong_label: str
    prior_margin: float
    prior_state: str
    confidence_slice: str
    authority: str
    instruction: str
    evidence_direction: str
    evidence_strength: str
    condition_id: str
    prompt_text: str
    target_metric_metadata: Dict[str, Any] = field(default_factory=dict)
    matched_control_uids: Tuple[str, ...] = field(default_factory=tuple)

    def to_record(self) -> Dict[str, Any]:
        record = asdict(self)
        record["matched_control_uids"] = list(self.matched_control_uids)
        return record


@dataclass(frozen=True)
class ActivationManifestRow:
    run_id: str
    model: str
    example_id: str
    condition_id: str
    source_or_base: str
    shard_path: str
    shard_index: int
    n_layers: int
    components: Tuple[str, ...]
    positions: Tuple[str, ...]
    dtype: str
    hidden_size: int
    prompt_hash: str
    seed: int

    def to_record(self) -> Dict[str, Any]:
        record = asdict(self)
        record["components"] = list(self.components)
        record["positions"] = list(self.positions)
        return record


@dataclass(frozen=True)
class InterventionResult:
    run_id: str
    model: str
    example_id: str
    intervention_type: str
    base_condition_id: str
    source_condition_id: str
    component: str
    layer_idx: int
    position_label: str
    alpha: float
    effect_logit_delta: float
    effect_fc_delta: float
    base_logit_correct: float
    patched_logit_correct: float
    base_fc_correct: float
    patched_fc_correct: float
    control_group: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


def write_panel_parquet(path: Path, rows: Sequence[PanelRow]) -> None:
    ensure_dir(path.parent)
    frame = pd.DataFrame([row.to_record() for row in rows])
    frame.to_parquet(path, index=False)


def read_panel_parquet(path: Path) -> List[PanelRow]:
    frame = pd.read_parquet(path)
    rows: List[PanelRow] = []
    for record in frame.to_dict(orient="records"):
        matched_raw = record.get("matched_control_uids")
        if matched_raw is None:
            matched_control_uids: Tuple[str, ...] = tuple()
        elif isinstance(matched_raw, np.ndarray):
            matched_control_uids = tuple(str(item) for item in matched_raw.tolist())
        elif isinstance(matched_raw, (list, tuple)):
            matched_control_uids = tuple(str(item) for item in matched_raw)
        else:
            matched_control_uids = (str(matched_raw),)
        rows.append(
            PanelRow(
                uid=str(record["uid"]),
                dataset=str(record["dataset"]),
                topic_domain=str(record["topic_domain"]),
                question=str(record["question"]),
                correct_answer=str(record["correct_answer"]),
                wrong_answer=str(record["wrong_answer"]),
                answer_format=str(record["answer_format"]),
                correct_label=str(record["correct_label"]),
                wrong_label=str(record["wrong_label"]),
                prior_margin=float(record["prior_margin"]),
                prior_state=str(record["prior_state"]),
                confidence_slice=str(record["confidence_slice"]),
                authority=str(record["authority"]),
                instruction=str(record["instruction"]),
                evidence_direction=str(record["evidence_direction"]),
                evidence_strength=str(record["evidence_strength"]),
                condition_id=str(record["condition_id"]),
                prompt_text=str(record["prompt_text"]),
                target_metric_metadata=dict(record.get("target_metric_metadata") or {}),
                matched_control_uids=matched_control_uids,
            )
        )
    return rows
