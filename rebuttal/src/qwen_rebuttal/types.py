from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

CueFamily = Literal["source", "user", "neutral", "none"]
Endorsement = Literal["correct", "wrong", "none"]
PromptPosition = Literal["before_question", "after_options"]
Phase = Literal["prefill", "answer", "all"]
Operation = Literal["none", "remove", "add", "coefficient_swap", "span_project"]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    dtype: str
    attention_backend: str
    expected_layers: int
    expected_hidden_size: int
    thinking: bool
    text_only: bool


@dataclass(frozen=True)
class InputSpec:
    dataset: str
    dataset_sha256: str
    eligible_masks: str
    fit_uids: str
    fit_uids_sha256: str
    historical_uids: str
    historical_uids_sha256: str
    eligible_mask_key: str


@dataclass(frozen=True)
class SplitSpec:
    fresh_seed_prefix: str
    fresh_count: int


@dataclass(frozen=True)
class ExperimentSpec:
    primary_layer: int
    robustness_layers: tuple[int, ...]
    direction_batch_size: int
    scoring_batch_size: int
    generation_batch_size: int
    generation_max_new_tokens: int
    bootstrap_resamples: int
    permutation_resamples: int
    statistics_seed: int
    random_seeds: tuple[int, ...]
    shuffle_seeds: tuple[int, ...]

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted({self.primary_layer, *self.robustness_layers}))


@dataclass(frozen=True)
class CAASpec:
    repository: str
    commit: str
    dataset_path: str
    source_sha256: str
    fit_count: int
    tune_count: int
    split_seed: int
    fit_uids_sha256: str
    tune_uids_sha256: str
    multipliers: tuple[float, ...]


@dataclass(frozen=True)
class PipelineConfig:
    schema_version: int
    model: ModelSpec
    inputs: InputSpec
    splits: SplitSpec
    experiment: ExperimentSpec
    caa: CAASpec


@dataclass(frozen=True)
class PromptExample:
    uid: str
    template_id: str
    cue_family: CueFamily
    endorsement: Endorsement
    position: PromptPosition
    rendered_text: str
    cue_text: str | None
    cue_char_start: int | None
    cue_char_end: int | None
    cue_token_start: int | None = None
    cue_token_end: int | None = None
    candidate_a: str = "A"
    candidate_b: str = "B"
    correct_label: str = ""
    endorsed_label: str = ""


@dataclass(frozen=True)
class DirectionSpec:
    direction_id: str
    layer: int
    kind: Literal[
        "source",
        "user",
        "shared",
        "identity",
        "shuffled",
        "random",
        "caa",
    ]
    fit_split_hash: str
    prompt_hash: str
    seed: int | None = None


@dataclass(frozen=True)
class InterventionSpec:
    intervention_id: str
    operation: Operation
    layer: int | None
    direction_id: str | None
    coefficient: float = 0.0
    target_coefficient: float | None = None
    phase: Phase = "prefill"
    token_scope: Literal["cue_endpoint", "cue_span", "answer_tokens", "none"] = "none"


@dataclass(frozen=True)
class RawResultRow:
    row_key: str
    run_id: str
    split: str
    uid: str
    measurement: Literal["margin", "generation"]
    template_id: str
    cue_family: CueFamily
    endorsement: Endorsement
    position: PromptPosition
    layer: int | None
    intervention_id: str
    direction_id: str | None
    seed: int | None
    endorsed_label: str
    other_label: str
    logp_endorsed: float | None = None
    logp_other: float | None = None
    compliance_margin: float | None = None
    generated_text: str | None = None
    parsed_label: str | None = None
    parse_status: str | None = None


@dataclass(frozen=True)
class StatisticalSummary:
    analysis_id: str
    estimand: str
    n_pairs: int
    estimate: float
    ci_low: float
    ci_high: float
    permutation_p: float
    resamples: int
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    created_utc: str
    git_commit: str
    dependency_lock_sha256: str
    config_sha256: str
    model: ModelSpec
    input_hashes: dict[str, str]
    split_hashes: dict[str, str]
    template_hash: str
    layers: tuple[int, ...]
    intervention_catalog_hash: str
    seeds: dict[str, tuple[int, ...] | int]
    run_id: str = ""

    def content_without_identity(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("created_utc")
        value.pop("run_id")
        return value


@dataclass(frozen=True)
class RunPaths:
    root: Path
    run_id: str

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_id

    @property
    def manifests(self) -> Path:
        return self.run_dir / "manifests"

    @property
    def checkpoints(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def directions(self) -> Path:
        return self.run_dir / "directions"

    @property
    def raw(self) -> Path:
        return self.run_dir / "raw"

    @property
    def analysis(self) -> Path:
        return self.run_dir / "analysis"

    @property
    def figures(self) -> Path:
        return self.run_dir / "figures"

    @property
    def bundle(self) -> Path:
        return self.run_dir / "bundle"

    def create(self) -> None:
        for path in (
            self.manifests,
            self.checkpoints,
            self.directions,
            self.raw,
            self.analysis,
            self.figures,
            self.bundle,
        ):
            path.mkdir(parents=True, exist_ok=True)
