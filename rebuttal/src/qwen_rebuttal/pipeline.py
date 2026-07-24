from __future__ import annotations

import json
import platform
import re
import tarfile
from collections.abc import Iterable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .analysis import (
    caa_comparison_summary,
    generation_summary,
    identity_control_summaries,
    load_raw_rows,
    neutral_decision_table,
    plot_core_effects,
    robustness_summaries,
    separation_summaries,
    write_statistical_summaries,
)
from .caa import (
    CAASelection,
    ensure_caa_checkout,
    fit_caa_directions,
    freeze_caa_data,
    tune_caa,
)
from .data import FrozenSplits, freeze_splits, load_dataset, write_splits
from .directions import (
    LayerDirections,
    collect_endpoint_activations,
    fit_layer_directions,
    matched_random_controls,
    save_directions,
    shuffled_identity_controls,
)
from .hooks import ResidualHook
from .io import (
    append_jsonl_atomic,
    assert_git_code_clean,
    atomic_torch_save,
    atomic_write_json,
    atomic_write_text,
    build_manifest,
    canonical_json,
    ensure_matching_manifest,
    load_config,
    sha256_file,
    sha256_text,
)
from .model import (
    QwenAdapter,
    assert_model_access,
    dependency_manifest,
    hardware_manifest,
)
from .prompts import (
    TEMPLATE_CATALOG,
    apply_chat_and_locate_cue,
    build_prompt,
    paired_identity_difference_is_declared,
    template_catalog_hash,
)
from .scoring import (
    RuntimeIntervention,
    generate_answers,
    score_margins,
)
from .types import PipelineConfig, PromptExample, RawResultRow, RunManifest, RunPaths

INTERVENTION_CATALOG = {
    "baseline": {"operation": "none"},
    "source_remove": {
        "operation": "project_out",
        "direction": "source_minus_neutral",
        "scope": "cue_endpoint",
        "coefficient": 1.0,
    },
    "user_remove": {
        "operation": "project_out",
        "direction": "user_minus_neutral",
        "scope": "cue_endpoint",
        "coefficient": 1.0,
    },
    "source_to_user": {
        "operation": "add",
        "direction": "orthogonal_identity",
        "coefficient": "user_fit_coefficient-source_fit_coefficient",
        "scope": "cue_endpoint",
    },
    "user_to_source": {
        "operation": "add",
        "direction": "orthogonal_identity",
        "coefficient": "source_fit_coefficient-user_fit_coefficient",
        "scope": "cue_endpoint",
    },
    "same_to_same": {
        "operation": "add",
        "direction": "orthogonal_identity",
        "coefficient": 0.0,
        "scope": "cue_endpoint",
    },
    "shuffled_identity": {"count": 5, "scope": "cue_endpoint"},
    "matched_norm_random": {"count": 5, "scope": "cue_endpoint"},
    "legacy_span_projection": {
        "operation": "project_out",
        "direction": "source_minus_neutral",
        "scope": "cue_span",
    },
    "caa": {
        "operation": "add",
        "direction": "native_final_token_matching_minus_not_matching",
        "scope": "answer_tokens",
        "selection": "CAA tune split only",
    },
}


def _resolve(repo_root: Path, value: str) -> Path:
    return (repo_root / value).resolve()


def _input_hashes(config: PipelineConfig, repo_root: Path) -> dict[str, str]:
    paths = {
        "dataset": _resolve(repo_root, config.inputs.dataset),
        "eligible_masks": _resolve(repo_root, config.inputs.eligible_masks),
        "fit_uids": _resolve(repo_root, config.inputs.fit_uids),
        "historical_uids": _resolve(repo_root, config.inputs.historical_uids),
    }
    values = {name: sha256_file(path) for name, path in paths.items()}
    values["caa_dataset_at_pinned_commit"] = config.caa.source_sha256
    return values


def prepare_run(
    *,
    repo_root: Path,
    config_path: Path,
    output_root: Path,
    require_clean: bool = True,
) -> tuple[PipelineConfig, FrozenSplits, RunManifest, RunPaths]:
    if output_root.resolve().is_relative_to(repo_root.resolve()):
        raise RuntimeError("The rebuttal output root must be outside the Git repository")
    config = load_config(config_path)
    if require_clean:
        assert_git_code_clean(repo_root)
    splits = freeze_splits(config, repo_root)
    lock = repo_root / "rebuttal" / "uv.lock"
    if not lock.exists():
        raise RuntimeError("rebuttal/uv.lock is required for an immutable evidence run")
    manifest = build_manifest(
        config=config,
        config_path=config_path,
        repo_root=repo_root,
        dependency_lock=lock,
        input_hashes=_input_hashes(config, repo_root),
        split_hashes={
            **splits.hashes(),
            "caa_fit": config.caa.fit_uids_sha256,
            "caa_tune": config.caa.tune_uids_sha256,
        },
        template_hash=template_catalog_hash(),
        intervention_catalog_hash=sha256_text(canonical_json(INTERVENTION_CATALOG)),
    )
    paths = RunPaths(output_root, manifest.run_id)
    paths.create()
    ensure_matching_manifest(paths.manifests / "run.json", manifest)
    atomic_write_json(paths.manifests / "config.json", config)
    atomic_write_json(paths.manifests / "templates.json", TEMPLATE_CATALOG)
    atomic_write_json(paths.manifests / "interventions.json", INTERVENTION_CATALOG)
    write_splits(splits, paths.manifests)
    atomic_write_text(output_root / "LATEST", f"{manifest.run_id}\n")
    return config, splits, manifest, paths


def preflight(
    *,
    config: PipelineConfig,
    output_root: Path,
    token: str | None,
    paths: RunPaths,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    software = dependency_manifest()
    hardware = hardware_manifest(output_root)
    access = assert_model_access(config.model, token)
    atomic_write_json(
        paths.manifests / "environment.json",
        {
            "software": software,
            "hardware": hardware,
            "model_access": access,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    )


def _tokenize_prompts(
    adapter: QwenAdapter, prompts: Iterable[PromptExample]
) -> list[PromptExample]:
    return [apply_chat_and_locate_cue(adapter.tokenizer, prompt) for prompt in prompts]


def _fit_prompts(
    adapter: QwenAdapter,
    dataset: dict[str, Any],
    uids: tuple[str, ...],
) -> dict[str, list[PromptExample]]:
    groups: dict[str, list[PromptExample]] = {"source": [], "user": [], "neutral": []}
    for uid in uids:
        example = dataset[uid]
        for endorsement in ("correct", "wrong"):
            source = build_prompt(
                example,
                template_id="primary",
                cue_family="source",
                endorsement=endorsement,
                position="after_options",
            )
            user = build_prompt(
                example,
                template_id="primary",
                cue_family="user",
                endorsement=endorsement,
                position="after_options",
            )
            if not paired_identity_difference_is_declared(source, user):
                raise RuntimeError(f"Matched prompt contract failed for {uid}/{endorsement}")
            groups["source"].append(source)
            groups["user"].append(user)
            groups["neutral"].append(
                build_prompt(
                    example,
                    template_id="primary",
                    cue_family="neutral",
                    endorsement=endorsement,
                    position="after_options",
                )
            )
    return {name: _tokenize_prompts(adapter, values) for name, values in groups.items()}


def _write_prompt_manifest(path: Path, groups: dict[str, list[PromptExample]]) -> None:
    rows = []
    for group, prompts in groups.items():
        for prompt in prompts:
            row = asdict(prompt)
            row["manifest_group"] = group
            row["row_key"] = sha256_text(canonical_json(row))
            rows.append(row)
    append_jsonl_atomic(path, rows)


def fit_source_user_directions(
    *,
    adapter: QwenAdapter,
    config: PipelineConfig,
    splits: FrozenSplits,
    repo_root: Path,
    paths: RunPaths,
) -> tuple[
    dict[int, LayerDirections],
    dict[int, dict[int, torch.Tensor]],
    dict[int, dict[int, torch.Tensor]],
]:
    direction_path = paths.directions / "source_user.pt"
    if direction_path.exists():
        checksum_path = paths.directions / "source_user.checksum.json"
        if not checksum_path.exists():
            raise RuntimeError("Direction tensor exists without its checksum")
        expected = json.loads(checksum_path.read_text())["sha256"]
        actual = sha256_file(direction_path)
        if actual != expected:
            raise RuntimeError(f"Direction tensor checksum mismatch: {actual} != {expected}")
        payload = torch.load(direction_path, map_location="cpu", weights_only=True)
        geometry = json.loads(direction_path.with_suffix(".geometry.json").read_text())
        fitted = {
            int(layer): LayerDirections(
                layer=int(layer),
                source=value["source"],
                user=value["user"],
                shared=value["shared"],
                identity=value["identity"],
                source_identity_coefficient=float(value["source_identity_coefficient"]),
                user_identity_coefficient=float(value["user_identity_coefficient"]),
                geometry=geometry[str(layer)],
            )
            for layer, value in payload["layers"].items()
        }
        return fitted, payload["shuffled"], payload["random"]
    dataset = load_dataset(_resolve(repo_root, config.inputs.dataset))
    prompt_groups = _fit_prompts(adapter, dataset, splits.fit)
    _write_prompt_manifest(paths.manifests / "direction_fit_prompts.jsonl", prompt_groups)
    activations = {
        group: collect_endpoint_activations(
            adapter,
            prompts,
            config.experiment.layers,
            config.experiment.direction_batch_size,
        )
        for group, prompts in prompt_groups.items()
    }
    fitted = fit_layer_directions(activations, config.experiment.layers)
    shuffled: dict[int, dict[int, torch.Tensor]] = {}
    random: dict[int, dict[int, torch.Tensor]] = {}
    for layer, value in fitted.items():
        shuffled[layer] = shuffled_identity_controls(
            activations["source"][layer],
            activations["user"][layer],
            value.shared,
            config.experiment.shuffle_seeds,
        )
        random[layer] = matched_random_controls(value.identity, config.experiment.random_seeds)
    tensor_hash = save_directions(direction_path, fitted, shuffled, random)
    atomic_write_json(paths.directions / "source_user.checksum.json", {"sha256": tensor_hash})
    return fitted, shuffled, random


def _evaluation_prompts(
    adapter: QwenAdapter,
    dataset: dict[str, Any],
    uids: tuple[str, ...],
    *,
    template_id: str,
    position: str,
    endorsements: tuple[str, ...] = ("correct", "wrong"),
    cue_families: tuple[str, ...] = ("source", "user"),
) -> list[PromptExample]:
    prompts = [
        build_prompt(
            dataset[uid],
            template_id=template_id,  # type: ignore[arg-type]
            cue_family=cue,  # type: ignore[arg-type]
            endorsement=endorsement,  # type: ignore[arg-type]
            position=position,  # type: ignore[arg-type]
        )
        for uid in uids
        for endorsement in endorsements
        for cue in cue_families
    ]
    if template_id == "primary" and set(cue_families) == {"source", "user"}:
        by_pair = {
            (prompt.uid, prompt.endorsement, prompt.cue_family): prompt
            for prompt in prompts
        }
        for uid in uids:
            for endorsement in endorsements:
                if not paired_identity_difference_is_declared(
                    by_pair[(uid, endorsement, "source")],
                    by_pair[(uid, endorsement, "user")],
                ):
                    raise RuntimeError(
                        f"Matched prompt contract failed for {uid}/{endorsement}/{position}"
                    )
    return _tokenize_prompts(adapter, prompts)


def _runtime_catalog(
    fitted: LayerDirections,
    shuffled: dict[int, torch.Tensor],
    random: dict[int, torch.Tensor],
) -> list[tuple[RuntimeIntervention, tuple[str, ...]]]:
    layer = fitted.layer
    source_to_user = fitted.user_identity_coefficient - fitted.source_identity_coefficient
    user_to_source = -source_to_user
    values: list[tuple[RuntimeIntervention, tuple[str, ...]]] = [
        (
            RuntimeIntervention(
                f"baseline_l{layer}", layer, "none", None, 0, "none", "prefill"
            ),
            ("source", "user"),
        ),
        (
            RuntimeIntervention(
                f"source_remove_l{layer}",
                layer,
                "project_out",
                fitted.source,
                1,
                "cue_endpoint",
                "prefill",
                f"source_l{layer}",
            ),
            ("source", "user"),
        ),
        (
            RuntimeIntervention(
                f"user_remove_l{layer}",
                layer,
                "project_out",
                fitted.user,
                1,
                "cue_endpoint",
                "prefill",
                f"user_l{layer}",
            ),
            ("source", "user"),
        ),
        (
            RuntimeIntervention(
                f"source_to_user_l{layer}",
                layer,
                "add",
                fitted.identity,
                source_to_user,
                "cue_endpoint",
                "prefill",
                f"identity_l{layer}",
            ),
            ("source",),
        ),
        (
            RuntimeIntervention(
                f"user_to_source_l{layer}",
                layer,
                "add",
                fitted.identity,
                user_to_source,
                "cue_endpoint",
                "prefill",
                f"identity_l{layer}",
            ),
            ("user",),
        ),
        (
            RuntimeIntervention(
                f"source_to_source_noop_l{layer}",
                layer,
                "add",
                fitted.identity,
                0,
                "cue_endpoint",
                "prefill",
                f"identity_l{layer}",
            ),
            ("source",),
        ),
        (
            RuntimeIntervention(
                f"user_to_user_noop_l{layer}",
                layer,
                "add",
                fitted.identity,
                0,
                "cue_endpoint",
                "prefill",
                f"identity_l{layer}",
            ),
            ("user",),
        ),
        (
            RuntimeIntervention(
                f"legacy_span_source_projection_l{layer}",
                layer,
                "project_out",
                fitted.source,
                1,
                "cue_span",
                "prefill",
                f"source_l{layer}",
            ),
            ("source", "user"),
        ),
    ]
    for seed, vector in shuffled.items():
        for cue, coefficient, label in (
            ("source", source_to_user, "source_to_user"),
            ("user", user_to_source, "user_to_source"),
        ):
            values.append(
                (
                    RuntimeIntervention(
                        f"shuffled_{label}_s{seed}_l{layer}",
                        layer,
                        "add",
                        vector,
                        coefficient,
                        "cue_endpoint",
                        "prefill",
                        f"shuffled_s{seed}_l{layer}",
                        seed,
                    ),
                    (cue,),
                )
            )
    for seed, vector in random.items():
        for cue, coefficient, label in (
            ("source", source_to_user, "source_to_user"),
            ("user", user_to_source, "user_to_source"),
        ):
            values.append(
                (
                    RuntimeIntervention(
                        f"random_{label}_s{seed}_l{layer}",
                        layer,
                        "add",
                        vector,
                        coefficient,
                        "cue_endpoint",
                        "prefill",
                        f"random_s{seed}_l{layer}",
                        seed,
                    ),
                    (cue,),
                )
            )
    return values


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _condition_file(
    paths: RunPaths,
    *,
    measurement: str,
    split: str,
    template: str,
    position: str,
    intervention_id: str,
    endorsement: str,
) -> Path:
    filename = "__".join(
        _safe_name(value)
        for value in (measurement, split, template, position, endorsement, intervention_id)
    )
    return paths.raw / f"{filename}.jsonl"


def _validate_existing_rows(path: Path, run_id: str, expected: int) -> bool:
    if not path.exists():
        return False
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected or any(row["run_id"] != run_id for row in rows):
        raise RuntimeError(f"Incomplete or foreign result file blocks resume: {path}")
    if len({row["row_key"] for row in rows}) != expected:
        raise RuntimeError(f"Duplicate row keys in {path}")
    return True


def _score_condition(
    *,
    adapter: QwenAdapter,
    prompts: list[PromptExample],
    intervention: RuntimeIntervention,
    run_id: str,
    split: str,
    batch_size: int,
    path: Path,
) -> None:
    if _validate_existing_rows(path, run_id, len(prompts)):
        return
    rows = score_margins(
        adapter=adapter,
        prompts=prompts,
        intervention=intervention,
        run_id=run_id,
        split=split,
        batch_size=batch_size,
    )
    append_jsonl_atomic(path, rows)


def run_margin_experiments(
    *,
    adapter: QwenAdapter,
    config: PipelineConfig,
    splits: FrozenSplits,
    manifest: RunManifest,
    paths: RunPaths,
    repo_root: Path,
    fitted: dict[int, LayerDirections],
    shuffled: dict[int, dict[int, torch.Tensor]],
    random: dict[int, dict[int, torch.Tensor]],
) -> None:
    dataset = load_dataset(_resolve(repo_root, config.inputs.dataset))
    for split_name, uids in (("historical", splits.historical), ("fresh", splits.fresh)):
        core = _evaluation_prompts(
            adapter,
            dataset,
            uids,
            template_id="primary",
            position="after_options",
        )
        _write_prompt_manifest(
            paths.manifests / "evaluation_prompts.jsonl",
            {f"{split_name}:primary:after_options": core},
        )
        for layer in config.experiment.layers:
            for intervention, cues in _runtime_catalog(
                fitted[layer], shuffled[layer], random[layer]
            ):
                selected = [prompt for prompt in core if prompt.cue_family in cues]
                for endorsement in ("correct", "wrong"):
                    subset = [prompt for prompt in selected if prompt.endorsement == endorsement]
                    path = _condition_file(
                        paths,
                        measurement="margin",
                        split=split_name,
                        template="primary",
                        position="after_options",
                        intervention_id=intervention.intervention_id,
                        endorsement=endorsement,
                    )
                    _score_condition(
                        adapter=adapter,
                        prompts=subset,
                        intervention=intervention,
                        run_id=manifest.run_id,
                        split=split_name,
                        batch_size=config.experiment.scoring_batch_size,
                        path=path,
                    )
        primary_layer = config.experiment.primary_layer
        robustness_conditions = _runtime_catalog(
            fitted[primary_layer], shuffled[primary_layer], random[primary_layer]
        )[:7]
        for template in ("primary", "submitted", "structured_paraphrase"):
            for position in ("before_question", "after_options"):
                if template == "primary" and position == "after_options":
                    continue
                prompts = _evaluation_prompts(
                    adapter, dataset, uids, template_id=template, position=position
                )
                _write_prompt_manifest(
                    paths.manifests / "evaluation_prompts.jsonl",
                    {f"{split_name}:{template}:{position}": prompts},
                )
                for intervention, cues in robustness_conditions:
                    selected = [prompt for prompt in prompts if prompt.cue_family in cues]
                    for endorsement in ("correct", "wrong"):
                        subset = [
                            prompt for prompt in selected if prompt.endorsement == endorsement
                        ]
                        path = _condition_file(
                            paths,
                            measurement="margin",
                            split=split_name,
                            template=template,
                            position=position,
                            intervention_id=intervention.intervention_id,
                            endorsement=endorsement,
                        )
                        _score_condition(
                            adapter=adapter,
                            prompts=subset,
                            intervention=intervention,
                            run_id=manifest.run_id,
                            split=split_name,
                            batch_size=config.experiment.scoring_batch_size,
                            path=path,
                        )
        no_cue = _evaluation_prompts(
            adapter,
            dataset,
            uids,
            template_id="no_cue",
            position="after_options",
            endorsements=("none",),
            cue_families=("none",),
        )
        _write_prompt_manifest(
            paths.manifests / "evaluation_prompts.jsonl",
            {f"{split_name}:no_cue:after_options": no_cue},
        )
        baseline = RuntimeIntervention(
            f"baseline_l{config.experiment.primary_layer}",
            config.experiment.primary_layer,
            "none",
            None,
            0,
            "none",
            "prefill",
        )
        path = _condition_file(
            paths,
            measurement="margin",
            split=split_name,
            template="no_cue",
            position="after_options",
            intervention_id=baseline.intervention_id,
            endorsement="none",
        )
        _score_condition(
            adapter=adapter,
            prompts=no_cue,
            intervention=baseline,
            run_id=manifest.run_id,
            split=split_name,
            batch_size=config.experiment.scoring_batch_size,
            path=path,
        )


def fit_and_tune_caa(
    *,
    adapter: QwenAdapter,
    config: PipelineConfig,
    manifest: RunManifest,
    paths: RunPaths,
) -> tuple[CAASelection, torch.Tensor]:
    selection_path = paths.directions / "caa_selection.json"
    tensor_path = paths.directions / "caa.pt"
    checksum_path = paths.directions / "caa.checksum.json"
    if selection_path.exists() and tensor_path.exists():
        expected = json.loads(checksum_path.read_text())["sha256"]
        actual = sha256_file(tensor_path)
        if actual != expected:
            raise RuntimeError(f"CAA tensor checksum mismatch: {actual} != {expected}")
        selection = CAASelection(**json.loads(selection_path.read_text()))
        directions = torch.load(tensor_path, map_location="cpu", weights_only=True)
        return selection, directions[selection.layer]
    checkout = ensure_caa_checkout(config.caa, paths.run_dir / "sources" / "CAA")
    frozen = freeze_caa_data(config.caa, checkout)
    atomic_write_json(
        paths.manifests / "caa_data.json",
        {
            "source_sha256": frozen.source_sha256,
            "fit_uids": [item.uid for item in frozen.fit],
            "tune_uids": [item.uid for item in frozen.tune],
            "repository": config.caa.repository,
            "commit": config.caa.commit,
        },
    )
    if tensor_path.exists():
        if not checksum_path.exists():
            raise RuntimeError("CAA tensor exists without its checksum")
        expected = json.loads(checksum_path.read_text())["sha256"]
        actual = sha256_file(tensor_path)
        if actual != expected:
            raise RuntimeError(f"CAA tensor checksum mismatch: {actual} != {expected}")
        directions = torch.load(tensor_path, map_location="cpu", weights_only=True)
    else:
        directions = fit_caa_directions(
            adapter, frozen, batch_size=config.experiment.direction_batch_size
        )
        atomic_torch_save(tensor_path, directions)
        atomic_write_json(checksum_path, {"sha256": sha256_file(tensor_path)})
    selection, candidates = tune_caa(
        adapter=adapter,
        frozen=frozen,
        directions=directions,
        multipliers=config.caa.multipliers,
        batch_size=config.experiment.scoring_batch_size,
        run_id=manifest.run_id,
    )
    atomic_write_json(selection_path, selection)
    atomic_write_json(paths.analysis / "caa_tuning_grid.json", candidates)
    return selection, directions[selection.layer]


def _caa_runtime(selection: CAASelection, direction: torch.Tensor) -> RuntimeIntervention:
    return RuntimeIntervention(
        intervention_id=f"caa_frozen_l{selection.layer}_m{selection.multiplier:g}",
        layer=selection.layer,
        mode="add",
        vector=direction,
        coefficient=selection.multiplier,
        token_scope="answer_tokens",
        phase="all",
        direction_id=f"caa_l{selection.layer}",
    )


def run_caa_fresh_margins(
    *,
    adapter: QwenAdapter,
    config: PipelineConfig,
    splits: FrozenSplits,
    manifest: RunManifest,
    paths: RunPaths,
    repo_root: Path,
    selection: CAASelection,
    direction: torch.Tensor,
) -> None:
    dataset = load_dataset(_resolve(repo_root, config.inputs.dataset))
    intervention = _caa_runtime(selection, direction)
    for template in ("primary", "submitted"):
        prompts = _evaluation_prompts(
            adapter,
            dataset,
            splits.fresh,
            template_id=template,
            position="after_options",
        )
        for endorsement in ("correct", "wrong"):
            subset = [prompt for prompt in prompts if prompt.endorsement == endorsement]
            path = _condition_file(
                paths,
                measurement="margin",
                split="fresh",
                template=template,
                position="after_options",
                intervention_id=intervention.intervention_id,
                endorsement=endorsement,
            )
            _score_condition(
                adapter=adapter,
                prompts=subset,
                intervention=intervention,
                run_id=manifest.run_id,
                split="fresh",
                batch_size=config.experiment.scoring_batch_size,
                path=path,
            )


def run_generation_experiments(
    *,
    adapter: QwenAdapter,
    config: PipelineConfig,
    splits: FrozenSplits,
    manifest: RunManifest,
    paths: RunPaths,
    repo_root: Path,
    fitted: LayerDirections,
    caa_selection: CAASelection,
    caa_direction: torch.Tensor,
) -> None:
    dataset = load_dataset(_resolve(repo_root, config.inputs.dataset))
    layer = fitted.layer
    catalog = {item.intervention_id: (item, cues) for item, cues in _runtime_catalog(
        fitted, {}, {}
    )}
    chosen: list[tuple[RuntimeIntervention, tuple[str, ...]]] = [
        catalog[f"baseline_l{layer}"],
        catalog[f"source_remove_l{layer}"],
        catalog[f"user_remove_l{layer}"],
        catalog[f"source_to_user_l{layer}"],
        catalog[f"user_to_source_l{layer}"],
        (_caa_runtime(caa_selection, caa_direction), ("source", "user")),
    ]
    for template in ("primary", "submitted"):
        prompts = _evaluation_prompts(
            adapter,
            dataset,
            splits.fresh,
            template_id=template,
            position="after_options",
        )
        for intervention, cues in chosen:
            for endorsement in ("correct", "wrong"):
                subset = [
                    prompt
                    for prompt in prompts
                    if prompt.cue_family in cues and prompt.endorsement == endorsement
                ]
                path = _condition_file(
                    paths,
                    measurement="generation",
                    split="fresh",
                    template=template,
                    position="after_options",
                    intervention_id=intervention.intervention_id,
                    endorsement=endorsement,
                )
                if _validate_existing_rows(path, manifest.run_id, len(subset)):
                    continue
                rows = generate_answers(
                    adapter=adapter,
                    prompts=subset,
                    intervention=intervention,
                    run_id=manifest.run_id,
                    split="fresh",
                    max_new_tokens=config.experiment.generation_max_new_tokens,
                )
                append_jsonl_atomic(path, rows)


def h100_smoke(
    *,
    adapter: QwenAdapter,
    config: PipelineConfig,
    splits: FrozenSplits,
    repo_root: Path,
    manifest: RunManifest,
    paths: RunPaths,
    token: str | None,
) -> None:
    dataset = load_dataset(_resolve(repo_root, config.inputs.dataset))
    prompts = _evaluation_prompts(
        adapter,
        dataset,
        splits.fit[:8],
        template_id="primary",
        position="after_options",
        endorsements=("wrong",),
        cue_families=("source",),
    )
    sample = prompts[0]
    encoded = adapter.tokenizer(
        sample.rendered_text, add_special_tokens=False, return_tensors="pt"
    )
    ids = encoded["input_ids"].to(adapter.device)
    mask = encoded["attention_mask"].to(adapter.device)
    with torch.inference_mode():
        reference_logits = adapter.model(
            input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1
        ).logits
    zero = torch.zeros(config.model.expected_hidden_size)
    no_op = ResidualHook(
        module=adapter.layers[config.experiment.primary_layer],
        layer=config.experiment.primary_layer,
        mode="add",
        vector=zero,
        coefficient=0,
        phase="prefill",
        token_positions=[sample.cue_token_end],
    )
    with torch.inference_mode(), no_op:
        no_op_logits = adapter.model(
            input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1
        ).logits
    if not torch.equal(reference_logits, no_op_logits):
        raise RuntimeError("No-op hook does not exactly reproduce no-hook logits")
    optimized_rows = score_margins(
        adapter=adapter,
        prompts=prompts,
        intervention=RuntimeIntervention(
            "smoke_optimized", None, "none", None, 0, "none", "prefill"
        ),
        run_id=manifest.run_id,
        split="smoke",
        batch_size=config.experiment.scoring_batch_size,
    )
    optimized = {row.uid: float(row.compliance_margin) for row in optimized_rows}
    del reference_logits, no_op_logits
    adapter.layers.clear()
    del adapter.model
    torch.cuda.empty_cache()
    reference_spec = replace(config.model, attention_backend="eager")
    reference_adapter = QwenAdapter.load(reference_spec, token)
    reference_prompts = _tokenize_prompts(
        reference_adapter,
        [
            build_prompt(
                dataset[uid],
                template_id="primary",
                cue_family="source",
                endorsement="wrong",
                position="after_options",
            )
            for uid in splits.fit[:8]
        ],
    )
    reference_rows = score_margins(
        adapter=reference_adapter,
        prompts=reference_prompts,
        intervention=RuntimeIntervention(
            "smoke_reference", None, "none", None, 0, "none", "prefill"
        ),
        run_id=manifest.run_id,
        split="smoke",
        batch_size=config.experiment.scoring_batch_size,
    )
    comparisons = []
    for row in reference_rows:
        ref = float(row.compliance_margin)
        opt = optimized[row.uid]
        comparisons.append(
            {
                "uid": row.uid,
                "optimized_margin": opt,
                "reference_margin": ref,
                "absolute_difference": abs(opt - ref),
                "ranking_agrees": (opt >= 0) == (ref >= 0),
            }
        )
    if any(not item["ranking_agrees"] for item in comparisons):
        raise RuntimeError("Optimized and reference attention disagree on answer ranking")
    if any(float(item["absolute_difference"]) > 0.05 for item in comparisons):
        raise RuntimeError("Attention backend compliance-margin difference exceeds 0.05")
    atomic_write_json(paths.analysis / "attention_parity_smoke.json", comparisons)
    reference_adapter.layers.clear()
    del reference_adapter.model
    torch.cuda.empty_cache()


def _generation_agreement(rows: list[RawResultRow], layer: int) -> bool:
    filtered = [
        row
        for row in rows
        if row.measurement == "generation"
        and row.template_id == "primary"
        and row.endorsement == "wrong"
    ]
    index: dict[tuple[str, str, str], int] = {}
    for row in filtered:
        index[(row.uid, row.cue_family, row.intervention_id)] = int(
            row.parsed_label == row.endorsed_label
        )
    checks = []
    for cue, intervention in (
        ("source", f"source_remove_l{layer}"),
        ("user", f"user_remove_l{layer}"),
    ):
        differences = [
            index[(uid, cue, f"baseline_l{layer}")] - index[(uid, cue, intervention)]
            for uid in {row.uid for row in filtered if row.cue_family == cue}
            if (uid, cue, f"baseline_l{layer}") in index
            and (uid, cue, intervention) in index
        ]
        checks.append(bool(differences) and float(np.mean(differences)) > 0)
    return all(checks)


def analyze_run(
    *,
    config: PipelineConfig,
    splits: FrozenSplits,
    paths: RunPaths,
    caa_selection: CAASelection,
) -> None:
    raw_paths = sorted(paths.raw.glob("*.jsonl"))
    rows = load_raw_rows(raw_paths)
    summaries = separation_summaries(
        rows,
        fresh_uids=splits.fresh,
        layer=config.experiment.primary_layer,
        resamples=config.experiment.bootstrap_resamples,
        seed=config.experiment.statistics_seed,
    )
    identity_summaries, identity_checks = identity_control_summaries(
        rows,
        fresh_uids=splits.fresh,
        layer=config.experiment.primary_layer,
        shuffle_seeds=config.experiment.shuffle_seeds,
        random_seeds=config.experiment.random_seeds,
        resamples=config.experiment.bootstrap_resamples,
        seed=config.experiment.statistics_seed,
    )
    caa_summary = caa_comparison_summary(
        rows,
        fresh_uids=splits.fresh,
        layer=config.experiment.primary_layer,
        resamples=config.experiment.bootstrap_resamples,
        seed=config.experiment.statistics_seed,
    )
    robustness = robustness_summaries(
        rows,
        resamples=config.experiment.bootstrap_resamples,
        seed=config.experiment.statistics_seed,
    )
    summaries.extend(identity_summaries)
    summaries.append(caa_summary)
    summaries.extend(robustness)
    write_statistical_summaries(paths.analysis / "statistical_summaries.json", summaries)
    generation = generation_summary(rows)
    atomic_write_json(paths.analysis / "generation_summary.json", generation)
    audit = [
        {
            "uid": row.uid,
            "template_id": row.template_id,
            "cue_family": row.cue_family,
            "intervention_id": row.intervention_id,
            "generated_text": row.generated_text,
            "parsed_label": row.parsed_label,
            "parse_status": row.parse_status,
        }
        for row in rows
        if row.measurement == "generation"
    ][:200]
    atomic_write_json(paths.analysis / "parser_audit_sample.json", audit)
    plot_core_effects(summaries, paths.figures / "core_effects.png")
    generation_agrees = _generation_agreement(rows, config.experiment.primary_layer)
    decision = neutral_decision_table(
        summaries,
        generation_agrees=generation_agrees,
        identity_checks=identity_checks,
        caa_comparison_ci_low=caa_summary.ci_low,
    )
    decision["caa_selection"] = asdict(caa_selection)
    atomic_write_json(paths.analysis / "decision_table.json", decision)
    _write_reviewer_summary(paths, summaries, generation, decision)
    atomic_write_json(
        paths.analysis / "raw_checksums.json",
        {path.name: sha256_file(path) for path in raw_paths},
    )


def _write_reviewer_summary(
    paths: RunPaths,
    summaries: list[Any],
    generation: list[dict[str, object]],
    decision: dict[str, object],
) -> None:
    by_id = {item.analysis_id: item for item in summaries}
    lines = [
        "# Plain-English reviewer mapping",
        "",
        "This report is generated from raw result rows in a separate analysis pass.",
        "",
        "## Reviewers txTS and 1474: is the small residual causally meaningful?",
        "",
        (
            f"The fresh-set source selectivity estimate is "
            f"{by_id['source_selectivity'].estimate:.4f} "
            f"(95% CI {by_id['source_selectivity'].ci_low:.4f}, "
            f"{by_id['source_selectivity'].ci_high:.4f})."
        ),
        (
            f"The user selectivity estimate is {by_id['user_selectivity'].estimate:.4f} "
            f"(95% CI {by_id['user_selectivity'].ci_low:.4f}, "
            f"{by_id['user_selectivity'].ci_high:.4f})."
        ),
        (
            f"The combined interaction is {by_id['combined_interaction'].estimate:.4f} "
            f"(95% CI {by_id['combined_interaction'].ci_low:.4f}, "
            f"{by_id['combined_interaction'].ci_high:.4f})."
        ),
        "",
        "## Reviewer fzp1: robustness, tuning, and generalization",
        "",
        "All registered layers, positions, templates, random controls, shuffled controls, "
        "and failures are present in the raw tables; no fresh item was used for selection.",
        "",
        "## Generation and parser sensitivity",
        "",
        f"Generation condition summaries: {len(generation)}. Each reports parsed rate and "
        "best/worst-case wrong-answer bounds for unparsed outputs.",
        "",
        "## Neutral claim decision",
        "",
        "```json",
        json.dumps(decision, indent=2, sort_keys=True),
        "```",
        "",
    ]
    atomic_write_text(paths.analysis / "reviewer_mapping.md", "\n".join(lines))


def bundle_run(paths: RunPaths) -> tuple[Path, str]:
    archive = paths.bundle / f"qwen-rebuttal-{paths.run_id}.tar.gz"
    included = [
        paths.manifests,
        paths.directions,
        paths.raw,
        paths.analysis,
        paths.figures,
    ]
    with tarfile.open(archive, "w:gz") as handle:
        for item in included:
            if item.exists():
                handle.add(item, arcname=item.relative_to(paths.run_dir))
    checksum = sha256_file(archive)
    checksum_path = archive.with_suffix(archive.suffix + ".sha256")
    atomic_write_text(checksum_path, f"{checksum}  {archive.name}\n")
    return archive, checksum
