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
    collect_final_activations,
    collect_span_activations,
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
from .paper_vectors import (
    PaperVectors,
    build_paper_vectors,
    load_paper_vectors,
    save_paper_vectors,
)
from .prompts import (
    TEMPLATE_CATALOG,
    apply_chat_and_locate_cue,
    build_paper_prompt,
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

ATTENTION_MARGIN_TOLERANCE = 0.125
ATTENTION_RANKING_TIE_BAND = 0.125


def _attention_parity_record(uid: str, optimized: float, reference: float) -> dict[str, Any]:
    ranking_agrees = (optimized >= 0) == (reference >= 0)
    ranking_is_decisive = (
        abs(optimized) > ATTENTION_RANKING_TIE_BAND
        and abs(reference) > ATTENTION_RANKING_TIE_BAND
    )
    difference = abs(optimized - reference)
    return {
        "uid": uid,
        "optimized_margin": optimized,
        "reference_margin": reference,
        "absolute_difference": difference,
        "within_original_0_05_tolerance": difference <= 0.05,
        "within_bf16_tolerance": difference <= ATTENTION_MARGIN_TOLERANCE,
        "ranking_agrees": ranking_agrees,
        "ranking_is_decisive": ranking_is_decisive,
        "passes_ranking_gate": ranking_agrees or not ranking_is_decisive,
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
    if config.inputs.assistant_axis:
        paths["assistant_axis"] = _resolve(repo_root, config.inputs.assistant_axis)
    values = {name: sha256_file(path) for name, path in paths.items()}
    values["caa_dataset_at_pinned_commit"] = config.caa.source_sha256
    if config.inputs.assistant_axis_sha256:
        declared = config.inputs.assistant_axis_sha256
        if values["assistant_axis"] != declared:
            raise RuntimeError(
                f"Assistant axis hashes to {values['assistant_axis']}, config declares {declared}"
            )
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


def fit_span_directions(
    *,
    adapter: QwenAdapter,
    config: PipelineConfig,
    splits: FrozenSplits,
    repo_root: Path,
    paths: RunPaths,
) -> dict[int, LayerDirections]:
    """Fit directions from the mean activation across each prompt's cue span.

    Span-scope interventions act at every cue token, so the endpoint-fitted
    directions are being applied where they were never fitted. Fitting a second
    set here lets both be run as separate conditions, which turns that mismatch
    from an unresolvable doubt into a measured comparison.
    """
    direction_path = paths.directions / "source_user_span.pt"
    checksum_path = paths.directions / "source_user_span.checksum.json"
    if direction_path.exists():
        if not checksum_path.exists():
            raise RuntimeError("Span direction tensor exists without its checksum")
        expected = json.loads(checksum_path.read_text())["sha256"]
        actual = sha256_file(direction_path)
        if actual != expected:
            raise RuntimeError(f"Span direction checksum mismatch: {actual} != {expected}")
        payload = torch.load(direction_path, map_location="cpu", weights_only=True)
        geometry = json.loads(direction_path.with_suffix(".geometry.json").read_text())
        return {
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
    dataset = load_dataset(_resolve(repo_root, config.inputs.dataset))
    prompt_groups = _fit_prompts(adapter, dataset, splits.fit)
    activations = {
        group: collect_span_activations(
            adapter,
            prompts,
            config.experiment.layers,
            config.experiment.direction_batch_size,
        )
        for group, prompts in prompt_groups.items()
    }
    fitted = fit_layer_directions(activations, config.experiment.layers)
    tensor_hash = save_directions(direction_path, fitted, {}, {})
    atomic_write_json(checksum_path, {"sha256": tensor_hash})
    return fitted


def fit_paper_reference_directions(
    *,
    adapter: QwenAdapter,
    config: PipelineConfig,
    splits: FrozenSplits,
    repo_root: Path,
    paths: RunPaths,
) -> dict[int, torch.Tensor]:
    """Rebuild the submitted paper's source direction, as a working positive control.

    The reviewer directions here are referenced against a *neutral* cue, which
    still contains the endorsement, so they isolate speaker identity alone.
    The paper references against *no cue at all*, so its direction carries the
    whole endorsement signal as well as identity. Removing the identity-only
    direction therefore barely moves compliance, while removing the paper's
    collapses it, and the two are not interchangeable as controls.

    Reproducing the paper's construction gives the run an intervention that is
    known to work, so a null on the identity direction can be read as a result
    rather than as a broken pipeline.
    """
    direction_path = paths.directions / "paper_source.pt"
    checksum_path = paths.directions / "paper_source.checksum.json"
    if direction_path.exists():
        if not checksum_path.exists():
            raise RuntimeError("Paper direction tensor exists without its checksum")
        expected = json.loads(checksum_path.read_text())["sha256"]
        actual = sha256_file(direction_path)
        if actual != expected:
            raise RuntimeError(f"Paper direction checksum mismatch: {actual} != {expected}")
        loaded = torch.load(direction_path, map_location="cpu", weights_only=True)
        return {int(layer): vector for layer, vector in loaded["layers"].items()}

    dataset = load_dataset(_resolve(repo_root, config.inputs.dataset))
    cue_prompts = _fit_prompts(adapter, dataset, splits.fit)["source"]
    no_cue_prompts = _tokenize_prompts(
        adapter,
        [
            build_prompt(
                dataset[uid],
                template_id="no_cue",
                cue_family="none",
                endorsement="none",
                position="after_options",
            )
            for uid in splits.fit
        ],
    )
    # Cue conditions are fitted across the cue span; the no-cue reference has no
    # span and is fitted at its final token, exactly as the paper does.
    cue_activations = collect_span_activations(
        adapter, cue_prompts, config.experiment.layers, config.experiment.direction_batch_size
    )
    no_cue_activations = collect_final_activations(
        adapter, no_cue_prompts, config.experiment.layers, config.experiment.direction_batch_size
    )
    directions: dict[int, torch.Tensor] = {}
    for layer in config.experiment.layers:
        difference = cue_activations[layer].mean(dim=0) - no_cue_activations[layer].mean(dim=0)
        norm = torch.linalg.vector_norm(difference)
        if norm <= 0:
            raise RuntimeError(f"Paper source direction at layer {layer} has zero norm")
        directions[layer] = difference / norm
    atomic_torch_save(direction_path, {"layers": directions})
    atomic_write_json(checksum_path, {"sha256": sha256_file(direction_path)})
    atomic_write_json(
        paths.directions / "paper_source.geometry.json",
        {
            str(layer): {
                "norm_before_normalisation": float(
                    torch.linalg.vector_norm(
                        cue_activations[layer].mean(dim=0)
                        - no_cue_activations[layer].mean(dim=0)
                    )
                )
            }
            for layer in config.experiment.layers
        },
    )
    return directions


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


def _span_scope_catalog(
    fitted: LayerDirections,
    span_fitted: LayerDirections,
    shuffled: dict[int, torch.Tensor],
    random: dict[int, torch.Tensor],
    paper_source: torch.Tensor,
) -> list[tuple[RuntimeIntervention, tuple[str, ...]]]:
    """Cue-span interventions: the properly powered identity test.

    The single-token variants in `_runtime_catalog` perturb the residual stream
    by roughly 0.16 L2, and matched-norm random controls move the model by the
    same negligible amount, so that test cannot detect an effect either way.
    These act at every cue token, matching the scope of the submitted paper's
    own span projection.

    `spanfit` conditions use a direction fitted across span positions rather
    than at the cue endpoint, so an ambiguous result cannot be blamed on
    applying an endpoint-fitted direction where it was never fitted.
    """
    layer = fitted.layer
    swap = fitted.user_identity_coefficient - fitted.source_identity_coefficient
    span_swap = (
        span_fitted.user_identity_coefficient - span_fitted.source_identity_coefficient
    )
    values: list[tuple[RuntimeIntervention, tuple[str, ...]]] = [
        # Working positive control: the paper's own no-cue-referenced direction,
        # which carries the whole endorsement signal rather than identity alone.
        (
            RuntimeIntervention(
                f"paper_source_remove_l{layer}",
                layer,
                "project_out",
                paper_source,
                1,
                "cue_span",
                "prefill",
                f"paper_source_l{layer}",
            ),
            ("source", "user"),
        ),
        (
            RuntimeIntervention(
                f"legacy_span_user_projection_l{layer}",
                layer,
                "project_out",
                fitted.user,
                1,
                "cue_span",
                "prefill",
                f"user_l{layer}",
            ),
            ("source", "user"),
        ),
        (
            RuntimeIntervention(
                f"source_to_user_span_l{layer}",
                layer,
                "add",
                fitted.identity,
                swap,
                "cue_span",
                "prefill",
                f"identity_l{layer}",
            ),
            ("source",),
        ),
        (
            RuntimeIntervention(
                f"user_to_source_span_l{layer}",
                layer,
                "add",
                fitted.identity,
                -swap,
                "cue_span",
                "prefill",
                f"identity_l{layer}",
            ),
            ("user",),
        ),
        (
            RuntimeIntervention(
                f"source_to_user_spanfit_l{layer}",
                layer,
                "add",
                span_fitted.identity,
                span_swap,
                "cue_span",
                "prefill",
                f"identity_spanfit_l{layer}",
            ),
            ("source",),
        ),
        (
            RuntimeIntervention(
                f"user_to_source_spanfit_l{layer}",
                layer,
                "add",
                span_fitted.identity,
                -span_swap,
                "cue_span",
                "prefill",
                f"identity_spanfit_l{layer}",
            ),
            ("user",),
        ),
        # Diagnostic: the whole source-user difference, not only the orthogonal
        # identity residual. If this moves behaviour while the residual does
        # not, the split rides on the shared endorsement component.
        (
            RuntimeIntervention(
                f"source_to_user_fulldiff_l{layer}",
                layer,
                "add",
                fitted.user - fitted.source,
                1.0,
                "cue_span",
                "prefill",
                f"fulldiff_l{layer}",
            ),
            ("source",),
        ),
        (
            RuntimeIntervention(
                f"user_to_source_fulldiff_l{layer}",
                layer,
                "add",
                fitted.source - fitted.user,
                1.0,
                "cue_span",
                "prefill",
                f"fulldiff_l{layer}",
            ),
            ("user",),
        ),
        (
            RuntimeIntervention(
                f"source_to_source_span_noop_l{layer}",
                layer,
                "add",
                fitted.identity,
                0,
                "cue_span",
                "prefill",
                f"identity_l{layer}",
            ),
            ("source",),
        ),
        (
            RuntimeIntervention(
                f"user_to_user_span_noop_l{layer}",
                layer,
                "add",
                fitted.identity,
                0,
                "cue_span",
                "prefill",
                f"identity_l{layer}",
            ),
            ("user",),
        ),
    ]
    for prefix, vectors in (("shuffled_span", shuffled), ("random_span", random)):
        for seed, vector in vectors.items():
            for cue, coefficient, label in (
                ("source", swap, "source_to_user"),
                ("user", -swap, "user_to_source"),
            ):
                values.append(
                    (
                        RuntimeIntervention(
                            f"{prefix}_{label}_s{seed}_l{layer}",
                            layer,
                            "add",
                            vector,
                            coefficient,
                            "cue_span",
                            "prefill",
                            f"{prefix}_s{seed}_l{layer}",
                            seed,
                        ),
                        (cue,),
                    )
                )
    return values


def _reviewer_span_core_catalog(
    fitted: LayerDirections,
    span_fitted: LayerDirections,
) -> list[tuple[RuntimeIntervention, tuple[str, ...]]]:
    """Five-condition core set used for the layer and prompt-template sweeps."""
    layer = fitted.layer
    wanted = {
        f"source_to_user_span_l{layer}",
        f"user_to_source_span_l{layer}",
        f"source_to_source_span_noop_l{layer}",
        f"user_to_user_span_noop_l{layer}",
    }
    core = [
        (intervention, cues)
        for intervention, cues in _span_scope_catalog(
            fitted, span_fitted, {}, {}, fitted.source
        )
        if intervention.intervention_id in wanted
    ]
    baseline = [
        (intervention, cues)
        for intervention, cues in _runtime_catalog(fitted, {}, {})
        if intervention.intervention_id == f"baseline_l{layer}"
    ]
    return [*baseline, *core]


def _reviewer_span_catalog(
    fitted: LayerDirections,
    span_fitted: LayerDirections,
    shuffled: dict[int, torch.Tensor],
    random: dict[int, torch.Tensor],
    paper_source: torch.Tensor,
) -> list[tuple[RuntimeIntervention, tuple[str, ...]]]:
    """Layer-5 primary set: positive controls plus every span-scope condition.

    The positive controls are the submitted paper's own interventions. They gate
    interpretation: if removing the source vector does not collapse source
    compliance here, this pipeline cannot detect a known-real effect and the
    swap conditions say nothing.
    """
    layer = fitted.layer
    controls = {
        f"baseline_l{layer}",
        f"source_remove_l{layer}",
        f"user_remove_l{layer}",
        f"legacy_span_source_projection_l{layer}",
    }
    positive = [
        (intervention, cues)
        for intervention, cues in _runtime_catalog(fitted, {}, {})
        if intervention.intervention_id in controls
    ]
    return [
        *positive,
        *_span_scope_catalog(fitted, span_fitted, shuffled, random, paper_source),
    ]


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


def _score_reviewer_catalog(
    *,
    adapter: QwenAdapter,
    config: PipelineConfig,
    manifest: RunManifest,
    paths: RunPaths,
    prompts: list[PromptExample],
    catalog: list[tuple[RuntimeIntervention, tuple[str, ...]]],
    template: str,
    position: str,
    split: str,
) -> None:
    """Score every condition over one identical prompt list.

    Batch composition perturbs exact margins, so a condition scored over a cue
    subset cannot be differenced against a baseline scored over the full list.
    Every condition therefore sees the same prompts, in the same order, in the
    same batches, and only the in-scope rows are written. Interventions act at
    each prompt's own cue endpoint, so out-of-scope rows are simply discarded.
    Holding composition fixed this way is what makes large batches safe.
    """
    for intervention, cues in catalog:
        expected = sum(1 for prompt in prompts if prompt.cue_family in cues)
        path = _condition_file(
            paths,
            measurement="margin",
            split=split,
            template=template,
            position=position,
            intervention_id=intervention.intervention_id,
            endorsement="wrong",
        )
        if _validate_existing_rows(path, manifest.run_id, expected):
            continue
        rows = score_margins(
            adapter=adapter,
            prompts=prompts,
            intervention=intervention,
            run_id=manifest.run_id,
            split=split,
            batch_size=config.experiment.scoring_batch_size,
        )
        append_jsonl_atomic(path, [row for row in rows if row.cue_family in cues])


def _assert_reviewer_noop_scoring(
    *,
    adapter: QwenAdapter,
    config: PipelineConfig,
    manifest: RunManifest,
    paths: RunPaths,
    prompts: list[PromptExample],
    fitted: LayerDirections,
    span_fitted: LayerDirections,
) -> None:
    """Verify the fixed-composition invariant the reviewer margins depend on.

    Conditions are differenced against one another, so what has to hold is that
    one prompt list scored twice yields identical margins, and that a
    zero-strength intervention over that same list reproduces the baseline
    exactly. Both are checked bitwise. If either moves, a condition difference
    would carry a batching artifact rather than a causal effect.
    """
    smoke_prompts = prompts[:16]
    catalog = {
        intervention.intervention_id: (intervention, cues)
        for intervention, cues in _reviewer_span_core_catalog(fitted, span_fitted)
    }
    baseline, _ = catalog[f"baseline_l{fitted.layer}"]

    def margins(intervention: RuntimeIntervention) -> dict[tuple[str, str], float]:
        rows = score_margins(
            adapter=adapter,
            prompts=smoke_prompts,
            intervention=intervention,
            run_id=manifest.run_id,
            split="smoke",
            batch_size=config.experiment.scoring_batch_size,
        )
        return {(row.uid, row.cue_family): float(row.compliance_margin) for row in rows}

    baseline_by_key = margins(baseline)
    checks: dict[str, dict[str, float | int]] = {}

    repeat = margins(baseline)
    repeat_maximum = max(
        (abs(value - baseline_by_key[key]) for key, value in repeat.items()),
        default=0.0,
    )
    checks["repeat_baseline"] = {"n": len(repeat), "max_abs_difference": repeat_maximum}
    if repeat_maximum != 0:
        raise RuntimeError(
            f"Rescoring the same prompt list moved a smoke margin by {repeat_maximum}"
        )

    for cue, intervention_id in (
        ("source", f"source_to_source_span_noop_l{fitted.layer}"),
        ("user", f"user_to_user_span_noop_l{fitted.layer}"),
    ):
        intervention, _ = catalog[intervention_id]
        no_op = margins(intervention)
        maximum = max(
            (abs(value - baseline_by_key[key]) for key, value in no_op.items()),
            default=0.0,
        )
        checks[cue] = {"n": len(no_op), "max_abs_noop_difference": maximum}
        if maximum != 0:
            raise RuntimeError(
                f"{cue} zero-strength no-op changed a smoke margin by {maximum}"
            )
    atomic_write_json(paths.analysis / "reviewer_noop_scoring_smoke.json", checks)


def run_reviewer_margin_experiments(
    *,
    adapter: QwenAdapter,
    config: PipelineConfig,
    splits: FrozenSplits,
    manifest: RunManifest,
    paths: RunPaths,
    repo_root: Path,
    fitted: dict[int, LayerDirections],
    span_fitted: dict[int, LayerDirections],
    paper_source: dict[int, torch.Tensor],
    shuffled: dict[int, dict[int, torch.Tensor]],
    random: dict[int, dict[int, torch.Tensor]],
) -> None:
    """Run only the new evidence explicitly requested in the three reviews.

    The recorded holdout evaluation UID partition is retained for direct
    comparability. Correct-endorsement preservation, removal experiments, and
    legacy span-wide projections are already present in the submitted results
    and are intentionally not recomputed here.
    """
    dataset = load_dataset(_resolve(repo_root, config.inputs.dataset))
    primary_layer = config.experiment.primary_layer

    # Reviewer 1474 called the Qwen3.5 confidence intervals excessively wide, so
    # every margin condition also runs on the untouched fresh partition. The
    # previously examined historical partition is retained so the new evidence
    # stays directly comparable with the submitted results.
    for split_name, uids in (("historical", splits.historical), ("fresh", splits.fresh)):
        primary = _evaluation_prompts(
            adapter,
            dataset,
            uids,
            template_id="primary",
            position="after_options",
            endorsements=("wrong",),
        )
        _write_prompt_manifest(
            paths.manifests / "reviewer_evaluation_prompts.jsonl",
            {f"{split_name}:primary:after_options:wrong": primary},
        )
        if split_name == "historical":
            _assert_reviewer_noop_scoring(
                adapter=adapter,
                config=config,
                manifest=manifest,
                paths=paths,
                prompts=primary,
                fitted=fitted[primary_layer],
                span_fitted=span_fitted[primary_layer],
            )

        # Primary causal result: positive controls, span-scope swaps in both
        # fitted variants, the full-difference diagnostic, and every control.
        _score_reviewer_catalog(
            adapter=adapter,
            config=config,
            manifest=manifest,
            paths=paths,
            prompts=primary,
            catalog=_reviewer_span_catalog(
                fitted[primary_layer],
                span_fitted[primary_layer],
                shuffled[primary_layer],
                random[primary_layer],
                paper_source[primary_layer],
            ),
            template="primary",
            position="after_options",
            split=split_name,
        )

        # Nearby-layer robustness: no post-hoc layer selection and no control sweep.
        for layer in config.experiment.robustness_layers:
            _score_reviewer_catalog(
                adapter=adapter,
                config=config,
                manifest=manifest,
                paths=paths,
                prompts=primary,
                catalog=_reviewer_span_core_catalog(fitted[layer], span_fitted[layer]),
                template="primary",
                position="after_options",
                split=split_name,
            )

        # Prompt wording and position checks at the frozen primary layer.
        prompt_conditions = (
            ("primary", "before_question"),
            ("submitted", "after_options"),
            ("structured_paraphrase", "after_options"),
        )
        for template, position in prompt_conditions:
            prompts = _evaluation_prompts(
                adapter,
                dataset,
                uids,
                template_id=template,
                position=position,
                endorsements=("wrong",),
            )
            _write_prompt_manifest(
                paths.manifests / "reviewer_evaluation_prompts.jsonl",
                {f"{split_name}:{template}:{position}:wrong": prompts},
            )
            _score_reviewer_catalog(
                adapter=adapter,
                config=config,
                manifest=manifest,
                paths=paths,
                prompts=prompts,
                catalog=_reviewer_span_core_catalog(
                    fitted[primary_layer], span_fitted[primary_layer]
                ),
                template=template,
                position=position,
                split=split_name,
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


def run_caa_reviewer_margins(
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
    """Evaluate the independently tuned native CAA baseline on both partitions."""
    dataset = load_dataset(_resolve(repo_root, config.inputs.dataset))
    intervention = _caa_runtime(selection, direction)
    for split_name, uids in (("historical", splits.historical), ("fresh", splits.fresh)):
        prompts = _evaluation_prompts(
            adapter,
            dataset,
            uids,
            template_id="primary",
            position="after_options",
            endorsements=("wrong",),
        )
        path = _condition_file(
            paths,
            measurement="margin",
            split=split_name,
            template="primary",
            position="after_options",
            intervention_id=intervention.intervention_id,
            endorsement="wrong",
        )
        _score_condition(
            adapter=adapter,
            prompts=prompts,
            intervention=intervention,
            run_id=manifest.run_id,
            split=split_name,
            batch_size=config.experiment.scoring_batch_size,
            path=path,
        )


def fit_paper_mitigation_vectors(
    *,
    adapter: QwenAdapter,
    config: PipelineConfig,
    splits: FrozenSplits,
    repo_root: Path,
    paths: RunPaths,
) -> PaperVectors:
    """Rebuild the submitted paper's own vectors at its declared mitigation layer.

    These are not the directions the rest of this pipeline fits. See
    `paper_vectors` for why the two constructions differ and why answering
    reviewer 1474 requires the paper's, not ours.
    """
    layer = config.experiment.mitigation_layer
    if layer < 0:
        raise RuntimeError("This config declares no mitigation_layer")
    if not config.inputs.assistant_axis:
        raise RuntimeError(
            "The residualized mitigation vector needs the paper's assistant axis; "
            "this config declares none"
        )
    tensor_path = paths.directions / "paper_mitigation.pt"
    checksum_path = paths.directions / "paper_mitigation.checksum.json"
    geometry_path = paths.directions / "paper_mitigation.geometry.json"
    if tensor_path.exists():
        if not checksum_path.exists():
            raise RuntimeError("Paper mitigation vectors exist without their checksum")
        expected = json.loads(checksum_path.read_text())["sha256"]
        actual = sha256_file(tensor_path)
        if actual != expected:
            raise RuntimeError(f"Paper mitigation checksum mismatch: {actual} != {expected}")
        vectors = load_paper_vectors(tensor_path, geometry_path)
        if vectors.layer != layer:
            raise RuntimeError(
                f"Cached paper vectors are at layer {vectors.layer}, config asks for {layer}"
            )
        return vectors

    dataset = load_dataset(_resolve(repo_root, config.inputs.dataset))

    # Fitted in exp20's own prompt format, which is the format the head-to-head
    # evaluates in. Fitting on the A/B-letter template instead produces a source
    # vector dominated by "a note exists in *that* template", nearly orthogonal
    # to the authority signal and inert when projected out of an exp20 prompt.
    def cued(cue_family: str) -> list[PromptExample]:
        return _tokenize_prompts(
            adapter,
            [
                build_paper_prompt(dataset[uid], cue_family=cue_family, endorsement=endorsement)
                for uid in splits.fit
                for endorsement in ("correct", "wrong")
            ],
        )

    no_cue = _tokenize_prompts(
        adapter,
        [
            build_paper_prompt(dataset[uid], cue_family="none", endorsement="none")
            for uid in splits.fit
        ],
    )
    vectors = build_paper_vectors(
        adapter=adapter,
        layer=layer,
        source_prompts=cued("source"),
        user_prompts=cued("user"),
        neutral_prompts=cued("neutral"),
        no_cue_prompts=no_cue,
        assistant_axis_path=_resolve(repo_root, config.inputs.assistant_axis),
        batch_size=config.experiment.direction_batch_size,
    )
    save_paper_vectors(tensor_path, vectors)
    atomic_write_json(checksum_path, {"sha256": sha256_file(tensor_path)})
    atomic_write_json(geometry_path, vectors.geometry)
    return vectors


def _head_to_head_catalog(
    vectors: PaperVectors,
    selection: CAASelection,
    caa_direction: torch.Tensor,
) -> list[tuple[RuntimeIntervention, tuple[str, ...]]]:
    """Every cell of the matched mitigation and causal-split comparison.

    All of these are evaluated on one prompt list, one metric and one frozen
    split, which is the whole point: reviewer 1474's objection is that the
    submitted CAA baseline and the submitted mitigation were not compared on
    equal terms, and the only answer to that is to compare them on equal terms.

    The two arms of the fairness question:
      * `auth_resid_remove` is the submitted Table 6 mitigation, at the layer
        and alpha the submission declares. It gets no search here at all.
      * `caa_tuned` is CAA at the layer and multiplier chosen by its own
        independent search over every layer and nine multipliers, scored on
        CAA's own held-out sycophancy data. It gets the entire search budget.
    So the search asymmetry runs in CAA's favour, not ours.

    `caa_paper` reproduces the submission's *own* CAA configuration -- unit
    vector, alpha=1, subtracted over the cue span at the mitigation layer --
    which is what `run_mitigation_pareto.py` runs under `--variants caa` with
    `mode="subtract"` and `--norm-scaling none`. Reporting it next to
    `caa_tuned` shows how much of the gap the extra search actually closes.

    `src_remove` / `usr_remove` are the Table 4 causal split, which uses the
    no-cue-referenced vectors rather than the mitigation vector.
    """
    layer = vectors.layer
    both = ("source", "user")
    unit_caa = caa_direction / torch.linalg.vector_norm(caa_direction)
    return [
        (
            RuntimeIntervention(
                f"h2h_baseline_l{layer}", layer, "none", None, 0, "none", "prefill"
            ),
            both,
        ),
        (
            RuntimeIntervention(
                f"h2h_auth_resid_remove_l{layer}",
                layer,
                "project_out",
                vectors.authority_resid,
                1,
                "cue_span",
                "prefill",
                f"auth_resid_l{layer}",
            ),
            both,
        ),
        (
            RuntimeIntervention(
                f"h2h_auth_remove_l{layer}",
                layer,
                "project_out",
                vectors.authority,
                1,
                "cue_span",
                "prefill",
                f"auth_l{layer}",
            ),
            both,
        ),
        (
            RuntimeIntervention(
                f"h2h_assistant_remove_l{layer}",
                layer,
                "project_out",
                vectors.assistant,
                1,
                "cue_span",
                "prefill",
                f"assistant_l{layer}",
            ),
            both,
        ),
        (
            RuntimeIntervention(
                f"h2h_caa_paper_l{layer}_a1",
                layer,
                "add",
                unit_caa,
                -1.0,
                "cue_span",
                "prefill",
                f"caa_unit_l{layer}",
            ),
            both,
        ),
        (
            RuntimeIntervention(
                f"h2h_caa_tuned_l{selection.layer}_m{selection.multiplier:g}",
                selection.layer,
                "add",
                caa_direction,
                selection.multiplier,
                "answer_tokens",
                "all",
                f"caa_l{selection.layer}",
            ),
            both,
        ),
        (
            RuntimeIntervention(
                f"h2h_src_remove_l{layer}",
                layer,
                "project_out",
                vectors.source,
                1,
                "cue_span",
                "prefill",
                f"src_l{layer}",
            ),
            both,
        ),
        (
            RuntimeIntervention(
                f"h2h_usr_remove_l{layer}",
                layer,
                "project_out",
                vectors.user,
                1,
                "cue_span",
                "prefill",
                f"usr_l{layer}",
            ),
            both,
        ),
    ]


def run_head_to_head_generation(
    *,
    adapter: QwenAdapter,
    config: PipelineConfig,
    manifest: RunManifest,
    paths: RunPaths,
    repo_root: Path,
    vectors: PaperVectors,
    selection: CAASelection,
    caa_direction: torch.Tensor,
    uids: tuple[str, ...],
    split_name: str,
) -> None:
    """Generate free-text answers for every head-to-head cell on one frozen split.

    The readout is the submitted paper's: its exp20 prompt, which forbids option
    letters and asks for a written sentence, so what is scored is which answer
    *text* the model produces. Only raw generations are written; parsing is done
    offline with the paper's own parser so the rates cannot drift from the
    submitted ones through a reimplementation.

    Every condition generates over the identical full prompt list. Batch
    composition shifts BF16 results by up to a tenth of a nat, so holding the
    list fixed is what makes the between-condition differences attributable to
    the intervention rather than to who shared a batch.
    """
    dataset = load_dataset(_resolve(repo_root, config.inputs.dataset))
    prompts = _tokenize_prompts(
        adapter,
        [
            build_paper_prompt(dataset[uid], cue_family=cue, endorsement="wrong")
            for uid in uids
            for cue in ("source", "user")
        ],
    )
    _write_prompt_manifest(
        paths.manifests / "head_to_head_prompts.jsonl",
        {f"{split_name}:paper_exp20:after_options:wrong": prompts},
    )
    atomic_write_json(
        paths.manifests / "head_to_head_conditions.json",
        {
            "mitigation_layer": vectors.layer,
            "caa_selected_layer": selection.layer,
            "caa_selected_multiplier": selection.multiplier,
            "caa_tune_items": selection.n_tune,
            "caa_raw_norm": float(torch.linalg.vector_norm(caa_direction)),
            "geometry": vectors.geometry,
            "conditions": [
                {
                    "intervention_id": intervention.intervention_id,
                    "layer": intervention.layer,
                    "mode": intervention.mode,
                    "coefficient": intervention.coefficient,
                    "token_scope": intervention.token_scope,
                    "phase": intervention.phase,
                }
                for intervention, _cues in _head_to_head_catalog(
                    vectors, selection, caa_direction
                )
            ],
        },
    )
    for intervention, cues in _head_to_head_catalog(vectors, selection, caa_direction):
        expected = sum(1 for prompt in prompts if prompt.cue_family in cues)
        path = _condition_file(
            paths,
            measurement="generation",
            split=split_name,
            template="paper_exp20",
            position="after_options",
            intervention_id=intervention.intervention_id,
            endorsement="wrong",
        )
        if _validate_existing_rows(path, manifest.run_id, expected):
            continue
        rows = generate_answers(
            adapter=adapter,
            prompts=prompts,
            intervention=intervention,
            run_id=manifest.run_id,
            split=split_name,
            max_new_tokens=config.experiment.generation_max_new_tokens,
            batch_size=config.experiment.generation_batch_size,
        )
        append_jsonl_atomic(path, [row for row in rows if row.cue_family in cues])


def run_caa_only_margins(
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
    """Score the no-steering baseline and the tuned CAA vector on both partitions.

    The baseline is scored here rather than reused from another run because a
    CAA margin is only interpretable against the same model's own unsteered
    behaviour, and this command is meant to stand alone for a new model.
    """
    dataset = load_dataset(_resolve(repo_root, config.inputs.dataset))
    caa = _caa_runtime(selection, direction)
    baseline = RuntimeIntervention(
        f"baseline_l{selection.layer}",
        selection.layer,
        "none",
        None,
        0,
        "none",
        "prefill",
    )
    for split_name, uids in (("historical", splits.historical), ("fresh", splits.fresh)):
        prompts = _evaluation_prompts(
            adapter,
            dataset,
            uids,
            template_id="primary",
            position="after_options",
            endorsements=("wrong",),
        )
        for intervention in (baseline, caa):
            path = _condition_file(
                paths,
                measurement="margin",
                split=split_name,
                template="primary",
                position="after_options",
                intervention_id=intervention.intervention_id,
                endorsement="wrong",
            )
            _score_condition(
                adapter=adapter,
                prompts=prompts,
                intervention=intervention,
                run_id=manifest.run_id,
                split=split_name,
                batch_size=config.experiment.scoring_batch_size,
                path=path,
            )


def run_reviewer_generation_experiments(
    *,
    adapter: QwenAdapter,
    config: PipelineConfig,
    splits: FrozenSplits,
    manifest: RunManifest,
    paths: RunPaths,
    repo_root: Path,
    fitted: LayerDirections,
    span_fitted: LayerDirections,
) -> None:
    """Paper-comparable free generation for only the primary causal conditions.

    Like the margin conditions, every condition generates over the same full
    prompt list and keeps only its in-scope rows, so left padding is identical
    across conditions and cannot bias the comparison between them.
    """
    dataset = load_dataset(_resolve(repo_root, config.inputs.dataset))
    prompts = _evaluation_prompts(
        adapter,
        dataset,
        splits.historical,
        template_id="primary",
        position="after_options",
        endorsements=("wrong",),
    )
    chosen = [
        (intervention, cues)
        for intervention, cues in _reviewer_span_core_catalog(fitted, span_fitted)
        if intervention.intervention_id
        in {
            f"baseline_l{fitted.layer}",
            f"source_to_user_span_l{fitted.layer}",
            f"user_to_source_span_l{fitted.layer}",
        }
    ]
    for intervention, cues in chosen:
        expected = sum(1 for prompt in prompts if prompt.cue_family in cues)
        path = _condition_file(
            paths,
            measurement="generation",
            split="historical",
            template="primary",
            position="after_options",
            intervention_id=intervention.intervention_id,
            endorsement="wrong",
        )
        if _validate_existing_rows(path, manifest.run_id, expected):
            continue
        rows = generate_answers(
            adapter=adapter,
            prompts=prompts,
            intervention=intervention,
            run_id=manifest.run_id,
            split="historical",
            max_new_tokens=config.experiment.generation_max_new_tokens,
            batch_size=config.experiment.generation_batch_size,
        )
        append_jsonl_atomic(path, [row for row in rows if row.cue_family in cues])


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


def h200_smoke(
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
    adapter.release()
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
        comparisons.append(_attention_parity_record(row.uid, opt, ref))
    atomic_write_json(paths.analysis / "attention_parity_smoke.json", comparisons)
    if any(not item["passes_ranking_gate"] for item in comparisons):
        raise RuntimeError(
            "Optimized and reference attention disagree on a decisive answer ranking"
        )
    maximum_difference = max(float(item["absolute_difference"]) for item in comparisons)
    if maximum_difference > ATTENTION_MARGIN_TOLERANCE:
        raise RuntimeError(
            "Attention backend compliance-margin difference exceeds the "
            f"BF16-aware tolerance of {ATTENTION_MARGIN_TOLERANCE} "
            f"(maximum={maximum_difference:.6f}); see attention_parity_smoke.json"
        )
    reference_adapter.release()


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
