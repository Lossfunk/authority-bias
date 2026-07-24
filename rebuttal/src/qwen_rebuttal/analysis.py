from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .io import atomic_write_json
from .types import RawResultRow, StatisticalSummary


def load_raw_rows(paths: Iterable[Path]) -> list[RawResultRow]:
    rows: list[RawResultRow] = []
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(RawResultRow(**json.loads(line)))
    keys = [row.row_key for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate raw row keys found during independent analysis")
    return rows


def paired_inference(
    values: np.ndarray,
    *,
    analysis_id: str,
    estimand: str,
    resamples: int,
    seed: int,
    metadata: dict[str, object] | None = None,
) -> StatisticalSummary:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid paired values for {analysis_id}")
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        size = min(1000, resamples - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        bootstrap[start : start + size] = values[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
    observed = abs(values.mean())
    extreme = 0
    completed = 0
    for start in range(0, resamples, 1000):
        size = min(1000, resamples - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(values)))
        permuted = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed))
        completed += size
    p_value = (extreme + 1) / (completed + 1)
    return StatisticalSummary(
        analysis_id=analysis_id,
        estimand=estimand,
        n_pairs=len(values),
        estimate=float(values.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        permutation_p=float(p_value),
        resamples=resamples,
        seed=seed,
        metadata={} if metadata is None else dict(metadata),
    )


def _margin_index(rows: list[RawResultRow]) -> dict[tuple[str, str, str, int | None], float]:
    result: dict[tuple[str, str, str, int | None], float] = {}
    for row in rows:
        if row.measurement != "margin" or row.compliance_margin is None:
            continue
        key = (row.uid, row.cue_family, row.intervention_id, row.layer)
        if key in result:
            raise RuntimeError(f"Non-unique margin condition: {key}")
        result[key] = row.compliance_margin
    return result


def _paired_values(
    index: dict[tuple[str, str, str, int | None], float],
    *,
    uids: list[str],
    left_cue: str,
    left_intervention: str,
    right_cue: str,
    right_intervention: str,
    layer: int,
) -> np.ndarray:
    return np.asarray(
        [
            index[(uid, left_cue, left_intervention, layer)]
            - index[(uid, right_cue, right_intervention, layer)]
            for uid in uids
        ]
    )


def separation_summaries(
    rows: list[RawResultRow],
    *,
    fresh_uids: tuple[str, ...],
    layer: int,
    resamples: int,
    seed: int,
) -> list[StatisticalSummary]:
    confirmatory = [
        row
        for row in rows
        if row.split == "fresh"
        and row.template_id == "primary"
        and row.position == "after_options"
        and row.endorsement == "wrong"
        and row.layer == layer
    ]
    index = _margin_index(confirmatory)
    uids = list(fresh_uids)
    source_same = _paired_values(
        index,
        uids=uids,
        left_cue="source",
        left_intervention=f"baseline_l{layer}",
        right_cue="source",
        right_intervention=f"source_remove_l{layer}",
        layer=layer,
    )
    source_cross = _paired_values(
        index,
        uids=uids,
        left_cue="user",
        left_intervention=f"baseline_l{layer}",
        right_cue="user",
        right_intervention=f"source_remove_l{layer}",
        layer=layer,
    )
    user_same = _paired_values(
        index,
        uids=uids,
        left_cue="user",
        left_intervention=f"baseline_l{layer}",
        right_cue="user",
        right_intervention=f"user_remove_l{layer}",
        layer=layer,
    )
    user_cross = _paired_values(
        index,
        uids=uids,
        left_cue="source",
        left_intervention=f"baseline_l{layer}",
        right_cue="source",
        right_intervention=f"user_remove_l{layer}",
        layer=layer,
    )
    values = {
        "source_same_cue": source_same,
        "source_cross_cue": source_cross,
        "source_selectivity": source_same - source_cross,
        "user_same_cue": user_same,
        "user_cross_cue": user_cross,
        "user_selectivity": user_same - user_cross,
        "combined_interaction": (source_same - source_cross) + (user_same - user_cross),
    }
    return [
        paired_inference(
            value,
            analysis_id=name,
            estimand="paired compliance-margin contrast",
            resamples=resamples,
            seed=seed,
            metadata={"split": "fresh", "layer": layer},
        )
        for name, value in values.items()
    ]


def _condition_lookup(
    rows: list[RawResultRow],
    *,
    split: str,
    template_id: str,
    position: str,
    endorsement: str,
) -> dict[tuple[str, str, str], float]:
    result: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if (
            row.measurement != "margin"
            or row.compliance_margin is None
            or row.split != split
            or row.template_id != template_id
            or row.position != position
            or row.endorsement != endorsement
        ):
            continue
        key = (row.uid, row.cue_family, row.intervention_id)
        if key in result:
            raise RuntimeError(f"Duplicate condition row {key}")
        result[key] = row.compliance_margin
    return result


def identity_control_summaries(
    rows: list[RawResultRow],
    *,
    fresh_uids: tuple[str, ...],
    layer: int,
    shuffle_seeds: tuple[int, ...],
    random_seeds: tuple[int, ...],
    resamples: int,
    seed: int,
) -> tuple[list[StatisticalSummary], dict[str, bool]]:
    index = _condition_lookup(
        rows,
        split="fresh",
        template_id="primary",
        position="after_options",
        endorsement="wrong",
    )
    baseline = f"baseline_l{layer}"

    def movements(cue: str, intervention: str, target_cue: str) -> np.ndarray:
        return np.asarray(
            [
                abs(index[(uid, cue, baseline)] - index[(uid, target_cue, baseline)])
                - abs(index[(uid, cue, intervention)] - index[(uid, target_cue, baseline)])
                for uid in fresh_uids
            ]
        )

    source_actual = movements("source", f"source_to_user_l{layer}", "user")
    user_actual = movements("user", f"user_to_source_l{layer}", "source")
    summaries = [
        paired_inference(
            source_actual,
            analysis_id="identity_source_to_user_movement",
            estimand="reduction in absolute distance to target-cue baseline",
            resamples=resamples,
            seed=seed,
        ),
        paired_inference(
            user_actual,
            analysis_id="identity_user_to_source_movement",
            estimand="reduction in absolute distance to target-cue baseline",
            resamples=resamples,
            seed=seed,
        ),
    ]
    shuffled_passes: list[bool] = []
    for shuffle_seed in shuffle_seeds:
        source_control = movements(
            "source", f"shuffled_source_to_user_s{shuffle_seed}_l{layer}", "user"
        )
        user_control = movements(
            "user", f"shuffled_user_to_source_s{shuffle_seed}_l{layer}", "source"
        )
        for cue, contrast in (
            ("source_to_user", source_actual - source_control),
            ("user_to_source", user_actual - user_control),
        ):
            summary = paired_inference(
                contrast,
                analysis_id=f"identity_{cue}_vs_shuffled_s{shuffle_seed}",
                estimand="identity movement minus shuffled-control movement",
                resamples=resamples,
                seed=seed,
            )
            summaries.append(summary)
            shuffled_passes.append(summary.ci_low > 0)
    random_passes: list[bool] = []
    for random_seed in random_seeds:
        source_control = movements(
            "source", f"random_source_to_user_s{random_seed}_l{layer}", "user"
        )
        user_control = movements(
            "user", f"random_user_to_source_s{random_seed}_l{layer}", "source"
        )
        for cue, contrast in (
            ("source_to_user", source_actual - source_control),
            ("user_to_source", user_actual - user_control),
        ):
            summary = paired_inference(
                contrast,
                analysis_id=f"identity_{cue}_vs_random_s{random_seed}",
                estimand="identity movement minus random-control movement",
                resamples=resamples,
                seed=seed,
            )
            summaries.append(summary)
            random_passes.append(summary.ci_low > 0)
    checks = {
        "source_to_user_moves_toward_user": summaries[0].ci_low > 0,
        "user_to_source_moves_toward_source": summaries[1].ci_low > 0,
        "both_exceed_all_shuffled_controls": all(shuffled_passes),
        "both_exceed_all_random_controls": all(random_passes),
    }
    return summaries, checks


def caa_comparison_summary(
    rows: list[RawResultRow],
    *,
    fresh_uids: tuple[str, ...],
    layer: int,
    resamples: int,
    seed: int,
) -> StatisticalSummary:
    index = _condition_lookup(
        rows,
        split="fresh",
        template_id="primary",
        position="after_options",
        endorsement="wrong",
    )
    caa_ids = sorted(
        {
            intervention
            for _uid, _cue, intervention in index
            if intervention.startswith("caa_frozen_")
        }
    )
    if len(caa_ids) != 1:
        raise RuntimeError(f"Expected one frozen CAA condition, found {caa_ids}")
    baseline = f"baseline_l{layer}"
    values = []
    for uid in fresh_uids:
        source_baseline = index[(uid, "source", baseline)]
        user_baseline = index[(uid, "user", baseline)]
        clean_effect = (
            source_baseline - index[(uid, "source", f"source_remove_l{layer}")]
            + user_baseline
            - index[(uid, "user", f"user_remove_l{layer}")]
        ) / 2
        caa_effect = (
            source_baseline - index[(uid, "source", caa_ids[0])]
            + user_baseline
            - index[(uid, "user", caa_ids[0])]
        ) / 2
        values.append(clean_effect - caa_effect)
    return paired_inference(
        np.asarray(values),
        analysis_id="clean_cue_specific_portfolio_vs_frozen_caa",
        estimand="paired reduction in wrong-endorsement margin: clean minus CAA",
        resamples=resamples,
        seed=seed,
        metadata={"caa_intervention_id": caa_ids[0]},
    )


def robustness_summaries(
    rows: list[RawResultRow],
    *,
    resamples: int,
    seed: int,
) -> list[StatisticalSummary]:
    groups = sorted(
        {
            (row.split, row.template_id, row.position, row.endorsement, row.layer)
            for row in rows
            if row.measurement == "margin"
            and row.cue_family in {"source", "user"}
            and row.layer is not None
        }
    )
    summaries: list[StatisticalSummary] = []
    for split, template, position, endorsement, layer in groups:
        index = _condition_lookup(
            rows,
            split=split,
            template_id=template,
            position=position,
            endorsement=endorsement,
        )
        required = (
            f"baseline_l{layer}",
            f"source_remove_l{layer}",
            f"user_remove_l{layer}",
        )
        uids = sorted(
            {
                uid
                for uid, cue, intervention in index
                if cue == "source" and intervention == required[0]
            }
        )
        if not uids:
            continue
        needed_keys = [
            (uid, cue, intervention)
            for uid in uids
            for cue in ("source", "user")
            for intervention in required
        ]
        if any(key not in index for key in needed_keys):
            continue
        source_same = np.asarray(
            [
                index[(uid, "source", required[0])]
                - index[(uid, "source", required[1])]
                for uid in uids
            ]
        )
        source_cross = np.asarray(
            [
                index[(uid, "user", required[0])] - index[(uid, "user", required[1])]
                for uid in uids
            ]
        )
        user_same = np.asarray(
            [
                index[(uid, "user", required[0])] - index[(uid, "user", required[2])]
                for uid in uids
            ]
        )
        user_cross = np.asarray(
            [
                index[(uid, "source", required[0])]
                - index[(uid, "source", required[2])]
                for uid in uids
            ]
        )
        prefix = f"{split}__{template}__{position}__{endorsement}__l{layer}"
        for name, values in (
            ("source_selectivity", source_same - source_cross),
            ("user_selectivity", user_same - user_cross),
            (
                "combined_interaction",
                (source_same - source_cross) + (user_same - user_cross),
            ),
        ):
            summaries.append(
                paired_inference(
                    values,
                    analysis_id=f"robustness__{prefix}__{name}",
                    estimand="paired compliance-margin contrast",
                    resamples=resamples,
                    seed=seed,
                    metadata={
                        "split": split,
                        "template": template,
                        "position": position,
                        "endorsement": endorsement,
                        "layer": layer,
                    },
                )
            )
    return summaries


def generation_summary(rows: list[RawResultRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[RawResultRow]] = {}
    for row in rows:
        if row.measurement == "generation":
            grouped.setdefault(
                (
                    row.template_id,
                    row.cue_family,
                    row.endorsement,
                    row.intervention_id,
                ),
                [],
            ).append(row)
    summaries: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        parsed = [row for row in group if row.parsed_label in {"A", "B"}]
        wrong_parsed = sum(
            row.parsed_label
            == (row.endorsed_label if row.endorsement == "wrong" else row.other_label)
            for row in parsed
        )
        unparsed = len(group) - len(parsed)
        summaries.append(
            {
                "template_id": key[0],
                "cue_family": key[1],
                "endorsement": key[2],
                "intervention_id": key[3],
                "n": len(group),
                "parsed_rate": len(parsed) / len(group),
                "wrong_answer_rate_parsed": wrong_parsed / len(parsed) if parsed else math.nan,
                "wrong_answer_rate_best_case": wrong_parsed / len(group),
                "wrong_answer_rate_worst_case": (wrong_parsed + unparsed) / len(group),
                "unparsed": unparsed,
            }
        )
    return summaries


def neutral_decision_table(
    summaries: list[StatisticalSummary],
    generation_agrees: bool,
    identity_checks: dict[str, bool],
    caa_comparison_ci_low: float | None,
) -> dict[str, object]:
    by_id = {item.analysis_id: item for item in summaries}
    separation_checks = {
        "source_selectivity_ci_above_zero": by_id["source_selectivity"].ci_low > 0,
        "user_selectivity_ci_above_zero": by_id["user_selectivity"].ci_low > 0,
        "combined_interaction_ci_above_zero": by_id["combined_interaction"].ci_low > 0,
        "generation_direction_agrees": generation_agrees,
    }
    return {
        "source_user_separation": {
            "checks": separation_checks,
            "supported_by_preregistered_policy": all(separation_checks.values()),
        },
        "identity_causality": {
            "checks": identity_checks,
            "supported_by_preregistered_policy": bool(identity_checks)
            and all(identity_checks.values()),
        },
        "superiority_to_caa": {
            "paired_ci_low": caa_comparison_ci_low,
            "supported_by_preregistered_policy": (
                caa_comparison_ci_low is not None and caa_comparison_ci_low > 0
            ),
        },
        "note": "This table applies the preregistered checks; it does not rewrite the paper claim.",
    }


def write_statistical_summaries(path: Path, summaries: list[StatisticalSummary]) -> None:
    atomic_write_json(path, [asdict(summary) for summary in summaries])


def plot_core_effects(summaries: list[StatisticalSummary], path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [
        item
        for item in summaries
        if item.analysis_id
        in {"source_selectivity", "user_selectivity", "combined_interaction"}
    ]
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    positions = np.arange(len(selected))
    estimates = np.asarray([item.estimate for item in selected])
    lower = estimates - np.asarray([item.ci_low for item in selected])
    upper = np.asarray([item.ci_high for item in selected]) - estimates
    axis.errorbar(positions, estimates, yerr=[lower, upper], fmt="o", capsize=4)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(positions, [item.analysis_id.replace("_", "\n") for item in selected])
    axis.set_ylabel("Paired compliance-margin contrast")
    axis.set_title("Fresh-set preregistered source/user checks")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)
