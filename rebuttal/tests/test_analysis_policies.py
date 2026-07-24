from __future__ import annotations

from qwen_rebuttal.analysis import (
    caa_comparison_summary,
    identity_control_summaries,
    separation_summaries,
)
from qwen_rebuttal.types import RawResultRow


def _row(uid: str, cue: str, intervention: str, margin: float, layer: int = 5):
    return RawResultRow(
        row_key=f"{uid}:{cue}:{intervention}",
        run_id="run",
        split="fresh",
        uid=uid,
        measurement="margin",
        template_id="primary",
        cue_family=cue,
        endorsement="wrong",
        position="after_options",
        layer=layer,
        intervention_id=intervention,
        direction_id=None,
        seed=None,
        endorsed_label="A",
        other_label="B",
        compliance_margin=margin,
    )


def test_preregistered_policy_contrasts_are_computed_from_raw_rows():
    uids = tuple(f"u{index}" for index in range(8))
    rows = []
    for uid in uids:
        conditions = {
            ("source", "baseline_l5"): 1.0,
            ("user", "baseline_l5"): 0.5,
            ("source", "source_remove_l5"): 0.5,
            ("user", "source_remove_l5"): 0.4,
            ("source", "user_remove_l5"): 0.9,
            ("user", "user_remove_l5"): 0.1,
            ("source", "source_to_user_l5"): 0.55,
            ("user", "user_to_source_l5"): 0.95,
            ("source", "shuffled_source_to_user_s11_l5"): 0.9,
            ("user", "shuffled_user_to_source_s11_l5"): 0.6,
            ("source", "random_source_to_user_s12_l5"): 0.9,
            ("user", "random_user_to_source_s12_l5"): 0.6,
            ("source", "caa_frozen_l10_m-1"): 0.8,
            ("user", "caa_frozen_l10_m-1"): 0.3,
        }
        rows.extend(
            _row(uid, cue, intervention, margin)
            for (cue, intervention), margin in conditions.items()
        )
    separation = separation_summaries(
        rows,
        fresh_uids=uids,
        layer=5,
        resamples=200,
        seed=7,
    )
    assert all(
        summary.ci_low > 0
        for summary in separation
        if summary.analysis_id
        in {"source_selectivity", "user_selectivity", "combined_interaction"}
    )
    identity, checks = identity_control_summaries(
        rows,
        fresh_uids=uids,
        layer=5,
        shuffle_seeds=(11,),
        random_seeds=(12,),
        resamples=200,
        seed=7,
    )
    assert len(identity) == 6
    assert all(checks.values())
    caa = caa_comparison_summary(
        rows,
        fresh_uids=uids,
        layer=5,
        resamples=200,
        seed=7,
    )
    assert caa.ci_low > 0
