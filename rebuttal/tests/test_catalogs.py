from __future__ import annotations

import torch

from qwen_rebuttal.directions import LayerDirections
from qwen_rebuttal.pipeline import (
    _reviewer_span_catalog,
    _reviewer_span_core_catalog,
)

HIDDEN = 8
SEEDS = (10, 11, 12, 13, 14)


def _directions(layer: int = 5, scale: float = 1.0) -> LayerDirections:
    unit = torch.zeros(HIDDEN)
    unit[0] = 1.0
    return LayerDirections(
        layer=layer,
        source=torch.full((HIDDEN,), 1.0 * scale),
        user=torch.full((HIDDEN,), 0.5 * scale),
        shared=torch.full((HIDDEN,), 0.75 * scale),
        identity=unit,
        source_identity_coefficient=0.2 * scale,
        user_identity_coefficient=-0.2 * scale,
        geometry={},
    )


def _controls() -> dict[int, torch.Tensor]:
    return {seed: torch.full((HIDDEN,), 0.1) for seed in SEEDS}


def _paper_source() -> torch.Tensor:
    unit = torch.zeros(HIDDEN)
    unit[1] = 1.0
    return unit


def _catalog(layer: int = 5):
    return _reviewer_span_catalog(
        _directions(layer=layer),
        _directions(layer=layer, scale=2.0),
        _controls(),
        _controls(),
        _paper_source(),
    )


def test_span_catalog_has_the_designed_condition_set():
    catalog = _catalog()
    ids = [intervention.intervention_id for intervention, _ in catalog]
    assert len(ids) == 34
    assert len(set(ids)) == 34

    # Positive controls: the submitted paper's own interventions, which gate
    # whether any swap result may be interpreted at all.
    for required in (
        "baseline_l5",
        "source_remove_l5",
        "user_remove_l5",
        "legacy_span_source_projection_l5",
        "legacy_span_user_projection_l5",
        "paper_source_remove_l5",
    ):
        assert required in ids

    # The test itself, in both fitted variants, plus the diagnostic.
    for required in (
        "source_to_user_span_l5",
        "user_to_source_span_l5",
        "source_to_user_spanfit_l5",
        "user_to_source_spanfit_l5",
        "source_to_user_fulldiff_l5",
        "user_to_source_fulldiff_l5",
    ):
        assert required in ids

    assert sum(1 for name in ids if name.startswith("shuffled_span_")) == 10
    assert sum(1 for name in ids if name.startswith("random_span_")) == 10


def test_every_span_condition_acts_on_the_whole_cue_span():
    catalog = _catalog()
    endpoint_only = {"source_remove_l5", "user_remove_l5"}
    for intervention, _ in catalog:
        if intervention.mode == "none" or intervention.intervention_id in endpoint_only:
            continue
        assert intervention.token_scope == "cue_span", intervention.intervention_id


def test_span_noops_carry_zero_coefficient():
    catalog = {
        intervention.intervention_id: intervention
        for intervention, _ in _catalog()
    }
    assert catalog["source_to_source_span_noop_l5"].coefficient == 0
    assert catalog["user_to_user_span_noop_l5"].coefficient == 0


def test_spanfit_conditions_use_the_span_fitted_direction():
    span = _directions(scale=2.0)
    catalog = {
        intervention.intervention_id: intervention
        for intervention, _ in _reviewer_span_catalog(
            _directions(), span, _controls(), _controls(), _paper_source()
        )
    }
    spanfit = catalog["source_to_user_spanfit_l5"]
    assert spanfit.direction_id == "identity_spanfit_l5"
    # Coefficient must come from the span fit, not the endpoint fit.
    assert spanfit.coefficient == (
        span.user_identity_coefficient - span.source_identity_coefficient
    )


def test_swap_directions_are_equal_and_opposite():
    catalog = {
        intervention.intervention_id: intervention
        for intervention, _ in _catalog()
    }
    assert (
        catalog["source_to_user_span_l5"].coefficient
        == -catalog["user_to_source_span_l5"].coefficient
    )


def test_core_catalog_is_the_five_sweep_conditions():
    catalog = _reviewer_span_core_catalog(_directions(layer=3), _directions(layer=3))
    ids = [intervention.intervention_id for intervention, _ in catalog]
    assert ids == [
        "baseline_l3",
        "source_to_user_span_l3",
        "user_to_source_span_l3",
        "source_to_source_span_noop_l3",
        "user_to_user_span_noop_l3",
    ]
