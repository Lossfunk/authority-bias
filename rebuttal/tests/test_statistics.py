from __future__ import annotations

import numpy as np

from qwen_rebuttal.analysis import paired_inference


def test_paired_statistics_are_seeded_and_recomputed_from_values():
    values = np.asarray([0.2, 0.4, 0.1, 0.5, 0.3])
    first = paired_inference(
        values,
        analysis_id="test",
        estimand="mean",
        resamples=1000,
        seed=17,
    )
    second = paired_inference(
        values,
        analysis_id="test",
        estimand="mean",
        resamples=1000,
        seed=17,
    )
    assert first == second
    assert first.estimate == np.mean(values)
    assert first.ci_low > 0
