from __future__ import annotations

from qwen_rebuttal.pipeline import _attention_parity_record


def test_attention_parity_accepts_small_drift_with_stable_ranking():
    record = _attention_parity_record("u", 2.0, 1.9)

    assert record["within_bf16_tolerance"] is True
    assert record["passes_ranking_gate"] is True


def test_attention_parity_treats_near_zero_sign_change_as_tie():
    record = _attention_parity_record("u", -0.10, 0.01)

    assert record["ranking_agrees"] is False
    assert record["ranking_is_decisive"] is False
    assert record["passes_ranking_gate"] is True


def test_attention_parity_rejects_decisive_ranking_reversal():
    record = _attention_parity_record("u", -0.2, 0.2)

    assert record["ranking_is_decisive"] is True
    assert record["passes_ranking_gate"] is False
