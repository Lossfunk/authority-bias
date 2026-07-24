from __future__ import annotations

import torch

from qwen_rebuttal.hooks import ResidualHook


class IdentityBlock(torch.nn.Module):
    def forward(self, hidden):
        return (hidden.clone(), "cache")


def test_hook_changes_only_selected_rows_and_positions():
    block = IdentityBlock()
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    vector = torch.ones(4)
    hook = ResidualHook(
        module=block,
        layer=5,
        mode="add",
        vector=vector,
        coefficient=2,
        phase="prefill",
        token_positions=[1, 2],
        row_mask=[True, False],
    )
    with hook:
        output, cache = block(hidden)
    expected = hidden.clone()
    expected[0, 1] += 2
    assert torch.equal(output, expected)
    assert cache == "cache"
    assert hook.audit.changed_rows == [0]
    assert hook.audit.changed_positions == [1]
    assert hook.audit.layer == 5


def test_noop_hook_exactly_matches_no_hook():
    block = IdentityBlock()
    hidden = torch.randn(2, 3, 4)
    baseline = block(hidden)[0]
    hook = ResidualHook(
        module=block,
        layer=3,
        mode="add",
        vector=torch.ones(4),
        coefficient=0,
        phase="prefill",
        token_positions=[0, 2],
    )
    with hook:
        result = block(hidden)[0]
    assert torch.equal(result, baseline)
    assert hook.audit.changed_calls == 0


def test_phase_gate_does_not_modify_decode_for_prefill_hook():
    block = IdentityBlock()
    hidden = torch.randn(1, 1, 4)
    hook = ResidualHook(
        module=block,
        layer=7,
        mode="add",
        vector=torch.ones(4),
        phase="prefill",
        token_positions=[0],
    )
    with hook:
        result = block(hidden)[0]
    assert torch.equal(result, hidden)
