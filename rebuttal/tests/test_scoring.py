from __future__ import annotations

import pytest
import torch

from qwen_rebuttal.scoring import BASELINE, score_margins
from qwen_rebuttal.types import PromptExample


class TinyTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == "A":
            return [1]
        if text == "multi-B":
            return [2, 3]
        return [4, 4]


class TinyAdapter:
    tokenizer = TinyTokenizer()
    device = torch.device("cpu")
    layers = []

    def scoring_logits(self, *, input_ids, logits_to_keep, **kwargs):
        return torch.zeros(input_ids.shape[0], logits_to_keep, 5)


def test_exact_margin_sums_complete_multi_token_candidate():
    adapter = TinyAdapter()
    prompt = PromptExample(
        uid="u",
        template_id="test",
        cue_family="none",
        endorsement="none",
        position="after_options",
        rendered_text="prompt",
        cue_text=None,
        cue_char_start=None,
        cue_char_end=None,
        candidate_a="A",
        candidate_b="multi-B",
        correct_label="A",
        endorsed_label="A",
    )
    rows = score_margins(
        adapter=adapter,
        prompts=[prompt],
        intervention=BASELINE,
        run_id="run",
        split="test",
        batch_size=1,
    )
    assert len(rows) == 1
    assert rows[0].logp_endorsed == pytest.approx(-torch.log(torch.tensor(5.0)).item())
    assert rows[0].logp_other == pytest.approx(-2 * torch.log(torch.tensor(5.0)).item())
    assert rows[0].compliance_margin == pytest.approx(torch.log(torch.tensor(5.0)).item())
