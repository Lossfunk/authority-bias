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
        if text == "B":
            return [2]
        if text == "multi-B":
            return [2, 3]
        if text == "source prompt":
            return [6, 6]
        if text == "user prompt":
            return [7, 7]
        if text == "user":
            return [7]
        return [4, 4]


class TinyAdapter:
    tokenizer = TinyTokenizer()
    device = torch.device("cpu")
    layers = []

    def scoring_logits(self, *, input_ids, logits_to_keep, **kwargs):
        return torch.zeros(input_ids.shape[0], logits_to_keep, 5)


class PromptSensitiveAdapter(TinyAdapter):
    def __init__(self):
        self.scored_shapes: list[tuple[int, ...]] = []

    def scoring_logits(self, *, input_ids, logits_to_keep, **kwargs):
        self.scored_shapes.append(tuple(input_ids.shape))
        logits = torch.zeros(input_ids.shape[0], logits_to_keep, 8)
        for row, tokens in enumerate(input_ids):
            if 6 in tokens:
                logits[row, :, 1] = 4
            elif 7 in tokens:
                logits[row, :, 2] = 4
            else:
                raise AssertionError("Test prompt identity token is missing")
        return logits


def _prompt(*, uid: str, cue_family: str, rendered_text: str) -> PromptExample:
    return PromptExample(
        uid=uid,
        template_id="test",
        cue_family=cue_family,
        endorsement="wrong",
        position="after_options",
        rendered_text=rendered_text,
        cue_text=None,
        cue_char_start=None,
        cue_char_end=None,
        candidate_a="A",
        candidate_b="B",
        correct_label="B",
        endorsed_label="A",
    )


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


def test_same_uid_prompts_keep_independent_candidate_scores():
    rows = score_margins(
        adapter=PromptSensitiveAdapter(),
        prompts=[
            _prompt(uid="shared", cue_family="source", rendered_text="source prompt"),
            _prompt(uid="shared", cue_family="user", rendered_text="user prompt"),
        ],
        intervention=BASELINE,
        run_id="run",
        split="test",
        batch_size=2,
    )
    assert [row.cue_family for row in rows] == ["source", "user"]
    assert rows[0].compliance_margin == pytest.approx(4.0)
    assert rows[1].compliance_margin == pytest.approx(-4.0)


def test_shorter_prompt_scores_correctly_inside_a_padded_batch():
    adapter = PromptSensitiveAdapter()
    rows = score_margins(
        adapter=adapter,
        prompts=[
            _prompt(uid="shared", cue_family="source", rendered_text="source prompt"),
            _prompt(uid="shared", cue_family="user", rendered_text="user"),
        ],
        intervention=BASELINE,
        run_id="run",
        split="test",
        batch_size=2,
    )
    # "source prompt" is two tokens and "user" is one, so both prompts go through
    # a single left-padded forward pass. The shorter prompt must still land on
    # its own candidate logits rather than on its padding.
    assert adapter.scored_shapes == [(4, 3)]
    assert [row.cue_family for row in rows] == ["source", "user"]
    assert rows[0].compliance_margin == pytest.approx(4.0)
    assert rows[1].compliance_margin == pytest.approx(-4.0)
