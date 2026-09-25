from __future__ import annotations

import pytest
import torch

from qwen_rebuttal.directions import (
    collect_endpoint_activations,
    collect_span_activations,
)
from qwen_rebuttal.types import PromptExample

HIDDEN = 4


class SpaceTokenizer:
    """Renders "3 4 5" as token ids [3, 4, 5] so activations stay predictable."""

    pad_token_id = 0

    def __call__(self, texts, add_special_tokens=False, padding=True, return_tensors="pt"):
        assert padding is True
        sequences = [[int(part) for part in text.split()] for text in texts]
        width = max(len(sequence) for sequence in sequences)
        ids, masks = [], []
        for sequence in sequences:
            pad = width - len(sequence)
            ids.append([self.pad_token_id] * pad + sequence)
            masks.append([0] * pad + [1] * len(sequence))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


class Block(torch.nn.Module):
    def forward(self, hidden):
        return (hidden.clone(), "cache")


class TinyModel(torch.nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=1):
        hidden = input_ids.unsqueeze(-1).float().repeat(1, 1, HIDDEN)
        for block in self.blocks:
            hidden = block(hidden)[0]
        return hidden


class TinyAdapter:
    def __init__(self):
        self.tokenizer = SpaceTokenizer()
        self.layers = [Block()]
        self.model = TinyModel(self.layers)
        self.device = torch.device("cpu")


def _prompt(text: str, start: int | None, end: int | None) -> PromptExample:
    return PromptExample(
        uid="u",
        template_id="test",
        cue_family="source",
        endorsement="wrong",
        position="after_options",
        rendered_text=text,
        cue_text="cue",
        cue_char_start=0,
        cue_char_end=1,
        cue_token_start=start,
        cue_token_end=end,
    )


def test_endpoint_collection_offsets_for_left_padding():
    # "7 8 9" is three tokens and "5 6" is two, so the shorter row is left padded
    # by one. Endpoint index 1 must still land on that row's own second token.
    adapter = TinyAdapter()
    activations = collect_endpoint_activations(
        adapter,
        [_prompt("7 8 9", 0, 1), _prompt("5 6", 0, 1)],
        layers=(0,),
        batch_size=2,
    )
    rows = activations[0]
    assert rows.shape == (2, HIDDEN)
    assert torch.allclose(rows[0], torch.full((HIDDEN,), 8.0))
    assert torch.allclose(rows[1], torch.full((HIDDEN,), 6.0))


def test_span_collection_averages_the_cue_span_under_padding():
    adapter = TinyAdapter()
    activations = collect_span_activations(
        adapter,
        [_prompt("7 8 9", 0, 2), _prompt("5 6", 0, 1)],
        layers=(0,),
        batch_size=2,
    )
    rows = activations[0]
    assert rows.shape == (2, HIDDEN)
    assert torch.allclose(rows[0], torch.full((HIDDEN,), 8.0))  # mean(7,8,9)
    assert torch.allclose(rows[1], torch.full((HIDDEN,), 5.5))  # mean(5,6)


def test_span_collection_requires_a_verified_span():
    adapter = TinyAdapter()
    with pytest.raises(RuntimeError, match="verified cue spans"):
        collect_span_activations(
            adapter, [_prompt("7 8 9", None, 2)], layers=(0,), batch_size=1
        )
