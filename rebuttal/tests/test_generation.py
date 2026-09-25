from __future__ import annotations

import torch

from qwen_rebuttal.scoring import BASELINE, generate_answers
from qwen_rebuttal.types import PromptExample


class GenTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [int(part) for part in text.split()]

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(str(int(token)) for token in tokens if int(token) != self.pad_token_id)


class ContinuationModel:
    """Continues each row from its own real tokens, ignoring any padding.

    Batched and unbatched calls must therefore agree, which is exactly the
    property batched generation has to preserve.
    """

    def generate(self, *, input_ids, attention_mask, max_new_tokens, **kwargs):
        rows = []
        for row in range(input_ids.shape[0]):
            real = input_ids[row][attention_mask[row].bool()]
            token = int(real.sum().item()) % 7 + 1
            rows.append(
                torch.cat(
                    [input_ids[row], torch.full((max_new_tokens,), token, dtype=input_ids.dtype)]
                )
            )
        return torch.stack(rows)


class GenAdapter:
    def __init__(self):
        self.tokenizer = GenTokenizer()
        self.model = ContinuationModel()
        self.layers = []
        self.device = torch.device("cpu")


def _prompt(uid: str, text: str) -> PromptExample:
    return PromptExample(
        uid=uid,
        template_id="test",
        cue_family="source",
        endorsement="wrong",
        position="after_options",
        rendered_text=text,
        cue_text=None,
        cue_char_start=None,
        cue_char_end=None,
        candidate_a="A",
        candidate_b="B",
        correct_label="B",
        endorsed_label="A",
    )


PROMPTS = [
    _prompt("a", "1 2 3"),
    _prompt("b", "4 5"),
    _prompt("c", "6 7 8 9"),
    _prompt("d", "2"),
]


def _texts(batch_size: int) -> dict[str, str]:
    rows = generate_answers(
        adapter=GenAdapter(),
        prompts=PROMPTS,
        intervention=BASELINE,
        run_id="run",
        split="test",
        max_new_tokens=3,
        batch_size=batch_size,
    )
    return {row.uid: row.generated_text for row in rows}


def test_batched_generation_matches_unbatched():
    # Prompts have different lengths, so batching forces left padding. The
    # decoded continuations must be unaffected by it.
    assert _texts(batch_size=4) == _texts(batch_size=1)
    assert _texts(batch_size=2) == _texts(batch_size=1)


def test_batched_generation_preserves_prompt_order_and_count():
    rows = generate_answers(
        adapter=GenAdapter(),
        prompts=PROMPTS,
        intervention=BASELINE,
        run_id="run",
        split="test",
        max_new_tokens=2,
        batch_size=3,
    )
    assert [row.uid for row in rows] == ["a", "b", "c", "d"]


def test_generated_text_excludes_the_prompt():
    rows = generate_answers(
        adapter=GenAdapter(),
        prompts=[_prompt("a", "1 2 3")],
        intervention=BASELINE,
        run_id="run",
        split="test",
        max_new_tokens=2,
        batch_size=1,
    )
    # sum(1,2,3) % 7 + 1 == 7, and only the two new tokens are returned.
    assert rows[0].generated_text == "7 7"
