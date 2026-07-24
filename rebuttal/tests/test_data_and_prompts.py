from __future__ import annotations

from pathlib import Path

from qwen_rebuttal.data import freeze_splits, load_dataset
from qwen_rebuttal.io import load_config
from qwen_rebuttal.prompts import (
    apply_chat_and_locate_cue,
    build_prompt,
    paired_identity_difference_is_declared,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config(REPO_ROOT / "rebuttal/configs/qwen_h100.json")


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return f"<user>{messages[0]['content']}</user><assistant>"

    def __call__(self, text, **kwargs):
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def test_recorded_partitions_are_disjoint_deterministic_and_complete():
    first = freeze_splits(CONFIG, REPO_ROOT)
    second = freeze_splits(CONFIG, REPO_ROOT)
    assert first == second
    assert len(first.fit) == 256
    assert len(first.historical) == 256
    assert len(first.fresh) == 512
    assert len(first.unused) == 331
    assert not set(first.fit) & set(first.historical)
    assert not set(first.fit) & set(first.fresh)
    assert not set(first.historical) & set(first.fresh)


def test_matched_primary_prompts_only_change_declared_identity():
    dataset = load_dataset(REPO_ROOT / CONFIG.inputs.dataset)
    example = next(iter(dataset.values()))
    source = build_prompt(
        example,
        template_id="primary",
        cue_family="source",
        endorsement="wrong",
        position="after_options",
    )
    user = build_prompt(
        example,
        template_id="primary",
        cue_family="user",
        endorsement="wrong",
        position="after_options",
    )
    assert paired_identity_difference_is_declared(source, user)


def test_cue_span_is_uniquely_mapped_after_chat_template():
    dataset = load_dataset(REPO_ROOT / CONFIG.inputs.dataset)
    prompt = build_prompt(
        next(iter(dataset.values())),
        template_id="primary",
        cue_family="source",
        endorsement="wrong",
        position="after_options",
    )
    located = apply_chat_and_locate_cue(CharacterTokenizer(), prompt)
    assert located.cue_token_start == located.cue_char_start
    assert located.cue_token_end == located.cue_char_end - 1
