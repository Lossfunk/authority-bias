from __future__ import annotations

import pytest
import torch

from qwen_rebuttal.caa import CAASelection
from qwen_rebuttal.paper_vectors import build_paper_vectors, load_assistant_axis
from qwen_rebuttal.pipeline import _head_to_head_catalog
from qwen_rebuttal.types import PromptExample

HIDDEN = 6
VOCAB = 20


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


class EmbeddingModel(torch.nn.Module):
    """Looks each token id up in a fixed generic table.

    The collinear "hidden = token id" model used elsewhere in these tests cannot
    exercise this module: every centroid difference would point the same way, so
    residualizing against the assistant axis would divide by zero and the two
    reference conditions would be indistinguishable.
    """

    def __init__(self, blocks):
        super().__init__()
        self.blocks = torch.nn.ModuleList(blocks)
        generator = torch.Generator().manual_seed(20260727)
        self.table = torch.randn(VOCAB, HIDDEN, generator=generator)

    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=1):
        hidden = self.table[input_ids]
        for block in self.blocks:
            hidden = block(hidden)[0]
        return hidden


class TinyAdapter:
    def __init__(self):
        self.tokenizer = SpaceTokenizer()
        self.layers = [Block()]
        self.model = EmbeddingModel(self.layers)
        self.device = torch.device("cpu")


def _prompt(text: str, family: str, start: int | None, end: int | None) -> PromptExample:
    return PromptExample(
        uid="u",
        template_id="test",
        cue_family=family,
        endorsement="wrong",
        position="after_options",
        rendered_text=text,
        cue_text="cue",
        cue_char_start=0,
        cue_char_end=1,
        cue_token_start=start,
        cue_token_end=end,
    )


def _axis_file(tmp_path, payload, name="axis.pt"):
    path = tmp_path / name
    torch.save(payload, path)
    return path


def test_load_assistant_axis_reads_every_shape_the_submission_stores(tmp_path):
    # Qwen stores a bare vector under "vector"; OLMo stores it under
    # "axis_layer"; GPT-OSS stores one row per layer in a 2-D tensor.
    raw = torch.arange(HIDDEN, dtype=torch.float32) + 1.0
    for key in ("vector", "axis_layer"):
        axis, declared = load_assistant_axis(
            _axis_file(tmp_path, {key: raw, "layer": 5}, f"{key}.pt"),
            layer=5,
            hidden_size=HIDDEN,
        )
        assert declared == 5
        assert torch.allclose(axis, raw / raw.norm())

    stack = torch.randn(8, HIDDEN, generator=torch.Generator().manual_seed(1))
    axis, declared = load_assistant_axis(
        _axis_file(tmp_path, stack, "stack.pt"), layer=3, hidden_size=HIDDEN
    )
    assert declared == 3
    assert torch.allclose(axis, stack[3] / stack[3].norm())


def test_load_assistant_axis_refuses_a_hidden_size_mismatch(tmp_path):
    path = _axis_file(tmp_path, torch.ones(HIDDEN + 1), "wrong.pt")
    with pytest.raises(RuntimeError, match="expected"):
        load_assistant_axis(path, layer=0, hidden_size=HIDDEN)


def _paper_vectors(tmp_path):
    adapter = TinyAdapter()
    axis_path = _axis_file(
        tmp_path, torch.randn(HIDDEN, generator=torch.Generator().manual_seed(7))
    )
    return adapter, build_paper_vectors(
        adapter=adapter,
        layer=0,
        source_prompts=[_prompt("3 4 5", "source", 0, 1), _prompt("6 7", "source", 0, 1)],
        user_prompts=[_prompt("8 9 10", "user", 0, 1), _prompt("11 12", "user", 0, 1)],
        neutral_prompts=[
            _prompt("13 14 15", "neutral", 0, 1),
            _prompt("16 17", "neutral", 0, 1),
        ],
        no_cue_prompts=[_prompt("18 19", "none", None, None), _prompt("2 1", "none", None, None)],
        assistant_axis_path=axis_path,
        batch_size=2,
    )


def test_the_two_references_produce_two_different_vectors(tmp_path):
    # The mitigation vector references a neutral cue and the causal-split vector
    # references no cue at all. Conflating them is exactly the error this module
    # exists to prevent, so they must not come out equal.
    adapter, vectors = _paper_vectors(tmp_path)
    table = adapter.model.table

    source_centroid = torch.stack(
        [table[[3, 4]].mean(dim=0), table[[6, 7]].mean(dim=0)]
    ).mean(dim=0)
    neutral_centroid = torch.stack(
        [table[[13, 14]].mean(dim=0), table[[16, 17]].mean(dim=0)]
    ).mean(dim=0)
    no_cue_centroid = torch.stack([table[19], table[1]]).mean(dim=0)

    expected_authority = source_centroid - neutral_centroid
    expected_source = source_centroid - no_cue_centroid
    assert torch.allclose(
        vectors.authority, expected_authority / expected_authority.norm(), atol=1e-6
    )
    assert torch.allclose(vectors.source, expected_source / expected_source.norm(), atol=1e-6)
    assert not torch.allclose(vectors.authority, vectors.source, atol=1e-3)


def test_every_returned_vector_is_a_unit_vector(tmp_path):
    _adapter, vectors = _paper_vectors(tmp_path)
    for name in ("authority", "assistant", "authority_resid", "source", "user"):
        norm = float(torch.linalg.vector_norm(getattr(vectors, name)))
        assert norm == pytest.approx(1.0, abs=1e-5), name


def test_residualized_authority_has_no_assistant_component(tmp_path):
    _adapter, vectors = _paper_vectors(tmp_path)
    assert float(torch.dot(vectors.authority_resid, vectors.assistant)) == pytest.approx(
        0.0, abs=1e-6
    )
    # Residualizing must actually change the vector, or the control is vacuous.
    assert abs(float(torch.dot(vectors.authority, vectors.assistant))) > 1e-3


def test_head_to_head_catalog_reproduces_the_submission_intervention_classes(tmp_path):
    _adapter, vectors = _paper_vectors(tmp_path)
    selection = CAASelection(
        layer=49, multiplier=-2.0, mean_anti_sycophancy_margin=0.5, n_tune=200
    )
    raw_caa = torch.randn(HIDDEN, generator=torch.Generator().manual_seed(11)) * 3.0
    catalog = {
        intervention.intervention_id: intervention
        for intervention, _cues in _head_to_head_catalog(vectors, selection, raw_caa)
    }
    assert len(catalog) == 8, "every condition must have a distinct id"

    # Authority removal is the submission's `project_out_direction` at alpha=1
    # over the cue span, which is its `endorsement_span` position mode.
    authority = catalog["h2h_auth_resid_remove_l0"]
    assert (authority.mode, authority.coefficient) == ("project_out", 1)
    assert (authority.token_scope, authority.phase) == ("cue_span", "prefill")

    # The submission's own CAA baseline is `subtract` at alpha=1 with
    # `--norm-scaling none`, i.e. h <- h - 1.0 * unit_vector, in the same slot.
    paper_caa = catalog["h2h_caa_paper_l0_a1"]
    assert (paper_caa.mode, paper_caa.coefficient) == ("add", -1.0)
    assert (paper_caa.token_scope, paper_caa.phase) == ("cue_span", "prefill")
    assert float(torch.linalg.vector_norm(paper_caa.vector)) == pytest.approx(1.0, abs=1e-5)

    # Native CAA steers the answer tokens with the raw, unnormalized
    # difference-of-means vector, so the tuned arm keeps that scale.
    tuned = catalog["h2h_caa_tuned_l49_m-2"]
    assert (tuned.mode, tuned.coefficient, tuned.layer) == ("add", -2.0, 49)
    assert (tuned.token_scope, tuned.phase) == ("answer_tokens", "all")
    assert torch.allclose(tuned.vector, raw_caa)

    # Both cue families are scored under every condition, so the source-vs-user
    # dissociation and the mitigation comparison come from one prompt list.
    for _intervention, cues in _head_to_head_catalog(vectors, selection, raw_caa):
        assert cues == ("source", "user")
