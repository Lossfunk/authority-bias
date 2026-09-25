from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F

from .hooks import ResidualHook
from .io import canonical_json, sha256_text
from .model import QwenAdapter
from .types import PromptExample, RawResultRow


@dataclass(frozen=True)
class RuntimeIntervention:
    intervention_id: str
    layer: int | None
    mode: Literal["none", "add", "project_out"]
    vector: torch.Tensor | None
    coefficient: float
    token_scope: Literal["none", "cue_endpoint", "cue_span", "answer_tokens"]
    phase: Literal["prefill", "answer", "all"]
    direction_id: str | None = None
    seed: int | None = None


BASELINE = RuntimeIntervention(
    intervention_id="baseline",
    layer=None,
    mode="none",
    vector=None,
    coefficient=0.0,
    token_scope="none",
    phase="prefill",
)


def _row_key(payload: dict[str, Any]) -> str:
    return sha256_text(canonical_json(payload))[:32]


def _candidate_ids(tokenizer: Any, label: str) -> list[int]:
    values = tokenizer.encode(label, add_special_tokens=False)
    if not values:
        raise RuntimeError(f"Candidate {label!r} tokenizes to an empty sequence")
    return values


def _positions_for_scope(
    intervention: RuntimeIntervention,
    prompt: PromptExample,
    *,
    left_padding: int,
    prompt_length: int,
    sequence_length: int,
) -> int | list[int]:
    if intervention.token_scope == "cue_endpoint":
        if prompt.cue_token_end is None:
            raise RuntimeError("Cue-endpoint intervention requested for a prompt without a cue")
        return left_padding + prompt.cue_token_end
    if intervention.token_scope == "cue_span":
        if prompt.cue_token_start is None or prompt.cue_token_end is None:
            raise RuntimeError("Cue-span intervention requested for a prompt without a cue")
        return list(
            range(
                left_padding + prompt.cue_token_start,
                left_padding + prompt.cue_token_end + 1,
            )
        )
    if intervention.token_scope == "answer_tokens":
        return list(
            range(
                left_padding + prompt_length - 1,
                left_padding + sequence_length - 1,
            )
        )
    raise RuntimeError(f"No positions defined for scope {intervention.token_scope}")


def score_margins(
    *,
    adapter: QwenAdapter,
    prompts: list[PromptExample],
    intervention: RuntimeIntervention,
    run_id: str,
    split: str,
    batch_size: int,
) -> list[RawResultRow]:
    tokenizer = adapter.tokenizer
    records: list[RawResultRow] = []
    for batch_start in range(0, len(prompts), batch_size):
        prompt_batch = prompts[batch_start : batch_start + batch_size]
        flattened: list[tuple[int, PromptExample, str, list[int], list[int]]] = []
        for prompt_index, prompt in enumerate(prompt_batch):
            prompt_ids = tokenizer.encode(prompt.rendered_text, add_special_tokens=False)
            for label, candidate_text in (
                ("A", prompt.candidate_a),
                ("B", prompt.candidate_b),
            ):
                candidate = _candidate_ids(tokenizer, candidate_text)
                flattened.append(
                    (prompt_index, prompt, label, prompt_ids, prompt_ids + candidate)
                )
        max_length = max(len(item[4]) for item in flattened)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError("Tokenizer has no padding token")
        ids: list[list[int]] = []
        masks: list[list[int]] = []
        hook_positions: list[int | list[int]] = []
        for _prompt_index, prompt, _label, prompt_ids, sequence in flattened:
            padding = max_length - len(sequence)
            ids.append([pad_id] * padding + sequence)
            masks.append([0] * padding + [1] * len(sequence))
            if intervention.mode != "none":
                hook_positions.append(
                    _positions_for_scope(
                        intervention,
                        prompt,
                        left_padding=padding,
                        prompt_length=len(prompt_ids),
                        sequence_length=len(sequence),
                    )
                )
        input_ids = torch.tensor(ids, device=adapter.device)
        attention_mask = torch.tensor(masks, device=adapter.device)
        maximum_candidate_length = max(
            len(sequence) - len(prompt_ids)
            for _prompt_index, _prompt, _label, prompt_ids, sequence in flattened
        )
        logits_to_keep = maximum_candidate_length + 1
        hook: ResidualHook | None = None
        if intervention.mode != "none":
            if intervention.layer is None or intervention.vector is None:
                raise RuntimeError("Active intervention lacks a layer or vector")
            hook = ResidualHook(
                module=adapter.layers[intervention.layer],
                layer=intervention.layer,
                mode=intervention.mode,
                vector=intervention.vector,
                coefficient=intervention.coefficient,
                phase=intervention.phase,
                token_positions=hook_positions,
            )
        with torch.inference_mode():
            if hook is not None:
                hook.__enter__()
            try:
                logits = adapter.scoring_logits(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=logits_to_keep,
                )
            finally:
                if hook is not None:
                    hook.__exit__(None, None, None)
        log_probs = F.log_softmax(logits, dim=-1)
        scores: dict[tuple[int, str], float] = {}
        for row, (prompt_index, _prompt, label, prompt_ids, sequence) in enumerate(
            flattened
        ):
            padding = max_length - len(sequence)
            candidate_length = len(sequence) - len(prompt_ids)
            first_target = padding + len(prompt_ids)
            positions = torch.arange(
                first_target - 1 - (max_length - logits_to_keep),
                first_target - 1 - (max_length - logits_to_keep) + candidate_length,
                device=adapter.device,
            )
            targets = input_ids[row, first_target : first_target + candidate_length]
            score = log_probs[row, positions, targets].sum().item()
            scores[(prompt_index, label)] = score
        for prompt_index, prompt in enumerate(prompt_batch):
            endorsed = prompt.endorsed_label
            other = "B" if endorsed == "A" else "A"
            payload = {
                "run_id": run_id,
                "split": split,
                "uid": prompt.uid,
                "measurement": "margin",
                "template_id": prompt.template_id,
                "cue_family": prompt.cue_family,
                "endorsement": prompt.endorsement,
                "position": prompt.position,
                "layer": intervention.layer,
                "intervention_id": intervention.intervention_id,
                "direction_id": intervention.direction_id,
                "seed": intervention.seed,
            }
            logp_endorsed = scores[(prompt_index, endorsed)]
            logp_other = scores[(prompt_index, other)]
            records.append(
                RawResultRow(
                    row_key=_row_key(payload),
                    **payload,
                    endorsed_label=endorsed,
                    other_label=other,
                    logp_endorsed=logp_endorsed,
                    logp_other=logp_other,
                    compliance_margin=logp_endorsed - logp_other,
                )
            )
    return records


_LABEL_PATTERN = re.compile(
    r"^\s*(?:the\s+answer\s+is\s+|answer\s*:\s*|option\s+)?([AB])(?:\b|[).,:])",
    flags=re.IGNORECASE,
)


def parse_generated_label(text: str) -> tuple[str | None, str]:
    match = _LABEL_PATTERN.search(text)
    if match is None:
        return None, "unparsed"
    return match.group(1).upper(), "parsed"


def generate_answers(
    *,
    adapter: QwenAdapter,
    prompts: list[PromptExample],
    intervention: RuntimeIntervention,
    run_id: str,
    split: str,
    max_new_tokens: int,
    batch_size: int = 1,
) -> list[RawResultRow]:
    """Greedily generate answers, batching prompts to keep the GPU busy.

    One prompt per call leaves decoding memory-bandwidth bound and wastes most
    of the device. Batching left-pads, so each row's hook positions are offset
    by that row's own padding, and new tokens are sliced from the shared padded
    width rather than from each prompt's unpadded length.
    """
    rows: list[RawResultRow] = []
    tokenizer = adapter.tokenizer
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("Tokenizer has no padding token")
    for batch_start in range(0, len(prompts), batch_size):
        batch = prompts[batch_start : batch_start + batch_size]
        encoded = [
            tokenizer.encode(prompt.rendered_text, add_special_tokens=False)
            for prompt in batch
        ]
        width = max(len(sequence) for sequence in encoded)
        ids: list[list[int]] = []
        masks: list[list[int]] = []
        paddings: list[int] = []
        for sequence in encoded:
            padding = width - len(sequence)
            paddings.append(padding)
            ids.append([pad_id] * padding + sequence)
            masks.append([0] * padding + [1] * len(sequence))
        input_ids = torch.tensor(ids, device=adapter.device)
        attention_mask = torch.tensor(masks, device=adapter.device)
        hook: ResidualHook | None = None
        if intervention.mode != "none":
            if intervention.layer is None or intervention.vector is None:
                raise RuntimeError("Active intervention lacks a layer or vector")
            positions: list[int | list[int]] | None
            if intervention.token_scope == "answer_tokens":
                positions = None
            else:
                positions = [
                    _positions_for_scope(
                        intervention,
                        prompt,
                        left_padding=padding,
                        prompt_length=width,
                        sequence_length=width,
                    )
                    for prompt, padding in zip(batch, paddings, strict=True)
                ]
            hook = ResidualHook(
                module=adapter.layers[intervention.layer],
                layer=intervention.layer,
                mode=intervention.mode,
                vector=intervention.vector,
                coefficient=intervention.coefficient,
                phase=intervention.phase,
                token_positions=positions,
            )
            hook.__enter__()
        try:
            with torch.inference_mode():
                generated = adapter.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=pad_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        finally:
            if hook is not None:
                hook.__exit__(None, None, None)
        for row, prompt in enumerate(batch):
            new_tokens = generated[row, width:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            parsed, status = parse_generated_label(text)
            endorsed = prompt.endorsed_label
            other = "B" if endorsed == "A" else "A"
            payload = {
                "run_id": run_id,
                "split": split,
                "uid": prompt.uid,
                "measurement": "generation",
                "template_id": prompt.template_id,
                "cue_family": prompt.cue_family,
                "endorsement": prompt.endorsement,
                "position": prompt.position,
                "layer": intervention.layer,
                "intervention_id": intervention.intervention_id,
                "direction_id": intervention.direction_id,
                "seed": intervention.seed,
            }
            rows.append(
                RawResultRow(
                    row_key=_row_key(payload),
                    **payload,
                    endorsed_label=endorsed,
                    other_label=other,
                    generated_text=text,
                    parsed_label=parsed,
                    parse_status=status,
                )
            )
    return rows
