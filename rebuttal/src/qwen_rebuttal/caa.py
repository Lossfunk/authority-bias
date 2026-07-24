from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .hooks import MultiResidualHooks, ResidualHook
from .io import canonical_json, hash_lines, sha256_text
from .model import QwenAdapter
from .scoring import RuntimeIntervention, score_margins
from .types import CAASpec, PromptExample


@dataclass(frozen=True)
class CAAExample:
    uid: str
    question: str
    matching: str
    not_matching: str

    @property
    def matching_label(self) -> str:
        return self.matching.strip("() ")

    @property
    def not_matching_label(self) -> str:
        return self.not_matching.strip("() ")


@dataclass(frozen=True)
class FrozenCAA:
    fit: tuple[CAAExample, ...]
    tune: tuple[CAAExample, ...]
    source_sha256: str


@dataclass(frozen=True)
class CAASelection:
    layer: int
    multiplier: float
    mean_anti_sycophancy_margin: float
    n_tune: int


def ensure_caa_checkout(spec: CAASpec, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        subprocess.run(
            ["git", "clone", "--no-checkout", spec.repository, str(destination)],
            check=True,
        )
    if not (destination / ".git").exists():
        raise RuntimeError(f"CAA destination is not a Git checkout: {destination}")
    subprocess.run(["git", "fetch", "--quiet", "origin", spec.commit], cwd=destination, check=True)
    subprocess.run(
        ["git", "checkout", "--quiet", "--detach", spec.commit],
        cwd=destination,
        check=True,
    )
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=destination, text=True
    ).strip()
    if actual != spec.commit:
        raise RuntimeError(f"CAA checkout is {actual}, expected {spec.commit}")
    return destination


def freeze_caa_data(spec: CAASpec, checkout: Path) -> FrozenCAA:
    source = checkout / spec.dataset_path
    raw_bytes = source.read_bytes()
    actual_source_hash = hashlib.sha256(raw_bytes).hexdigest()
    if actual_source_hash != spec.source_sha256:
        raise RuntimeError(
            f"CAA dataset SHA-256 mismatch: {actual_source_hash} != {spec.source_sha256}"
        )
    raw = json.loads(raw_bytes)
    if len(raw) != spec.fit_count + spec.tune_count:
        raise RuntimeError(
            f"Expected {spec.fit_count + spec.tune_count} CAA pairs, found {len(raw)}"
        )
    examples: list[CAAExample] = []
    for row in raw:
        uid = sha256_text(canonical_json(row))[:24]
        example = CAAExample(
            uid=uid,
            question=row["question"],
            matching=row["answer_matching_behavior"],
            not_matching=row["answer_not_matching_behavior"],
        )
        if {example.matching_label, example.not_matching_label} != {"A", "B"}:
            raise RuntimeError(f"CAA row {uid} does not contain an A/B contrast")
        examples.append(example)
    ordered = sorted(
        examples,
        key=lambda item: (
            hashlib.sha256(f"{spec.split_seed}:{item.uid}".encode()).hexdigest(),
            item.uid,
        ),
    )
    fit = tuple(ordered[: spec.fit_count])
    tune = tuple(ordered[spec.fit_count :])
    fit_hash = hash_lines(item.uid for item in fit)
    tune_hash = hash_lines(item.uid for item in tune)
    if fit_hash != spec.fit_uids_sha256 or tune_hash != spec.tune_uids_sha256:
        raise RuntimeError("CAA deterministic split hashes do not match the frozen specification")
    return FrozenCAA(
        fit=fit,
        tune=tune,
        source_sha256=actual_source_hash,
    )


def _full_assistant_render(
    tokenizer: Any, question: str, answer: str
) -> tuple[str, int]:
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    answer_start = rendered.rfind(answer)
    if answer_start < 0:
        raise RuntimeError("CAA answer is absent from the rendered chat")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    answer_end = answer_start + len(answer)
    covered = [
        index
        for index, (start, end) in enumerate(encoded["offset_mapping"])
        if end > answer_start and start < answer_end
    ]
    if not covered:
        raise RuntimeError("CAA answer has no token span")
    return rendered, covered[-1]


def _collect_caa(
    adapter: QwenAdapter,
    examples: tuple[CAAExample, ...],
    *,
    answer_attribute: str,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    layers = tuple(range(adapter.spec.expected_layers))
    all_rows: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    tokenizer = adapter.tokenizer
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        rendered_and_positions = [
            _full_assistant_render(tokenizer, item.question, getattr(item, answer_attribute))
            for item in batch
        ]
        texts = [item[0] for item in rendered_and_positions]
        raw_positions = [item[1] for item in rendered_and_positions]
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(adapter.device)
        mask = encoded["attention_mask"].to(adapter.device)
        positions = [
            ids.shape[1] - int(row_mask.sum()) + raw_position
            for row_mask, raw_position in zip(mask, raw_positions, strict=True)
        ]
        hooks = [
            ResidualHook(
                module=adapter.layers[layer],
                layer=layer,
                mode="collect",
                phase="prefill",
                token_positions=positions,
            )
            for layer in layers
        ]
        with torch.inference_mode(), MultiResidualHooks(hooks):
            adapter.model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=False,
                logits_to_keep=1,
            )
        for layer, hook in zip(layers, hooks, strict=True):
            all_rows[layer].extend(hook.collected)
    return {layer: torch.stack(values) for layer, values in all_rows.items()}


def fit_caa_directions(
    adapter: QwenAdapter, frozen: FrozenCAA, batch_size: int
) -> dict[int, torch.Tensor]:
    matching = _collect_caa(
        adapter, frozen.fit, answer_attribute="matching", batch_size=batch_size
    )
    not_matching = _collect_caa(
        adapter, frozen.fit, answer_attribute="not_matching", batch_size=batch_size
    )
    return {
        layer: (matching[layer] - not_matching[layer]).mean(dim=0)
        for layer in range(adapter.spec.expected_layers)
    }


def caa_tuning_prompts(
    adapter: QwenAdapter, examples: tuple[CAAExample, ...]
) -> list[PromptExample]:
    prompts: list[PromptExample] = []
    for item in examples:
        rendered = adapter.tokenizer.apply_chat_template(
            [{"role": "user", "content": item.question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(
            PromptExample(
                uid=item.uid,
                template_id="caa_native",
                cue_family="none",
                endorsement="none",
                position="after_options",
                rendered_text=rendered,
                cue_text=None,
                cue_char_start=None,
                cue_char_end=None,
                candidate_a="(A)",
                candidate_b="(B)",
                correct_label="",
                endorsed_label=item.not_matching_label,
            )
        )
    return prompts


def tune_caa(
    *,
    adapter: QwenAdapter,
    frozen: FrozenCAA,
    directions: dict[int, torch.Tensor],
    multipliers: tuple[float, ...],
    batch_size: int,
    run_id: str,
) -> tuple[CAASelection, list[dict[str, float | int]]]:
    prompts = caa_tuning_prompts(adapter, frozen.tune)
    candidates: list[dict[str, float | int]] = []
    for layer in range(adapter.spec.expected_layers):
        for multiplier in multipliers:
            intervention = RuntimeIntervention(
                intervention_id=f"caa_tune_l{layer}_m{multiplier:g}",
                layer=layer,
                mode="add",
                vector=directions[layer],
                coefficient=multiplier,
                token_scope="answer_tokens",
                phase="all",
                direction_id=f"caa_l{layer}",
            )
            rows = score_margins(
                adapter=adapter,
                prompts=prompts,
                intervention=intervention,
                run_id=run_id,
                split="caa_tune",
                batch_size=batch_size,
            )
            mean_margin = sum(float(row.compliance_margin) for row in rows) / len(rows)
            candidates.append(
                {
                    "layer": layer,
                    "multiplier": multiplier,
                    "mean_anti_sycophancy_margin": mean_margin,
                }
            )
    best = min(
        candidates,
        key=lambda item: (
            -float(item["mean_anti_sycophancy_margin"]),
            abs(float(item["multiplier"])),
            int(item["layer"]),
        ),
    )
    return (
        CAASelection(
            layer=int(best["layer"]),
            multiplier=float(best["multiplier"]),
            mean_anti_sycophancy_margin=float(best["mean_anti_sycophancy_margin"]),
            n_tune=len(prompts),
        ),
        candidates,
    )
