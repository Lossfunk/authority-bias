from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd
import torch

from src.exp7.scoring import _tokenize_for_scoring
from src.mechanism.common import (
    ActivationManifestRow,
    PanelRow,
    ensure_dir,
    model_tag,
    prompt_hash,
    read_panel_parquet,
    stable_run_id,
)
from src.mechanism.hooks import COMPONENT_ORDER, ModelHookRunner
from src.models.llama_loader import load_model_and_tokenizer


DEFAULT_POSITION_LABELS: Tuple[str, ...] = ("decision", "endorsement")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect mechanistic activations into sharded caches")
    parser.add_argument("--panel-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--components", type=str, default=",".join(COMPONENT_ORDER))
    parser.add_argument("--positions", type=str, default=",".join(DEFAULT_POSITION_LABELS))
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--slice", type=str, default="")
    return parser.parse_args()


def _parse_csv(raw: str) -> List[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _find_substring_span(prompt_text: str, needle: str) -> Tuple[int, int]:
    start = prompt_text.rfind(needle)
    if start < 0:
        raise ValueError(f"Could not find substring {needle!r} in prompt")
    return start, start + len(needle)


def _token_positions_for_span(offsets: Sequence[Tuple[int, int]], span: Tuple[int, int]) -> List[int]:
    start, end = span
    positions: List[int] = []
    for token_idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= tok_start:
            continue
        if tok_end <= start or tok_start >= end:
            continue
        positions.append(token_idx)
    return positions


def extract_token_positions(tokenizer, row: PanelRow) -> Dict[str, List[int]]:
    encoded = tokenizer(row.prompt_text, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"]
    seq_len = int(input_ids.shape[1])
    if tokenizer.eos_token_id is not None and int(input_ids[0, -1]) == tokenizer.eos_token_id and seq_len > 1:
        seq_len -= 1
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"][0].tolist()[:seq_len]]
    positions: Dict[str, List[int]] = {"decision": [seq_len - 1]}

    if row.evidence_direction == "neutral":
        positions["endorsement"] = [seq_len - 1]
        return positions

    if row.answer_format == "ab":
        endorsed = row.correct_label if row.evidence_direction == "corrective" else row.wrong_label
    else:
        endorsed = row.correct_answer if row.evidence_direction == "corrective" else row.wrong_answer
    try:
        cue = f"I think it's {endorsed}"
        if cue in row.prompt_text:
            span = _find_substring_span(row.prompt_text, cue)
            span = (span[0] + len("I think it's "), span[1])
        else:
            span = _find_substring_span(row.prompt_text, endorsed)
        token_positions = _token_positions_for_span(offsets, span)
        positions["endorsement"] = token_positions or [seq_len - 1]
    except ValueError:
        positions["endorsement"] = [seq_len - 1]
    return positions


def collect_activations(
    *,
    panel_rows: Sequence[PanelRow],
    output_dir: Path,
    model_name: str,
    components: Sequence[str],
    position_labels: Sequence[str],
    shard_size: int,
    seed: int,
    device: str,
    dtype: str,
) -> Dict[str, str]:
    output_dir = ensure_dir(output_dir)
    model, tokenizer = load_model_and_tokenizer(model_name, device=device, dtype=dtype)
    runner = ModelHookRunner(model)

    run_id = stable_run_id(
        "collect",
        {
            "model": model_name,
            "panel_hash": hashlib.sha256("\n".join(f"{row.uid}:{row.condition_id}" for row in panel_rows).encode("utf-8")).hexdigest(),
            "components": list(components),
            "positions": list(position_labels),
            "seed": seed,
        },
    )

    manifest_rows: List[ActivationManifestRow] = []
    shard_dir = ensure_dir(output_dir / f"{run_id}_shards")
    shard_records: List[torch.Tensor] = []
    shard_meta: List[PanelRow] = []
    shard_index = 0

    def flush_shard() -> None:
        nonlocal shard_records, shard_meta, shard_index
        if not shard_records:
            return
        shard_path = shard_dir / f"shard_{shard_index:04d}.pt"
        tensor = torch.stack(shard_records, dim=0).to(torch.float16).cpu()
        payload = {
            "activations": tensor,
            "example_ids": [row.uid for row in shard_meta],
            "condition_ids": [row.condition_id for row in shard_meta],
            "components": list(components),
            "positions": list(position_labels),
        }
        torch.save(payload, shard_path)
        for local_idx, row in enumerate(shard_meta):
            manifest_rows.append(
                ActivationManifestRow(
                    run_id=run_id,
                    model=model_name,
                    example_id=row.uid,
                    condition_id=row.condition_id,
                    source_or_base="base",
                    shard_path=str(shard_path),
                    shard_index=local_idx,
                    n_layers=tensor.shape[1],
                    components=tuple(components),
                    positions=tuple(position_labels),
                    dtype="float16",
                    hidden_size=tensor.shape[-1],
                    prompt_hash=prompt_hash(row.prompt_text),
                    seed=seed,
                )
            )
        shard_records = []
        shard_meta = []
        shard_index += 1

    for row in panel_rows:
        model_inputs = {
            key: value.to(next(model.parameters()).device)
            for key, value in _tokenize_for_scoring(tokenizer, row.prompt_text).items()
        }
        raw_positions = extract_token_positions(tokenizer, row)
        token_positions = {label: raw_positions[label] for label in position_labels if label in raw_positions}
        activations = runner.collect(model_inputs=model_inputs, token_positions=token_positions, components=components)
        shard_records.append(activations)
        shard_meta.append(row)
        if len(shard_records) >= shard_size:
            flush_shard()
    flush_shard()

    manifest_path = output_dir / f"{run_id}_manifest.parquet"
    pd.DataFrame([row.to_record() for row in manifest_rows]).to_parquet(manifest_path, index=False)
    summary_path = output_dir / f"{run_id}_summary.json"
    summary = {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "n_examples": len(panel_rows),
        "n_manifest_rows": len(manifest_rows),
        "n_shards": shard_index,
        "components": list(components),
        "positions": list(position_labels),
        "model": model_name,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def main() -> None:
    args = parse_args()
    panel_rows = read_panel_parquet(args.panel_path)
    if args.slice:
        allowed = {token.strip() for token in args.slice.split(",") if token.strip()}
        panel_rows = [row for row in panel_rows if row.confidence_slice in allowed]
    summary = collect_activations(
        panel_rows=panel_rows,
        output_dir=args.output_dir,
        model_name=args.model,
        components=_parse_csv(args.components),
        position_labels=_parse_csv(args.positions),
        shard_size=args.shard_size,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
