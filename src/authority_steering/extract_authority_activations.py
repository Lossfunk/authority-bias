from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from src.authority_steering.dynamic_parser import discover_model_markers, dynamic_parse_row
from src.lexical_controls.scoring import _tokenize_for_scoring, locate_prompt_text_span
from src.mechanism.hooks import get_transformer_layers
from src.mechanism.hooks_v2 import SelectiveLayerCapture, setup_h100_optimizations, try_compile_model
from src.models.llama_loader import load_model_and_tokenizer


DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_MC_DATASET = Path("data/exp7_mc_dataset.jsonl")
DEFAULT_V2_RESULTS = Path("results/authority/v2_gpt_oss_freegen/openai__gpt-oss-20b_dissociation_rows.jsonl")
DEFAULT_WEAK_RESULTS = Path("results/authority/gradient_gpt_oss_weak/openai__gpt-oss-20b_dissociation_rows.jsonl")
DEFAULT_UNCERTAIN_RESULTS = Path("results/authority/gradient_gpt_oss_uncertain/openai__gpt-oss-20b_dissociation_rows.jsonl")
DEFAULT_ASSERTIVE_RESULTS = Path("results/authority/gradient_gpt_oss_assertive/openai__gpt-oss-20b_dissociation_rows.jsonl")
DEFAULT_THINK_RESULTS = Path("results/authority/gpt_oss_freegen_scaled/openai__gpt-oss-20b_dissociation_rows.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/authority/mechanism/gpt_oss_authority_activations")

STYLE_ORDER = (
    "authoritative_verified",
    "weak",
    "uncertain",
    "assertive",
    "think",
)
REQUIRED_STYLES = ("authoritative_verified", "weak", "uncertain", "assertive")
STYLE_PATH_KEYS = {
    "authoritative_verified": "authoritative_results_path",
    "weak": "weak_results_path",
    "uncertain": "uncertain_results_path",
    "assertive": "assertive_results_path",
    "think": "think_results_path",
}

CONDITION_CODES = ("N0_note", "C1_note", "W1_note")
POINT_POSITION_NAMES = ("endorsement_start", "endorsed_answer", "endorsement_end", "answer_position")
ALL_POSITION_NAMES = (*POINT_POSITION_NAMES, "endorsement_mean")

NOTE_LINE_RE = re.compile(r"^Note:.*$", re.MULTILINE)


@dataclass(frozen=True)
class Record:
    uid: str
    style: str
    condition_code: str
    prompt_text: str
    correct_label: str
    wrong_label: str
    final_answer_label: Optional[str]
    parse_valid: bool
    final_is_correct: Optional[bool]
    w1_flip: Optional[bool]
    c1_correction: Optional[bool]
    n0_correct: Optional[bool]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract authority-conditioned activations for exp16 free-generation runs.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--mc-dataset-path", type=Path, default=DEFAULT_MC_DATASET)
    p.add_argument("--v2-results-path", type=Path, default=DEFAULT_V2_RESULTS)
    p.add_argument("--weak-results-path", type=Path, default=DEFAULT_WEAK_RESULTS)
    p.add_argument("--uncertain-results-path", type=Path, default=DEFAULT_UNCERTAIN_RESULTS)
    p.add_argument("--assertive-results-path", type=Path, default=DEFAULT_ASSERTIVE_RESULTS)
    p.add_argument("--think-results-path", type=Path, default=DEFAULT_THINK_RESULTS)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--arm", type=str, default="natural")
    p.add_argument("--target-layers", nargs="+", type=int, default=[2, 5, 10, 15, 18, 20, 22, 24, 26, 28, 30, 32, 35, 38])
    p.add_argument("--max-batch-tokens", type=int, default=8192)
    p.add_argument("--fallback-batch-size", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--loader-dtype", type=str, default="bfloat16")
    p.add_argument("--save-dtype", type=str, default="float16")
    p.add_argument("--uids-file", type=Path, default=None)
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _normalize_label(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"A", "B"}:
        return text
    return None


def _dtype_from_arg(raw: str) -> torch.dtype:
    val = getattr(torch, raw, None)
    if not isinstance(val, torch.dtype):
        raise ValueError(f"Unknown dtype: {raw}")
    return val


def _subsequence_span(haystack: Sequence[int], needle: Sequence[int]) -> Optional[Tuple[int, int]]:
    if not needle or len(needle) > len(haystack):
        return None
    last = len(haystack) - len(needle) + 1
    for start in range(last):
        if list(haystack[start : start + len(needle)]) == list(needle):
            return start, start + len(needle) - 1
    return None


def _load_style_rows(
    path: Path,
    *,
    style: str,
    arm: str,
) -> Dict[str, Dict[str, Record]]:
    filtered_rows: List[Dict] = []
    for row in _iter_jsonl(path):
        if row.get("arm") != arm:
            continue
        condition_code = row.get("condition_code")
        if condition_code not in CONDITION_CODES:
            continue
        filtered_rows.append(row)

    markers = discover_model_markers(filtered_rows)
    by_uid: Dict[str, Dict[str, Record]] = defaultdict(dict)
    for row in filtered_rows:
        condition_code = row["condition_code"]
        final_answer_label, _ = dynamic_parse_row(row, markers)
        parse_valid = final_answer_label in {"A", "B"}

        correct_label = str(row["correct_label"]).strip().upper()
        wrong_label = str(row["wrong_label"]).strip().upper()

        final_is_correct: Optional[bool] = None
        if parse_valid:
            final_is_correct = final_answer_label == correct_label

        w1_flip: Optional[bool] = None
        c1_correction: Optional[bool] = None
        n0_correct: Optional[bool] = None
        if condition_code == "W1_note" and parse_valid:
            w1_flip = final_answer_label == wrong_label
        if condition_code == "C1_note" and parse_valid:
            c1_correction = final_answer_label == correct_label
        if condition_code == "N0_note" and parse_valid:
            n0_correct = final_answer_label == correct_label

        rec = Record(
            uid=row["uid"],
            style=style,
            condition_code=condition_code,
            prompt_text=row["prompt_text"],
            correct_label=correct_label,
            wrong_label=wrong_label,
            final_answer_label=final_answer_label,
            parse_valid=parse_valid,
            final_is_correct=final_is_correct,
            w1_flip=w1_flip,
            c1_correction=c1_correction,
            n0_correct=n0_correct,
        )
        by_uid[row["uid"]][condition_code] = rec
    return by_uid


def _load_all_styles(args: argparse.Namespace) -> Tuple[Dict[str, Dict[str, Dict[str, Record]]], List[str]]:
    paths = {
        "authoritative_verified": args.v2_results_path,
        "weak": args.weak_results_path,
        "uncertain": args.uncertain_results_path,
        "assertive": args.assertive_results_path,
        "think": args.think_results_path,
    }
    out: Dict[str, Dict[str, Dict[str, Record]]] = {}
    included_styles: List[str] = []
    for style in STYLE_ORDER:
        path = paths[style]
        if not path.exists():
            if style in REQUIRED_STYLES:
                raise FileNotFoundError(f"Missing rows file for style={style}: {path}")
            print(f"[extract_authority_activations] Optional style '{style}' not found at {path}; skipping.")
            continue
        rows = _load_style_rows(path, style=style, arm=args.arm)
        out[style] = rows
        included_styles.append(style)
    return out, included_styles


def _style_uid_set(rows_by_style: Dict[str, Dict[str, Dict[str, Record]]], style: str) -> set[str]:
    return set(rows_by_style[style].keys())


def _require_condition_triplet(rows_by_style: Dict[str, Dict[str, Dict[str, Record]]], uid: str, style: str) -> bool:
    conds = rows_by_style[style].get(uid, {})
    return all(code in conds for code in CONDITION_CODES)


def _build_outcomes(rows_by_style: Dict[str, Dict[str, Dict[str, Record]]], uid: str, style: str) -> Dict[str, object]:
    conds = rows_by_style[style][uid]
    n0 = conds["N0_note"]
    c1 = conds["C1_note"]
    w1 = conds["W1_note"]
    return {
        "n0_parse_valid": n0.parse_valid,
        "c1_parse_valid": c1.parse_valid,
        "w1_parse_valid": w1.parse_valid,
        "all3_parse_valid": bool(n0.parse_valid and c1.parse_valid and w1.parse_valid),
        "n0_correct": n0.n0_correct,
        "c1_correction": c1.c1_correction,
        "w1_flip": w1.w1_flip,
    }


def _build_label_masks(
    selected_uids: List[str],
    rows_by_style: Dict[str, Dict[str, Dict[str, Record]]],
    included_styles: Sequence[str],
) -> Dict[str, object]:
    outcomes: Dict[str, Dict[str, Dict[str, object]]] = {}
    for uid in selected_uids:
        outcomes[uid] = {}
        for style in included_styles:
            if not _require_condition_triplet(rows_by_style, uid, style):
                continue
            outcomes[uid][style] = _build_outcomes(rows_by_style, uid, style)

    def _all3(uid: str, style: str) -> bool:
        return bool(outcomes.get(uid, {}).get(style, {}).get("all3_parse_valid", False))

    def _n0_correct(uid: str, style: str) -> bool:
        return bool(outcomes.get(uid, {}).get(style, {}).get("n0_correct", False))

    primary_all3 = [uid for uid in selected_uids if _all3(uid, "authoritative_verified")]
    primary_w1 = [uid for uid in primary_all3 if _n0_correct(uid, "authoritative_verified")]
    primary_n0_incorrect = [uid for uid in primary_all3 if not _n0_correct(uid, "authoritative_verified")]
    primary_c1 = list(primary_n0_incorrect) if len(primary_n0_incorrect) >= 50 else list(primary_all3)

    non_think_styles = ("authoritative_verified", "weak", "uncertain", "assertive")
    cross_non_think = [
        uid for uid in selected_uids
        if all(_all3(uid, style) for style in non_think_styles)
    ]
    cross_non_think_w1 = [uid for uid in cross_non_think if _n0_correct(uid, "authoritative_verified")]
    cross_non_think_c1 = [uid for uid in cross_non_think if not _n0_correct(uid, "authoritative_verified")]
    if len(cross_non_think_c1) < 50:
        cross_non_think_c1 = list(cross_non_think)

    cross_all5: List[str] = []
    if "think" in included_styles:
        cross_all5 = [
            uid for uid in selected_uids
            if all(_all3(uid, style) for style in included_styles)
        ]

    mask = {
        "mask_primary": primary_all3,
        "mask_primary_w1": primary_w1,
        "mask_primary_c1": primary_c1,
        "mask_cross": cross_non_think,
        "mask_cross_w1": cross_non_think_w1,
        "mask_cross_c1": cross_non_think_c1,
        "mask_think": cross_all5,
        "outcomes_by_uid_style": outcomes,
        "counts": {
            "selected_uids": len(selected_uids),
            "mask_primary": len(primary_all3),
            "mask_primary_w1": len(primary_w1),
            "mask_primary_c1": len(primary_c1),
            "mask_cross": len(cross_non_think),
            "mask_cross_w1": len(cross_non_think_w1),
            "mask_cross_c1": len(cross_non_think_c1),
            "mask_think": len(cross_all5),
        },
    }
    return mask


def _resolve_positions(
    tokenizer,
    *,
    prompt_text: str,
    condition_code: str,
    correct_label: str,
    wrong_label: str,
) -> Tuple[Dict[str, Optional[int]], Optional[Tuple[int, int]]]:
    tokenized = _tokenize_for_scoring(tokenizer, prompt_text)
    input_ids = tokenized["input_ids"][0].tolist()
    prompt_start, prompt_end = locate_prompt_text_span(tokenizer, prompt_text, input_ids)

    positions: Dict[str, Optional[int]] = {name: None for name in POINT_POSITION_NAMES}
    positions["answer_position"] = prompt_end

    note_match = NOTE_LINE_RE.search(prompt_text)
    if note_match is None:
        return positions, None

    note_line = note_match.group(0)
    note_start_char = note_match.start()

    prefix = prompt_text[:note_start_char]
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    note_ids = tokenizer(note_line, add_special_tokens=False).input_ids
    if not note_ids:
        return positions, None

    note_start = prompt_start + len(prefix_ids)
    note_end = note_start + len(note_ids) - 1

    if note_end >= len(input_ids) or note_start < 0:
        found_note = _subsequence_span(input_ids, note_ids)
        if found_note is None:
            return positions, None
        note_start, note_end = found_note

    positions["endorsement_start"] = note_start
    positions["endorsement_end"] = note_end

    answer_segment = "\nAnswer:"
    answer_idx = prompt_text.find(answer_segment)
    if answer_idx >= 0:
        answer_prefix = prompt_text[: answer_idx + len(answer_segment)]
        answer_prefix_ids = tokenizer(answer_prefix, add_special_tokens=False).input_ids
        answer_pos = prompt_start + len(answer_prefix_ids) - 1
        if 0 <= answer_pos < len(input_ids):
            positions["answer_position"] = answer_pos

    if condition_code in {"C1_note", "W1_note"}:
        endorsed = correct_label if condition_code == "C1_note" else wrong_label
        endorsed_ids = tokenizer(f" {endorsed}", add_special_tokens=False).input_ids
        if not endorsed_ids:
            endorsed_ids = tokenizer(endorsed, add_special_tokens=False).input_ids
        if endorsed_ids:
            note_token_slice = input_ids[note_start : note_end + 1]
            found = _subsequence_span(note_token_slice, endorsed_ids)
            if found is not None:
                rel_start, _ = found
                positions["endorsed_answer"] = note_start + rel_start

    return positions, (note_start, note_end)


def _dynamic_batch_iter(records: List[Dict], max_batch_tokens: int, min_batch: int):
    i = 0
    while i < len(records):
        batch = [records[i]]
        max_seq = records[i]["prompt_token_count"]
        i += 1
        while i < len(records):
            candidate_len = records[i]["prompt_token_count"]
            new_max = max(max_seq, candidate_len)
            new_total = new_max * (len(batch) + 1)
            if new_total > max_batch_tokens and len(batch) >= min_batch:
                break
            batch.append(records[i])
            max_seq = new_max
            i += 1
        yield batch


def _pad_batch(batch: Sequence[Dict], device: torch.device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    input_ids_list = [r["input_ids"] for r in batch]
    attn_list = [r["attention_mask"] for r in batch]
    max_len = max(ids.shape[0] for ids in input_ids_list)
    bs = len(batch)

    input_ids = torch.zeros((bs, max_len), dtype=torch.long)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
    for i, (ids, mask) in enumerate(zip(input_ids_list, attn_list)):
        sl = ids.shape[0]
        input_ids[i, :sl] = ids
        attention_mask[i, :sl] = mask

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }, attention_mask


def _build_prepared_records(
    tokenizer,
    selected_uids: Sequence[str],
    rows_by_style: Dict[str, Dict[str, Dict[str, Record]]],
    included_styles: Sequence[str],
) -> List[Dict]:
    prepared: List[Dict] = []
    for uid in selected_uids:
        for style in included_styles:
            cond_map = rows_by_style.get(style, {}).get(uid)
            if cond_map is None:
                continue
            for condition_code in CONDITION_CODES:
                rec = cond_map.get(condition_code)
                if rec is None:
                    continue
                positions, endorsement_span = _resolve_positions(
                    tokenizer,
                    prompt_text=rec.prompt_text,
                    condition_code=condition_code,
                    correct_label=rec.correct_label,
                    wrong_label=rec.wrong_label,
                )
                tokenized = _tokenize_for_scoring(tokenizer, rec.prompt_text)
                prepared.append(
                    {
                        "uid": rec.uid,
                        "style": style,
                        "condition_code": condition_code,
                        "prompt_text": rec.prompt_text,
                        "correct_label": rec.correct_label,
                        "wrong_label": rec.wrong_label,
                        "final_answer_label": rec.final_answer_label,
                        "parse_valid": rec.parse_valid,
                        "final_is_correct": rec.final_is_correct,
                        "w1_flip": rec.w1_flip,
                        "c1_correction": rec.c1_correction,
                        "n0_correct": rec.n0_correct,
                        "positions": positions,
                        "endorsement_span": endorsement_span,
                        "input_ids": tokenized["input_ids"].squeeze(0).cpu(),
                        "attention_mask": tokenized["attention_mask"].squeeze(0).cpu(),
                        "prompt_token_count": int(tokenized["input_ids"].shape[1]),
                    }
                )
    prepared.sort(key=lambda r: r["prompt_token_count"])
    return prepared


def _extract_with_hooks(
    *,
    capture: SelectiveLayerCapture,
    batch: Sequence[Dict],
    target_layers: Sequence[int],
    save_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    bs = len(batch)
    n_target = len(target_layers)
    h_dim = next(iter(capture._captured.values())).shape[-1]  # noqa: SLF001
    out: Dict[str, torch.Tensor] = {
        name: torch.zeros((bs, n_target, h_dim), dtype=save_dtype) for name in ALL_POSITION_NAMES
    }

    positions_per_item = [row["positions"] for row in batch]
    point_views = capture.extract_positions(
        positions_per_item=positions_per_item,
        batch_size=bs,
        position_names=POINT_POSITION_NAMES,
    )
    for name in POINT_POSITION_NAMES:
        out[name] = point_views[name]

    mean_buf = out["endorsement_mean"]
    for item_idx, row in enumerate(batch):
        span = row["endorsement_span"]
        if span is None:
            continue
        start, end = span
        if end < start:
            continue
        for li, layer_idx in enumerate(target_layers):
            layer_hidden = capture._captured[layer_idx][item_idx]  # noqa: SLF001
            mean_vec = layer_hidden[start : end + 1, :].mean(dim=0)
            mean_buf[item_idx, li] = mean_vec.to(dtype=save_dtype)
    return out


def main() -> None:
    args = parse_args()
    setup_h100_optimizations()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_style, included_styles = _load_all_styles(args)
    base_uids = sorted(_style_uid_set(rows_by_style, "authoritative_verified"))
    if args.uids_file is not None:
        allowed = {line.strip() for line in args.uids_file.read_text().splitlines() if line.strip()}
        base_uids = [uid for uid in base_uids if uid in allowed]
    if args.max_items > 0:
        base_uids = base_uids[: args.max_items]
    if not base_uids:
        raise SystemExit("No UIDs selected for extraction.")

    masks = _build_label_masks(base_uids, rows_by_style, included_styles)
    (args.output_dir / "label_masks.json").write_text(json.dumps(masks, indent=2))
    (args.output_dir / "selected_uids.txt").write_text("".join(f"{uid}\n" for uid in base_uids))

    summary = {
        "model": args.model,
        "arm": args.arm,
        "n_selected_uids": len(base_uids),
        "included_styles": list(included_styles),
        "style_rows_available": {style: len(rows_by_style.get(style, {})) for style in included_styles},
        "mask_counts": masks["counts"],
        "target_layers": list(args.target_layers),
        "max_batch_tokens": args.max_batch_tokens,
        "fallback_batch_size": args.fallback_batch_size,
        "loader_dtype": args.loader_dtype,
        "save_dtype": args.save_dtype,
    }

    if args.dry_run:
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    save_dtype = _dtype_from_arg(args.save_dtype)

    print(f"Loading model {args.model}...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        device=args.device,
        dtype=args.loader_dtype,
    )
    model.eval()
    if not args.no_compile:
        model = try_compile_model(model)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    device = next(model.parameters()).device
    n_model_layers = len(get_transformer_layers(model))
    target_layers = list(args.target_layers)
    bad_layers = [idx for idx in target_layers if idx < 0 or idx >= n_model_layers]
    if bad_layers:
        print(
            f"[extract_authority_activations] Dropping invalid target layers for model with "
            f"{n_model_layers} layers: {bad_layers}"
        )
        target_layers = [idx for idx in target_layers if 0 <= idx < n_model_layers]
    if not target_layers:
        raise ValueError(
            f"No valid target layers remain for model with {n_model_layers} layers. "
            "Pass --target-layers with in-range indices."
        )

    prepared = _build_prepared_records(tokenizer, base_uids, rows_by_style, included_styles)
    if not prepared:
        raise SystemExit("No prepared records after filtering.")

    token_lengths = [r["prompt_token_count"] for r in prepared]
    print(
        f"Prepared {len(prepared)} records | prompt_len min={min(token_lengths)} "
        f"max={max(token_lengths)} mean={sum(token_lengths) / len(token_lengths):.1f}"
    )

    batches = list(_dynamic_batch_iter(prepared, args.max_batch_tokens, args.fallback_batch_size))
    batch_sizes = [len(b) for b in batches]
    print(
        f"Dynamic batches: n={len(batches)} min={min(batch_sizes)} max={max(batch_sizes)} "
        f"mean={sum(batch_sizes) / len(batch_sizes):.1f}"
    )

    activation_lists: Dict[str, List[torch.Tensor]] = {name: [] for name in ALL_POSITION_NAMES}
    metadata_path = args.output_dir / "metadata.jsonl"
    style_counter = Counter()
    condition_counter = Counter()
    t_start = time.time()
    total_records = 0

    with metadata_path.open("w") as metadata_file:
        for batch in tqdm(batches, desc=f"Extracting {args.model}"):
            model_inputs, _ = _pad_batch(batch, device)
            capture = SelectiveLayerCapture(
                model,
                target_layers=target_layers,
                save_dtype=save_dtype,
            )
            with capture:
                with torch.inference_mode():
                    _ = model(**model_inputs, use_cache=False)

            views = _extract_with_hooks(
                capture=capture,
                batch=batch,
                target_layers=target_layers,
                save_dtype=save_dtype,
            )
            capture.clear()

            for i, row in enumerate(batch):
                for name in ALL_POSITION_NAMES:
                    activation_lists[name].append(views[name][i])

                style_counter[row["style"]] += 1
                condition_counter[row["condition_code"]] += 1
                payload = {
                    "uid": row["uid"],
                    "style": row["style"],
                    "condition_code": row["condition_code"],
                    "prompt_text": row["prompt_text"],
                    "prompt_token_count": row["prompt_token_count"],
                    "positions": row["positions"],
                    "endorsement_span": row["endorsement_span"],
                    "correct_label": row["correct_label"],
                    "wrong_label": row["wrong_label"],
                    "final_answer_label": row["final_answer_label"],
                    "parse_valid": row["parse_valid"],
                    "final_is_correct": row["final_is_correct"],
                    "w1_flip": row["w1_flip"],
                    "c1_correction": row["c1_correction"],
                    "n0_correct": row["n0_correct"],
                }
                metadata_file.write(json.dumps(payload) + "\n")
                total_records += 1

    elapsed = time.time() - t_start
    print(f"Extraction completed: {total_records} records in {elapsed:.1f}s ({total_records / elapsed:.2f} rec/s)")

    print("Saving activation tensors...")
    activations = {
        name: torch.stack(tensors, dim=0)
        for name, tensors in activation_lists.items()
    }
    tensor_payload = {
        "model": args.model,
        "position_names": list(ALL_POSITION_NAMES),
        "target_layers": target_layers,
        "n_model_layers": n_model_layers,
        "save_dtype": args.save_dtype,
        "activations": activations,
    }
    activations_path = args.output_dir / "activations.pt"
    torch.save(tensor_payload, activations_path)

    summary.update(
        {
            "n_records": total_records,
            "elapsed_seconds": round(elapsed, 2),
            "records_per_second": round(total_records / elapsed, 4),
            "style_counter": dict(style_counter),
            "condition_counter": dict(condition_counter),
            "metadata_path": str(metadata_path),
            "activations_path": str(activations_path),
            "label_masks_path": str(args.output_dir / "label_masks.json"),
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
