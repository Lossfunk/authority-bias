from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch


DEFAULT_EXTRACTION_DIR = Path(
    "new-phase-results/mechanism/qwen_correction_gating_i1a_i1c_note"
)
DEFAULT_OUTPUT_PATH = Path(
    "new-phase-results/mechanism/qwen_within_i1a_vector_custom.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build within-i1a correction-gating vectors from filtered subsets."
    )
    parser.add_argument("--extraction-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument(
        "--position",
        type=str,
        default="endorsement_last",
        choices=("last_token", "instruction_last", "endorsement_last"),
    )
    parser.add_argument("--layer-index", type=int, default=23)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--min-abs-mn0", type=float, default=None)
    parser.add_argument("--max-abs-mn0", type=float, default=None)
    parser.add_argument("--entrenching-min-mc1", type=float, default=None)
    parser.add_argument("--entrenching-max-mc1", type=float, default=None)
    parser.add_argument("--correcting-min-mc1", type=float, default=None)
    parser.add_argument("--correcting-max-mc1", type=float, default=None)
    return parser.parse_args()


def _load_metadata(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _passes_shared_filters(row: Dict, args: argparse.Namespace) -> bool:
    sel = row["selection"]
    abs_m_n0 = float(sel["abs_m_n0"])
    if args.min_abs_mn0 is not None and abs_m_n0 < args.min_abs_mn0:
        return False
    if args.max_abs_mn0 is not None and abs_m_n0 > args.max_abs_mn0:
        return False
    return True


def _passes_label_filters(row: Dict, args: argparse.Namespace) -> bool:
    sel = row["selection"]
    m_c1 = float(sel["m_c1"])
    label = row["item_label"]
    if label == "entrenching":
        if args.entrenching_min_mc1 is not None and m_c1 < args.entrenching_min_mc1:
            return False
        if args.entrenching_max_mc1 is not None and m_c1 > args.entrenching_max_mc1:
            return False
    elif label == "correcting":
        if args.correcting_min_mc1 is not None and m_c1 < args.correcting_min_mc1:
            return False
        if args.correcting_max_mc1 is not None and m_c1 > args.correcting_max_mc1:
            return False
    return True


def _select_rows(metadata: Sequence[Dict], args: argparse.Namespace) -> tuple[list[int], list[int]]:
    correcting_rows: List[int] = []
    entrenching_rows: List[int] = []
    for idx, row in enumerate(metadata):
        if row["variant_label"] != "i1a":
            continue
        if not _passes_shared_filters(row, args):
            continue
        if not _passes_label_filters(row, args):
            continue
        if row["item_label"] == "correcting":
            correcting_rows.append(idx)
        elif row["item_label"] == "entrenching":
            entrenching_rows.append(idx)
    return correcting_rows, entrenching_rows


def main() -> None:
    args = parse_args()
    metadata = _load_metadata(args.extraction_dir / "metadata.jsonl")
    payload = torch.load(args.extraction_dir / "activations.pt", map_location="cpu")
    acts = payload["activations"][args.position]

    correcting_rows, entrenching_rows = _select_rows(metadata, args)
    if not correcting_rows or not entrenching_rows:
        raise SystemExit(
            f"Need both classes after filtering; got correcting={len(correcting_rows)}, "
            f"entrenching={len(entrenching_rows)}"
        )

    correcting_mean = acts[torch.tensor(correcting_rows), args.layer_index, :].float().mean(dim=0)
    entrenching_mean = acts[torch.tensor(entrenching_rows), args.layer_index, :].float().mean(dim=0)
    vector = correcting_mean - entrenching_mean

    out = {
        "vector": vector.cpu(),
        "position": args.position,
        "layer_index": args.layer_index,
        "n_correcting": len(correcting_rows),
        "n_entrenching": len(entrenching_rows),
        "correcting_rows": correcting_rows,
        "entrenching_rows": entrenching_rows,
        "filters": {
            "min_abs_mn0": args.min_abs_mn0,
            "max_abs_mn0": args.max_abs_mn0,
            "entrenching_min_mc1": args.entrenching_min_mc1,
            "entrenching_max_mc1": args.entrenching_max_mc1,
            "correcting_min_mc1": args.correcting_min_mc1,
            "correcting_max_mc1": args.correcting_max_mc1,
        },
        "vector_norm": float(vector.norm().item()),
        "extraction_dir": str(args.extraction_dir),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output_path)
    print(json.dumps({k: v for k, v in out.items() if k not in {"vector", "correcting_rows", "entrenching_rows"}}, indent=2))


if __name__ == "__main__":
    main()
