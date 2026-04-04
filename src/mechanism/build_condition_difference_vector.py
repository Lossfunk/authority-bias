from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a mean-difference vector from extracted prompt conditions."
    )
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument(
        "--position",
        type=str,
        default="endorsement_last",
        choices=("last_token", "instruction_last", "endorsement_last"),
    )
    parser.add_argument("--layer-index", type=int, default=23)
    parser.add_argument("--positive-condition-codes", nargs="+", required=True)
    parser.add_argument("--negative-condition-codes", nargs="+", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def _load_metadata(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _select_rows(metadata: Sequence[Dict], codes: Sequence[str]) -> List[int]:
    allowed = set(codes)
    return [idx for idx, row in enumerate(metadata) if row["condition_code"] in allowed]


def main() -> None:
    args = parse_args()
    metadata = _load_metadata(args.extraction_dir / "metadata.jsonl")
    payload = torch.load(args.extraction_dir / "activations.pt", map_location="cpu")
    acts = payload["activations"][args.position]

    pos_rows = _select_rows(metadata, args.positive_condition_codes)
    neg_rows = _select_rows(metadata, args.negative_condition_codes)
    if not pos_rows or not neg_rows:
        raise SystemExit(
            f"Need both groups after filtering; got positive={len(pos_rows)} negative={len(neg_rows)}"
        )

    pos_mean = acts[torch.tensor(pos_rows), args.layer_index, :].float().mean(dim=0)
    neg_mean = acts[torch.tensor(neg_rows), args.layer_index, :].float().mean(dim=0)
    vector = pos_mean - neg_mean

    out = {
        "vector": vector.cpu(),
        "position": args.position,
        "layer_index": args.layer_index,
        "n_positive": len(pos_rows),
        "n_negative": len(neg_rows),
        "positive_condition_codes": list(args.positive_condition_codes),
        "negative_condition_codes": list(args.negative_condition_codes),
        "vector_norm": float(vector.norm().item()),
        "extraction_dir": str(args.extraction_dir),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output_path)
    print(
        json.dumps(
            {k: v for k, v in out.items() if k != "vector"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
