"""Rebuild an exp7/exp10-compatible MC dataset from existing results JSONL.

This lets us reuse the mechanism pipeline on datasets like PIQA while preserving
the exact label assignment and option ordering used in the original results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lexical_controls.dataset_mc import MCExample, save_mc_dataset, verify_mc_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an exp7-style MC dataset from exp10 results JSONL."
    )
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _to_example(row: Dict) -> MCExample:
    correct_label = row["correct_label"]
    wrong_label = row["wrong_label"]
    correct_answer = row["correct_answer"]
    wrong_answer = row["wrong_answer"]

    if correct_label == "A":
        option_a = correct_answer
        option_b = wrong_answer
    elif correct_label == "B":
        option_a = wrong_answer
        option_b = correct_answer
    else:
        raise ValueError(f"Unexpected correct_label={correct_label!r} for uid={row['uid']}")

    return MCExample(
        uid=row["uid"],
        question=row["question"],
        correct_answer=correct_answer,
        wrong_answer=wrong_answer,
        correct_label=correct_label,
        wrong_label=wrong_label,
        option_a=option_a,
        option_b=option_b,
        endorsed_label=wrong_label,
        metadata=dict(row.get("metadata") or {}),
    )


def main() -> None:
    args = parse_args()
    examples: List[MCExample] = [_to_example(row) for row in _iter_jsonl(args.results_path)]
    save_mc_dataset(examples, args.output_path)
    payload = {
        "results_path": str(args.results_path),
        "output_path": str(args.output_path),
        "n_examples": len(examples),
        "verification": verify_mc_dataset(examples),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
