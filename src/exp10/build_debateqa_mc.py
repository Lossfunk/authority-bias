"""Convert DebateQA into MCExample JSONL for the exp10 pipeline.

For debatable questions there is no ground truth. We pick the two most
opposing perspectives, randomly assign them to A/B, and arbitrarily
label one as "correct" so the existing pipeline runs unchanged. The
metadata field `no_ground_truth=True` marks these items so downstream
analysis uses position-shift metrics instead of correction rate.

The endorsement always points to answer_b_perspective (mapped to
`wrong_answer` in MCExample terms) so the pipeline's C1 condition
endorses that side.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_INPUT = Path("external/debateqa/test.jsonl")
DEFAULT_OUTPUT = Path("data/debateqa_mc_dataset.jsonl")


def load_debateqa(path: Path) -> List[Dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _shorten_pov(pov: str, max_words: int = 15) -> str:
    """Trim a point_of_view string to a concise option label."""
    pov = pov.strip().rstrip(".")
    words = pov.split()
    if len(words) <= max_words:
        return pov
    return " ".join(words[:max_words]).rstrip(",;:") + "..."


def _select_two_perspectives(
    partial_answers: List[Dict],
) -> Tuple[str, str]:
    """Pick the two most opposing perspectives from the list.

    Heuristic: for questions with a clear yes/no structure, pick the
    first perspective that leans affirmative and the first that leans
    negative. If we can't detect polarity, just take the first two.
    """
    povs = [pa["point_of_view"] for pa in partial_answers]

    neg_markers = re.compile(
        r"\bnot\b|\bnor\b|\bno\b|\bnever\b|\bdoes not\b|\bisn't\b|"
        r"\bcannot\b|\bcan't\b|\bshould not\b|\bshouldn't\b|\bharmful\b|"
        r"\bnegative\b|\bworse\b|\bless\b|\bwon't\b|\bdon't\b",
        re.IGNORECASE,
    )

    pos_idx, neg_idx = None, None
    for i, pov in enumerate(povs):
        if neg_markers.search(pov):
            if neg_idx is None:
                neg_idx = i
        else:
            if pos_idx is None:
                pos_idx = i
        if pos_idx is not None and neg_idx is not None:
            break

    if pos_idx is not None and neg_idx is not None:
        return povs[pos_idx], povs[neg_idx]

    return povs[0], povs[1] if len(povs) > 1 else povs[0]


def build_mc_dataset(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    seed: int = 42,
    max_examples: int = 0,
) -> int:
    rng = random.Random(seed)
    rows = load_debateqa(input_path)

    if max_examples > 0:
        rng.shuffle(rows)
        rows = rows[:max_examples]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with output_path.open("w") as f:
        for row in rows:
            qid = row.get("id", row.get("question", "")[:40])
            question = row["question"]
            partial_answers = row["partial_answers"]

            if len(partial_answers) < 2:
                continue

            pov_a_full, pov_b_full = _select_two_perspectives(partial_answers)
            pov_a = _shorten_pov(pov_a_full)
            pov_b = _shorten_pov(pov_b_full)

            # Skip if both perspectives are identical after shortening
            if pov_a == pov_b:
                continue

            # Randomly assign which perspective is "correct" (label A or B)
            correct_is_a = rng.choice([True, False])

            if correct_is_a:
                correct_label = "A"
                wrong_label = "B"
                option_a = pov_a
                option_b = pov_b
                correct_answer = pov_a
                wrong_answer = pov_b
            else:
                correct_label = "B"
                wrong_label = "A"
                option_a = pov_b
                option_b = pov_a
                correct_answer = pov_a
                wrong_answer = pov_b

            obj = {
                "uid": f"debateqa::{qid}",
                "question": question,
                "correct_answer": correct_answer,
                "wrong_answer": wrong_answer,
                "correct_label": correct_label,
                "wrong_label": wrong_label,
                "option_a": option_a,
                "option_b": option_b,
                "endorsed_label": wrong_label,
                "metadata": {
                    "dataset": "debateqa",
                    "source": row.get("source", "unknown"),
                    "no_ground_truth": True,
                    "pov_a_full": pov_a_full,
                    "pov_b_full": pov_b_full,
                    "n_perspectives": len(partial_answers),
                },
            }
            f.write(json.dumps(obj) + "\n")
            written += 1

    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()

    n = build_mc_dataset(args.input, args.output, args.seed, args.max_examples)
    print(f"Wrote {n} MC examples to {args.output}")
