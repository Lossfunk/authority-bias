"""Build an exp7/exp10-compatible MC dataset from commonsense benchmarks.

Supported datasets:
- COPA (SuperGLUE)
- WinoGrande
- PIQA

This writes the same JSONL schema as `data/exp7_mc_dataset.jsonl`, so the
existing exp10/exp11 pipeline can be reused by pointing `--mc-dataset-path`
at the generated file.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.exp7.dataset_mc import MCExample, save_mc_dataset, verify_mc_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a commonsense benchmark into exp7 MC JSONL format."
    )
    parser.add_argument(
        "--dataset",
        choices=["copa", "winogrande", "piqa"],
        required=True,
        help="Commonsense benchmark to convert.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help=(
            "Datasets split expression passed through to Hugging Face datasets. "
            "Examples: validation, train, train+validation."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Where to write the exp7-style MC JSONL file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for A/B label assignment.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Optional cap after loading (0 = all).",
    )
    parser.add_argument(
        "--winogrande-subset",
        default="winogrande_debiased",
        help="Subset name for WinoGrande.",
    )
    return parser.parse_args()


def _randomize_labels(
    uid: str,
    question: str,
    correct_answer: str,
    wrong_answer: str,
    metadata: Dict,
    rng: random.Random,
) -> MCExample:
    correct_is_a = rng.choice([True, False])
    if correct_is_a:
        correct_label = "A"
        wrong_label = "B"
        option_a = correct_answer
        option_b = wrong_answer
    else:
        correct_label = "B"
        wrong_label = "A"
        option_a = wrong_answer
        option_b = correct_answer

    return MCExample(
        uid=uid,
        question=question,
        correct_answer=correct_answer,
        wrong_answer=wrong_answer,
        correct_label=correct_label,
        wrong_label=wrong_label,
        option_a=option_a,
        option_b=option_b,
        endorsed_label=wrong_label,
        metadata=metadata,
    )


def _load_copa(split: str, rng: random.Random) -> List[MCExample]:
    ds = load_dataset("super_glue", "copa", split=split)
    examples: List[MCExample] = []
    for row in ds:
        question = f"Premise: {row['premise']}\nWhat was the {row['question']}?"
        correct_answer = row["choice1"] if row["label"] == 0 else row["choice2"]
        wrong_answer = row["choice2"] if row["label"] == 0 else row["choice1"]
        metadata = {
            "dataset": "copa",
            "source_split": split,
            "idx": row["idx"],
            "question_type": row["question"],
        }
        examples.append(
            _randomize_labels(
                uid=f"copa::{row['idx']}",
                question=question,
                correct_answer=correct_answer,
                wrong_answer=wrong_answer,
                metadata=metadata,
                rng=rng,
            )
        )
    return examples


def _load_winogrande(split: str, subset: str, rng: random.Random) -> List[MCExample]:
    ds = load_dataset("allenai/winogrande", subset, split=split)
    examples: List[MCExample] = []
    for idx, row in enumerate(ds):
        answer = row["answer"]
        correct_answer = row["option1"] if answer == "1" else row["option2"]
        wrong_answer = row["option2"] if answer == "1" else row["option1"]
        metadata = {
            "dataset": "winogrande",
            "source_split": split,
            "subset": subset,
            "source_index": idx,
        }
        examples.append(
            _randomize_labels(
                uid=f"winogrande::{subset}::{idx}",
                question=row["sentence"],
                correct_answer=correct_answer,
                wrong_answer=wrong_answer,
                metadata=metadata,
                rng=rng,
            )
        )
    return examples


def _load_piqa(split: str, rng: random.Random) -> List[MCExample]:
    ds = load_dataset("lighteval/piqa", split=split)
    examples: List[MCExample] = []
    for idx, row in enumerate(ds):
        correct_answer = row["sol1"] if row["label"] == 0 else row["sol2"]
        wrong_answer = row["sol2"] if row["label"] == 0 else row["sol1"]
        metadata = {
            "dataset": "piqa",
            "source_split": split,
            "source_index": idx,
        }
        examples.append(
            _randomize_labels(
                uid=f"piqa::{idx}",
                question=row["goal"],
                correct_answer=correct_answer,
                wrong_answer=wrong_answer,
                metadata=metadata,
                rng=rng,
            )
        )
    return examples


def _truncate(examples: List[MCExample], max_examples: int, rng: random.Random) -> List[MCExample]:
    if max_examples <= 0 or len(examples) <= max_examples:
        return examples
    rng.shuffle(examples)
    return examples[:max_examples]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.dataset == "copa":
        examples = _load_copa(args.split, rng)
    elif args.dataset == "winogrande":
        examples = _load_winogrande(args.split, args.winogrande_subset, rng)
    else:
        examples = _load_piqa(args.split, rng)

    examples = _truncate(examples, args.max_examples, rng)
    save_mc_dataset(examples, args.output_path)

    summary = verify_mc_dataset(examples)
    print(json.dumps(
        {
            "dataset": args.dataset,
            "split": args.split,
            "n_examples": len(examples),
            "output_path": str(args.output_path),
            "verification": summary,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
