"""
Utilities for parsing the sycophancy dataset into experiment-ready splits.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass
class UserWrongExample:
    """
    Minimal record describing a user-wrong prompt with aligned answers.
    """

    uid: str
    question: str
    user_wrong_prompt: str
    correct_answer: str
    wrong_answer: str
    metadata: Dict

    def to_dict(self) -> Dict:
        return asdict(self)


def read_jsonl(path: Path) -> List[Dict]:
    """
    Load a JSONL file into memory.
    """
    records: List[Dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    """
    Write dictionaries to disk as JSONL.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            json.dump(row, f)
            f.write("\n")


def is_user_wrong_prompt(prompt_text: str, wrong_answer: str | None) -> bool:
    """
    Heuristic for identifying UserWrong examples (user explicitly stakes a belief).

    We accept two common templates in the dataset:
      1) "I think the answer is {wrong_answer}"
      2) "I don't think the answer is {correct_answer}"

    The second pattern does not contain `wrong_answer`, so we detect it
    independently of the wrong answer content.
    """
    if not prompt_text:
        return False
    norm_prompt = prompt_text.lower().replace("’", "'")
    has_positive_claim = "i think the answer is" in norm_prompt
    has_negative_claim = "i don't think the answer is" in norm_prompt
    has_wrong_token = bool(wrong_answer) and wrong_answer.lower() in norm_prompt
    # Accept if either template is present; require the wrong answer token only
    # for the positive-claim template to avoid false positives elsewhere.
    if has_positive_claim and has_wrong_token:
        return True
    if has_negative_claim:
        return True
    return False


def filter_user_wrong_examples(raw_path: Path) -> List[UserWrongExample]:
    """
    Load raw records and return the subset where the user asserts an incorrect belief.
    """
    raw_records = read_jsonl(raw_path)
    examples: List[UserWrongExample] = []
    for idx, row in enumerate(raw_records):
        prompt_items = row.get("prompt", [])
        if not prompt_items:
            continue
        prompt_text = prompt_items[0].get("content", "").strip()
        base = row.get("base", {})
        wrong_answer = base.get("incorrect_answer")
        correct_answer = base.get("correct_answer")
        question = base.get("question", prompt_text)
        if not (correct_answer and wrong_answer):
            continue
        if not is_user_wrong_prompt(prompt_text, wrong_answer):
            continue
        uid = f"{base.get('dataset', 'unknown')}::{idx}"
        examples.append(
            UserWrongExample(
                uid=uid,
                question=question.strip(),
                user_wrong_prompt=prompt_text,
                correct_answer=correct_answer.strip(),
                wrong_answer=wrong_answer.strip(),
                metadata=row.get("metadata", {}),
            )
        )
    return examples


_BELIEF_MARKERS = (
    "i think the answer is",
    "i don't think the answer is",
    "i do not think the answer is",
)


def make_neutral_prompt(example: UserWrongExample) -> str:
    """
    Construct a neutral prompt from a user-wrong example.
    """
    prompt = (example.user_wrong_prompt or "").strip()
    if prompt:
        norm = prompt.lower().replace("’", "'")
        idxs = [norm.find(marker) for marker in _BELIEF_MARKERS]
        cut = min((i for i in idxs if i != -1), default=-1)
        if cut != -1:
            neutral = prompt[:cut].rstrip()
            if neutral:
                return neutral

    question = (example.question or "").strip()
    return question or prompt


def split_examples(
    examples: Sequence[UserWrongExample], seed: int = 42, fractions: Sequence[float] = (0.2, 0.2, 0.6)
) -> Dict[str, List[UserWrongExample]]:
    """
    Deterministically split into train/dev/test style partitions (A/B/C).
    """
    if len(fractions) != 3:
        raise ValueError("Expected three fractions for splits A/B/C.")
    total = sum(fractions)
    if not abs(total - 1.0) < 1e-6:
        raise ValueError("Split fractions must sum to 1.0.")
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)

    n = len(examples)
    n_a = int(fractions[0] * n)
    n_b = int(fractions[1] * n)
    idx_a = set(indices[:n_a])
    idx_b = set(indices[n_a : n_a + n_b])

    split_a: List[UserWrongExample] = []
    split_b: List[UserWrongExample] = []
    split_c: List[UserWrongExample] = []
    for i, ex in enumerate(examples):
        if i in idx_a:
            split_a.append(ex)
        elif i in idx_b:
            split_b.append(ex)
        else:
            split_c.append(ex)
    return {"A": split_a, "B": split_b, "C": split_c}


def ensure_splits_exist(
    raw_path: Path,
    split_a_path: Path,
    split_b_path: Path,
    split_c_path: Path,
    seed: int = 42,
) -> Dict[str, Path]:
    """
    Create split files if they are missing.
    """
    if split_a_path.exists() and split_b_path.exists() and split_c_path.exists():
        return {"A": split_a_path, "B": split_b_path, "C": split_c_path}

    user_wrong_examples = filter_user_wrong_examples(raw_path)
    if not user_wrong_examples:
        raise RuntimeError(f"No user wrong examples found in {raw_path}")
    splits = split_examples(user_wrong_examples, seed=seed)
    write_jsonl(split_a_path, (ex.to_dict() for ex in splits["A"]))
    write_jsonl(split_b_path, (ex.to_dict() for ex in splits["B"]))
    write_jsonl(split_c_path, (ex.to_dict() for ex in splits["C"]))
    return {"A": split_a_path, "B": split_b_path, "C": split_c_path}


def load_user_wrong_split(path: Path) -> List[UserWrongExample]:
    """
    Load a saved split back into dataclass objects.
    """
    rows = read_jsonl(path)
    return [
        UserWrongExample(
            uid=row["uid"],
            question=row["question"],
            user_wrong_prompt=row["user_wrong_prompt"],
            correct_answer=row["correct_answer"],
            wrong_answer=row["wrong_answer"],
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]


