from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.mechanism.common import normalize_answer, read_panel_parquet
from src.models.llama_loader import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mechanism robustness helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("regression")
    reg.add_argument("--panel-path", type=Path, required=True)
    reg.add_argument("--output-path", type=Path, required=True)

    subset = sub.add_parser("openended-subset")
    subset.add_argument("--panel-path", type=Path, required=True)
    subset.add_argument("--output-path", type=Path, required=True)
    subset.add_argument("--max-answer-tokens", type=int, default=4)
    subset.add_argument(
        "--alias-source",
        type=Path,
        default=Path("external/sycophancy-eval/datasets/answer.jsonl"),
    )

    return parser.parse_args()


def make_option_number_prompt(row) -> str:
    text = row.prompt_text
    text = text.replace("Options: A)", "Options: 1)")
    text = text.replace(", B)", ", 2)")
    if row.evidence_direction != "neutral":
        text = text.replace(f"it's {row.correct_label}", f"it's {'1' if row.correct_label == 'A' else '2'}")
        text = text.replace(f"{row.wrong_label}.", f"{'1' if row.wrong_label == 'A' else '2'}." )
        text = text.replace(f"not {row.correct_label}", f"not {'1' if row.correct_label == 'A' else '2'}")
        text = text.replace(f"not {row.wrong_label}", f"not {'1' if row.wrong_label == 'A' else '2'}")
    return text.replace("Answer:", "Answer with Option 1 or Option 2:")


def make_full_string_prompt(row) -> str:
    prompt = row.prompt_text
    prompt = prompt.replace(
        f"Options: A) {row.correct_answer if row.correct_label == 'A' else row.wrong_answer}, B) {row.wrong_answer if row.wrong_label == 'B' else row.correct_answer}.",
        f"Candidate answers: {row.correct_answer}; {row.wrong_answer}.",
    )
    if row.evidence_direction == "corrective":
        prompt = prompt.replace(f"it's {row.correct_label}, not {row.wrong_label}", f"it's {row.correct_answer}, not {row.wrong_answer}")
    elif row.evidence_direction == "misleading":
        prompt = prompt.replace(f"it's {row.wrong_label}, not {row.correct_label}", f"it's {row.wrong_answer}, not {row.correct_answer}")
    return prompt.replace("Answer:", "Answer with the exact answer text:")


def _continuation_logprob(model, tokenizer, prompt_text: str, continuation: str, device: torch.device) -> float:
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    full_ids = tokenizer(prompt_text + continuation, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    continuation_ids = full_ids[:, prompt_ids.shape[1]:]
    with torch.inference_mode():
        outputs = model(full_ids[:, :-1], use_cache=False)
        logits = outputs.logits[0]
    total_logprob = 0.0
    start = prompt_ids.shape[1] - 1
    for offset in range(continuation_ids.shape[1]):
        token_id = int(continuation_ids[0, offset])
        logp = torch.log_softmax(logits[start + offset], dim=-1)[token_id]
        total_logprob += float(logp.item())
    denom = max(continuation_ids.shape[1], 1)
    return total_logprob / denom


def score_candidate_strings(model, tokenizer, prompt_text: str, candidates: Sequence[str], device: torch.device) -> Dict[str, float]:
    scores = np.array([_continuation_logprob(model, tokenizer, prompt_text, candidate, device) for candidate in candidates], dtype=float)
    probs = np.exp(scores - scores.max())
    probs = probs / probs.sum()
    return {candidate: float(prob) for candidate, prob in zip(candidates, probs)}


def load_triviaqa_aliases(path: Path) -> Dict[str, List[str]]:
    aliases: Dict[str, List[str]] = {}
    if not path.exists():
        return aliases
    with path.open() as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                continue
            base = payload.get("base") or {}
            question = str(base.get("question", "")).strip()
            answer_list = base.get("answer") or []
            if not question or not isinstance(answer_list, list) or not answer_list:
                continue
            aliases.setdefault(question, [str(alias) for alias in answer_list if str(alias).strip()])
    return aliases


def select_exact_match_friendly_subset(
    panel_path: Path,
    output_path: Path,
    *,
    max_answer_tokens: int,
    alias_source: Path,
) -> Dict[str, object]:
    rows = read_panel_parquet(panel_path)
    trivia_aliases = load_triviaqa_aliases(alias_source)
    selected = []
    seen = set()
    for row in rows:
        if row.uid in seen:
            continue
        if len(row.correct_answer.split()) > max_answer_tokens:
            continue
        if len(row.wrong_answer.split()) > max_answer_tokens:
            continue
        if normalize_answer(row.correct_answer) == normalize_answer(row.wrong_answer):
            continue
        aliases = trivia_aliases.get(row.question, [row.correct_answer])
        seen.add(row.uid)
        selected.append(
            {
                "uid": row.uid,
                "dataset": row.dataset,
                "question": row.question,
                "correct_answer": row.correct_answer,
                "wrong_answer": row.wrong_answer,
                "aliases": aliases,
            }
        )
    pd.DataFrame(selected).to_parquet(output_path, index=False)
    summary = {
        "output_path": str(output_path),
        "n_rows": len(selected),
        "max_answer_tokens": max_answer_tokens,
        "alias_source": str(alias_source),
    }
    return summary


def fit_additive_regression(panel_path: Path, output_path: Path) -> Dict[str, object]:
    rows = read_panel_parquet(panel_path)
    frame = pd.DataFrame(
        {
            "uid": [row.uid for row in rows],
            "authority_note": [1.0 if row.authority == "Note" else 0.0 for row in rows],
            "instruction": [float(row.instruction) for row in rows],
            "evidence_strength": [float(["bare", "reason1", "reason2", "reason_data"].index(row.evidence_strength)) for row in rows],
            "direction_corrective": [1.0 if row.evidence_direction == "corrective" else 0.0 for row in rows],
            "prior_margin": [row.prior_margin for row in rows],
            "y": [float(row.target_metric_metadata.get("logit_correct", 0.0)) for row in rows],
        }
    )
    X = np.column_stack(
        [
            np.ones(len(frame)),
            frame["prior_margin"].to_numpy(),
            frame["instruction"].to_numpy(),
            frame["direction_corrective"].to_numpy(),
            frame["authority_note"].to_numpy(),
            frame["evidence_strength"].to_numpy(),
            (frame["instruction"] * frame["direction_corrective"]).to_numpy(),
            (frame["prior_margin"] * frame["direction_corrective"]).to_numpy(),
        ]
    )
    y = frame["y"].to_numpy()
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ coef
    sigma2 = float((residual @ residual) / max(len(y) - X.shape[1], 1))
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    stderr = np.sqrt(np.diag(cov))
    names = [
        "intercept",
        "prior_margin",
        "instruction",
        "direction_corrective",
        "authority_note",
        "evidence_strength",
        "instruction_x_direction",
        "prior_margin_x_direction",
    ]
    rows_out = []
    for name, value, se in zip(names, coef, stderr):
        rows_out.append({"term": name, "coef": float(value), "stderr": float(se)})
    pd.DataFrame(rows_out).to_parquet(output_path, index=False)
    return {"output_path": str(output_path), "n_rows": len(frame), "terms": len(rows_out)}


def main() -> None:
    args = parse_args()
    if args.command == "regression":
        summary = fit_additive_regression(args.panel_path, args.output_path)
    elif args.command == "openended-subset":
        summary = select_exact_match_friendly_subset(
            args.panel_path,
            args.output_path,
            max_answer_tokens=args.max_answer_tokens,
            alias_source=args.alias_source,
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
