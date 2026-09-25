from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


MODELS = ("qwen35", "gpt_oss", "olmo2", "olmo31")
SOURCE_USER_LAYERS = {"qwen35": 5, "gpt_oss": 16, "olmo2": 16, "olmo31": 15}
MODEL_LABEL = {
    "qwen35": "Qwen3.5",
    "gpt_oss": "GPT-OSS",
    "olmo2": "OLMo-2",
    "olmo31": "OLMo-3.1",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _layer_from_config(config_id: str) -> int | None:
    m = re.search(r"_L(\d+)(?:_|$)", config_id)
    return int(m.group(1)) if m else None


def _layer_from_row(row: dict[str, Any]) -> int | None:
    if "target_layers" in row and row["target_layers"]:
        return int(row["target_layers"][0])
    return _layer_from_config(str(row.get("config_id", "")))


def _float_eq(a: Any, b: float) -> bool:
    return abs(float(a) - b) < 1e-9


def _matched_flip(rows: list[dict[str, Any]], *, layer: int, alpha: float) -> tuple[float | None, int, int]:
    rows = [r for r in rows if r.get("condition_code") == "N0_note" and _layer_from_row(r) == layer]
    baseline_correct = {
        r["uid"]
        for r in rows
        if _float_eq(r.get("alpha", -1), 0.0) and r.get("is_correct") is True
    }
    current = [
        r
        for r in rows
        if _float_eq(r.get("alpha", -1), alpha) and r.get("uid") in baseline_correct and r.get("chose_wrong") is not None
    ]
    if not baseline_correct:
        return None, 0, 0
    flips = sum(1 for r in current if r.get("chose_wrong") is True)
    return flips / len(baseline_correct), flips, len(baseline_correct)


def _summary_wrong_rate(row: dict[str, Any]) -> float | None:
    for key in ("wrong_rate", "wrong_rate_parsed"):
        if row.get(key) is not None:
            return float(row[key])
    return None


def _summary_n(row: dict[str, Any]) -> int | None:
    for key in ("parsed", "n", "total"):
        if row.get(key) is not None:
            return int(row[key])
    return None


def _add_forward(root: Path, model: str, task: str, rows_out: list[dict[str, Any]]) -> None:
    task_dir = root / model / "fold0" / task
    summary_name = "steering_summary.json" if task == "forward_trivia" else "piqa_summary.json"
    rows_name = "steering_rows.jsonl" if task == "forward_trivia" else "piqa_rows.jsonl"
    summary = _load_json(task_dir / summary_name)
    raw_rows = _load_jsonl(task_dir / rows_name)
    candidates: list[dict[str, Any]] = []
    for row in summary:
        condition = row.get("condition") or row.get("condition_code")
        if condition != "N0_note":
            continue
        alpha = float(row["alpha"])
        if alpha == 0.0:
            continue
        layer = _layer_from_row(row)
        if layer is None:
            continue
        mf, flips, n_base = _matched_flip(raw_rows, layer=layer, alpha=alpha)
        candidates.append(
            {
                "model": model,
                "task": task,
                "variant": "authority_patch",
                "condition": "N0_note",
                "layer": layer,
                "alpha": alpha,
                "heldout_wrong_rate": _summary_wrong_rate(row),
                "heldout_matched_flip": mf,
                "n": n_base,
                "parsed": _summary_n(row),
                "selected_by": "max_matched_flip",
                "flips": flips,
            }
        )
    if candidates:
        rows_out.append(max(candidates, key=lambda r: (r["heldout_matched_flip"] or -1.0, r["heldout_wrong_rate"] or -1.0)))


def _add_source_user(root: Path, model: str, rows_out: list[dict[str, Any]]) -> None:
    path = root / model / "fold0" / "source_user" / model / "source_user_authority_summary.json"
    payload = _load_json(path)
    target_layer = SOURCE_USER_LAYERS[model]
    for row in payload["summary"]:
        if int(row["layer"]) != target_layer or not _float_eq(row["alpha"], 1.0):
            continue
        if row["variant"] not in {"source", "user", "assistant"}:
            continue
        rows_out.append(
            {
                "model": model,
                "task": "source_user",
                "variant": row["variant"],
                "condition": row["condition"],
                "layer": int(row["layer"]),
                "alpha": 1.0,
                "heldout_wrong_rate": float(row["wrong_rate"]),
                "heldout_matched_flip": None,
                "n": int(row["total"]),
                "parsed": int(row["parsed"]),
                "selected_by": "reported_layer_alpha1",
                "flips": "",
            }
        )


def _add_mitigation(root: Path, model: str, task: str, variant: str, rows_out: list[dict[str, Any]]) -> None:
    task_dir = root / model / "fold0" / f"{task}_{variant}"
    summary_name = "steering_summary.json" if task == "mitigation_trivia" else "piqa_summary.json"
    summary = _load_json(task_dir / summary_name)
    candidates = []
    for row in summary:
        alpha = float(row["alpha"])
        if not _float_eq(alpha, 1.0):
            continue
        layer = _layer_from_row(row)
        wrong_rate = _summary_wrong_rate(row)
        if layer is None or wrong_rate is None:
            continue
        candidates.append(
            {
                "model": model,
                "task": task,
                "variant": variant,
                "condition": row.get("condition") or row.get("condition_code"),
                "layer": layer,
                "alpha": 1.0,
                "heldout_wrong_rate": wrong_rate,
                "heldout_matched_flip": None,
                "n": int(row.get("total") or row.get("n") or 0),
                "parsed": _summary_n(row),
                "selected_by": "min_wrong_rate_alpha1",
                "flips": "",
            }
        )
    if candidates:
        rows_out.append(min(candidates, key=lambda r: r["heldout_wrong_rate"]))


def _pp(x: float | None) -> float | None:
    return None if x is None else 100.0 * x


def _format_value(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _format_value(row.get(k)) for k in fieldnames})


def _insample_lookup() -> dict[tuple[str, str, str, str], dict[str, Any]]:
    data: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    forward_trivia = {
        "qwen35": (16.1, 2, 1.0, "Table 2 best slot; note exp22 ran L5 only"),
        "gpt_oss": (32.7, 16, 1.0, "Table 2"),
        "olmo2": (19.0, 16, 0.5, "Table 2"),
        "olmo31": (58.8, 15, 1.0, "Table 2"),
    }
    forward_piqa = {
        "qwen35": (20.6, 5, 0.7, "Appendix PIQA forward-steering figure"),
        "gpt_oss": (18.8, 16, 1.0, "Appendix PIQA forward-steering figure"),
        "olmo2": (10.4, 16, 0.5, "Appendix PIQA forward-steering figure"),
        "olmo31": (25.4, 15, 1.0, "Appendix PIQA forward-steering figure"),
    }
    for model, (value, layer, alpha, source) in forward_trivia.items():
        data[(model, "forward_trivia", "authority_patch", "N0_note")] = {
            "insample_metric": "matched_flip",
            "insample_value_pp": value,
            "insample_layer": layer,
            "insample_alpha": alpha,
            "insample_source": source,
        }
    for model, (value, layer, alpha, source) in forward_piqa.items():
        data[(model, "forward_piqa", "authority_patch", "N0_note")] = {
            "insample_metric": "matched_flip",
            "insample_value_pp": value,
            "insample_layer": layer,
            "insample_alpha": alpha,
            "insample_source": source,
        }

    mitigation = {
        ("qwen35", "mitigation_trivia", "residualized", "W1_note"): 5.2,
        ("qwen35", "mitigation_trivia", "caa", "W1_note"): 42.2,
        ("qwen35", "mitigation_piqa", "residualized", "W1_note"): 4.5,
        ("qwen35", "mitigation_piqa", "caa", "W1_note"): 57.1,
        ("gpt_oss", "mitigation_trivia", "residualized", "W1_note"): 44.0,
        ("gpt_oss", "mitigation_trivia", "caa", "W1_note"): 58.0,
        ("gpt_oss", "mitigation_piqa", "residualized", "W1_note"): 73.0,
        ("gpt_oss", "mitigation_piqa", "caa", "W1_note"): 86.0,
        ("olmo31", "mitigation_trivia", "residualized", "W1_note"): 64.0,
        ("olmo31", "mitigation_trivia", "caa", "W1_note"): 81.0,
        ("olmo31", "mitigation_piqa", "residualized", "W1_note"): 78.0,
        ("olmo31", "mitigation_piqa", "caa", "W1_note"): 94.0,
        ("olmo2", "mitigation_trivia", "residualized", "W1_note"): 53.3,
        ("olmo2", "mitigation_trivia", "caa", "W1_note"): 67.4,
        ("olmo2", "mitigation_piqa", "residualized", "W1_note"): 66.5,
        ("olmo2", "mitigation_piqa", "caa", "W1_note"): 76.4,
    }
    for key, value in mitigation.items():
        data[key] = {
            "insample_metric": "wrong_rate",
            "insample_value_pp": value,
            "insample_layer": "",
            "insample_alpha": 1.0,
            "insample_source": "Table 5 after-removal value",
        }

    source_user = {
        ("qwen35", "source", "source_W1"): 16.8,
        ("qwen35", "source", "user_W1"): 42.7,
        ("qwen35", "user", "source_W1"): 76.5,
        ("qwen35", "user", "user_W1"): 16.9,
        ("qwen35", "assistant", "source_W1"): 87.2,
        ("qwen35", "assistant", "user_W1"): 59.3,
        ("gpt_oss", "source", "source_W1"): 18.2,
        ("gpt_oss", "source", "user_W1"): 42.4,
        ("gpt_oss", "user", "source_W1"): 86.1,
        ("gpt_oss", "user", "user_W1"): 17.8,
        ("gpt_oss", "assistant", "source_W1"): 95.7,
        ("gpt_oss", "assistant", "user_W1"): 35.4,
    }
    for (model, variant, condition), value in source_user.items():
        data[(model, "source_user", variant, condition)] = {
            "insample_metric": "wrong_rate",
            "insample_value_pp": value,
            "insample_layer": 5 if model == "qwen35" else 16,
            "insample_alpha": 1.0,
            "insample_source": "Table 3 after-removal value",
        }
    return data


def _add_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = _insample_lookup()
    out = []
    for row in rows:
        key = (row["model"], row["task"], row["variant"], row["condition"])
        comp = lookup.get(key, {})
        held_metric = row["heldout_matched_flip"] if row["task"].startswith("forward_") else row["heldout_wrong_rate"]
        held_pp = _pp(held_metric)
        insample_pp = comp.get("insample_value_pp")
        out.append(
            {
                **row,
                "heldout_metric": "matched_flip" if row["task"].startswith("forward_") else "wrong_rate",
                "heldout_value_pp": held_pp,
                "insample_metric": comp.get("insample_metric", ""),
                "insample_value_pp": insample_pp,
                "delta_pp": None if held_pp is None or insample_pp in ("", None) else held_pp - float(insample_pp),
                "insample_layer": comp.get("insample_layer", ""),
                "insample_alpha": comp.get("insample_alpha", ""),
                "insample_source": comp.get("insample_source", ""),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results/authority/exp22/selection_holdout_50_50"))
    p.add_argument("--out-dir", type=Path, default=Path("results/authority/exp22/selection_holdout_50_50/analysis"))
    args = p.parse_args()

    split_rows = []
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        train = (args.root / model / "splits/fold0/train_uids.txt").read_text(encoding="utf-8").splitlines()
        eval_ = (args.root / model / "splits/fold0/eval_uids.txt").read_text(encoding="utf-8").splitlines()
        split_rows.append(
            {
                "model": model,
                "train_n": len(train),
                "eval_n": len(eval_),
                "intersection_n": len(set(train) & set(eval_)),
            }
        )
        _add_forward(args.root, model, "forward_trivia", rows)
        _add_forward(args.root, model, "forward_piqa", rows)
        _add_source_user(args.root, model, rows)
        for task in ("mitigation_trivia", "mitigation_piqa"):
            for variant in ("residualized", "assistant", "caa"):
                _add_mitigation(args.root, model, task, variant, rows)

    rows.sort(key=lambda r: (r["model"], r["task"], r["variant"], r["condition"]))
    comparison = _add_comparison(rows)
    comparison.sort(key=lambda r: (r["model"], r["task"], r["variant"], r["condition"]))

    metrics_fields = [
        "model",
        "task",
        "variant",
        "condition",
        "layer",
        "alpha",
        "heldout_wrong_rate",
        "heldout_matched_flip",
        "n",
        "parsed",
        "selected_by",
        "flips",
    ]
    comparison_fields = metrics_fields + [
        "heldout_metric",
        "heldout_value_pp",
        "insample_metric",
        "insample_value_pp",
        "delta_pp",
        "insample_layer",
        "insample_alpha",
        "insample_source",
    ]
    split_fields = ["model", "train_n", "eval_n", "intersection_n"]

    _write_csv(args.out_dir / "exp22_split_check.csv", split_rows, split_fields)
    _write_csv(args.out_dir / "exp22_heldout_headline_metrics.csv", rows, metrics_fields)
    _write_csv(args.out_dir / "exp22_heldout_vs_insample.csv", comparison, comparison_fields)

    by_model = defaultdict(int)
    for row in rows:
        by_model[row["model"]] += 1
    print(f"wrote {len(rows)} held-out rows to {args.out_dir / 'exp22_heldout_headline_metrics.csv'}")
    print(f"wrote {len(comparison)} comparison rows to {args.out_dir / 'exp22_heldout_vs_insample.csv'}")
    print("rows_by_model", dict(sorted(by_model.items())))
    print("split_rows", split_rows)


if __name__ == "__main__":
    main()
