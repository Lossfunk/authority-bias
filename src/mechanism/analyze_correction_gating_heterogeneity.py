from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


DEFAULT_ENTRENCHING_DIR = Path(
    "new-phase-results/mechanism/qwen_correction_gating_patching_endorsement_l23"
)
DEFAULT_CORRECTING_DIR = Path(
    "new-phase-results/mechanism/qwen_correction_gating_patching_c1note_correcting"
)
DEFAULT_OUTPUT_PATH = Path(
    "new-phase-results/mechanism/qwen_correction_gating_heterogeneity_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze item-level heterogeneity in correction-gating patching runs."
    )
    parser.add_argument("--entrenching-dir", type=Path, default=DEFAULT_ENTRENCHING_DIR)
    parser.add_argument("--correcting-dir", type=Path, default=DEFAULT_CORRECTING_DIR)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--control", type=str, default="real")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_rows(item_results_path: Path, control: str, alpha: float) -> List[Dict]:
    rows = [
        row
        for row in _iter_jsonl(item_results_path)
        if row["control"] == control and math.isclose(float(row["alpha"]), alpha)
    ]
    if not rows:
        raise SystemExit(
            f"No rows found in {item_results_path} for control={control!r}, alpha={alpha:g}"
        )
    return rows


def _safe_mean(xs: Sequence[float]) -> float:
    return float(mean(xs)) if xs else math.nan


def _safe_median(xs: Sequence[float]) -> float:
    return float(median(xs)) if xs else math.nan


def _margin_shift(row: Dict) -> float:
    return float(row["patched_signed_margin"]) - float(row["baseline_signed_margin"])


def _uid_source(uid: str) -> str:
    return uid.split("::", 1)[0] if "::" in uid else uid


def _stats_for_rows(rows: Sequence[Dict]) -> Dict[str, float]:
    margins = [float(r["baseline_signed_margin"]) for r in rows]
    shifts = [_margin_shift(r) for r in rows]
    return {
        "n_items": len(rows),
        "mean_baseline_margin": _safe_mean(margins),
        "median_baseline_margin": _safe_median(margins),
        "mean_margin_shift": _safe_mean(shifts),
        "median_margin_shift": _safe_median(shifts),
    }


def _summarize_split(
    rows: Sequence[Dict],
    predicate: Callable[[Dict], bool],
    positive_name: str,
    negative_name: str,
) -> Dict[str, object]:
    pos = [r for r in rows if predicate(r)]
    neg = [r for r in rows if not predicate(r)]
    return {
        positive_name: _stats_for_rows(pos),
        negative_name: _stats_for_rows(neg),
    }


def _group_summary(
    rows: Sequence[Dict],
    key_fn: Callable[[Dict], str],
    extra_fn: Callable[[Sequence[Dict]], Dict[str, object]] | None = None,
) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    out: Dict[str, Dict[str, object]] = {}
    for key, group in sorted(grouped.items()):
        record: Dict[str, object] = _stats_for_rows(group)
        record["positive_shift_rate"] = _safe_mean(
            [1.0 if _margin_shift(r) > 0 else 0.0 for r in group]
        )
        record["flip_to_correct_rate"] = _safe_mean(
            [1.0 if (not r["baseline_correct"] and r["patched_correct"]) else 0.0 for r in group]
        )
        record["harm_rate"] = _safe_mean(
            [1.0 if (r["baseline_correct"] and not r["patched_correct"]) else 0.0 for r in group]
        )
        if extra_fn is not None:
            record.update(extra_fn(group))
        out[key] = record
    return out


def _bin_label(value: float, boundaries: Sequence[float]) -> str:
    lower = -math.inf
    for upper in boundaries:
        if value <= upper:
            return f"({lower:g}, {upper:g}]"
        lower = upper
    return f"({lower:g}, inf)"


def _entrenching_summary(rows: Sequence[Dict]) -> Dict[str, object]:
    flipped = [r for r in rows if (not r["baseline_correct"] and r["patched_correct"])]
    non_flipped = [r for r in rows if not (not r["baseline_correct"] and r["patched_correct"])]
    positive_shift = [r for r in rows if _margin_shift(r) > 0]

    def _extra(group: Sequence[Dict]) -> Dict[str, object]:
        return {
            "n_flipped": sum(1 for r in group if (not r["baseline_correct"] and r["patched_correct"])),
            "n_positive_shift": sum(1 for r in group if _margin_shift(r) > 0),
        }

    return {
        "overall": {
            **_stats_for_rows(rows),
            "n_flipped": len(flipped),
            "flip_rate": len(flipped) / len(rows),
            "n_positive_shift": len(positive_shift),
            "positive_shift_rate": len(positive_shift) / len(rows),
        },
        "flipped_vs_nonflipped": {
            "flipped": _stats_for_rows(flipped),
            "non_flipped": _stats_for_rows(non_flipped),
        },
        "by_source": _group_summary(rows, key_fn=lambda r: _uid_source(r["uid"]), extra_fn=_extra),
        "by_margin_bin": _group_summary(
            rows,
            key_fn=lambda r: _bin_label(float(r["baseline_signed_margin"]), [-12.0, -8.0, -4.0, -2.0, 0.0]),
            extra_fn=_extra,
        ),
    }


def _correcting_summary(rows: Sequence[Dict]) -> Dict[str, object]:
    harmed = [r for r in rows if (r["baseline_correct"] and not r["patched_correct"])]
    unharmed = [r for r in rows if not (r["baseline_correct"] and not r["patched_correct"])]
    flipped = [r for r in rows if (not r["baseline_correct"] and r["patched_correct"])]

    def _extra(group: Sequence[Dict]) -> Dict[str, object]:
        return {
            "n_harmed": sum(1 for r in group if (r["baseline_correct"] and not r["patched_correct"])),
            "n_flipped": sum(1 for r in group if (not r["baseline_correct"] and r["patched_correct"])),
            "n_baseline_correct": sum(1 for r in group if r["baseline_correct"]),
        }

    return {
        "overall": {
            **_stats_for_rows(rows),
            "n_harmed": len(harmed),
            "harm_rate": len(harmed) / len(rows),
            "n_flipped": len(flipped),
            "flip_rate": len(flipped) / len(rows),
            "n_baseline_correct": sum(1 for r in rows if r["baseline_correct"]),
            "baseline_correct_rate": _safe_mean([1.0 if r["baseline_correct"] else 0.0 for r in rows]),
        },
        "harmed_vs_unharmed": {
            "harmed": _stats_for_rows(harmed),
            "unharmed": _stats_for_rows(unharmed),
        },
        "by_baseline_correctness": _group_summary(
            rows,
            key_fn=lambda r: "baseline_correct" if r["baseline_correct"] else "baseline_wrong",
            extra_fn=_extra,
        ),
        "by_source": _group_summary(rows, key_fn=lambda r: _uid_source(r["uid"]), extra_fn=_extra),
        "by_margin_bin": _group_summary(
            rows,
            key_fn=lambda r: _bin_label(float(r["baseline_signed_margin"]), [0.0, 1.0, 3.0, 6.0, 9.0]),
            extra_fn=_extra,
        ),
    }


def main() -> None:
    args = parse_args()

    entrenching_rows = _load_rows(
        args.entrenching_dir / "item_results.jsonl", control=args.control, alpha=args.alpha
    )
    correcting_rows = _load_rows(
        args.correcting_dir / "item_results.jsonl", control=args.control, alpha=args.alpha
    )

    summary = {
        "alpha": args.alpha,
        "control": args.control,
        "entrenching_input_dir": str(args.entrenching_dir),
        "correcting_input_dir": str(args.correcting_dir),
        "entrenching": _entrenching_summary(entrenching_rows),
        "correcting": _correcting_summary(correcting_rows),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
