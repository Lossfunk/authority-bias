from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

try:
    from src.authority_steering.run_harmbench_interventions import classify_harmbench_response
except ModuleNotFoundError:
    # Allow direct script execution: `python src/authority_steering/relabel_harmbench_rows.py ...`
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.authority_steering.run_harmbench_interventions import classify_harmbench_response


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-label existing HarmBench rows with the current response parser."
    )
    p.add_argument("--rows-path", type=Path, required=True, help="Input harmbench_rows.jsonl path.")
    p.add_argument(
        "--output-rows-path",
        type=Path,
        default=None,
        help="Output rows path (defaults to <rows-stem>.reparsed.jsonl).",
    )
    p.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Output summary path (defaults near output rows).",
    )
    p.add_argument("--in-place", action="store_true", help="Rewrite --rows-path in place.")
    p.add_argument(
        "--backup-suffix",
        type=str,
        default=".bak",
        help="Backup suffix used when --in-place is set.",
    )
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs.")
    p.add_argument("--print-top", type=int, default=12, help="How many top compliance configs to print.")
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.in_place:
        rows_out = args.rows_path
    elif args.output_rows_path is not None:
        rows_out = args.output_rows_path
    else:
        rows_out = args.rows_path.with_name(f"{args.rows_path.stem}.reparsed{args.rows_path.suffix}")

    if args.summary_path is not None:
        summary_out = args.summary_path
    else:
        if args.in_place:
            summary_out = args.rows_path.with_name("harmbench_summary.reparsed.json")
        else:
            stem = rows_out.stem.replace("rows", "summary")
            if stem == rows_out.stem:
                stem = f"{rows_out.stem}.summary"
            summary_out = rows_out.with_name(f"{stem}.json")
    return rows_out, summary_out


def _iter_rows(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    args = parse_args()
    rows_out, summary_out = _resolve_paths(args)

    if not args.rows_path.exists():
        raise SystemExit(f"Input rows file not found: {args.rows_path}")

    if args.in_place:
        backup_path = args.rows_path.with_name(args.rows_path.name + args.backup_suffix)
        if backup_path.exists() and not args.overwrite:
            raise SystemExit(f"Backup already exists: {backup_path} (use --overwrite to replace).")
        backup_path.write_bytes(args.rows_path.read_bytes())

    if rows_out.exists() and rows_out != args.rows_path and not args.overwrite:
        raise SystemExit(f"Output rows exists: {rows_out} (use --overwrite to replace).")
    if summary_out.exists() and not args.overwrite:
        raise SystemExit(f"Summary exists: {summary_out} (use --overwrite to replace).")

    rows_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    tmp_out = rows_out.with_name(rows_out.name + ".tmp")

    class_counts = Counter()
    grouped = defaultdict(
        lambda: {
            "n": 0,
            "refusal": 0,
            "compliance": 0,
            "likely": 0,
            "degenerate": 0,
            "uncertain": 0,
            "len_sum": 0.0,
            "example": None,
        }
    )

    n_rows = 0
    with tmp_out.open("w") as fw:
        for row in _iter_rows(args.rows_path):
            n_rows += 1
            response = str(row.get("response") or row.get("raw_text") or "")
            response_class = classify_harmbench_response(
                response,
                prompt=str(row.get("prompt") or ""),
                apply_phase=str(row.get("apply_phase") or ""),
            )
            is_refusal = response_class in {"refusal", "likely_refusal_truncated"}
            is_compliance = response_class == "compliance"

            row["response_class"] = response_class
            row["is_refusal"] = is_refusal
            row["is_compliance"] = is_compliance
            row["response_len_chars"] = int(row.get("response_len_chars") or len(response.strip()))

            cfg = str(row.get("config_id") or "unknown")
            g = grouped[cfg]
            g["n"] += 1
            g["len_sum"] += float(row["response_len_chars"])
            if g["example"] is None:
                g["example"] = row
            if response_class == "refusal":
                g["refusal"] += 1
            elif response_class == "likely_refusal_truncated":
                g["likely"] += 1
            elif response_class == "degenerate":
                g["degenerate"] += 1
            elif response_class == "compliance":
                g["compliance"] += 1
            else:
                g["uncertain"] += 1
            class_counts[response_class] += 1

            fw.write(json.dumps(row, default=str) + "\n")

    tmp_out.replace(rows_out)

    summary = []
    for cfg in sorted(grouped):
        g = grouped[cfg]
        n = max(int(g["n"]), 1)
        ex = g["example"] or {}
        summary.append(
            {
                "config_id": cfg,
                "n": int(g["n"]),
                "mode": ex.get("mode"),
                "alpha": ex.get("alpha"),
                "direction_kind": ex.get("direction_kind"),
                "sparse_topk": ex.get("sparse_topk"),
                "target_layers": ex.get("target_layers"),
                "position_mode": ex.get("position_mode"),
                "apply_phase": ex.get("apply_phase"),
                "refusal_rate": (g["refusal"] + g["likely"]) / n,
                "compliance_rate": g["compliance"] / n,
                "effective_compliance_rate": g["compliance"] / n,
                "likely_refusal_truncated_rate": g["likely"] / n,
                "degenerate_rate": g["degenerate"] / n,
                "uncertain_rate": g["uncertain"] / n,
                "mean_response_len_chars": g["len_sum"] / n,
            }
        )

    summary_out.write_text(json.dumps(summary, indent=2))

    print(f"Rows relabeled: {n_rows}")
    print(f"Wrote rows: {rows_out}")
    print(f"Wrote summary: {summary_out}")
    print("Class counts:", dict(class_counts))

    total = max(n_rows, 1)
    refusal_plus_likely = (class_counts["refusal"] + class_counts["likely_refusal_truncated"]) / total
    print(f"Global refusal+likely rate: {refusal_plus_likely:.4f}")

    top_n = max(args.print_top, 0)
    if top_n:
        print(f"\nTop {top_n} configs by compliance_rate:")
        for s in sorted(summary, key=lambda x: x["compliance_rate"], reverse=True)[:top_n]:
            print(
                f"{s['config_id'][:42]:42s} "
                f"comp={s['compliance_rate']:.3f} "
                f"ref+likely={s['refusal_rate']:.3f} "
                f"likely={s['likely_refusal_truncated_rate']:.3f} "
                f"deg={s['degenerate_rate']:.3f}"
            )


if __name__ == "__main__":
    main()
