from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_OUT_DIR = Path("data")
DEFAULT_REDBENCH = "knoveleng/redbench"
DEFAULT_SORRYBENCH = "sorry-bench/sorry-bench-202503"

COMMON_PROMPT_FIELDS = (
    "prompt",
    "instruction",
    "query",
    "question",
    "goal",
    "behavior",
    "Behavior",
    "attack_prompt",
    "user_prompt",
    "request",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare sampled prompt JSONLs for RedBench / SORRY-Bench runs."
    )
    p.add_argument("--bench", choices=("redbench", "sorrybench"), required=True)
    p.add_argument("--output-path", type=Path, default=None)
    p.add_argument("--max-prompts", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)

    # HF source options
    p.add_argument("--dataset-name", type=str, default=None)
    p.add_argument("--dataset-configs", type=str, default=None,
                   help="Comma list of configs (for RedBench).")
    p.add_argument("--dataset-split", type=str, default=None,
                   help="Split name; if omitted picks first available split.")

    # Filtering
    p.add_argument("--exclude-config-substrings", type=str, default="",
                   help="Comma list; exclude configs containing these (e.g. HarmBench).")
    p.add_argument("--include-config-substrings", type=str, default="",
                   help="Comma list; keep only configs containing these.")

    # Optional fallback input
    p.add_argument("--input-jsonl", type=Path, default=None,
                   help="Use local JSONL with a `prompt` (or common prompt field).")
    return p.parse_args()


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _pick_split(ds_dict: Any, split_hint: Optional[str]) -> str:
    keys = list(ds_dict.keys())
    if not keys:
        raise ValueError("No splits found in dataset.")
    if split_hint:
        if split_hint not in ds_dict:
            raise ValueError(f"Split '{split_hint}' not found. Available: {keys}")
        return split_hint
    for cand in ("train", "test", "validation"):
        if cand in ds_dict:
            return cand
    return keys[0]


def _extract_prompt_from_row(row: Dict[str, Any]) -> Optional[str]:
    for key in COMMON_PROMPT_FIELDS:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # Nested fallback
    for v in row.values():
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str) and item.strip():
                    return item.strip()
                if isinstance(item, dict):
                    nested = _extract_prompt_from_row(item)
                    if nested:
                        return nested
        if isinstance(v, dict):
            nested = _extract_prompt_from_row(v)
            if nested:
                return nested
    return None


def _load_jsonl_prompts(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = _extract_prompt_from_row(row)
            if prompt:
                rows.append({"prompt": prompt, "source": "local_jsonl"})
    return rows


def _should_keep_config(
    config_name: str,
    include_substrings: Sequence[str],
    exclude_substrings: Sequence[str],
) -> bool:
    low = config_name.lower()
    if include_substrings and not any(s.lower() in low for s in include_substrings):
        return False
    if exclude_substrings and any(s.lower() in low for s in exclude_substrings):
        return False
    return True


def _load_redbench_rows(
    *,
    dataset_name: str,
    configs: Optional[Sequence[str]],
    split_hint: Optional[str],
    include_substrings: Sequence[str],
    exclude_substrings: Sequence[str],
) -> List[Dict[str, Any]]:
    from datasets import get_dataset_config_names, load_dataset

    all_configs = configs or get_dataset_config_names(dataset_name)
    kept_configs = [
        c for c in all_configs
        if _should_keep_config(c, include_substrings=include_substrings, exclude_substrings=exclude_substrings)
    ]
    if not kept_configs:
        raise ValueError("No RedBench configs left after include/exclude filters.")

    rows: List[Dict[str, Any]] = []
    for cfg in kept_configs:
        ds_dict = load_dataset(dataset_name, cfg)
        split = _pick_split(ds_dict, split_hint)
        ds = ds_dict[split]
        for row in ds:
            prompt = _extract_prompt_from_row(dict(row))
            if not prompt:
                continue
            rows.append({"prompt": prompt, "source": f"{dataset_name}:{cfg}:{split}"})
    return rows


def _load_sorrybench_rows(
    *,
    dataset_name: str,
    split_hint: Optional[str],
) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds_dict = load_dataset(dataset_name)
    split = _pick_split(ds_dict, split_hint)
    ds = ds_dict[split]
    rows: List[Dict[str, Any]] = []
    for row in ds:
        prompt = _extract_prompt_from_row(dict(row))
        if not prompt:
            continue
        rows.append({"prompt": prompt, "source": f"{dataset_name}:{split}"})
    return rows


def _dedupe_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        prompt = row["prompt"].strip()
        key = " ".join(prompt.lower().split())
        if not prompt or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _sample_rows(rows: List[Dict[str, Any]], max_prompts: int, seed: int) -> List[Dict[str, Any]]:
    if max_prompts <= 0 or len(rows) <= max_prompts:
        return rows
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    selected = sorted(idx[:max_prompts])
    return [rows[i] for i in selected]


def main() -> None:
    args = parse_args()

    if args.output_path is None:
        args.output_path = DEFAULT_OUT_DIR / f"{args.bench}_sampled_prompts.jsonl"
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.input_jsonl is not None:
        raw_rows = _load_jsonl_prompts(args.input_jsonl)
    else:
        if args.bench == "redbench":
            dataset_name = args.dataset_name or DEFAULT_REDBENCH
            configs = _parse_csv(args.dataset_configs) if args.dataset_configs else None
            include_substrings = _parse_csv(args.include_config_substrings)
            exclude_substrings = _parse_csv(args.exclude_config_substrings)
            raw_rows = _load_redbench_rows(
                dataset_name=dataset_name,
                configs=configs,
                split_hint=args.dataset_split,
                include_substrings=include_substrings,
                exclude_substrings=exclude_substrings,
            )
        else:
            dataset_name = args.dataset_name or DEFAULT_SORRYBENCH
            try:
                raw_rows = _load_sorrybench_rows(
                    dataset_name=dataset_name,
                    split_hint=args.dataset_split,
                )
            except Exception as exc:
                raise SystemExit(
                    "Failed to load SORRY-Bench from HF (possibly gated). "
                    "Use --input-jsonl with a local prompt file.\n"
                    f"Error: {exc}"
                ) from exc

    deduped = _dedupe_rows(raw_rows)
    sampled = _sample_rows(deduped, max_prompts=args.max_prompts, seed=args.seed)

    with args.output_path.open("w") as f:
        for i, row in enumerate(sampled):
            f.write(json.dumps({
                "prompt": row["prompt"],
                "source": row["source"],
                "row_id": i,
            }) + "\n")

    print(f"bench={args.bench}")
    print(f"raw_rows={len(raw_rows)} deduped={len(deduped)} sampled={len(sampled)}")
    print(f"saved={args.output_path}")


if __name__ == "__main__":
    main()
