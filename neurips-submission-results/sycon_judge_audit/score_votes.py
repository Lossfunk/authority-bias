#!/usr/bin/env python3
"""Score exported SYCON human audit votes against stored LLM-judge labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def cohen_kappa(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    cm = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    for h, j in pairs:
        cm[(h, j)] += 1
    n = len(pairs)
    agree = cm[(0, 0)] + cm[(1, 1)]
    if n == 0:
        return {"n": 0, "agreement": None, "kappa": None, "confusion": cm}
    p0 = agree / n
    h0, h1 = cm[(0, 0)] + cm[(0, 1)], cm[(1, 0)] + cm[(1, 1)]
    j0, j1 = cm[(0, 0)] + cm[(1, 0)], cm[(0, 1)] + cm[(1, 1)]
    pe = (h0 / n) * (j0 / n) + (h1 / n) * (j1 / n)
    kappa = None if abs(1 - pe) < 1e-12 else (p0 - pe) / (1 - pe)
    return {
        "n": n,
        "agreement": p0,
        "chance_agreement": pe,
        "kappa": kappa,
        "confusion_human_judge": {f"{h},{j}": v for (h, j), v in cm.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("votes_json", type=Path)
    args = p.parse_args()
    payload = json.loads(args.votes_json.read_text(encoding="utf-8"))
    pairs: list[tuple[int, int]] = []
    skipped = 0
    for row in payload.get("votes", []):
        if row.get("skipped"):
            skipped += 1
            continue
        h = row.get("human_label")
        j = row.get("judge_label")
        if h in (0, 1) and j in (0, 1):
            pairs.append((int(h), int(j)))
    out = cohen_kappa(pairs)
    out["skipped"] = skipped
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
