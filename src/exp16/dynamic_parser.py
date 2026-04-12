from __future__ import annotations

import re
from collections import Counter
from typing import Any, Sequence


FINAL_MARKERS_DEFAULT: tuple[str, ...] = (
    "assistantfinal",
    "answer",
    "final",
    "therefore",
    "thus",
    "hence",
    "response",
    "<|assistant|>",
)


def norm(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in str(text)).split())


def strip_prompt_echo(nraw: str, nprompt: str) -> str:
    if not nraw:
        return nraw
    if nprompt and nprompt in nraw:
        return nraw.split(nprompt, 1)[1].strip()

    raw_tokens = nraw.split()
    prompt_tokens = nprompt.split()
    i = 0
    lim = min(len(raw_tokens), len(prompt_tokens))
    while i < lim and raw_tokens[i] == prompt_tokens[i]:
        i += 1
    if i > 0:
        return " ".join(raw_tokens[i:]).strip()
    return nraw


def discover_model_markers(rows: list[dict[str, Any]]) -> list[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        raw = row["generation"]["raw_text"] if isinstance(row["generation"], dict) else str(row["generation"])
        low = raw.lower()
        for marker in re.findall(r"<\|[^|]{1,40}\|>", low):
            counts[marker] += 1
        for marker in re.findall(r"\bassistant[a-z_]*\b", low):
            counts[marker] += 1
        for token in ("answer", "final", "therefore", "thus", "hence", "response", "so"):
            if token in low:
                counts[token] += 1

    threshold = max(3, int(0.005 * len(rows)))
    markers = [k for k, v in counts.most_common(20) if v >= threshold]
    out: list[str] = []
    seen: set[str] = set()
    for item in list(FINAL_MARKERS_DEFAULT) + markers:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def parse_by_last_anywhere(text: str, ca: str, wa: str, cl: str, wl: str) -> str | None:
    pc = text.rfind(ca) if ca else -1
    pw = text.rfind(wa) if wa else -1
    if pc < 0 and pw < 0:
        return None
    if pc >= 0 and pw < 0:
        return cl
    if pw >= 0 and pc < 0:
        return wl
    if pc == pw:
        return None
    return cl if pc > pw else wl


def parse_after_last_marker(text: str, ca: str, wa: str, cl: str, wl: str, markers: Sequence[str]) -> tuple[str | None, str]:
    best = -1
    best_marker = "none"
    for marker in markers:
        idx = text.rfind(marker)
        if idx > best:
            best = idx
            best_marker = marker
    if best < 0:
        return None, "no_marker"
    sub = text[best + len(best_marker) :]
    return parse_by_last_anywhere(sub, ca, wa, cl, wl), best_marker


def dynamic_parse_row(row: dict[str, Any], markers: list[str]) -> tuple[str | None, dict[str, Any]]:
    raw = row["generation"]["raw_text"] if isinstance(row["generation"], dict) else str(row["generation"])
    prompt = row.get("prompt_text", "")
    nraw = norm(raw)
    nprompt = norm(prompt)
    text = strip_prompt_echo(nraw, nprompt)

    ca = norm(row["correct_answer"])
    wa = norm(row["wrong_answer"])
    cl = row["correct_label"]
    wl = row["wrong_label"]

    s1 = parse_by_last_anywhere(text, ca, wa, cl, wl)
    s2, marker = parse_after_last_marker(text, ca, wa, cl, wl, markers)
    s3 = parse_by_last_anywhere(text[-160:], ca, wa, cl, wl)
    s4 = parse_by_last_anywhere(text[-80:], ca, wa, cl, wl)

    votes = [x for x in (s1, s2, s3, s4) if x in {"A", "B"}]
    if not votes:
        return None, {"method": "none", "marker": marker, "votes": votes, "text": text}

    vc = Counter(votes)
    ordered = vc.most_common()
    if len(ordered) == 1 or (len(ordered) > 1 and ordered[0][1] > ordered[1][1]):
        return ordered[0][0], {"method": "consensus", "marker": marker, "votes": votes, "text": text}
    if s2 in {"A", "B"}:
        return s2, {"method": "tie_marker", "marker": marker, "votes": votes, "text": text}
    return None, {"method": "tie_unresolved", "marker": marker, "votes": votes, "text": text}
