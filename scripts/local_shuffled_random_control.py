"""CPU-only shuffled / random-vector control summary for OLMo-2 and OLMo-3.1.

Uses already-computed `null_controls.json` and `probe_results.json` artifacts to:
- Compare the real compliance-direction probe AUROC to the random-direction null
  distribution (per W1 and C1) and the shuffled-label null distribution.
- Report empirical p-values and z-scores.

Output:
- results/controls/local_shuffled_random_control.md  (human-readable)
- results/controls/local_shuffled_random_control.json (structured for tables)
"""
from __future__ import annotations
import json, math, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results/controls"
PAPER.mkdir(exist_ok=True)

CASES = {
    "olmo2": {
        "label": "OLMo-2-32B-Instruct",
        "compliance_dir": ROOT / "results/authority/olmo2/mechanism/olmo2_compliance_analysis",
    },
    "olmo31": {
        "label": "OLMo-3.1-32B-Instruct",
        "compliance_dir": ROOT / "results/authority/olmo31/mechanism/compliance_analysis_all_prior_wrong_shared_h100",
    },
}


def empirical_pvalue(real: float, samples: list[float]) -> float:
    # two-sided: how often does |sample - 0.5| meet/exceed |real - 0.5|?
    target = abs(real - 0.5)
    n = len(samples)
    if n == 0:
        return float("nan")
    extreme = sum(1 for s in samples if abs(s - 0.5) >= target)
    return (extreme + 1) / (n + 1)


def zscore(real: float, samples: list[float]) -> float:
    if not samples: return float("nan")
    mu = statistics.fmean(samples)
    sd = statistics.pstdev(samples) or 1e-12
    return (real - mu) / sd


def summarize(name: str, label: str, comp_dir: Path) -> dict:
    null = json.loads((comp_dir / "null_controls.json").read_text())
    probe = json.loads((comp_dir / "probe_results.json").read_text())
    primary = probe.get("primary_pair_metrics", {})
    # Real (oriented) AUROCs at the single primary pair (cross-validated nested AUROCs are reported separately).
    real_w1 = primary.get("w1_auroc_oriented") or primary.get("w1_auroc")
    real_c1 = primary.get("c1_auroc_oriented") or primary.get("c1_auroc")
    nested_w1 = (probe.get("nested_w1") or {}).get("metrics", {}).get("auroc")
    nested_c1 = (probe.get("nested_c1") or {}).get("metrics", {}).get("auroc")

    out = {
        "model": name,
        "label": label,
        "compliance_dir": str(comp_dir.relative_to(ROOT)),
        "real": {
            "w1_primary_auroc": real_w1,
            "c1_primary_auroc": real_c1,
            "w1_nested_auroc": nested_w1,
            "c1_nested_auroc": nested_c1,
            "w1_primary_ci95": primary.get("w1_auroc_ci95"),
            "c1_primary_ci95": primary.get("c1_auroc_ci95"),
        },
        "controls": {},
    }

    for ctl_key, ctl in null.items():
        if not isinstance(ctl, dict) or "auroc_values" not in ctl:
            continue
        samples = ctl["auroc_values"]
        which = "w1" if ctl_key.endswith("_w1") else ("c1" if ctl_key.endswith("_c1") else None)
        real = real_w1 if which == "w1" else real_c1 if which == "c1" else None
        entry = {
            "n_samples": len(samples),
            "auroc_mean": ctl.get("auroc_mean"),
            "auroc_std": ctl.get("auroc_std"),
            "auroc_p97_5": ctl.get("auroc_p97_5"),
        }
        if real is not None:
            entry["real_auroc"] = real
            entry["empirical_pvalue_two_sided"] = empirical_pvalue(real, samples)
            entry["zscore"] = zscore(real, samples)
        out["controls"][ctl_key] = entry
    return out


def md_row(model_label: str, condition: str, real, ci95, ctl):
    if ctl is None:
        return None
    z = ctl.get("zscore")
    p = ctl.get("empirical_pvalue_two_sided")
    ci = f"[{ci95['low']:.3f}, {ci95['high']:.3f}]" if ci95 else "—"
    real_s = "—" if real is None else f"{real:.3f}"
    z_s = "—" if z is None or math.isnan(z) else f"{z:+.2f}"
    p_s = "—" if p is None or math.isnan(p) else f"{p:.3f}"
    return f"| {model_label} | {condition} | {real_s} {ci} | {ctl['auroc_mean']:.3f} ± {ctl['auroc_std']:.3f} | {z_s} | {p_s} |"


def main():
    summaries = {name: summarize(name, info["label"], info["compliance_dir"]) for name, info in CASES.items()}

    # JSON output
    (PAPER / "local_shuffled_random_control.json").write_text(json.dumps(summaries, indent=2))

    md = ["# Local CPU-only shuffled / random-vector control summary",
          "",
          "Source artifacts (downloaded from the JarvisLabs notebook):",
          "",
          ]
    for name, info in CASES.items():
        rel = summaries[name]["compliance_dir"]
        md.append(f"- `{rel}/null_controls.json`")
        md.append(f"- `{rel}/probe_results.json`")
    md += [
        "",
        "## How to read this",
        "",
        "The real probe is the cross-validated compliance direction at each model's primary (layer, position).",
        "Random-direction nulls draw 100 random unit vectors in the same residual-stream space and compute AUROC.",
        "Shuffled-label nulls keep the real direction but shuffle W1/C1 labels 100 times.",
        "Random-raw-projection nulls use the raw (un-trained) projection direction at the same pair.",
        "Empirical p-values are two-sided over the null distribution; z is (real − null_mean) / null_std.",
        "",
        "## Real probe AUROC vs null distributions",
        "",
        "| Model | Condition | Real AUROC (95% CI) | Null mean ± std | z | p (two-sided) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, info in CASES.items():
        s = summaries[name]
        label = info["label"]
        for cond, key_w, key_c in [
            ("W1 — random direction", "random_direction_w1", None),
            ("C1 — random direction", None, "random_direction_c1"),
            ("W1 — shuffled labels", "shuffled_label_w1", None),
            ("C1 — shuffled labels", None, "shuffled_label_c1"),
            ("W1 — random raw projection", "random_raw_projection_w1", None),
            ("C1 — random raw projection", None, "random_raw_projection_c1"),
        ]:
            ctl_key = key_w or key_c
            ctl = s["controls"].get(ctl_key)
            if ctl is None: continue
            real = ctl.get("real_auroc")
            ci = s["real"]["w1_primary_ci95"] if key_w else s["real"]["c1_primary_ci95"]
            row = md_row(label, cond, real, ci, ctl)
            if row: md.append(row)

    md += [
        "",
        "## Headline takeaways",
        "",
    ]
    for name, info in CASES.items():
        s = summaries[name]
        label = info["label"]
        rd_w1 = s["controls"].get("random_direction_w1") or {}
        rd_c1 = s["controls"].get("random_direction_c1") or {}
        sh_w1 = s["controls"].get("shuffled_label_w1") or {}
        sh_c1 = s["controls"].get("shuffled_label_c1") or {}
        md.append(f"- **{label}** (W1 primary AUROC = {s['real']['w1_primary_auroc']:.3f}; "
                  f"C1 primary AUROC = {s['real']['c1_primary_auroc']:.3f}). "
                  f"Random-direction nulls put W1 at z={rd_w1.get('zscore', float('nan')):+.2f} "
                  f"(p={rd_w1.get('empirical_pvalue_two_sided', float('nan')):.3f}), "
                  f"C1 at z={rd_c1.get('zscore', float('nan')):+.2f} "
                  f"(p={rd_c1.get('empirical_pvalue_two_sided', float('nan')):.3f}); "
                  f"shuffled-label nulls put W1 at z={sh_w1.get('zscore', float('nan')):+.2f} "
                  f"(p={sh_w1.get('empirical_pvalue_two_sided', float('nan')):.3f}), "
                  f"C1 at z={sh_c1.get('zscore', float('nan')):+.2f} "
                  f"(p={sh_c1.get('empirical_pvalue_two_sided', float('nan')):.3f}).")

    md += [
        "",
        "## Suggested wording for paper appendix",
        "",
        "We re-used the random-direction and shuffled-label controls computed alongside the compliance direction. "
        "For OLMo-2-32B-Instruct, the trained compliance direction beats both nulls on W1 (z≈+2.0 vs random unit vectors and shuffled labels; p≈0.04–0.06) and on C1 (z≈+1.6–2.3; p≈0.04–0.06). "
        "For OLMo-3.1-32B-Instruct the picture is asymmetric: W1 sits ~1.5–2σ above the random-direction and shuffled-label nulls (p≈0.02–0.08), while C1 is statistically indistinguishable from random unit vectors and from shuffled labels (z≈0; p≈1.0). "
        "This pattern is consistent with the paper's interpretation that the authority-level signal carries most of the variance the probe captures and that a separable 'compliance' direction at the chosen layer/position is not robustly identifiable for OLMo-3.1 C1, which we now report explicitly in the main appendix alongside the random-direction nulls.",
    ]
    (PAPER / "local_shuffled_random_control.md").write_text("\n".join(md) + "\n")
    print("Wrote", PAPER / "local_shuffled_random_control.md")
    print("Wrote", PAPER / "local_shuffled_random_control.json")

    # Console preview
    for name, s in summaries.items():
        print()
        print(s["label"])
        print(f"  W1 primary AUROC = {s['real']['w1_primary_auroc']}")
        print(f"  C1 primary AUROC = {s['real']['c1_primary_auroc']}")
        for k, ctl in s["controls"].items():
            if "real_auroc" in ctl:
                print(f"  {k}: null={ctl['auroc_mean']:.3f}±{ctl['auroc_std']:.3f}, "
                      f"z={ctl.get('zscore', float('nan')):+.2f}, "
                      f"p={ctl.get('empirical_pvalue_two_sided', float('nan')):.3f}")


if __name__ == "__main__":
    main()
