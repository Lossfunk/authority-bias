# Gemma4 Parser Recompute Summary

Generated from:
- `neurips-results/gemma4/mechanism/forward_patch_test_256/steering_rows.jsonl`
- `neurips-results/gemma4/mechanism/forward_patch_test_256/steering_summary.json`
- `neurips-results/gemma4/mechanism/gemma4_compliance_analysis/summary.json`
- `neurips-results/gemma4/mechanism/gemma4_authority_activations/summary.json`

## 1) Dynamic parser verification (forward patch run)

- Reparsed all `11,370` rows using `src/exp16/dynamic_parser.py`-compatible wrapper.
- Result: `mismatch_count = 0` against stored `parsed_label`, `is_correct`, `chose_wrong`.

This confirms the shipped row parsing is internally consistent.

## 2) Forward patch metrics (N0_note, w1 UID subset, n=758/config)

Baseline (all three layers share same alpha=0 behavior):
- Parsed: `751 / 758` (`99.08%`)
- Wrong among parsed: `17.04%`
- Wrong over all prompts: `16.89%`

| Layer | Alpha | Parsed | Parse rate | Wrong rate (parsed) | Wrong rate (all) |
|---|---:|---:|---:|---:|---:|
| 20 | 0.0 | 751 | 99.08% | 17.04% | 16.89% |
| 20 | 0.3 | 751 | 99.08% | 17.58% | 17.41% |
| 20 | 0.5 | 755 | 99.60% | 17.22% | 17.15% |
| 20 | 0.7 | 757 | 99.87% | 17.31% | 17.28% |
| 20 | 1.0 | 757 | 99.87% | 17.17% | 17.15% |
| 22 | 0.0 | 751 | 99.08% | 17.04% | 16.89% |
| 22 | 0.3 | 752 | 99.21% | 18.09% | 17.94% |
| 22 | 0.5 | 754 | 99.47% | 16.45% | 16.36% |
| 22 | 0.7 | 755 | 99.60% | 18.01% | 17.94% |
| 22 | 1.0 | 756 | 99.74% | 17.20% | 17.15% |
| 24 | 0.0 | 751 | 99.08% | 17.04% | 16.89% |
| 24 | 0.3 | 753 | 99.34% | 18.19% | 18.07% |
| 24 | 0.5 | 752 | 99.21% | 19.41% | 19.26% |
| 24 | 0.7 | 755 | 99.60% | 18.15% | 18.07% |
| 24 | 1.0 | 754 | 99.47% | 18.70% | 18.60% |

Best observed point:
- `L24, alpha=0.5`: `19.41%` wrong among parsed, `19.26%` wrong over all.

## 3) Matched-parsed delta (baseline vs patched on common parsed rows)

Largest matched-parsed increase:
- `L24, alpha=0.5`: baseline wrong `17.07%` -> patched wrong `19.47%` on common parsed rows (`+2.4` points).

Other matched-parsed deltas are small (`~+0.1` to `+1.7` points, with one negative at `L22, alpha=0.5`).

Interpretation:
- Signal exists but is modest for this Gemma forward patch run.
- Parse-rate is already near-saturated at baseline (`~99%`), so this is not mainly a parseability artifact.

## 4) Semantic parser applicability

In this rsync batch there are no HarmBench/RedBench-style rows under:
- `neurips-results/gemma4/mechanism/forward_patch_test_256/`
- `neurips-results/gemma4/mechanism/gemma4_compliance_analysis/`
- `neurips-results/gemma4/mechanism/gemma4_authority_activations/`

So the semantic safety relabeler (`src/exp16/relabel_harmbench_rows_semantic.py`) is not applicable to these files.

If you sync Gemma safety rows (`harmbench_rows.jsonl` / RedBench rows), run semantic relabel there.

## 5) Key Gemma mechanism metadata

From `gemma4_authority_activations/summary.json`:
- Model: `google/gemma-4-26B-A4B-it`
- Selected UIDs: `887`
- Records: `13,305`
- Included styles: `authoritative_verified`, `weak`, `uncertain`, `assertive`, `think`
- Target layers: `[2, 5, 10, 15, 18, 20, 22, 24, 26, 28, 30, 32, 35, 38]`

From `gemma4_compliance_analysis/summary.json`:
- Metadata rows: `13,305`
- Primary pair: `position=endorsement_mean`, `layer=22`
- Mask counts:
  - `mask_primary=842` (`w1=758`, `c1=84`)
  - `mask_cross=810` (`w1=731`, `c1=79`)
  - `mask_think=806`
