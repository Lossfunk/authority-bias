# Forward Patch Test 256: Recomputed Summary

This summary was verified against the steering parser in `src/exp16/dynamic_parser.py`.

I reran the parser logic over `steering_rows.jsonl` using:

- `discover_model_markers(...)`
- `dynamic_parse_row(...)`

and compared the outputs to the stored row fields:

- `parsed_label`
- `is_correct`
- `chose_wrong`

Result:

- `mismatch_count = 0`

So the stored parsed fields already match the steering parser exactly, and the percentages below are parser-consistent.

The main causal metric for this forward test is:

- **Flip rate among baseline-correct items**: among items answered correctly at `alpha=0`, what fraction switch to the wrong answer after patching in the W1 authority state.

Notes:

- The shipped `steering_summary.json` reports `accuracy` and `wrong_rate` correctly, but `flip_rate` is `null` because the code only fills it for condition `W1_note`, while this run uses condition `N0_note`.
- Total items per config: `212`
- Baseline-correct items at `alpha=0`: `168`

## Recomputed Results

| Layers | Alpha | Accuracy | Wrong rate | Baseline-correct n | Flips to wrong | Flip rate |
|---|---:|---:|---:|---:|---:|---:|
| `L16` | 0.0 | 79.25% | 20.75% | 168 | 0 | 0.00% |
| `L16` | 0.3 | 78.30% | 21.70% | 168 | 10 | 5.95% |
| `L16` | 0.5 | 78.77% | 21.23% | 168 | 15 | 8.93% |
| `L16` | 0.7 | 75.47% | 24.53% | 168 | 30 | 17.86% |
| `L16` | 1.0 | 62.74% | 37.26% | 168 | 55 | 32.74% |
| `L18` | 0.0 | 79.25% | 20.75% | 168 | 0 | 0.00% |
| `L18` | 0.3 | 77.36% | 22.64% | 168 | 10 | 5.95% |
| `L18` | 0.5 | 79.72% | 20.28% | 168 | 15 | 8.93% |
| `L18` | 0.7 | 80.19% | 19.81% | 168 | 15 | 8.93% |
| `L18` | 1.0 | 78.77% | 21.23% | 168 | 19 | 11.31% |
| `L20` | 0.0 | 79.25% | 20.75% | 168 | 0 | 0.00% |
| `L20` | 0.3 | 78.77% | 21.23% | 168 | 7 | 4.17% |
| `L20` | 0.5 | 82.08% | 17.92% | 168 | 5 | 2.98% |
| `L20` | 0.7 | 79.25% | 20.75% | 168 | 10 | 5.95% |
| `L20` | 1.0 | 81.60% | 18.40% | 168 | 7 | 4.17% |

## Interpretation

- `L16` shows a strong forward causal effect with a clear dose response:
  - `0.00% -> 5.95% -> 8.93% -> 17.86% -> 32.74%`
- `L18` shows a weaker but still nonzero forward effect:
  - peaks at `11.31%` flip rate at `alpha=1.0`
- `L20` shows little to no forward harmful effect in this setup:
  - stays in the `2.98% - 5.95%` range and even improves raw accuracy at some alphas

## Main Takeaway

Within-domain forward causality is present, but it is highly layer-dependent:

- strongest at `L16`
- modest at `L18`
- weak / near-flat at `L20`

That closes the causal loop more cleanly than the old summary, because it measures exactly the subset you care about: baseline-correct `N0` items that are pushed into wrong answers by adding the W1 authority state.
