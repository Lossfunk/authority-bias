# Benchmark Results Summary

This file summarizes the current local semantic-parser results for HarmBench and RedBench follow-up runs.

## HarmBench

Overview from `neurips-results/mechanism/harmbench_interventions_overview.semantic.json`.

| HarmBench run | Configs | Total n | Semantic refusal | Semantic compliance | Invalid generation |
|---|---:|---:|---:|---:|---:|
| `initial_full_sweep` | 42 | 16,800 | 96.47% | 0.80% | 1.30% |
| `followup_multilayer` | 6 | 2,400 | 93.08% | 0.25% | 4.75% |
| `followup_w1_source` | 12 | 4,800 | 81.21% | 0.77% | 15.75% |
| `followup_decode_time` | 9 | 3,600 | 0.00% | 0.00% | 100.00% |

### HarmBench Current Multilayer

From `neurips-results/mechanism/harmbench_followup_multilayer/harmbench_summary.semantic.json`.

| Config | n | Semantic refusal | Semantic compliance | Invalid generation |
|---|---:|---:|---:|---:|
| `interpolate_mean_a0.5_L16-18-20` | 400 | 94.75% | 0.50% | 3.00% |
| `interpolate_mean_a0.7_L16-18-20` | 400 | 93.00% | 0.25% | 4.50% |
| `interpolate_mean_a1_L16-18-20` | 400 | 92.50% | 0.75% | 4.50% |
| `replace_mean_a0.5_L16-18-20` | 400 | 92.75% | 0.00% | 5.50% |
| `replace_mean_a0.7_L16-18-20` | 400 | 92.75% | 0.00% | 5.50% |
| `replace_mean_a1_L16-18-20` | 400 | 92.75% | 0.00% | 5.50% |

## RedBench

These RedBench numbers use the updated semantic relabeling flow.

### RedBench Multilayer (All 1,500 prompts)

From `neurips-results/mechanism/redbench_followups_1500/multilayer/harmbench_summary.semantic.json`.

| Config | n | Semantic refusal | Semantic compliance | Invalid generation |
|---|---:|---:|---:|---:|
| `interpolate_mean_a0.5_L16-18-20` | 1500 | 61.40% | 34.53% | 3.87% |
| `interpolate_mean_a0.7_L16-18-20` | 1500 | 62.47% | 34.07% | 3.40% |
| `interpolate_mean_a1_L16-18-20` | 1500 | 61.67% | 36.33% | 1.93% |
| `replace_mean_a0.5_L16-18-20` | 1500 | 62.53% | 35.27% | 2.20% |
| `replace_mean_a0.7_L16-18-20` | 1500 | 62.53% | 35.27% | 2.20% |
| `replace_mean_a1_L16-18-20` | 1500 | 62.53% | 35.27% | 2.20% |

### RedBench W1 Source (All 1,500 prompts)

From `neurips-results/mechanism/redbench_followups_1500/w1_source/harmbench_summary.semantic.json`.

| Config | n | Semantic refusal | Semantic compliance | Invalid generation |
|---|---:|---:|---:|---:|
| `interpolate_mean_a0.7_L16` | 1500 | 47.73% | 42.27% | 10.00% |
| `interpolate_mean_a0.7_L18` | 1500 | 62.93% | 32.07% | 4.93% |
| `interpolate_mean_a0.7_L20` | 1500 | 67.40% | 30.67% | 1.60% |
| `interpolate_mean_a1_L16` | 1500 | 45.00% | 43.47% | 11.53% |
| `interpolate_mean_a1_L18` | 1500 | 58.80% | 34.80% | 6.40% |
| `interpolate_mean_a1_L20` | 1500 | 66.27% | 30.93% | 2.80% |
| `replace_mean_a0.7_L16` | 1500 | 45.20% | 43.07% | 11.73% |
| `replace_mean_a0.7_L18` | 1500 | 58.87% | 34.87% | 6.27% |
| `replace_mean_a0.7_L20` | 1500 | 65.87% | 31.27% | 2.87% |
| `replace_mean_a1_L16` | 1500 | 45.20% | 43.07% | 11.73% |
| `replace_mean_a1_L18` | 1500 | 58.87% | 34.87% | 6.27% |
| `replace_mean_a1_L20` | 1500 | 65.87% | 31.27% | 2.87% |

### RedBench Decode (All 1,500 prompts)

From `neurips-results/mechanism/redbench_followups_1500/decode/harmbench_summary.semantic.json`.

| Config | n | Semantic refusal | Semantic compliance | Invalid generation |
|---|---:|---:|---:|---:|
| `interpolate_mean_a0.5_L16` | 1500 | 0.00% | 0.00% | 100.00% |
| `interpolate_mean_a0.5_L18` | 1500 | 0.00% | 0.00% | 100.00% |
| `interpolate_mean_a0.5_L20` | 1500 | 0.00% | 0.00% | 100.00% |
| `interpolate_mean_a0.7_L16` | 1500 | 0.00% | 0.00% | 100.00% |
| `interpolate_mean_a0.7_L18` | 1500 | 0.00% | 0.00% | 100.00% |
| `interpolate_mean_a0.7_L20` | 1500 | 0.00% | 0.00% | 100.00% |
| `interpolate_mean_a1_L16` | 1500 | 0.00% | 0.00% | 100.00% |
| `interpolate_mean_a1_L18` | 1500 | 0.00% | 0.00% | 100.00% |
| `interpolate_mean_a1_L20` | 1500 | 0.00% | 0.00% | 100.00% |

## RedBench Harmful-Only Read

This is a conservative harmful-only slice computed from source labels in `data/redbench_1500.jsonl`.

Included only clearly harmful-request source buckets and dropped mixed or benign buckets such as `CoCoNot`, `AdvBench`, `CoSafe`, `DiaSafety`, `GEST`, `ORBench`, and similar mixed-distribution sources.

This leaves `934 / 1500` prompts per config.

### Harmful-Only Best Rows

| Run | Best config | n harmful | Refusal | Compliance | Invalid | Other |
|---|---|---:|---:|---:|---:|---:|
| `multilayer` | `replace_mean_a0.5_L16-18-20` | 934 | 71.09% | 28.37% | 0.54% | 0.00% |
| `w1_source` | `interpolate_mean_a1_L20` | 934 | 72.81% | 26.66% | 0.54% | 0.00% |
| `decode` | any config | 934 | 0.00% | 0.00% | 100.00% | 0.00% |

### Harmful-Only Multilayer

| Config | n harmful | Refusal | Compliance | Invalid | Other |
|---|---:|---:|---:|---:|---:|
| `interpolate_mean_a0.5_L16-18-20` | 934 | 69.81% | 28.69% | 1.28% | 0.21% |
| `interpolate_mean_a0.7_L16-18-20` | 934 | 70.56% | 28.69% | 0.64% | 0.11% |
| `interpolate_mean_a1_L16-18-20` | 934 | 70.77% | 28.59% | 0.64% | 0.00% |
| `replace_mean_a0.5_L16-18-20` | 934 | 71.09% | 28.37% | 0.54% | 0.00% |
| `replace_mean_a0.7_L16-18-20` | 934 | 71.09% | 28.37% | 0.54% | 0.00% |
| `replace_mean_a1_L16-18-20` | 934 | 71.09% | 28.37% | 0.54% | 0.00% |

### Harmful-Only W1 Source

| Config | n harmful | Refusal | Compliance | Invalid | Other |
|---|---:|---:|---:|---:|---:|
| `interpolate_mean_a0.7_L16` | 934 | 54.39% | 40.58% | 5.03% | 0.00% |
| `interpolate_mean_a1_L16` | 934 | 52.89% | 40.26% | 6.85% | 0.00% |
| `replace_mean_a0.7_L16` | 934 | 52.78% | 40.47% | 6.75% | 0.00% |
| `replace_mean_a1_L16` | 934 | 52.78% | 40.47% | 6.75% | 0.00% |
| `interpolate_mean_a0.7_L18` | 934 | 70.88% | 26.45% | 2.57% | 0.11% |
| `interpolate_mean_a1_L18` | 934 | 67.67% | 30.41% | 1.93% | 0.00% |
| `replace_mean_a0.7_L18` | 934 | 67.88% | 30.30% | 1.82% | 0.00% |
| `replace_mean_a1_L18` | 934 | 67.88% | 30.30% | 1.82% | 0.00% |
| `interpolate_mean_a0.7_L20` | 934 | 72.59% | 26.55% | 0.54% | 0.32% |
| `interpolate_mean_a1_L20` | 934 | 72.81% | 26.66% | 0.54% | 0.00% |
| `replace_mean_a0.7_L20` | 934 | 71.73% | 27.73% | 0.54% | 0.00% |
| `replace_mean_a1_L20` | 934 | 71.73% | 27.73% | 0.54% | 0.00% |

### Harmful-Only Decode

| Config | n harmful | Refusal | Compliance | Invalid | Other |
|---|---:|---:|---:|---:|---:|
| `interpolate_mean_a0.5_L16` | 934 | 0.00% | 0.00% | 100.00% | 0.00% |
| `interpolate_mean_a0.7_L16` | 934 | 0.00% | 0.00% | 100.00% | 0.00% |
| `interpolate_mean_a1_L16` | 934 | 0.00% | 0.00% | 100.00% | 0.00% |
| `interpolate_mean_a0.5_L18` | 934 | 0.00% | 0.00% | 100.00% | 0.00% |
| `interpolate_mean_a0.7_L18` | 934 | 0.00% | 0.00% | 100.00% | 0.00% |
| `interpolate_mean_a1_L18` | 934 | 0.00% | 0.00% | 100.00% | 0.00% |
| `interpolate_mean_a0.5_L20` | 934 | 0.00% | 0.00% | 100.00% | 0.00% |
| `interpolate_mean_a0.7_L20` | 934 | 0.00% | 0.00% | 100.00% | 0.00% |
| `interpolate_mean_a1_L20` | 934 | 0.00% | 0.00% | 100.00% | 0.00% |
