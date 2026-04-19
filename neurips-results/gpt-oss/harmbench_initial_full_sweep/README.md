# HarmBench Initial Full Sweep

This folder contains the initial full-sweep HarmBench intervention results copied from local audit artifacts.

Files:
- `harmbench_rows.jsonl`: raw rows copied from `/tmp/hb_audit/harmbench_rows.remote.jsonl`
- `harmbench_rows.reparsed.degfilter.jsonl`: rows relabeled with the improved parser and degeneration filter
- `harmbench_summary.reparsed.degfilter.json`: per-config summary including
  - `refusal_rate`
  - `compliance_rate`
  - `effective_compliance_rate`
  - `likely_refusal_truncated_rate`
  - `degenerate_rate`
  - `uncertain_rate`
