# Exp14D Social vs Epistemic Analysis

Includes 2x2 factorial decomposition and paired contrasts per tag: tau(correct)-tau(wrong) and (E_high-E_low)_correct-(E_high-E_low)_wrong.

## Qwen/Qwen3-4B-Instruct-2507
- source: `new-phase-results/exp14_assertive_v2/Qwen__Qwen3-4B-Instruct-2507_results.jsonl`
- design: high_tag=expert, low_tag=note, factorial_evidence=(bare->reason2), instruction=I1
- paired diagnostics: tau_levels=bare,reason1,reason2,reason_data, magnitude_levels=bare->reason_data
- slice counts: all=1813, prior_correct=1357, prior_wrong=450

### Slice: all
- expert: n=1813; tau gap mean=-0.462 [-0.494, -0.431], pos_frac=0.177; magnitude gap mean=-8.814 [-9.287, -8.334], pos_frac=0.206
- note: n=1813; tau gap mean=-0.721 [-0.766, -0.679], pos_frac=0.186; magnitude gap mean=-5.959 [-6.390, -5.535], pos_frac=0.218

### Slice: prior_wrong
- expert: n=450; tau gap mean=0.140 [0.082, 0.199], pos_frac=0.460; magnitude gap mean=0.953 [0.112, 1.815], pos_frac=0.567
- note: n=450; tau gap mean=0.317 [0.242, 0.393], pos_frac=0.584; magnitude gap mean=3.564 [2.654, 4.473], pos_frac=0.673

## Qwen/Qwen3-4B-Thinking-2507
- source: `new-phase-results/exp14_assertive_v2/Qwen__Qwen3-4B-Thinking-2507_results.jsonl`
- design: high_tag=expert, low_tag=note, factorial_evidence=(bare->reason2), instruction=I1
- paired diagnostics: tau_levels=bare,reason1,reason2,reason_data, magnitude_levels=bare->reason_data
- slice counts: all=1813, prior_correct=1344, prior_wrong=441

### Slice: all
- expert: n=1813; tau gap mean=-0.076 [-0.101, -0.053], pos_frac=0.290; magnitude gap mean=-0.539 [-0.616, -0.460], pos_frac=0.389
- note: n=1813; tau gap mean=-0.130 [-0.155, -0.105], pos_frac=0.303; magnitude gap mean=-0.351 [-0.426, -0.280], pos_frac=0.413

### Slice: prior_wrong
- expert: n=441; tau gap mean=0.056 [0.011, 0.102], pos_frac=0.381; magnitude gap mean=-0.050 [-0.206, 0.103], pos_frac=0.503
- note: n=441; tau gap mean=0.175 [0.132, 0.217], pos_frac=0.522; magnitude gap mean=0.588 [0.440, 0.730], pos_frac=0.664

## meta-llama/Llama-3.1-8B-Instruct
- source: `new-phase-results/exp14_assertive_v2/meta-llama__Llama-3.1-8B-Instruct_results.jsonl`
- design: high_tag=expert, low_tag=note, factorial_evidence=(bare->reason2), instruction=I1
- paired diagnostics: tau_levels=bare,reason1,reason2,reason_data, magnitude_levels=bare->reason_data
- slice counts: all=1813, prior_correct=1328, prior_wrong=451

### Slice: all
- expert: n=1813; tau gap mean=-0.157 [-0.172, -0.142], pos_frac=0.165; magnitude gap mean=-1.186 [-1.280, -1.087], pos_frac=0.268
- note: n=1813; tau gap mean=-0.217 [-0.238, -0.193], pos_frac=0.193; magnitude gap mean=-1.311 [-1.445, -1.182], pos_frac=0.300

### Slice: prior_wrong
- expert: n=451; tau gap mean=-0.021 [-0.051, 0.013], pos_frac=0.313; magnitude gap mean=0.575 [0.405, 0.744], pos_frac=0.647
- note: n=451; tau gap mean=0.124 [0.081, 0.168], pos_frac=0.430; magnitude gap mean=1.507 [1.270, 1.739], pos_frac=0.734

