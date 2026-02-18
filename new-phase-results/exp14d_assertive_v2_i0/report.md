# Exp14D Social vs Epistemic Analysis

Includes 2x2 factorial decomposition and paired contrasts per tag: tau(correct)-tau(wrong) and (E_high-E_low)_correct-(E_high-E_low)_wrong.

## Qwen/Qwen3-4B-Instruct-2507
- source: `new-phase-results/exp14_assertive_v2/Qwen__Qwen3-4B-Instruct-2507_results.jsonl`
- design: high_tag=expert, low_tag=note, factorial_evidence=(bare->reason2), instruction=I0
- paired diagnostics: tau_levels=bare,reason1,reason2,reason_data, magnitude_levels=bare->reason_data
- slice counts: all=1813, prior_correct=1357, prior_wrong=450

### Slice: all
- expert: n=1813; tau gap mean=-0.428 [-0.463, -0.395], pos_frac=0.202; magnitude gap mean=-5.625 [-5.950, -5.323], pos_frac=0.202
- note: n=1813; tau gap mean=-0.531 [-0.571, -0.487], pos_frac=0.223; magnitude gap mean=-6.545 [-6.956, -6.146], pos_frac=0.245

### Slice: prior_wrong
- expert: n=450; tau gap mean=-0.008 [-0.068, 0.054], pos_frac=0.404; magnitude gap mean=-1.607 [-2.403, -0.839], pos_frac=0.536
- note: n=450; tau gap mean=0.357 [0.276, 0.429], pos_frac=0.609; magnitude gap mean=3.322 [2.632, 4.033], pos_frac=0.718

## Qwen/Qwen3-4B-Thinking-2507
- source: `new-phase-results/exp14_assertive_v2/Qwen__Qwen3-4B-Thinking-2507_results.jsonl`
- design: high_tag=expert, low_tag=note, factorial_evidence=(bare->reason2), instruction=I0
- paired diagnostics: tau_levels=bare,reason1,reason2,reason_data, magnitude_levels=bare->reason_data
- slice counts: all=1813, prior_correct=1344, prior_wrong=441

### Slice: all
- expert: n=1813; tau gap mean=-0.133 [-0.155, -0.111], pos_frac=0.257; magnitude gap mean=-1.934 [-2.071, -1.802], pos_frac=0.268
- note: n=1813; tau gap mean=-0.055 [-0.077, -0.034], pos_frac=0.308; magnitude gap mean=-0.908 [-1.020, -0.800], pos_frac=0.351

### Slice: prior_wrong
- expert: n=441; tau gap mean=0.090 [0.047, 0.133], pos_frac=0.397; magnitude gap mean=0.750 [0.486, 1.010], pos_frac=0.664
- note: n=441; tau gap mean=0.130 [0.088, 0.170], pos_frac=0.444; magnitude gap mean=1.246 [1.037, 1.465], pos_frac=0.755

## meta-llama/Llama-3.1-8B-Instruct
- source: `new-phase-results/exp14_assertive_v2/meta-llama__Llama-3.1-8B-Instruct_results.jsonl`
- design: high_tag=expert, low_tag=note, factorial_evidence=(bare->reason2), instruction=I0
- paired diagnostics: tau_levels=bare,reason1,reason2,reason_data, magnitude_levels=bare->reason_data
- slice counts: all=1813, prior_correct=1328, prior_wrong=451

### Slice: all
- expert: n=1813; tau gap mean=-0.273 [-0.292, -0.254], pos_frac=0.162; magnitude gap mean=-1.942 [-2.066, -1.824], pos_frac=0.221
- note: n=1813; tau gap mean=-0.181 [-0.198, -0.162], pos_frac=0.183; magnitude gap mean=-1.380 [-1.503, -1.259], pos_frac=0.277

### Slice: prior_wrong
- expert: n=451; tau gap mean=0.015 [-0.026, 0.056], pos_frac=0.386; magnitude gap mean=0.830 [0.604, 1.048], pos_frac=0.659
- note: n=451; tau gap mean=-0.003 [-0.040, 0.037], pos_frac=0.308; magnitude gap mean=0.972 [0.695, 1.234], pos_frac=0.665

