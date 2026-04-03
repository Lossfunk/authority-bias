# Evidence-Gating Analysis: Llama-3.1-8B Base vs Instruct

## 1. Overview

This analysis compares Llama-3.1-8B (base) and Llama-3.1-8B-Instruct on the evidence-gating phenomenon:
does instruction tuning (RLHF) *create* or merely *amplify* a tendency to reject correct evidence
when the model's prior belief is wrong?

**Key question:** On prior-wrong items, does the model integrate the correct evidence provided
(frac_dr_pos > 0.5) or entrench its prior belief (frac_dr_pos < 0.5)?

**Metrics reminder:**
- `r_w`: endorsement rate shift when wrong answer is endorsed by source
- `r_c`: endorsement rate shift when correct answer is endorsed by source
- `dr = r_w - r_c`: differential rate (positive = wrong-answer endorsement stronger)
- `frac_dr_pos`: fraction of items where dr > 0 (>0.5 means model follows wrong endorsement more)
- Values are `[point_estimate, [CI_lower, CI_upper]]`

## 2. Overall Metrics (Part A — All Items, Sign-Consistent)

### Table 2a: Expert Tag — Overall

| Metric | Base | Instruct |
|--------|------|----------|
| r_w_mean | 0.181 [0.140, 0.222] | 0.231 [-0.732, 0.788] |
| r_w_median | 0.199 [0.177, 0.219] | 0.682 [0.661, 0.707] |
| r_c_mean | 0.438 [0.313, 0.630] | 1.066 [0.038, 2.524] |
| r_c_median | 0.306 [0.285, 0.324] | 0.554 [0.528, 0.582] |
| dr_mean | -0.265 [-0.472, -0.125] | -1.104 [-3.086, 0.392] |
| dr_median | -0.120 [-0.152, -0.076] | 0.035 [0.002, 0.064] |
| frac_dr_pos | 0.432 [0.408, 0.456] | 0.528 [0.501, 0.554] |
| frac_eff_w_gt_c | 0.461 [0.438, 0.484] | 0.681 [0.660, 0.702] |
| baseline_shift_mean | 0.112 [0.078, 0.146] | -0.691 [-0.809, -0.569] |

### Table 2b: Note Tag — Overall

| Metric | Base | Instruct |
|--------|------|----------|
| r_w_mean | 0.019 [-0.008, 0.045] | 0.994 [0.917, 1.061] |
| r_w_median | 0.080 [0.066, 0.105] | 0.933 [0.898, 0.985] |
| r_c_mean | 0.147 [0.117, 0.181] | 0.416 [-1.295, 2.013] |
| r_c_median | 0.184 [0.167, 0.200] | 0.561 [0.523, 0.616] |
| dr_mean | -0.125 [-0.168, -0.084] | 0.415 [-1.336, 2.297] |
| dr_median | -0.089 [-0.129, -0.062] | 0.166 [0.115, 0.243] |
| frac_dr_pos | 0.426 [0.404, 0.449] | 0.586 [0.556, 0.615] |
| frac_eff_w_gt_c | 0.440 [0.418, 0.463] | 0.720 [0.699, 0.741] |
| baseline_shift_mean | 0.093 [0.058, 0.128] | -1.104 [-1.239, -0.967] |

### Interpretation (Overall)

- **Base model (expert):** frac_dr_pos = 0.432 — below 0.5, meaning correct evidence is slightly
  favored over wrong evidence on average. dr_mean = -0.265 (negative = correct wins).
- **Instruct model (expert):** frac_dr_pos = 0.528 — roughly at 0.5, with wide CIs on dr_mean.
  The instruct model is more ambivalent overall with expert tags.
- **Base model (note):** frac_dr_pos = 0.426 — also below 0.5, correct evidence wins.
- **Instruct model (note):** frac_dr_pos = 0.586 — clearly above 0.5, meaning the instruct model
  follows wrong endorsements more than correct ones when the source is a note. This is striking.

## 3. Prior-Wrong Items (Part B — Inverted Prior)

These are items where the model's baseline prior favors the wrong answer. This is the critical
test for evidence gating: when the model is wrong, does it update toward the correct evidence?

### Table 3a: Expert Tag — Prior-Wrong (full slice)

| Metric | Base (n=737) | Instruct (n=454) |
|--------|------|----------|
| r_w_mean | 0.350 [0.265, 0.439] | -1.741 [-6.200, 0.749] |
| r_w_median | 0.396 [0.347, 0.440] | 0.503 [0.442, 0.546] |
| r_c_mean | 0.040 [-0.002, 0.077] | 0.611 [0.566, 0.659] |
| r_c_median | 0.143 [0.105, 0.169] | 0.587 [0.533, 0.619] |
| dr_mean | 0.287 [0.190, 0.389] | -2.327 [-6.786, 0.160] |
| dr_median | 0.250 [0.201, 0.286] | -0.072 [-0.111, -0.002] |
| frac_dr_pos | 0.668 [0.632, 0.704] | 0.447 [0.397, 0.497] |
| frac_eff_w_gt_c | 0.620 [0.585, 0.655] | 0.267 [0.227, 0.308] |
| baseline_shift_mean | -0.284 [-0.331, -0.238] | 1.337 [1.117, 1.561] |

### Table 3b: Note Tag — Prior-Wrong (full slice)

| Metric | Base (n=702) | Instruct (n=444) |
|--------|------|----------|
| r_w_mean | -0.060 [-0.115, -0.012] | 0.408 [0.104, 0.631] |
| r_w_median | 0.039 [0.000, 0.067] | 0.498 [0.403, 0.570] |
| r_c_mean | 0.165 [0.135, 0.193] | 0.854 [0.786, 0.929] |
| r_c_median | 0.200 [0.170, 0.221] | 0.753 [0.709, 0.821] |
| dr_mean | -0.224 [-0.288, -0.166] | -0.362 [-0.679, -0.129] |
| dr_median | -0.181 [-0.225, -0.138] | -0.154 [-0.281, -0.076] |
| frac_dr_pos | 0.347 [0.311, 0.382] | 0.383 [0.333, 0.435] |
| frac_eff_w_gt_c | 0.338 [0.303, 0.373] | 0.196 [0.160, 0.234] |
| baseline_shift_mean | -0.222 [-0.273, -0.173] | 1.603 [1.388, 1.823] |

### Table 3c: High-Confidence Prior-Wrong (top 25% strongest priors)

| Metric | Base Expert | Base Note | Instruct Expert | Instruct Note |
|--------|-------------|-----------|-----------------|---------------|
| r_w_mean | 0.766 [0.590, 0.971] | -0.080 [-0.201, 0.027] | 1.596 [-0.806, 5.119] | -0.058 [-0.547, 0.393] |
| r_c_mean | -0.012 [-0.063, 0.038] | 0.196 [0.157, 0.235] | 0.713 [0.658, 0.779] | 0.953 [0.883, 1.030] |
| dr_mean | 0.772 [0.584, 0.993] | -0.276 [-0.405, -0.160] | 0.904 [-1.491, 4.431] | -0.899 [-1.376, -0.452] |
| dr_median | 0.546 [0.439, 0.604] | -0.172 [-0.292, -0.086] | 0.002 [-0.274, 0.116] | -0.671 [-1.151, -0.332] |
| frac_dr_pos | 0.843 [0.786, 0.896] | 0.349 [0.278, 0.420] | 0.500 [0.367, 0.633] | 0.154 [0.049, 0.276] |
| frac_eff_w_gt_c | 0.768 [0.708, 0.827] | 0.312 [0.244, 0.381] | 0.035 [0.009, 0.070] | 0.009 [0.000, 0.027] |

### Interpretation (Prior-Wrong Items)

**Critical finding — Base model on prior-wrong items:**

- **Base + Expert tag:** frac_dr_pos = 0.668, dr_mean = 0.287
  → frac_dr_pos > 0.5: The base model shows **entrenchment** — it follows wrong endorsements more than correct ones.
  The evidence-gating phenomenon EXISTS in the base model with expert tags.

- **Base + Note tag:** frac_dr_pos = 0.347, dr_mean = -0.224
  → frac_dr_pos < 0.5: The base model does NOT show entrenchment with note tags — correct evidence wins.

- **Instruct + Expert tag:** frac_dr_pos = 0.447, dr_mean = -2.327
  → Entrenchment pattern present

- **Instruct + Note tag:** frac_dr_pos = 0.383, dr_mean = -0.362
  → Entrenchment pattern present: correct evidence rejected

**High-confidence prior-wrong items (top 25%):**

| Model | Tag | frac_dr_pos | dr_mean | Entrenchment? |
|-------|-----|-------------|---------|---------------|
| Base | Expert | 0.843 | 0.772 | YES |
| Base | Note | 0.349 | -0.276 | NO |
| Instruct | Expert | 0.500 | 0.904 | NO |
| Instruct | Note | 0.154 | -0.899 | NO |

### Source-Authority Differential (Expert vs Note)

A key signature of evidence gating is whether the model treats expert and note sources differently.

**On prior-wrong items (full slice):**
- Base: Expert frac_dr_pos = 0.668, Note frac_dr_pos = 0.347
  → Δ = +0.321 (expert - note)
- Instruct: Expert frac_dr_pos = 0.447, Note frac_dr_pos = 0.383
  → Δ = +0.064 (expert - note)

**On high-confidence prior-wrong items:**
- Base: Expert frac_dr_pos = 0.843, Note frac_dr_pos = 0.349
  → Δ = +0.495
- Instruct: Expert frac_dr_pos = 0.500, Note frac_dr_pos = 0.154
  → Δ = +0.346

## 4. Confidence-Conditioned Behavior (Part C)

How does evidence integration vary across confidence quartiles? Q1 = lowest confidence, Q4 = highest.

### Table 4a: Expert Tag — frac_dr_pos by Confidence Quartile

| Quartile | Base | Instruct |
|----------|------|----------|
| Q1 (|m_N0| ~ 0.29 / 0.85) | 0.514 [0.467, 0.560] | 0.447 [0.402, 0.492] |
| Q2 (|m_N0| ~ 0.88 / 3.08) | 0.477 [0.430, 0.525] | 0.479 [0.432, 0.526] |
| Q3 (|m_N0| ~ 1.44 / 6.57) | 0.472 [0.422, 0.521] | 0.647 [0.593, 0.701] |
| Q4 (|m_N0| ~ 2.79 / 10.23) | 0.252 [0.209, 0.295] | 0.753 [0.667, 0.837] |

### Table 4b: Note Tag — frac_dr_pos by Confidence Quartile

| Quartile | Base | Instruct |
|----------|------|----------|
| Q1 (|m_N0| ~ 0.29 / 0.87) | 0.420 [0.376, 0.464] | 0.546 [0.499, 0.592] |
| Q2 (|m_N0| ~ 0.86 / 3.31) | 0.436 [0.390, 0.484] | 0.578 [0.529, 0.628] |
| Q3 (|m_N0| ~ 1.49 / 7.02) | 0.432 [0.386, 0.477] | 0.652 [0.587, 0.716] |
| Q4 (|m_N0| ~ 2.94 / 11.40) | 0.418 [0.372, 0.463] | 0.821 [0.667, 0.957] |

### Table 4c: dr_mean by Confidence Quartile

| Quartile | Base Expert | Base Note | Instruct Expert | Instruct Note |
|----------|-------------|-----------|-----------------|---------------|
| Q1 | 0.097 [0.006, 0.211] | -0.104 [-0.176, -0.029] | -0.071 [-0.159, 0.011] | 0.129 [0.026, 0.231] |
| Q2 | -0.024 [-0.158, 0.105] | -0.150 [-0.266, -0.057] | -4.094 [-9.639, -0.142] | -1.590 [-5.428, 0.964] |
| Q3 | -0.053 [-0.206, 0.105] | -0.119 [-0.203, -0.040] | 0.351 [-0.953, 1.817] | 4.603 [-0.022, 13.033] |
| Q4 | -1.147 [-1.921, -0.643] | -0.129 [-0.209, -0.049] | 2.385 [-4.933, 10.487] | 1.842 [-2.175, 5.596] |

### Table 4d: r_w_mean and r_c_mean by Confidence Quartile

| Quartile | Metric | Base Expert | Base Note | Instruct Expert | Instruct Note |
|----------|--------|-------------|-----------|-----------------|---------------|
| Q1 | r_w_mean | 0.190 [0.142, 0.235] | -0.007 [-0.055, 0.043] | 0.464 [0.423, 0.505] | 0.724 [0.651, 0.799] |
| Q1 | r_c_mean | 0.077 [-0.032, 0.160] | 0.104 [0.053, 0.153] | 0.536 [0.460, 0.619] | 0.622 [0.539, 0.716] |
| Q2 | r_w_mean | 0.089 [-0.017, 0.175] | 0.002 [-0.058, 0.056] | -1.510 [-5.223, 0.572] | 0.760 [0.500, 0.950] |
| Q2 | r_c_mean | 0.106 [0.018, 0.183] | 0.166 [0.095, 0.269] | 2.504 [0.616, 6.211] | 2.229 [-0.174, 5.869] |
| Q3 | r_w_mean | 0.201 [0.116, 0.312] | 0.017 [-0.045, 0.067] | 0.888 [0.582, 1.351] | 1.106 [1.006, 1.215] |
| Q3 | r_c_mean | 0.264 [0.143, 0.373] | 0.134 [0.078, 0.184] | 0.563 [-0.637, 1.646] | -2.778 [-9.515, 0.923] |
| Q4 | r_w_mean | 0.244 [0.171, 0.328] | 0.066 [0.013, 0.113] | 1.114 [0.991, 1.268] | 1.410 [1.299, 1.533] |
| Q4 | r_c_mean | 1.380 [0.919, 2.116] | 0.187 [0.130, 0.243] | -0.459 [-6.506, 5.020] | 0.040 [-1.996, 2.244] |

### Interpretation (Confidence-Conditioned)

**Confidence gradient (Q1 → Q4 frac_dr_pos):**
- Base Expert: 0.514 → 0.252 (Δ = -0.262)
- Base Note: 0.420 → 0.418 (Δ = -0.002)
- Instruct Expert: 0.447 → 0.753 (Δ = +0.306)
- Instruct Note: 0.546 → 0.821 (Δ = +0.276)

The base model shows a monotonic decrease in frac_dr_pos as confidence increases (Q1→Q4),
meaning higher-confidence items have relatively stronger correct-evidence integration.
The instruct model shows a dramatically different pattern: at high confidence (Q3, Q4), frac_dr_pos
shoots up (especially with note tag), indicating the instruct model increasingly follows wrong
endorsements at high confidence.

## 5. Synthesis: Does RLHF Create or Amplify Evidence Gating?

### The Evidence

**1. Prior-wrong items — Overall (Part B, full slice):**

The base model with expert tags shows frac_dr_pos = 0.668 on prior-wrong items —
this is ABOVE 0.5, indicating the base model already exhibits a mild form of the gating
pattern where wrong-answer endorsements are followed more than correct-answer ones.
With note tags, the base model shows frac_dr_pos = 0.347 — below 0.5, so
it does update correctly for note-tagged evidence.

The instruct model with expert tags shows frac_dr_pos = 0.447 — below 0.5,
which actually suggests evidence integration. But with note tags, frac_dr_pos = 0.383
— also below 0.5 but with a much stronger negative dr_mean (-0.362 vs base -0.224).

**2. High-confidence prior-wrong items (strongest test):**

At high confidence, the base model with expert tags shows frac_dr_pos = 0.843
(strong entrenchment!), while with note tags it shows 0.349 (correct evidence wins).
This means the base model ALREADY has source-dependent evidence gating: it entrenches
with expert endorsements but corrects with note endorsements on high-confidence wrong items.

The instruct model at high confidence shows expert frac_dr_pos = 0.500
(exactly at chance) and note = 0.154 (strong correct-evidence integration).
This is a reversal from the overall pattern and suggests complex dynamics at high confidence.

**3. Confidence gradient (Part C):**

The base model shows a pattern where higher confidence leads to LOWER frac_dr_pos (more
correct-evidence integration), while the instruct model shows the OPPOSITE at extreme
confidence levels — higher frac_dr_pos, especially with expert tags.

### Summary Table: Evidence-Gating Signatures

| Signature | Base Model | Instruct Model | RLHF Effect |
|-----------|-----------|----------------|-------------|
| Overall frac_dr_pos < 0.5 | Expert: 0.432, Note: 0.426 | Expert: 0.528, Note: 0.586 | Instruct shifts note tag toward wrong-endorse |
| Prior-wrong entrenchment | Expert: 0.668 (YES), Note: 0.347 (NO) | Expert: 0.447 (NO), Note: 0.383 (NO) | Complex: base has expert-gating, instruct reduces it |
| High-conf prior-wrong | Expert: 0.843 (STRONG YES), Note: 0.349 (NO) | Expert: 0.500 (NEUTRAL), Note: 0.154 (NO) | RLHF reduces expert-tag gating at high conf |
| Expert-Note differential | Δ = +0.321 | Δ = +0.064 | RLHF reduces source-authority differential |

### Conclusions

1. **The base model ALREADY shows evidence gating with expert tags on prior-wrong items.**
   At full slice: frac_dr_pos = 0.668 (>0.5). At high confidence: 0.843.
   This is the strongest version of the gating pattern in the data. The base model treats expert
   endorsements of wrong answers as MORE persuasive than expert endorsements of correct answers,
   specifically when its prior is wrong.

2. **The base model does NOT gate with note tags.**
   frac_dr_pos = 0.347 on prior-wrong items (correct evidence wins).
   This creates a clear expert-note asymmetry: Δ = +0.321.

3. **RLHF/instruction tuning changes the gating pattern rather than simply amplifying it.**
   The instruct model actually REDUCES expert-tag gating on prior-wrong items
   (frac_dr_pos drops from 0.668 to 0.447).
   However, it introduces strong overall wrong-answer bias with note tags
   (overall frac_dr_pos goes from 0.426 to 0.586).

4. **The hypothesis that RLHF creates evidence gating is NOT supported in its simple form.**
   The base model already has the gating pattern with expert tags. However, RLHF
   transforms the pattern: it reduces expert-specific gating while introducing a general
   tendency to follow wrong endorsements from note sources, and creates much larger
   endorsement effects overall (r_w and r_c magnitudes are much larger for instruct).

5. **A more nuanced view: RLHF reorganizes evidence weighting.**
   - Pre-RLHF: Gating is source-dependent (expert tags gate, note tags don't)
   - Post-RLHF: The instruct model shows larger effect sizes, a reduced expert-note
     differential on prior-wrong items, but a new overall bias where note-tagged
     wrong endorsements are very persuasive (frac_dr_pos = 0.586 overall)
   - The baseline_shift_mean is strongly negative for instruct (meaning the instruct model
     is much more confident a priori), which may drive some of these dynamics.

## 6. Raw Numbers Reference

### Part B — Low-Confidence Prior-Wrong Items

| Metric | Base Expert | Base Note | Instruct Expert | Instruct Note |
|--------|-------------|-----------|-----------------|---------------|
| r_w_mean | 0.146 [0.050, 0.228] | -0.041 [-0.110, 0.019] | 0.453 [0.389, 0.516] | 0.653 [0.550, 0.764] |
| r_c_mean | 0.078 [0.005, 0.137] | 0.139 [0.087, 0.186] | 0.546 [0.470, 0.629] | 0.736 [0.617, 0.871] |
| dr_mean | 0.030 [-0.081, 0.140] | -0.178 [-0.264, -0.095] | -0.089 [-0.188, 0.008] | -0.012 [-0.143, 0.121] |
| frac_dr_pos | 0.562 [0.510, 0.613] | 0.350 [0.300, 0.400] | 0.456 [0.392, 0.520] | 0.459 [0.393, 0.526] |
| frac_eff_w_gt_c | 0.527 [0.476, 0.579] | 0.356 [0.308, 0.405] | 0.423 [0.357, 0.489] | 0.342 [0.279, 0.405] |

### Part B — Sample Sizes

| Slice | Base Expert | Base Note | Instruct Expert | Instruct Note |
|-------|-------------|-----------|-----------------|---------------|
| full | 737 | 702 | 454 | 444 |
| low_conf | 368 | 351 | 227 | 222 |
| high_conf_top25 | 185 | 176 | 114 | 111 |
| top10 | 74 | 71 | 46 | 45 |
