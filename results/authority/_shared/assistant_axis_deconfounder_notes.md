# Assistant-axis deconfounder — figure design notes

This note exists so the figure question ("which of the three layouts fits?")
can be answered from first principles against what the deconfounder **actually
computes** in code, not against a mental model of it.

## Where the code lives

Two files drive the deconfounder:

- Computation (producing `overlap_summary.json`):
  - `src/exp16/compute_assistant_axis_overlap.py`
- Visualisation (the Image-1-style single-axis figure we already have):
  - `src/analysis/plot_assistant_axis_deconfound.py`

Example inputs/outputs:
- GPT-OSS overlap → `external/assistant-axis/hardened/overlap_summary.json`
- Gemma-4 overlap → `neurips-results/gemma4/mechanism/assistant_axis_hardened/overlap_summary.json`
- OLMo-2 overlap → `neurips-results/olmo2/mechanism/assistant_axis_hardened/overlap_summary.json`
- Current figure → `figures/neurips/v2/candidates/assistant_axis_deconfound.{png,pdf,svg}`

## What the deconfounder actually computes

Exactly, from `compute_assistant_axis_overlap.py`:

1. For each non-default role $r$ (comedian, detective, …), take the mean
   activation vector at the primary mechanism layer
   (score-filtered when a judge file is available).
2. Take the `default` role's mean activation at the same layer.
3. Compute **one direction**, the *Assistant Axis*:
   $$
     a \;=\; \mu_\text{default} \;-\; \tfrac{1}{R}\sum_r \mu_r
   $$
   i.e. `default − mean(roles)`.
4. Run PCA on the centered role-vector matrix and keep the top-5 PCs
   (`pca_variance_explained`) **purely as a secondary diagnostic** — the
   deconfounder's headline quantity never uses them.
5. For every compliance-direction kind $d$ (`shared_within_label`,
   `authority_given_correct`, `pooled_authority`, …) compute:
   - $\cos(d, a)$ — main output
   - $\cos(d, \mu_r)$ for each role (used in the figure as the role spectrum)
   - $\cos(d, \text{PC}_1)$ (only logged, not plotted)
   - `subspace_norm_fraction` = $\lVert P_5 d\rVert / \lVert d\rVert$
     (fraction of $d$ inside the top-5 PC subspace, logged only)
   - `pc_coords` — $d$'s projection onto the top-5 PCs (logged only)

So the primary deconfounder output is a **single scalar** per model:
$\cos(\text{authority direction}, a)$.

### Numbers on hand (layer 18 / 22 / 16 for GPT-OSS / Gemma-4 / OLMo-2):

| Model   | cos(authority\_given\_correct, Assistant Axis) |
|---------|-----------------------------------------------|
| GPT-OSS | +0.0118                                       |
| Gemma-4 | ≈ −0.037                                      |
| OLMo-2  | ≈ −0.037                                      |

All three are ≈ 0 ⇒ the claim is "the authority direction is orthogonal to the
Assistant Axis", which is what Image 1 shows.

## Answers to the three framing questions

### 1) Does the deconfounder produce one direction, or a subspace?

**One direction.** The headline claim is a single scalar per model,
$\cos(d, a)$, computed against the 1-D Assistant Axis. The PCA block exists,
but it only produces auxiliary diagnostics (PC1 cosine, subspace-norm
fraction, coordinates) that never enter the main result we report.

### 2) Which figure style matches?

**Image 1 — single-axis spectrum.**
Reasons:
- Math is 1-D, so the plot should be 1-D too. Collapsing role activations to
  one coordinate is exactly what the Assistant Axis *is*.
- The natural "bookends" of Image 1 (Role-playing ← … → Assistant-like) are
  the actual two endpoints the axis is constructed from (mean role vs
  default), so the visual metaphor is faithful.
- Stacking PC1/PC2/PC3 panels (Image 2) would imply the deconfounder uses
  multiple independent directions equally; it doesn't.

### 3) When would we move off Image 1?

Two situations, and only two:

- If the story became: "authority is orthogonal to PC1 but not to PC2" →
  Image 2 (stacked panels), because then the honest claim *is* about a
  subspace.
- If a reviewer asks for robustness evidence → **hero + supporting**
  (the middle path). The hero panel stays as Image 1; two thin strips
  below show $\cos(d, \text{PC}_2)$ and $\cos(d, \text{PC}_3)$ for
  completeness, so the reader sees we aren't hiding structure.

## Recommendation

Keep Image 1 as the hero. If a reviewer asks for a robustness supplement,
add two thin PC2/PC3 strips below in the same styling (light grey axis, tiny
markers at each model's projection, no legend). We already log everything
needed for that in `overlap_summary.json` under
`directions.<kind>.pc_coords` and `directions.<kind>.subspace_norm_fraction`;
no recomputation required.

## Snippet of the code path, for reference

The single-direction computation:

```python
# src/exp16/compute_assistant_axis_overlap.py
def _compute_axis(role_vectors, default_vec):
    mean_role = np.mean(np.stack(list(role_vectors.values()), axis=0), axis=0)
    return default_vec - mean_role
```

The deconfounder's headline quantity:

```python
# per compliance-direction kind d
cos_with_axis = _cosine(d, axis)   # one scalar -> figure
```

The PCA block is informational only and is not part of the claim we plot:

```python
pca = PCA(n_components=k); pca.fit(centered)
subspace_norm_fraction = np.linalg.norm(pcs @ d) / np.linalg.norm(d)
cos_with_pc1 = _cosine(pcs[0], d)
```
