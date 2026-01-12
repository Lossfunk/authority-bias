## Big update: Llama‑3.1 sycophancy steering results (Instruct vs Base)

**Topline:** On **Llama‑3.1‑8B‑Instruct**, every attention‑based intervention we tried either does nothing or moves **in the wrong direction** (more sycophantic). The strongest intervention—**patching all 1024 heads with neutral activations**—makes sycophancy *worse* (**−0.107**). On **Llama‑3.1‑8B‑Base**, the *same* full‑head patch moves in the **correct direction** (**+0.035**, weak/noisy).

### Key results (Llama‑3.1‑8B‑Instruct)

**CAA (residual stream) fails**
- Early injection (Layer 2): **−0.065** (α=16)
- Late injection (Layers 29–31): ~0 to **−0.071**

**Distributed test / ablations (n=200 per α)**
- Baseline steering gets *more* wrong with higher α: **−0.013 → −0.094** (α=4→24)
- With **all attention ablated**, effects flip **positive** (correct sign, but model is broken)
- With **MLPs ablated**, effects stay negative / worsen

**CCM / causal head selection doesn’t find a usable circuit**
- Per‑head indirect effects are tiny: max |IE| **< 0.10**, signal/noise **< 0.25**
- “Top mediators” are mostly **early layers** (Layer 0 dominates), not late‑layer “behavior heads”

**Steer only “causal” heads (top‑50) still fails**
- Mean‑diff steering (top‑50): **−0.108 → +0.054** across α (inconsistent, not significant)
- Activation patching (top‑50): **−0.079**

**Upper bound / stress test**
- Patch **ALL 1024 heads** (replace with neutral activations): **−0.107** (worst + most decisive)

### Base vs Instruct (same intervention)

Full‑model patching (replace mode, n=200):

| Model | Heads patched | Mean effect | Std | Direction |
|---|---:|---:|---:|---|
| Instruct | 1024 | **−0.107** | 1.29 | ❌ wrong |
| Base | 1024 | **+0.035** | 1.49 | ✅ correct (weak) |

### Takeaway
- **RLHF / instruction‑tuning changes something fundamental:** the same “neutralize attention” intervention flips sign between **Base** and **Instruct**.
- If sycophancy is steerable on **Instruct**, it’s likely **not** via simple attention‑head patching / mean‑diff vectors; the signal looks either **too distributed** or **primarily outside attention** (MLPs/embeddings), and the metric is extremely noisy.

**Attached:** `llama-results/RESULTS.md` + “Do Persona Vectors and Attention Circuits Align?” (PDF)
