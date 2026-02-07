# Exp8 (Order/Recency vs Endorsement) — Results Summary

This write‑up reflects the **matched‑template order control** version of Exp8, using the new decomposition:

- **order** = C2_order − C0_order
- **endorse** = C1_contrast − C2_order
- **contrast** = C1_plain − C1_contrast  (negative ⇒ “not A” strengthens endorsement)
- **fragment_leak** = C2_fragment − C2_order
- **total_contrast** = C1_contrast − C0_order
- **total_plain** = C1_plain − C0_order

All effects are in **logit space** of P(wrong). Positive values mean a shift toward the wrong label.

---

## 1) Main findings (overall)
### Instruct (Llama‑3.1‑8B‑Instruct)
- **order:** **+0.338** (pos 65%)
- **endorse:** **+1.066** (pos 89%)
- **contrast:** **−0.440** (pos 30%)
- **fragment_leak:** **+0.352** (pos 61%)
- **total_contrast:** **+1.404** (pos 91%)
- **total_plain:** **+0.964** (pos 82%)

**Interpretation:** endorsement remains large and robust under matched order control. The contrastive “not A” *strengthens* the endorsement effect (negative contrast). Order and fragment leakage are non‑zero but clearly smaller than endorsement.

### Base (Llama‑3.1‑8B)
- **order:** **+0.069** (pos 54%)
- **endorse:** **+0.523** (pos 94%)
- **contrast:** **−0.135** (pos 31%)
- **fragment_leak:** **+0.098** (pos 60%)
- **total_contrast:** **+0.592** (pos 94%)
- **total_plain:** **+0.457** (pos 81%)

**Interpretation:** endorsement survives and dominates; order/fragment effects are small. Contrastive negation again strengthens endorsement (negative contrast).

**Decision rule outcomes (from Step 6):**
- `endorse` **remains large and consistently positive** ⇒ endorsement/compliance channel is real.
- `order` is smaller than `endorse` ⇒ order/recency is **not** the main driver.
- `fragment_leak` exists (especially in Instruct) ⇒ the old fragment control is indeed leaky, but does **not** explain the full effect.

---

## 2) Condition means (forced‑choice P(wrong))
These show the raw shift across conditions.

### Instruct
- C0_order: **0.315**
- C2_order: **0.337**
- C1_contrast: **0.524**
- C1_plain: **0.472**
- C2_fragment: **0.401**

### Base
- C0_order: **0.476**
- C2_order: **0.490**
- C1_contrast: **0.606**
- C1_plain: **0.580**
- C2_fragment: **0.513**

**Interpretation:**
- Order effect is visible but small (C2_order − C0_order).
- Endorsement strongly increases P(wrong) (C1_contrast vs C2_order).
- Contrastive negation boosts more than plain endorsement (C1_contrast > C1_plain).
- Fragment condition is non‑neutral (C2_fragment > C2_order), confirming leakage.

---

## 3) Dataset heterogeneity (mean logit components)
**Base**
- TriviaQA: order 0.137, endorse 0.492, contrast −0.098, fragment_leak 0.088
- TruthfulQA: order −0.013, endorse 0.560, contrast −0.180, fragment_leak 0.110

**Instruct**
- TriviaQA: order 0.523, endorse 1.090, contrast −0.623, fragment_leak 0.313
- TruthfulQA: order 0.112, endorse 1.036, contrast −0.215, fragment_leak 0.400

**Interpretation:** endorsement is robust across datasets; order and contrast effects are larger on TriviaQA for Instruct, but endorsement stays strong in both datasets.

---

## 4) Label assignment effects (mean logit components)
**Base**
- Correct=A: order 0.119, endorse 0.631, contrast +0.069, fragment_leak 0.242
- Correct=B: order 0.023, endorse 0.422, contrast −0.325, fragment_leak −0.036

**Instruct**
- Correct=A: order 0.372, endorse 1.381, contrast −0.308, fragment_leak 0.880
- Correct=B: order 0.306, endorse 0.773, contrast −0.562, fragment_leak −0.139

**Interpretation:** label bias still modulates effect size (larger shifts when correct=A for both models), but endorsement stays positive in both label settings.

---

## 5) Baseline confidence bins (|C0_order|)
**Base** (mean endorse by bin):
- [0,0.5): 0.453
- [0.5,1): 0.561
- [1,2): 0.548
- [2,4): 0.679

**Instruct** (mean endorse by bin):
- [0,0.5): 1.033
- [0.5,1): 1.130
- [1,2): 1.145
- [2,4): 1.096
- ≥4: 0.781

**Interpretation:** endorsement stays strong across confidence levels; order and fragment effects tend to grow with high baseline confidence for some bins, especially in Instruct.

---

## 6) Key conclusions (what these results actually show)
1. **Endorsement/compliance is the dominant effect**, not order/recency. This holds for both Base and Instruct.
2. **Contrastive negation (“not A”) strengthens endorsement** (negative contrast component).
3. The **fragment control is leaky**, but the leakage is smaller than the endorsement component.
4. **Instruct >> Base** in endorsement magnitude, consistent with stronger compliance in RLHF‑tuned models.

---

## 7) Cautions / limits
- These results still do not distinguish **epistemic belief updating** vs **cooperative compliance**; they only show stance‑sensitive behavioral shifts.
- Label assignment effects indicate that **format bias** is still a confound in magnitude, even if it does not erase the effect.

---

## 8) Bottom line
With matched‑template order controls, the endorsement effect **survives clearly and strongly**. Order/recency contributes, but it is **not** the main driver. The contrastive wording (“not A”) makes endorsement stronger, and the fragment condition is measurably leaky but insufficient to explain the main effect.
