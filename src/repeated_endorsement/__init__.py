"""Exp12: Persistence/Washout test for endorsement-induced state carryover.

Goal:
- Distinguish transient compliance from persistent internal-state carryover.

Core idea:
- Measure immediate endorsement shift at turn 0.
- Re-ask the same task over follow-up turns after endorsement is removed.
- Compare follow-up answers between neutral-history and endorsed-history contexts.
- Support timing controls for instruction onset (`none`, `t0`, `t1`, `t2`).
- Run fresh probes independently per branch to avoid result-sharing artifacts.

Primary outputs:
- Per-turn residual shift in logit margin (wrong-history vs neutral-history).
- Decay/persistence ratios relative to immediate shift.
- Flip rates in wrong-prior slices.
"""
