"""Exp12: Persistence/Washout test for endorsement-induced state carryover.

Goal:
- Distinguish transient compliance from persistent internal-state carryover.

Core idea:
- Measure immediate endorsement shift at turn 0.
- Re-ask the same task over follow-up turns after endorsement is removed.
- Compare follow-up answers between neutral-history and endorsed-history contexts.

Primary outputs:
- Per-turn residual shift in logit margin (wrong-history vs neutral-history).
- Decay/persistence ratios relative to immediate shift.
- Flip rates in wrong-prior slices.
"""

