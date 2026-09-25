"""Experiment 9: Instruction Override Test (Compliance vs Belief-Updating).

Tests whether the endorsement effect is instruction-overridable compliance
vs belief-like updating by adding an explicit "be correct" instruction.

Design: 2x2 factorial per speaker tag
- Factor 1: Endorsement (Neutral vs Endorse)
- Factor 2: Instruction (absent vs present)

Conditions:
- I0: Neutral, no instruction
- I1: Neutral, with instruction
- E0: Endorse, no instruction
- E1: Endorse, with instruction

Tags tested: Expert, Note (strongest effect and format-conditioned)

Key metrics:
- Endorse_no_instr = E0 - I0
- Endorse_instr = E1 - I1
- Instruction efficacy (diff-in-diff): Δ = (E0 - I0) - (E1 - I1)
- Baseline shift (sanity): I1 - I0

Interpretation:
- Δ > 0 and large: Instruction suppresses endorsement (compliance)
- Δ ≈ 0: Effect persists (deep belief-like integration)
- Δ partial: Mixed mechanism
"""
