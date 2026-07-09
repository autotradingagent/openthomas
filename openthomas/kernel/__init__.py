"""The kernel plane: what the self-improvement loop can never touch.

OpenThomas is split into two planes (docs/RSI.md):

- **kernel** (this package, plus `risk/`, `weather/verification.py`, and the
  row-collection half of `weather/replay.py`): the evaluator, ground-truth
  pipeline, parameter bounds, promotion gate, and risk engine. Owned by the
  operator; the evolution loop has no write path here — structurally, not by
  convention. The operator's ongoing role is tuning kernel policy (bounds,
  gate thresholds, risk caps), never editing the agent plane by hand.

- **agent** (everything else, today mutated at parameter level via
  `improve/`): how to scan, forecast, decide, and evolve. Every change must
  pass the kernel gate on leak-free replay before it takes effect.

The agent plane can change everything about *how*; it can never change *what
counts as good*. That single invariant is what blocks reward hacking.
"""
