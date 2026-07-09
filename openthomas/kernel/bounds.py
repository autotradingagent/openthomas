"""Kernel policy: which parameters the evolution loop may touch, and how far.

A parameter earns a slot in PARAM_SPACE only when the gate can actually
discriminate on it in replay — tuning what you cannot measure is drift, not
improvement. The loop explores INSIDE this box; widening the box is an
operator decision, never the loop's.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bound:
    lo: float
    hi: float

    def clamp(self, v: float) -> float:
        return max(self.lo, min(self.hi, float(v)))


PARAM_SPACE: dict[str, Bound] = {
    "risk.min_edge": Bound(0.04, 0.15),
    "risk.market_prior_weight": Bound(0.30, 0.80),
}


def clamp_params(params: dict) -> dict:
    """Drop unknown keys, coerce and clamp known ones into their bounds."""
    out: dict[str, float] = {}
    for key, value in (params or {}).items():
        bound = PARAM_SPACE.get(key)
        if bound is None:
            continue
        try:
            out[key] = round(bound.clamp(float(value)), 4)
        except (TypeError, ValueError):
            continue
    return out
