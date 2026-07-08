"""Statistical temperature baseline: P(YES) for a strike from multi-model
consensus, learned station bias/spread, and today's observed extreme.

The LLM never replaces this number — it may only nudge it within a bounded
band for information the statistics can't see (late-breaking obs, a model
consensus gone stale). Design rule from the Bridgewater AIA result: blends
beat pure LLM forecasts.
"""

from __future__ import annotations

from datetime import datetime
from statistics import NormalDist

from .strikes import Strike

# Official CLI temperatures are integers; score strikes at half-integer
# thresholds so "> 83" means "≥ 84" gets the mass from 83.5 up.
_HALF = 0.5


def _mix_cdf(x: float, mu: float, sigma: float) -> float:
    """Normal core with a fat tail (20% mass at double spread): temperature
    busts — blown fronts, stalled sea breezes — happen more than a Gaussian
    admits."""
    return 0.8 * NormalDist(mu, sigma).cdf(x) + 0.2 * NormalDist(mu, 2 * sigma).cdf(x)


def hour_factor(kind: str, now_local: datetime, target_is_today: bool) -> float:
    """Same-day highs are mostly realized by late afternoon — spread shrinks
    as the day plays out. Lows keep full spread (an evening front can still
    set the day's minimum before the market closes)."""
    if not target_is_today or kind != "high":
        return 1.0
    h = now_local.hour
    if h >= 18:
        return 0.25
    if h >= 15:
        return 0.5
    if h >= 12:
        return 0.8
    return 1.0


def strike_probability(strike: Strike, mu: float, sigma: float, kind: str,
                       observed: float | None = None) -> float:
    """P(YES) under the mixture distribution, truncated by today's observed
    extreme: an official high can't end below what's already on the books."""
    def cdf(t: float) -> float:
        return _mix_cdf(t, mu, sigma)

    if observed is not None:
        f_obs = cdf(observed)
        if kind == "high":
            if f_obs >= 0.999:  # no headroom left in the distribution
                return float(strike.covers(observed))
            def cdf(t: float, _f=f_obs):  # noqa: E731 — truncate below observed
                return 0.0 if t < observed else (_mix_cdf(t, mu, sigma) - _f) / (1 - _f)
        else:
            if f_obs <= 0.001:
                return float(strike.covers(observed))
            def cdf(t: float, _f=f_obs):  # truncate above observed
                return 1.0 if t >= observed else _mix_cdf(t, mu, sigma) / _f

    if strike.kind == "greater":
        return 1.0 - cdf(strike.lo + _HALF)
    if strike.kind == "less":
        return cdf(strike.hi - _HALF)
    return max(0.0, cdf(strike.hi + _HALF) - cdf(strike.lo - _HALF))
