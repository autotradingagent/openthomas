"""The promotion gate: scores candidate strategies, decides what goes live.

KERNEL PLANE — the evolution loop may tune parameters inside PARAM_SPACE; it
has no write path to this module, the truth pipeline it scores with, or these
thresholds. An optimizer that can edit its own judge will find it easier to
lower the bar than to clear it.

The gate scores a *strategy callable* over kernel-collected replay rows, so
it is already shaped for later evolution levels: today the callable is the
stock decision rule with mutated parameters; later it can be an evolved
decision function or workflow — the gate does not change, only the mutation
operator does.

Scoring is total replay PnL, not PnL per contract: a rule that makes two
lucky trades must not outrank one that compounds a small edge across a
hundred — and a rule that trades nothing must not win on drawdown. The
held-out window is the most recent settled days; it rolls forward daily, so
repeated meta-cycles keep facing genuinely new data rather than re-fitting
one static split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

from ..weather.replay import ReplayRow, ReplayTrade, summarize

Strategy = Callable[[list[ReplayRow]], list[ReplayTrade]]

HELD_OUT_DAYS = 7  # most recent settled days, never used to pick a winner
MIN_HELD_IN_TRADES = 20  # below this, any win is indistinguishable from luck
PROMOTION_MARGIN = 0.50  # $ of held-in total PnL the challenger must ADD
HELD_OUT_TOLERANCE = 0.25  # $ of held-out total PnL the challenger may give up
ROLLBACK_MARGIN = 0.50  # $ by which the parent must beat the active gen to revert


@dataclass
class Score:
    params: dict
    held_in: dict = field(default_factory=dict)
    held_out: dict = field(default_factory=dict)

    @property
    def pnl_in(self) -> float:
        return self.held_in.get("total_pnl", 0.0)

    @property
    def pnl_out(self) -> float:
        return self.held_out.get("total_pnl", 0.0)

    def as_dict(self) -> dict:
        return {"params": self.params, "held_in": self.held_in,
                "held_out": self.held_out}


def split_rows(rows: list[ReplayRow],
               held_out_days: int = HELD_OUT_DAYS) -> tuple[list[ReplayRow], list[ReplayRow]]:
    """Held-in = older days (pick winners here); held-out = the freshest days
    (only allowed to veto). Split by calendar day so both sides of a market's
    book land in the same window."""
    if not rows:
        return [], []
    latest = max(date.fromisoformat(r.day) for r in rows)
    cutoff = (latest - timedelta(days=held_out_days)).isoformat()
    return ([r for r in rows if r.day <= cutoff],
            [r for r in rows if r.day > cutoff])


def score(rows_in: list[ReplayRow], rows_out: list[ReplayRow],
          params: dict, strategy: Strategy) -> Score:
    """Run the candidate strategy over both windows; truth math stays here."""
    return Score(
        params=dict(params),
        held_in=summarize(strategy(rows_in)),
        held_out=summarize(strategy(rows_out)),
    )


def beats(challenger: Score, champion: Score) -> tuple[bool, str]:
    """Deterministic promotion rule. Both scores must come from the same rows."""
    n = challenger.held_in.get("n", 0)
    if n < MIN_HELD_IN_TRADES:
        return False, f"held-in n={n} < {MIN_HELD_IN_TRADES}: not enough trades to trust"
    gain = challenger.pnl_in - champion.pnl_in
    if gain < PROMOTION_MARGIN:
        return False, f"held-in gain ${gain:+.2f} < ${PROMOTION_MARGIN:.2f} margin"
    slip = champion.pnl_out - challenger.pnl_out
    if slip > HELD_OUT_TOLERANCE:
        return False, f"held-out regression ${slip:.2f} > ${HELD_OUT_TOLERANCE:.2f} tolerance"
    return True, (f"held-in ${challenger.pnl_in:+.2f} vs ${champion.pnl_in:+.2f} "
                  f"(+${gain:.2f}), held-out ${challenger.pnl_out:+.2f} "
                  f"vs ${champion.pnl_out:+.2f}")


def should_rollback(active: Score, parent: Score) -> tuple[bool, str]:
    """Out-of-sample regret check on the full fresh window. The window has
    rolled forward since promotion, so it now contains days the promotion
    never saw; if the parent wins on those, the promotion was overfit."""
    active_total = active.pnl_in + active.pnl_out
    parent_total = parent.pnl_in + parent.pnl_out
    regret = parent_total - active_total
    if regret > ROLLBACK_MARGIN:
        return True, (f"parent ${parent_total:+.2f} beats active "
                      f"${active_total:+.2f} by ${regret:.2f} on the fresh window")
    return False, f"active holds: regret ${regret:+.2f} within ${ROLLBACK_MARGIN:.2f}"
