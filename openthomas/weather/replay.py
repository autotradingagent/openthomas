"""Market replay: what would the live decision rule have earned on settled
temperature markets?

Mirrors the production pipeline, not a fantasy one: station bias/sigma are
learned strictly BEFORE the replay window (no peeking), the baseline is
blended 50/50 with the market price exactly as the loop does, and only then
does the edge bar apply. Prices come from Kalshi hourly candlesticks at a
fixed decision snapshot. Still conservative vs live: no intraday
observations, no LLM adjustment, yesterday's model run only.

The first, unblended run of this replay lost -$0.038/contract on 417 trades
— the market at 11am already knows the morning obs. That number is why the
market-prior blend is not optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ..markets.base import Market, Side
from ..markets.kalshi import KalshiConnector
from .baseline import strike_probability
from .stations import KALSHI_SERIES, STATIONS, Station
from .strikes import parse_strike
from .verification import VerificationStore, prior_sigma

SNAPSHOT_UTC_HOUR = 15  # ~11am ET: the morning run is priced in, the day is young
REPLAY_LEAD = 1  # freshest guidance that is certainly pre-snapshot


@dataclass
class ReplayTrade:
    ticker: str
    station: str
    side: Side
    price: float  # per contract, for our side
    fee: float
    p: float  # our P(side wins)
    outcome_win: bool
    pnl: float  # per contract


def _snapshot_quotes(kalshi: KalshiConnector, series: str, ticker: str,
                     day: datetime) -> tuple[float, float] | None:
    """(yes_bid, yes_ask) at the decision snapshot, from hourly candles."""
    ts = int(day.replace(hour=SNAPSHOT_UTC_HOUR, tzinfo=timezone.utc).timestamp())
    data = kalshi.http.get(
        f"/series/{series}/markets/{ticker}/candlesticks",
        params={"start_ts": ts - 3600, "end_ts": ts, "period_interval": 60},
    ).json()
    candles = data.get("candlesticks") or []
    if not candles:
        return None
    last = candles[-1]
    try:
        bid = float(last["yes_bid"]["close_dollars"])
        ask = float(last["yes_ask"]["close_dollars"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 < bid and 0 < ask < 1) or ask - bid > 0.10:
        return None  # no real book at the snapshot
    return bid, ask


def replay_station(kalshi: KalshiConnector, store: VerificationStore, series: str,
                   station: Station, kind: str, days: int = 30,
                   min_edge: float = 0.08,
                   market_prior_weight: float = 0.5) -> list[ReplayTrade]:
    window_start = (datetime.now(timezone.utc) - timedelta(days=days))
    min_close = int(window_start.timestamp())
    # Bias/sigma learned only from days before the replay window — no peeking.
    cutoff = window_start.date().isoformat()
    bias, sigma_stat, _ = store.stats(station.key, kind, REPLAY_LEAD, before=cutoff)

    data = kalshi.http.get("/markets", params={
        "series_ticker": series, "status": "settled",
        "min_close_ts": min_close, "limit": 500,
    }).json()

    trades: list[ReplayTrade] = []
    for raw in data.get("markets", []):
        if raw.get("result") not in ("yes", "no"):
            continue
        market: Market = kalshi._to_market(raw)
        strike = parse_strike(market)
        parts = market.id.split("-")
        if strike is None or len(parts) < 3:
            continue
        try:
            day = datetime.strptime(parts[1], "%y%b%d")
        except ValueError:
            continue

        guidance = store.guidance(station.key, kind, day.date(), REPLAY_LEAD)
        if guidance is None:
            continue
        mean, spread = guidance
        sigma = max(sigma_stat, 0.8 * spread)
        p_model = strike_probability(strike, mean + bias, sigma, kind)

        quotes = _snapshot_quotes(kalshi, series, market.id, day)
        if quotes is None:
            continue
        bid, ask = quotes
        # The loop's rule exactly: the crowd is information, not noise.
        p_yes = market_prior_weight * (bid + ask) / 2 + (1 - market_prior_weight) * p_model
        outcome_yes = raw["result"] == "yes"

        for side, price, p in ((Side.YES, ask, p_yes), (Side.NO, 1 - bid, 1 - p_yes)):
            fee = kalshi.fee(price, 1)
            if p - price - fee < min_edge or not (0.05 <= price <= 0.95):
                continue
            win = outcome_yes if side is Side.YES else not outcome_yes
            pnl = (1 - price - fee) if win else (-price - fee)
            trades.append(ReplayTrade(market.id, station.key, side, price, fee,
                                      p, win, pnl))
            break  # at most one side per market
    return trades


def replay_all(store: VerificationStore, days: int = 30,
               min_edge: float = 0.08) -> dict[str, list[ReplayTrade]]:
    kalshi = KalshiConnector()
    out: dict[str, list[ReplayTrade]] = {}
    for series, (station_key, kind) in KALSHI_SERIES.items():
        out[series] = replay_station(kalshi, store, series, STATIONS[station_key],
                                     kind, days, min_edge)
    return out


def summarize(trades: list[ReplayTrade]) -> dict:
    n = len(trades)
    if not n:
        return {"n": 0}
    wins = sum(t.outcome_win for t in trades)
    pnl = sum(t.pnl for t in trades)
    return {
        "n": n, "win_rate": wins / n, "pnl_per_contract": pnl / n, "total_pnl": pnl,
        "avg_edge_priced": sum(t.p - t.price - t.fee for t in trades) / n,
    }
