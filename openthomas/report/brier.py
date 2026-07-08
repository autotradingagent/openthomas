"""Bucketed forecast skill: Brier by station × lead, model vs market.

The market's own price at forecast time is the baseline to beat — the
Prediction Arena finding is that beating it at all is rare. Buckets localize
where any edge actually lives (which station, which lead) so the playbook,
the risk profile, and the roadmap act on facts instead of vibes.
"""

from __future__ import annotations

from datetime import datetime

from ..memory.journal import Journal
from ..weather.stations import KALSHI_SERIES


def weather_skill(journal: Journal) -> list[dict]:
    """One row per (station, lead bucket): n, Brier(model), Brier(market),
    and skill = 1 − model/market (positive → we beat the price)."""
    rows = journal.db.execute(
        """SELECT f.market_id, f.ts, f.p_calibrated AS p, f.mid,
                  CASE WHEN s.outcome = 'yes' THEN 1 ELSE 0 END AS y
           FROM settlements s
           JOIN forecasts f ON f.id = (SELECT id FROM forecasts
                                       WHERE market_id = s.market_id
                                       ORDER BY ts LIMIT 1)"""
    ).fetchall()

    buckets: dict[tuple[str, str], dict] = {}
    for r in rows:
        parts = r["market_id"].split("-")
        hit = KALSHI_SERIES.get(parts[0])
        if hit is None or len(parts) < 3:
            continue
        station_key, _kind = hit
        try:
            target = datetime.strptime(parts[1], "%y%b%d").date()
        except ValueError:
            continue
        lead = (target - datetime.fromisoformat(r["ts"]).date()).days
        lead_bucket = "3+" if lead >= 3 else str(max(lead, 0))
        b = buckets.setdefault((station_key, lead_bucket),
                               {"n": 0, "se_model": 0.0, "n_mkt": 0, "se_market": 0.0})
        b["n"] += 1
        b["se_model"] += (r["p"] - r["y"]) ** 2
        if r["mid"] is not None:
            b["n_mkt"] += 1
            b["se_market"] += (r["mid"] - r["y"]) ** 2

    out = []
    for (station, lead_bucket), b in sorted(buckets.items()):
        brier_model = b["se_model"] / b["n"]
        brier_market = (b["se_market"] / b["n_mkt"]) if b["n_mkt"] else None
        out.append({
            "station": station, "lead": lead_bucket, "n": b["n"],
            "brier_model": brier_model, "brier_market": brier_market,
            "skill": (1 - brier_model / brier_market)
            if brier_market not in (None, 0) else None,
        })
    return out


def summarize_skill(buckets: list[dict]) -> dict | None:
    scored = [b for b in buckets if b["brier_market"] is not None]
    if not scored:
        return None
    n = sum(b["n"] for b in scored)
    model = sum(b["brier_model"] * b["n"] for b in scored) / n
    market = sum(b["brier_market"] * b["n"] for b in scored) / n
    return {"n": n, "brier_model": model, "brier_market": market,
            "skill": 1 - model / market if market else None}
