"""The public feed: what openthomas.com publishes about the running agent.

Build in public, trade in public. The agent's private state is a journal, a
lineage archive, and a token ledger; this module projects those into one
JSON document a static site can render — the positions we hold, the edges we
think we've found and why, what the evolution loop is currently trying to
improve, and what the compute cost.

Two rules govern what lands here:

1. **Whitelist, never dump.** Every field is named explicitly below. The
   journal holds prompt inputs, news text, and venue identifiers that have no
   business on a public page, and a `SELECT *` reaching the feed would be a
   leak, not a bug in rendering.
2. **Never guess.** Account value comes from the last recorded cycle, not from
   a fresh mark-to-market — publishing must not depend on venue APIs being up,
   and a stale-but-true number beats a fresh-but-invented one. `as_of` says
   when that number was real.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..config import Settings
from ..forecast.calibration import brier_score
from ..improve.genome import GenerationStore, display_params
from ..memory import board as board_store
from ..memory import heartbeat
from ..memory.journal import Journal
from ..memory.usage import UsageLedger, summarize
from ..report import dispatch
from ..report.vital import max_drawdown
from ..weather.geo import locate
from ..weather.temps import global_grid

SCHEMA_VERSION = 4


def _downsample(curve: list[tuple[str, float]], limit: int) -> list[list]:
    """Keep the shape and both endpoints; the site plots a sparkline, not a tape."""
    if len(curve) <= limit:
        keep = curve
    else:
        stride = len(curve) / limit
        idx = sorted({int(i * stride) for i in range(limit)} | {len(curve) - 1})
        keep = [curve[i] for i in idx]
    return [[ts, round(v, 2)] for ts, v in keep]


def _thesis(row: dict, settings: Settings, status: str) -> dict:
    """One market view, stripped to what a reader needs to judge it later.

    `mid` is the market's price when we formed the view; freezing it here is
    the point — a claimed edge is only checkable against the price that was
    actually on offer at the time.
    """
    mid = row.get("mid")
    p = row["p_calibrated"]
    edge = None if mid is None else p - mid
    reasoning = (row.get("reasoning") or "").strip()
    cap = settings.site.max_reasoning_chars
    return {
        "ts": row["ts"],
        "status": status,  # held | pending | passed
        "platform": row["platform"],
        "question": row["question"],
        "category": row["category"] or "",
        "side": None if edge is None else ("yes" if edge > 0 else "no"),
        "p_model": round(p, 4),
        "p_raw": round(row["p_raw"], 4),
        "p_market": None if mid is None else round(mid, 4),
        "edge": None if edge is None else round(edge, 4),
        "confidence": round(row["confidence"], 3),
        "base_rate": row["base_rate"],
        "why": row.get("market_gap_reason") or "",
        "invalidation": row.get("invalidation") or "",
        "reasoning": reasoning[:cap] + ("…" if len(reasoning) > cap else ""),
        "model": row.get("model") or "",
        # Where the weather is — for the globe. Derived from the market, never
        # the returned id, so no venue handle leaks.
        "loc": locate(row.get("market_id"), row["platform"], row["question"]),
    }


def _theses(journal: Journal, settings: Settings) -> list[dict]:
    """Live market views: what we hold, and what currently clears the edge bar.

    A forecast is `pending` only while it is still actionable — inside the
    staleness window and not yet traded or settled. Yesterday's untaken edge is
    not a bet we are "about to place", and publishing it as one would be a
    claim we never made.
    """
    held = {p.market_id for p in journal.positions()}
    traded, settled = journal.traded_market_ids(), journal.settled_market_ids()
    fresh = datetime.now(timezone.utc) - timedelta(hours=24)

    out = []
    for row in journal.recent_forecasts(limit=200):
        mid, mkt = row.get("mid"), row["market_id"]
        if mkt in held:
            status = "held"
        elif mkt in settled or mkt in traded:
            continue  # closed book: it belongs in the track record, not the outlook
        elif (mid is not None
              and abs(row["p_calibrated"] - mid) >= settings.risk.min_edge
              and datetime.fromisoformat(row["ts"]) >= fresh):
            status = "pending"
        else:
            continue
        out.append(_thesis(row, settings, status))

    rank = {"held": 0, "pending": 1}
    out.sort(key=lambda t: (rank[t["status"]], -abs(t["edge"] or 0)))
    return out[: settings.site.max_theses]


def _board(journal: Journal, settings: Settings) -> dict:
    """Every weather market the agent is watching, placed on the globe with our
    view attached: which we forecast, hold, or have already won or lost.

    Prefers the live snapshot the trading loop writes each cycle (the whole
    book, with real prices); falls back to the markets we have forecast when no
    snapshot exists yet, so the globe is never empty. Market ids are the join
    key and stay server-side — only place, price, and our published view ship.
    Plain markets (no view of ours) carry price only; ours carry the analysis.
    """
    snap = board_store.read(settings.home) or {}
    rows = snap.get("markets")
    from_snapshot = rows is not None
    if not rows:  # no snapshot yet: plot the markets we have opinions on
        seen: set[str] = set()
        rows = []
        for r in journal.recent_forecasts(limit=400):
            if r["market_id"] in seen:
                continue
            seen.add(r["market_id"])
            rows.append({"id": r["market_id"], "platform": r["platform"],
                         "question": r["question"], "category": r["category"],
                         "yes_bid": None, "yes_ask": None, "mid": r.get("mid"),
                         "volume_24h": None, "close_time": None})

    positions = {p.market_id: p for p in journal.positions()}
    settled = {s["market_id"]: s for s in journal.recent_settlements(limit=200)}
    latest: dict[str, dict] = {}
    for r in journal.recent_forecasts(limit=400):  # DESC ts → first seen is newest
        latest.setdefault(r["market_id"], r)

    fresh = datetime.now(timezone.utc) - timedelta(hours=24)
    cap = settings.site.max_reasoning_chars
    out = []
    for m in rows:
        loc = locate(m.get("id"), m["platform"], m["question"])
        if loc is None:
            continue  # can't place it → no pin
        mid = m.get("mid")
        if mid is None and m.get("yes_bid") is not None and m.get("yes_ask") is not None:
            mid = (m["yes_bid"] + m["yes_ask"]) / 2
        f, s, pos = latest.get(m["id"]), settled.get(m["id"]), positions.get(m["id"])
        p_model = round(f["p_calibrated"], 4) if f else None
        edge = None if (p_model is None or mid is None) else round(p_model - mid, 4)

        if s is not None:
            state = "won" if s["pnl"] >= 0 else "lost"
        elif pos is not None:
            state = "held"
        elif (f and edge is not None and abs(edge) >= settings.risk.min_edge
              and datetime.fromisoformat(f["ts"]) >= fresh):
            state = "pending"
        else:
            state = "market"

        entry = {
            "loc": loc, "place": loc["place"], "platform": m["platform"],
            "question": m["question"], "state": state,
            "mid": None if mid is None else round(mid, 4),
            "yes_bid": m.get("yes_bid"), "yes_ask": m.get("yes_ask"),
            "volume": m.get("volume_24h"),
            "close": m.get("close_time"),
        }
        if state != "market":  # our view — kept lean for plain book markets
            reasoning = (f.get("reasoning") if f else "") or ""
            entry.update({
                "p_model": p_model, "edge": edge,
                "side": None if edge is None else ("yes" if edge > 0 else "no"),
                "why": (f.get("market_gap_reason") if f else "") or "",
                "reasoning": reasoning[:cap] + ("…" if len(reasoning) > cap else ""),
                "outcome": s["outcome"] if s else None,
                "pnl": None if s is None else round(s["pnl"], 2),
            })
        out.append(entry)

    rank = {"held": 0, "pending": 1, "won": 2, "lost": 3, "market": 4}
    out.sort(key=lambda e: (rank[e["state"]], -abs(e.get("edge") or 0)))
    return {"from_snapshot": from_snapshot, "as_of": snap.get("ts"),
            "markets": out[: settings.site.max_board]}


def _sample_grid(g: dict, lat: float, lon: float) -> float | None:
    """Bilinear temperature at (lat, lon) from a lon/lat grid, or None if a
    corner has no data. Longitude wraps; latitude clamps to the poles."""
    import math
    nx, ny, temps = g["nx"], g["ny"], g["temps"]
    fx = (lon - g["lon0"]) / g["dlon"]
    fy = (lat - g["lat0"]) / g["dlat"]
    x0 = math.floor(fx)
    y0 = max(0, min(ny - 2, math.floor(fy)))
    tx, ty = fx - x0, fy - y0
    xa, xb = x0 % nx, (x0 + 1) % nx
    c = [temps[y0 * nx + xa], temps[y0 * nx + xb],
         temps[(y0 + 1) * nx + xa], temps[(y0 + 1) * nx + xb]]
    if any(v is None for v in c):
        return None
    return (c[0] * (1 - tx) + c[1] * tx) * (1 - ty) + (c[2] * (1 - tx) + c[3] * tx) * ty


def _anomaly(settings: Settings, grid: dict | None) -> list[dict]:
    """How far each city sits from its monthly normal (°C). Red-hot, blue-cold on
    the globe — unusual weather is where markets are slow to reprice.

    The comparison must be mean-to-mean or the day/night cycle drowns it out, so
    we use today's daily-mean temperature per city when a fetch has cached one;
    otherwise we estimate the daily mean from the current grid snapshot by
    subtracting a modelled diurnal swing (warm ~15:00 local, cool ~03:00). Needs
    cached normals; empty without them.
    """
    import math

    from ..weather.anomaly import MONTHS, current_temps, known_coords, normals
    nrm = normals(settings.home)
    if not nrm:
        return []
    daily = current_temps(settings.home)
    if not daily and not grid:
        return []

    now = datetime.now(timezone.utc)
    month = MONTHS[now.month - 1]
    try:
        snap = datetime.fromisoformat(grid["as_of"]) if grid else now
        uh = snap.astimezone(timezone.utc).hour + snap.minute / 60
    except (KeyError, ValueError, TypeError):
        uh = now.hour + now.minute / 60

    out = []
    for place, lat, lon in known_coords():
        key = f"{lat},{lon}"
        n = nrm.get(key)
        if not n or month not in n:
            continue
        mean = daily.get(key)
        if mean is None and grid is not None:
            cur = _sample_grid(grid, lat, lon)
            if cur is None:
                continue
            local = (uh + lon / 15.0) % 24
            mean = cur - 4.5 * math.cos(2 * math.pi * (local - 15) / 24)  # → est. daily mean
        if mean is None:
            continue
        out.append({"place": place, "lat": lat, "lon": lon,
                    "temp": round(mean, 1), "normal": n[month],
                    "anomaly": round(mean - n[month], 1),
                    "estimated": key not in daily})
    out.sort(key=lambda x: -abs(x["anomaly"]))
    return out


def _skill(journal: Journal, settings: Settings) -> list[dict]:
    """Per place: how contrarian we are, and whether it pays.

    `disagreement` is the average gap between our probability and the market's,
    over every recent forecast there — data-rich, it maps where we bet against
    the crowd. `skill` is 1 − Brier(us)/Brier(market) on *settled* markets: the
    Prediction Arena baseline is the price itself, so skill > 0 means we've
    actually beaten it here. Sparse until markets settle, and honest about it
    (`n_settled`). The globe colours cities by skill, sizes them by disagreement.
    """
    from ..report.brier import weather_skill
    from ..weather.stations import KALSHI_SERIES, STATIONS

    agg: dict[str, dict] = {}

    def cell(place, lat, lon):
        return agg.setdefault(place, {
            "place": place, "lat": lat, "lon": lon, "dsum": 0.0, "n_forecasts": 0,
            "n_settled": 0, "skill": None, "brier_model": None, "brier_market": None,
            "win": 0, "settled": 0, "pnl": 0.0})

    for r in journal.recent_forecasts(limit=500):
        if r.get("mid") is None:
            continue
        loc = locate(r.get("market_id"), r["platform"], r["question"])
        if loc is None:
            continue
        c = cell(loc["place"], loc["lat"], loc["lon"])
        c["dsum"] += abs(r["p_calibrated"] - r["mid"])
        c["n_forecasts"] += 1

    by_station: dict[str, dict] = {}
    for b in weather_skill(journal):
        s = by_station.setdefault(b["station"], {"n": 0, "sm": 0.0, "nk": 0, "smk": 0.0})
        s["n"] += b["n"]; s["sm"] += b["brier_model"] * b["n"]
        if b["brier_market"] is not None:
            s["nk"] += b["n"]; s["smk"] += b["brier_market"] * b["n"]
    for key, s in by_station.items():
        st = STATIONS[key]
        c = cell(st.name.split(",")[0], st.lat, st.lon)
        c["n_settled"] = s["n"]
        c["brier_model"] = round(s["sm"] / s["n"], 3)
        if s["nk"]:
            bk = s["smk"] / s["nk"]
            c["brier_market"] = round(bk, 3)
            c["skill"] = round(1 - (s["sm"] / s["n"]) / bk, 3) if bk else None

    for row in journal.recent_settlements(limit=200):
        hit = KALSHI_SERIES.get(row["market_id"].split("-")[0])
        if hit is None:
            continue
        st = STATIONS[hit[0]]
        c = cell(st.name.split(",")[0], st.lat, st.lon)
        c["settled"] += 1
        c["win"] += 1 if row["pnl"] >= 0 else 0
        c["pnl"] += row["pnl"]

    out = []
    for c in agg.values():
        out.append({
            "place": c["place"], "lat": c["lat"], "lon": c["lon"],
            "disagreement": round(c["dsum"] / c["n_forecasts"], 3) if c["n_forecasts"] else None,
            "n_forecasts": c["n_forecasts"], "n_settled": c["n_settled"],
            "skill": c["skill"], "brier_model": c["brier_model"], "brier_market": c["brier_market"],
            "win_rate": round(c["win"] / c["settled"], 2) if c["settled"] else None,
            "pnl": round(c["pnl"], 2) if c["settled"] else None,
        })
    out.sort(key=lambda x: -(abs(x["skill"] or 0) * 5 + (x["disagreement"] or 0)))
    return out[: settings.site.max_board]


def _board_mids(settings: Settings) -> dict[str, float]:
    """`{market_id: mid}` from the live board snapshot — the current price the
    trading loop last saw, so open positions can be marked to market. Keyed by
    the venue id, which stays server-side; only the derived value ships."""
    snap = board_store.read(settings.home) or {}
    out: dict[str, float] = {}
    for m in snap.get("markets") or []:
        bid, ask = m.get("yes_bid"), m.get("yes_ask")
        if bid is not None and ask is not None:
            out[m["id"]] = (bid + ask) / 2
    return out


def _positions(journal: Journal, mids: dict[str, float]) -> tuple[list[dict], float, float]:
    """Open positions marked to market, and the book's value and unrealized PnL.

    A YES contract is worth today's mid; a NO contract, one minus it. When no
    live price is on hand we hold the position at cost — a missing quote is not
    a move, and inventing one would put a fake gain or loss on the page. Returns
    the rows plus (positions_value, unrealized_pnl) so the ledger and the list
    read from the same marks.
    """
    rows: list[dict] = []
    value = cost = 0.0
    for p in journal.positions():
        mid = mids.get(p.market_id)
        mark = p.avg_cost if mid is None else (mid if p.side.value == "yes" else 1 - mid)
        worth = p.qty * mark
        value += worth
        cost += p.cost_basis
        rows.append({
            "platform": p.platform, "question": p.question, "category": p.category,
            "side": p.side.value, "qty": p.qty, "avg_cost": round(p.avg_cost, 4),
            "cost_basis": round(p.cost_basis, 2),
            "mark": round(mark, 4), "priced": mid is not None,
            "value": round(worth, 2), "unrealized": round(worth - p.cost_basis, 2),
            "loc": locate(p.market_id, p.platform, p.question),
        })
    rows.sort(key=lambda r: -abs(r["unrealized"]))
    return rows, round(value, 2), round(value - cost, 2)


def _activity(journal: Journal, limit: int = 40) -> list[dict]:
    """The tape: every entry, exit, and resolution, newest first. Buys and sells
    come from the fill log with the price paid; settlements carry the realized
    PnL. Market ids join the two server-side and never ship — only the place, the
    action, and the money do, the same shape a reader sees on the venue."""
    out: list[dict] = []
    for t in journal.recent_trades(limit=limit):
        out.append({
            "ts": t["ts"], "kind": t["action"],  # buy | sell
            "platform": t["platform"], "question": t["question"],
            "category": t["category"] or "", "side": t["side"], "qty": t["qty"],
            "price": round(t["price"], 4), "cost": round(t["qty"] * t["price"], 2),
            "loc": locate(t["market_id"], t["platform"], t["question"]),
        })
    for s in journal.recent_settlements(limit=limit):
        out.append({
            "ts": s["ts"], "kind": "settle", "platform": s["platform"],
            "question": s["question"], "category": s["category"] or "",
            "outcome": s["outcome"], "pnl": round(s["pnl"], 2),
            "loc": locate(s["market_id"], s["platform"], s["question"]),
        })
    out.sort(key=lambda e: e["ts"], reverse=True)
    return out[:limit]


def _performance(journal: Journal, settings: Settings,
                 positions_value: float, unrealized_pnl: float) -> dict:
    curve = journal.equity_curve()
    stats = journal.settlement_stats()
    pairs = journal.forecast_outcome_pairs()
    value = curve[-1][1] if curve else settings.bankroll
    realized = round(stats["pnl"], 2)
    return {
        "as_of": curve[-1][0] if curve else None,
        "account_value": round(value, 2),
        "bankroll": settings.bankroll,
        "return_pct": round(value / settings.bankroll - 1, 6) if settings.bankroll else 0.0,
        "peak_value": round(journal.peak_value(), 2),
        "max_drawdown": round(max_drawdown(curve), 6),
        "cycles": len(curve),
        "settled_trades": stats["n"],
        "realized_pnl": realized,
        "unrealized_pnl": round(unrealized_pnl, 2),
        "total_pnl": round(realized + unrealized_pnl, 2),
        "positions_value": round(positions_value, 2),
        "win_rate": round(stats["win_rate"], 4),
        "avg_win": round(stats["avg_win"], 2),
        "avg_loss": round(stats["avg_loss"], 2),
        # Biggest win/loss are over the whole settled book, not the shown tail —
        # the single trade that most defines the record, win or lose.
        "biggest_win": None if not stats["n"] else round(stats["best"], 2),
        "biggest_loss": None if not stats["n"] else round(stats["worst"], 2),
        "brier": round(brier_score(pairs), 4) if pairs else None,
        "forecasts_scored": len(pairs),
        "equity_curve": _downsample(curve, settings.site.max_curve_points),
    }


def _reports(settings: Settings) -> list[dict]:
    """The daily dispatch log — build-in-public field notes, newest first.

    Every field is templated from the journal upstream (see report/dispatch.py),
    so nothing here escapes the same whitelist the rest of the feed obeys."""
    return dispatch.read_reports(settings.home, dispatch.MAX_REPORTS)


def _track_record(journal: Journal) -> list[dict]:
    return [
        {"ts": s["ts"], "platform": s["platform"], "question": s["question"],
         "category": s["category"] or "", "outcome": s["outcome"],
         "pnl": round(s["pnl"], 2), "cost_basis": round(s["cost_basis"], 2),
         "loc": locate(s["market_id"], s["platform"], s["question"])}
        for s in journal.recent_settlements(limit=25)
    ]


def _rsi(settings: Settings) -> dict:
    """What the evolution loop is trying to improve, and what it has proven.

    An empty lineage is the honest answer before the first meta-cycle: the
    operator's config is in force and nothing has been promoted past it.
    """
    gens = GenerationStore(settings.home).all()
    active = next((g for g in gens if g.status == "active"), None)
    log_path = settings.home / "improve-log.jsonl"
    meta_cycles = []
    if log_path.exists():
        for line in log_path.read_text().splitlines()[-20:]:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta_cycles.append({
                "ts": entry.get("ts"), "operator": entry.get("operator"),
                "replay_rows": entry.get("rows"), "promoted": entry.get("promoted"),
                "rollback": entry.get("rollback") or "", "reason": entry.get("reason") or "",
                "candidates": [
                    {"proposer": c.get("proposer"), "verdict": c.get("verdict"),
                     "params": display_params(c.get("params") or {}),
                     "pnl_in": (c.get("held_in") or {}).get("total_pnl"),
                     "pnl_out": (c.get("held_out") or {}).get("total_pnl"),
                     "brier": (c.get("held_in") or {}).get("brier")}
                    for c in entry.get("candidates") or []
                ],
            })
    return {
        "active_generation": None if active is None else {
            "id": active.id, "parent": active.parent, "operator": active.operator,
            "proposer": active.proposer, "created": active.created,
            "rationale": active.rationale, "evidence": active.evidence,
            "scores": active.scores, "params": display_params(active.params),
        },
        "generations": [
            {"id": g.id, "parent": g.parent, "status": g.status, "operator": g.operator,
             "proposer": g.proposer, "created": g.created,
             "rationale": g.rationale or g.note}
            for g in gens
        ],
        "meta_cycles": list(reversed(meta_cycles)),
    }


def _status(settings: Settings) -> dict:
    """Is the loop alive, and when did it last act? Straight from the heartbeat.

    `last_cycle` is the honest liveness signal — the site decides "live" from
    how recent it is, rather than this builder asserting a state it cannot see.
    An empty beat is the truthful answer before the loop's first cycle.
    """
    beat = heartbeat.read(settings.home) or {}
    return {
        "run_started": beat.get("run_started"),
        "last_cycle": beat.get("last_cycle"),
        "cycles_this_run": beat.get("cycles_this_run") or 0,
    }


def _compute(settings: Settings, journal: Journal, status: dict) -> dict:
    """Token spend — all-time, and just this run — with the facts that keep a
    small number from lying.

    `ledger_started` dates the accounting: the agent forecast for weeks before
    it counted tokens, and a reader seeing "0 tokens" deserves to know whether
    that means "cheap" or "not measured yet". `session` is spend since the
    current process started (see status.run_started), so a reader can watch
    what this run costs without doing the subtraction. `forecasts_recorded` is
    the journal's own count, which predates the ledger.
    """
    rows = UsageLedger(settings.home).read()
    started = status.get("run_started")
    session_rows = [r for r in rows if started and r.ts >= started]
    return {
        **summarize(rows),
        "session": {**summarize(session_rows), "since": started},
        "ledger_started": min((r.ts for r in rows), default=None),
        "forecasts_recorded": journal.forecast_count(),
    }


def build_feed(settings: Settings, journal: Journal | None = None) -> dict:
    journal = journal or Journal(settings.db_path)
    status = _status(settings)
    temperature = global_grid(settings.home)
    positions, book_value, unrealized = _positions(journal, _board_mids(settings))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "agent": {
            "mode": settings.mode,
            "focus": settings.focus,
            "platforms": settings.platforms,
            "goal": settings.goal,
            "risk_profile": settings.risk.name,
            "forecaster": {
                "label": settings.site.model_label or settings.forecaster.model,
                "url": settings.site.model_url,
            },
            "cycle_minutes": settings.cycle_minutes,
            "min_edge": settings.risk.min_edge,
            "kelly_fraction": settings.risk.kelly_fraction,
            "market_prior_weight": settings.risk.market_prior_weight,
        },
        "performance": _performance(journal, settings, book_value, unrealized),
        "temperature": temperature,
        "board": _board(journal, settings),
        "skill": _skill(journal, settings),
        "anomaly": _anomaly(settings, temperature),
        "positions": positions,
        "theses": _theses(journal, settings),
        "track_record": _track_record(journal),
        "activity": _activity(journal),
        "reports": _reports(settings),
        "rsi": _rsi(settings),
        "compute": _compute(settings, journal, status),
        "links": {
            "github": settings.site.github,
            "huggingface": settings.site.huggingface,
            "x": f"https://x.com/{settings.site.x_handle}" if settings.site.x_handle else "",
        },
    }


def publish(settings: Settings, out_dir: str | Path) -> Path:
    """Write feed.json atomically — the site reads this file while we rewrite it.

    Refresh today's dispatch first so the daily field note ships with the feed on
    the same cron; past days are already frozen in the log and are not rebuilt.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    journal = Journal(settings.db_path)
    try:
        dispatch.ensure_today(journal, settings)
    except Exception:  # a dispatch hiccup must never block the feed itself
        pass
    path = out / "feed.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(build_feed(settings, journal), indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path
