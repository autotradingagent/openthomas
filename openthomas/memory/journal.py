"""The journal: every forecast, fill, settlement, and cycle in SQLite.

This is the agent's ground truth for PnL and its training data for
self-improvement (calibration fitting, lesson distillation, local fine-tuning).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..markets.base import Fill, Market, Position, Side

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY, ts TEXT, market_id TEXT, platform TEXT, question TEXT,
  category TEXT, side TEXT, action TEXT, qty INTEGER, price REAL, fee REAL, reason TEXT
);
CREATE TABLE IF NOT EXISTS forecasts (
  id INTEGER PRIMARY KEY, ts TEXT, market_id TEXT, platform TEXT, question TEXT,
  category TEXT, p_raw REAL, p_calibrated REAL, confidence REAL, base_rate REAL,
  market_gap_reason TEXT, invalidation TEXT, reasoning TEXT, model TEXT
);
CREATE TABLE IF NOT EXISTS settlements (
  market_id TEXT PRIMARY KEY, ts TEXT, platform TEXT, question TEXT, category TEXT,
  outcome TEXT, payout REAL, cost_basis REAL, pnl REAL
);
CREATE TABLE IF NOT EXISTS cycles (
  id INTEGER PRIMARY KEY, ts TEXT, account_value REAL, cash REAL,
  n_positions INTEGER, notes TEXT
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Journal:
    def __init__(self, path: Path | str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        # WAL: the trading loop and the MCP server share this file across processes.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript(SCHEMA)
        try:  # v0.2: market price at forecast time — the baseline Brier to beat
            self.db.execute("ALTER TABLE forecasts ADD COLUMN mid REAL")
        except sqlite3.OperationalError:
            pass  # column already present

    # --- writes ---------------------------------------------------------------
    def record_fill(self, fill: Fill, market: Market) -> None:
        o = fill.order
        self.db.execute(
            "INSERT INTO trades (ts, market_id, platform, question, category, side, action,"
            " qty, price, fee, reason) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (_now(), o.market_id, o.platform, market.question, market.category,
             o.side.value, o.action.value, fill.qty, fill.price, fill.fee, o.reason),
        )
        self.db.commit()

    def record_forecast(self, f, market: Market) -> None:
        self.db.execute(
            "INSERT INTO forecasts (ts, market_id, platform, question, category, p_raw,"
            " p_calibrated, confidence, base_rate, market_gap_reason, invalidation,"
            " reasoning, model, mid) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (_now(), f.market_id, market.platform, market.question, market.category,
             f.p_raw, f.p_calibrated, f.confidence, f.base_rate, f.market_gap_reason,
             f.invalidation, f.reasoning, f.model, market.mid),
        )
        self.db.commit()

    def record_settlement(self, position: Position, outcome: Side) -> float:
        payout = position.qty * (1.0 if position.side == outcome else 0.0)
        pnl = payout - position.cost_basis
        self.db.execute(
            "INSERT OR REPLACE INTO settlements (market_id, ts, platform, question,"
            " category, outcome, payout, cost_basis, pnl) VALUES (?,?,?,?,?,?,?,?,?)",
            (position.market_id, _now(), position.platform, position.question,
             position.category, outcome.value, payout, position.cost_basis, pnl),
        )
        self.db.commit()
        return pnl

    def record_cycle(self, account_value: float, cash: float, n_positions: int, notes: str = "") -> None:
        self.db.execute(
            "INSERT INTO cycles (ts, account_value, cash, n_positions, notes) VALUES (?,?,?,?,?)",
            (_now(), account_value, cash, n_positions, notes),
        )
        peak = max(self.peak_value(), account_value)
        self.db.execute("INSERT OR REPLACE INTO kv (key, value) VALUES ('peak_value', ?)", (str(peak),))
        self.db.commit()

    # --- reads ------------------------------------------------------------------
    def positions(self) -> list[Position]:
        """Net open positions from the trade log, excluding settled markets."""
        rows = self.db.execute(
            """SELECT market_id, platform, side, question, category,
                      SUM(CASE WHEN action='buy' THEN qty ELSE -qty END) AS net_qty,
                      SUM(CASE WHEN action='buy' THEN qty*price ELSE 0 END) AS buy_cost,
                      SUM(CASE WHEN action='buy' THEN qty ELSE 0 END) AS buy_qty
               FROM trades
               WHERE market_id NOT IN (SELECT market_id FROM settlements)
               GROUP BY market_id, side HAVING net_qty > 0"""
        ).fetchall()
        return [
            Position(
                market_id=r["market_id"], platform=r["platform"], side=Side(r["side"]),
                qty=r["net_qty"], avg_cost=r["buy_cost"] / r["buy_qty"] if r["buy_qty"] else 0,
                question=r["question"], category=r["category"] or "",
            )
            for r in rows
        ]

    def cash(self, bankroll: float) -> float:
        flows = self.db.execute(
            """SELECT COALESCE(SUM(CASE WHEN action='buy' THEN -(qty*price+fee)
                                        ELSE qty*price-fee END), 0) AS net
               FROM trades"""
        ).fetchone()["net"]
        payouts = self.db.execute("SELECT COALESCE(SUM(payout), 0) AS p FROM settlements").fetchone()["p"]
        return bankroll + flows + payouts

    def peak_value(self) -> float:
        row = self.db.execute("SELECT value FROM kv WHERE key='peak_value'").fetchone()
        return float(row["value"]) if row else 0.0

    def forecast_outcome_pairs(self, category: str | None = None) -> list[tuple[float, int]]:
        """(raw forecast, outcome) pairs for calibration fitting — first forecast
        per settled market only, mirroring 'initial prediction accuracy'."""
        query = """
            SELECT f.p_raw AS p, CASE WHEN s.outcome='yes' THEN 1 ELSE 0 END AS y
            FROM settlements s
            JOIN forecasts f ON f.id = (
              SELECT id FROM forecasts WHERE market_id = s.market_id ORDER BY ts LIMIT 1)
        """
        args: tuple = ()
        if category:
            query += " WHERE s.category = ?"
            args = (category,)
        return [(r["p"], r["y"]) for r in self.db.execute(query, args).fetchall()]

    def settlement_stats(self) -> dict:
        row = self.db.execute(
            """SELECT COUNT(*) AS n, COALESCE(SUM(pnl), 0) AS pnl,
                      COALESCE(AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END), 0) AS win_rate,
                      COALESCE(AVG(CASE WHEN pnl > 0 THEN pnl END), 0) AS avg_win,
                      COALESCE(AVG(CASE WHEN pnl <= 0 THEN pnl END), 0) AS avg_loss
               FROM settlements"""
        ).fetchone()
        return dict(row)

    def category_stats(self) -> list[dict]:
        rows = self.db.execute(
            """SELECT category, COUNT(*) AS n, SUM(pnl) AS pnl,
                      AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) AS win_rate
               FROM settlements GROUP BY category ORDER BY pnl"""
        ).fetchall()
        return [dict(r) for r in rows]

    def scope_performance(self, scope: str, since_ts: str) -> dict:
        """Post-adoption track record for a playbook rule: settlements after
        `since_ts` whose question or category contains `scope`."""
        like = f"%{scope.lower()}%"
        row = self.db.execute(
            """SELECT COUNT(*) AS n, COALESCE(SUM(pnl), 0) AS pnl,
                      COALESCE(AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END), 0) AS win_rate
               FROM settlements
               WHERE ts >= ? AND (LOWER(question) LIKE ? OR LOWER(category) LIKE ?)""",
            (since_ts, like, like),
        ).fetchone()
        return dict(row)

    def equity_curve(self) -> list[tuple[str, float]]:
        rows = self.db.execute("SELECT ts, account_value FROM cycles ORDER BY ts").fetchall()
        return [(r["ts"], r["account_value"]) for r in rows]

    def recent_settlements(self, limit: int = 20) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM settlements ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def has_recent_forecast(self, market_id: str, hours: float = 12) -> bool:
        row = self.db.execute(
            "SELECT ts FROM forecasts WHERE market_id=? ORDER BY ts DESC LIMIT 1", (market_id,)
        ).fetchone()
        if not row:
            return False
        age = datetime.now(timezone.utc) - datetime.fromisoformat(row["ts"])
        return age.total_seconds() < hours * 3600
