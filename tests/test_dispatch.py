"""The daily dispatch: honest, bounded, one row per day, and on the feed."""

from __future__ import annotations

import pytest

from openthomas.config import Settings
from openthomas.markets.base import Action, Fill, Market, Order, Side
from openthomas.memory.journal import Journal
from openthomas.report.dispatch import (
    LIMIT,
    daily_report,
    daily_text,
    ensure_today,
    read_reports,
    record_report,
)
from openthomas.site.feed import build_feed


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(bankroll=1000.0, home=tmp_path)


def _win(j: Journal) -> float:
    m = Market(id="M1", platform="kalshi", question="Will the high be >80F?",
               category="climate/weather", yes_bid=0.39, yes_ask=0.41)
    order = Order(market_id="M1", platform="kalshi", side=Side.YES, action=Action.BUY,
                  qty=10, limit_price=0.41, reason="edge")
    j.record_fill(Fill(order=order, qty=10, price=0.41, fee=0.0), m)
    return j.record_settlement(j.positions()[0], Side.YES)


def test_daily_text_shows_the_loss_and_fits_the_limit(settings):
    """A down day is content too. The return is published with its sign, the post
    fits X's ceiling, and it points back to the site of record."""
    j = Journal(settings.db_path)
    j.record_cycle(account_value=990.0, cash=990.0, n_positions=0)  # -1% on $1,000

    text = daily_text(j, settings)
    assert len(text) <= LIMIT
    assert "-1.00%" in text  # the loss is stated, not hidden
    assert "openthomas.com" in text
    assert settings.mode in text  # "paper" — never implies real money


def test_daily_report_counts_only_the_given_days_settlements(settings):
    j = Journal(settings.db_path)
    pnl = _win(j)  # a win dated today

    today = j.recent_settlements(1)[0]["ts"][:10]
    report = daily_report(j, settings, day=today)
    assert report["date"] == today
    assert report["stats"]["settled_today"] == 1
    assert any("resolved today" in p for p in report["body"])
    assert len(report["tweet"]) <= LIMIT
    assert pnl > 0

    empty = daily_report(j, settings, day="2000-01-01")
    assert empty["stats"]["settled_today"] == 0
    assert any("Nothing resolved today" in p for p in empty["body"])


def test_the_log_keeps_one_row_per_day_newest_first(settings):
    """Re-running a day upserts, it doesn't append — the timeline stays one entry
    per date, and the newest day sorts first."""
    record_report(settings.home, {"date": "2026-07-14", "title": "old", "body": [], "tweet": "a"})
    record_report(settings.home, {"date": "2026-07-15", "title": "v1", "body": [], "tweet": "b"})
    record_report(settings.home, {"date": "2026-07-15", "title": "v2", "body": [], "tweet": "c"})

    rows = read_reports(settings.home)
    assert [r["date"] for r in rows] == ["2026-07-15", "2026-07-14"]  # newest first, deduped
    assert rows[0]["title"] == "v2"  # last write for the date wins


def test_ensure_today_puts_the_dispatch_on_the_feed(settings):
    j = Journal(settings.db_path)
    j.record_cycle(account_value=1005.0, cash=1005.0, n_positions=0)
    ensure_today(j, settings)

    feed = build_feed(settings, j)
    assert feed["schema_version"] == 4
    (note,) = feed["reports"]
    assert note["stats"]["return_pct"] == pytest.approx(0.005)
    assert note["body"] and note["tweet"]
