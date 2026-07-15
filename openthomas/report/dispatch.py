"""The daily dispatch: a short, honest field note on how the agent is doing,
published to the site as a running log and copy-pasteable to X.

Build in public means the losses ship too. Each day's entry is templated from
the journal — the same source the public feed reads — so it costs no model
tokens and can't claim a profit the record doesn't show. Today's entry refreshes
as the day trades; once the date rolls over it freezes, and the log becomes an
uneditable timeline of what was claimed, when.

`daily_report` builds one day's entry and `ensure_today` upserts it into the log
the feed publishes. There is no auto-post: the operator copies the `tweet` line
to X by hand, so nothing leaves the box without a human.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from ..config import Settings
from ..memory.journal import Journal

SITE = "openthomas.com"
LIMIT = 280  # X's per-post character ceiling
MAX_REPORTS = 30  # entries carried in the public feed; the log on disk keeps all


def _money(v: float) -> str:
    """Signed, whole-dollar, ASCII — the sign is the news on a P&L line."""
    return f"{'+' if v >= 0 else '-'}${abs(v):,.0f}"


def _log_path(home: Path | str) -> Path:
    return Path(home) / "reports.jsonl"


def read_reports(home: Path | str, limit: int | None = None) -> list[dict]:
    """Every recorded dispatch, newest first, one per day (last write wins)."""
    path = _log_path(home)
    if not path.exists():
        return []
    by_date: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_date[entry["date"]] = entry  # a re-run for a date replaces its row
    out = sorted(by_date.values(), key=lambda e: e["date"], reverse=True)
    return out[:limit] if limit else out


def record_report(home: Path | str, entry: dict) -> None:
    """Upsert one day's entry, rewriting the log so a date keeps a single row.

    Written atomically (temp file, rename): the publisher reads this file while
    the next cycle rewrites it, and a half-written log is a wrong log.
    """
    by_date = {r["date"]: r for r in read_reports(home)}
    by_date[entry["date"]] = entry
    rows = sorted(by_date.values(), key=lambda r: r["date"])
    path = _log_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def _today_settlements(journal: Journal, day: str) -> list[dict]:
    return [s for s in journal.recent_settlements(limit=200) if s["ts"][:10] == day]


def daily_text(journal: Journal, settings: Settings, day: str | None = None) -> str:
    """The X-ready line for `day` (UTC date, defaults to today).

    Headline and link are always kept; the day's activity and the standing call
    are added only while they fit under the character limit, longest-lived facts
    first. Everything is drawn from settled, recorded state.
    """
    from ..site.feed import _theses  # local import: avoids a feed→dispatch cycle

    day = day or datetime.now(timezone.utc).date().isoformat()
    curve = journal.equity_curve()
    value = curve[-1][1] if curve else settings.bankroll
    ret = value / settings.bankroll - 1 if settings.bankroll else 0.0
    stats = journal.settlement_stats()

    head = f"OpenThomas · {settings.mode} weather-trading agent"
    value_line = f"${value:,.0f} book, {ret:+.2%} since ${settings.bankroll:,.0f} start."
    link = f"{SITE} — every claim timestamped before it settles"

    optional: list[str] = []
    today = _today_settlements(journal, day)
    if today:
        line = f"Today: {len(today)} settled, {_money(sum(s['pnl'] for s in today))}."
        if stats["n"]:
            line += f" {stats['win_rate']:.0%} win over {stats['n']}."
        optional.append(line)
    else:
        held = len(journal.positions())
        optional.append(f"No settlements today; holding {held} position"
                        f"{'' if held == 1 else 's'}, watching the board.")

    theses = _theses(journal, settings)
    if theses:
        t = theses[0]
        subject = (t.get("loc") or {}).get("place") or t["question"].rstrip("? ")[:34]
        optional.append(f"Widest call: {subject} {t['side']} — we say "
                        f"{t['p_model'] * 100:.0f}¢ vs market {t['p_market'] * 100:.0f}¢.")

    body = [head, value_line]
    for line in optional:  # add each only if the whole post still fits
        if len("\n".join([*body, line, link])) <= LIMIT:
            body.append(line)
    return "\n".join([*body, link])


def daily_report(journal: Journal, settings: Settings, day: str | None = None) -> dict:
    """One day's field note: a short prose body for the site, plus the `tweet`
    line to copy to X, plus the numbers behind it. Same journal, same day."""
    from ..site.feed import _theses

    day = day or datetime.now(timezone.utc).date().isoformat()
    curve = journal.equity_curve()
    value = curve[-1][1] if curve else settings.bankroll
    ret = value / settings.bankroll - 1 if settings.bankroll else 0.0
    stats = journal.settlement_stats()
    today = _today_settlements(journal, day)
    day_pnl = sum(s["pnl"] for s in today)
    wins = sum(1 for s in today if s["pnl"] >= 0)
    held = len(journal.positions())

    start = curve[0][0][:10] if curve else day
    n = (date.fromisoformat(day) - date.fromisoformat(start)).days + 1

    body = [f"Paper book at ${value:,.2f}, {ret:+.2%} since the "
            f"${settings.bankroll:,.0f} start."]
    if today:
        s = "" if len(today) == 1 else "s"
        body.append(f"{len(today)} market{s} resolved today for {_money(day_pnl)} "
                    f"({wins} won, {len(today) - wins} lost). Realized to date "
                    f"{_money(stats['pnl'])} over {stats['n']} settled, "
                    f"{stats['win_rate']:.0%} of them winners.")
    else:
        s = "" if held == 1 else "s"
        body.append(f"Nothing resolved today; {held} position{s} open, "
                    f"{_money(stats['pnl'])} realized so far over {stats['n']} settled.")

    theses = _theses(journal, settings)
    if theses:
        t = theses[0]
        subject = (t.get("loc") or {}).get("place") or t["question"].rstrip("? ")[:60]
        line = (f"Widest standing call: {subject} — the model reads "
                f"{t['p_model'] * 100:.0f}¢ against the market's "
                f"{t['p_market'] * 100:.0f}¢ on {t['side'].upper()}.")
        if t.get("why"):
            line += f" {t['why']}"
        body.append(line)

    return {
        "date": day,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "title": f"Day {max(n, 1)} · {ret:+.2%}, {held} held",
        "tweet": daily_text(journal, settings, day=day),
        "body": body,
        "stats": {"account_value": round(value, 2), "return_pct": round(ret, 6),
                  "day_pnl": round(day_pnl, 2), "settled_today": len(today),
                  "positions": held},
    }


def ensure_today(journal: Journal, settings: Settings, day: str | None = None) -> dict:
    """Build today's dispatch and upsert it into the log the feed publishes.

    Called on every publish: today's entry stays current as the day trades, and
    freezes into the timeline once the date rolls over (past days are never
    rebuilt — only the row for `day` is replaced)."""
    entry = daily_report(journal, settings, day=day)
    record_report(settings.home, entry)
    return entry
