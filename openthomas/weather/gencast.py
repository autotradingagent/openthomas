"""GenCast ensemble spread → flow-dependent forecast uncertainty for the desk.

GenCast is our probabilistic model. `extract_gencast.py` distils its ensemble into
`gencast-spread.json`: per station-day, the standard deviation of the daily high
and low across members, plus a per-lead reference spread. This turns that into a
sigma *multiplier*: on days the members fan out more than a typical day at that
lead, widen the strike distribution (less overconfident → the risk engine sizes
down); on unusually tight days, narrow it. Clamped, and absent/stale data yields
1.0 — the deterministic baseline is never made worse, only flow-aware.

We use the ensemble's *relative* spread (this day vs typical at this lead), not its
absolute magnitude: ensembles are systematically under-dispersed, but the ratio
still says which days are genuinely uncertain — and the magnitude stays anchored
to the verification-learned sigma it multiplies.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

FRESHNESS_HOURS = 30  # a GenCast run older than this is stale guidance
FLOOR, CEIL = 0.7, 1.6  # one ensemble may never collapse or explode the sigma


class GencastSpread:
    def __init__(self, path: Path | str, freshness_hours: float = FRESHNESS_HOURS):
        self.path = Path(path)
        self.freshness = timedelta(hours=freshness_hours)
        self._data: dict | None = None
        self._mtime: float | None = None

    def _load(self) -> dict | None:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            self._data, self._mtime = None, None
            return None
        if self._data is not None and mtime == self._mtime:
            return self._data
        self._mtime = mtime
        try:
            d = json.loads(self.path.read_text())
            issued = datetime.fromisoformat(d["issued_at"])
            if issued.tzinfo is None:
                issued = issued.replace(tzinfo=timezone.utc)
            self._data = None if datetime.now(timezone.utc) - issued > self.freshness else d
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            self._data = None
        return self._data

    def factor(self, station_key: str, date: str, kind: str, lead: int) -> float:
        """Flow-dependent sigma multiplier in [FLOOR, CEIL]; 1.0 when no fresh
        ensemble spread covers this (station, date, kind)."""
        d = self._load()
        if not d:
            return 1.0
        try:
            sig = float(d["spread"][station_key][date][kind])
            ref = d["ref_sigma"].get(str(lead))
        except (KeyError, TypeError, ValueError):
            return 1.0
        if not ref or float(ref) <= 0:
            return 1.0
        return max(FLOOR, min(CEIL, sig / float(ref)))
