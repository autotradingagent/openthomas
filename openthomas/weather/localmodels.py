"""Local AI-NWP model source: forecasts produced by our own inference runs
(scripts/nwp/) merged into the multi-model consensus.

The seam is deliberately dumb — an append-only JSONL of station-day extremes
with an issue timestamp. Whatever produces rows (Pangu on CPU today,
GraphCast/Aurora on freed GPUs later) plugs in without touching the desk,
and the verification store scores every local model per station exactly
like it scores GFS or ECMWF. A model earns its place in the consensus by
its bias/sigma record, not by being ours.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

FRESHNESS_HOURS = 30  # ignore runs older than this — stale guidance is noise


class LocalModelSource:
    def __init__(self, path: Path, freshness_hours: float = FRESHNESS_HOURS):
        self.path = Path(path)
        self.freshness = timedelta(hours=freshness_hours)

    def record(self, station_key: str, target_date: str, kind: str, model: str,
               value_f: float, issued_at: str | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({
                "station": station_key, "target_date": target_date, "kind": kind,
                "model": model, "value": round(value_f, 1),
                "issued_at": issued_at or datetime.now(timezone.utc).isoformat(),
            }) + "\n")

    def extremes(self, station_key: str) -> dict[str, dict[str, dict[str, float]]]:
        """Same shape as OpenMeteoClient.daily_extremes: {date: {kind: {model: °F}}}.
        Latest fresh row wins per (date, kind, model)."""
        if not self.path.exists():
            return {}
        cutoff = datetime.now(timezone.utc) - self.freshness
        out: dict[str, dict[str, dict[str, float]]] = {}
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["station"] != station_key:
                continue
            try:
                issued = datetime.fromisoformat(r["issued_at"])
            except ValueError:
                continue
            if issued < cutoff:
                continue
            out.setdefault(r["target_date"], {"high": {}, "low": {}}) \
               .setdefault(r["kind"], {})[r["model"]] = float(r["value"])
        return out
