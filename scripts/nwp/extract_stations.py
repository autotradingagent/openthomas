#!/usr/bin/env python3
"""Extract station-day temperature extremes from an AI-NWP GRIB output and
append them to OpenThomas's local model source.

Usage (from the nwp venv, which has cfgrib/xarray via ai-models):
    python extract_stations.py pangu.grib --model pangu_local

Reads 2t (2-metre temperature) steps, samples the nearest grid point to each
settlement station, converts to °F, buckets hours into station-local calendar
days, and writes daily max/min for days with enough coverage. The trading
side picks these up through LocalModelSource — no OpenThomas import needed
here beyond the station registry, so this script runs in the heavyweight
NWP environment while the agent stays lean.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# Load the station registry directly from its file — one source of truth,
# without importing the openthomas package (whose deps don't live in the
# heavyweight NWP venv).
_spec = importlib.util.spec_from_file_location(
    "ot_stations",
    Path(__file__).resolve().parents[2] / "openthomas" / "weather" / "stations.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["ot_stations"] = _mod  # dataclass decorator introspects sys.modules
_spec.loader.exec_module(_mod)
STATIONS = _mod.STATIONS

MIN_HOURS_PER_DAY = 18  # partial local days give false extremes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("grib", help="path to the model output GRIB file")
    ap.add_argument("--model", default="pangu_local", help="model name in the consensus")
    ap.add_argument("--out", default=str(Path.home() / ".openthomas" / "local-models.jsonl"))
    args = ap.parse_args()

    import xarray as xr

    ds = xr.open_dataset(args.grib, engine="cfgrib",
                         filter_by_keys={"shortName": "2t"})
    # valid_time = base time + step; dims typically (step, latitude, longitude)
    lats, lons = ds.latitude.values, ds.longitude.values

    issued_at = datetime.now(timezone.utc).isoformat()
    rows = 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for station in STATIONS.values():
            lon = station.lon % 360 if lons.max() > 180 else station.lon
            point = ds.sel(latitude=station.lat, longitude=lon, method="nearest")
            temps_f = point["t2m"].values * 9 / 5 - 459.67  # K → °F
            times = point.valid_time.values

            by_day: dict[str, list[float]] = {}
            tz = ZoneInfo(station.timezone)
            # numpy datetime64 → aware station-local datetime
            for t, v in zip(times.astype("datetime64[s]").astype(int), temps_f):
                local = datetime.fromtimestamp(int(t), tz)
                by_day.setdefault(local.date().isoformat(), []).append(float(v))

            for day, values in sorted(by_day.items()):
                if len(values) < MIN_HOURS_PER_DAY / 6:  # Pangu steps are 6-hourly
                    continue
                for kind, value in (("high", max(values)), ("low", min(values))):
                    f.write(json.dumps({
                        "station": station.key, "target_date": day, "kind": kind,
                        "model": args.model, "value": round(value, 1),
                        "issued_at": issued_at,
                    }) + "\n")
                    rows += 1
    print(f"wrote {rows} rows for {len(STATIONS)} stations → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
