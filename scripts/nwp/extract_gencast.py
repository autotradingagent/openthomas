#!/usr/bin/env python3
"""Turn a GenCast ensemble GRIB into a per-station forecast-SPREAD file.

GenCast is the probabilistic model: its ensemble spread is a flow-dependent
uncertainty — some days the members agree tightly, some days they fan out — that
the deterministic baseline's climatological sigma can't see. This writes, per
station-day, the ensemble standard deviation of the daily high and low (across
members), plus a per-lead reference spread so the desk can read "how uncertain is
this day *relative to* a typical day at this lead".

Deliberately NOT written: GenCast's daily-extreme MEAN. GenCast's 12-hourly steps
undersample the afternoon peak, so its point high is biased low — but the spread
*across members* shares that timing and is unaffected. The point forecast stays
with GraphCast (6-hourly); GenCast contributes only uncertainty.

    python extract_gencast.py gencast.grib            # -> ~/.openthomas/gencast-spread.json

Run in the GenCast venv (cfgrib/xarray via ai-models).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

_spec = importlib.util.spec_from_file_location(
    "ot_stations",
    Path(__file__).resolve().parents[2] / "openthomas" / "weather" / "stations.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["ot_stations"] = _mod
_spec.loader.exec_module(_mod)
STATIONS = _mod.STATIONS

MIN_STEPS_PER_DAY = 2  # need both 00z+12z samples for a daily extreme (GenCast is 12-hourly)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("grib", help="path to the GenCast ensemble GRIB")
    ap.add_argument("--out", default=str(Path.home() / ".openthomas" / "gencast-spread.json"))
    args = ap.parse_args()

    import numpy as np
    import xarray as xr

    ds = xr.open_dataset(args.grib, engine="cfgrib",
                         filter_by_keys={"shortName": "2t"},
                         backend_kwargs={"indexpath": ""})
    lons = ds.longitude.values
    if "number" not in ds.dims:
        print("no ensemble dimension ('number') — is this a GenCast ensemble GRIB?",
              file=sys.stderr)
        return 2
    n_members = ds.sizes["number"]
    issued_at = datetime.now(timezone.utc).isoformat()
    init_day = np.datetime64(ds.time.values).astype("datetime64[D]")

    spread: dict[str, dict[str, dict[str, float]]] = {}
    per_lead: dict[int, list[float]] = {}  # lead(days) -> list of sigma_ens, for the reference

    for station in STATIONS.values():
        lon = station.lon % 360 if lons.max() > 180 else station.lon
        point = ds.sel(latitude=station.lat, longitude=lon, method="nearest")
        temps_f = point["t2m"].values * 9 / 5 - 459.67  # (number, step) K -> °F
        times = point.valid_time.values.astype("datetime64[s]").astype("int64")
        tz = ZoneInfo(station.timezone)

        # bucket step indices by station-local day
        days: dict[str, list[int]] = {}
        for si, t in enumerate(times):
            local = datetime.fromtimestamp(int(t), tz).date().isoformat()
            days.setdefault(local, []).append(si)

        for day, sidx in sorted(days.items()):
            if len(sidx) < MIN_STEPS_PER_DAY:
                continue
            # per member: that member's daily high/low across the day's steps.
            # Some (member, step) cells are NaN (e.g. the analysis step exists for
            # one member only) — drop NaN before taking each member's extreme.
            member_high, member_low = [], []
            for m in range(n_members):
                vals = temps_f[m, sidx]
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    member_high.append(float(vals.max()))
                    member_low.append(float(vals.min()))
            if len(member_high) < 2 or len(member_low) < 2:
                continue
            s_hi = statistics.stdev(member_high)
            s_lo = statistics.stdev(member_low)
            lead = int((np.datetime64(day) - init_day) / np.timedelta64(1, "D"))
            spread.setdefault(station.key, {}).setdefault(day, {})["high"] = round(s_hi, 2)
            spread[station.key][day]["low"] = round(s_lo, 2)
            per_lead.setdefault(lead, []).extend([s_hi, s_lo])

    ref = {str(k): round(statistics.median(v), 2) for k, v in per_lead.items() if v}
    out = {"issued_at": issued_at, "source": "OpenThomas · GenCast",
           "members": int(n_members), "ref_sigma": ref, "spread": spread}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote gencast spread for {len(spread)} stations "
          f"({n_members} members, leads {sorted(int(k) for k in ref)}) → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
