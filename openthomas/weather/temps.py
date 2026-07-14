"""Current global surface temperature — the live heatmap the globe wears.

A coarse lon/lat grid of 2 m temperatures. `refresh()` fetches it from Open-Meteo
and caches it on disk; `global_grid()` only ever reads that cache, so building the
public feed touches no network (and the tests stay hermetic). The refresh runs
publish-side on its own cadence — a stale-but-real grid beats a fresh-but-invented
one, and a missing grid just falls the globe back to a plain blue planet.

Open-Meteo's current temperature is an observation-grade nowcast; the agent's own
Pangu forecast could feed this same grid later — the shape is the contract.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..memory.usage import now

DLON, DLAT = 10, 10  # grid spacing in degrees


def _cache_path(home: Path | str) -> Path:
    return Path(home) / "tempgrid.json"


def global_grid(home: Path | str) -> dict | None:
    """The temperature grid the globe wears, or None. Read-only — safe in the feed.

    Our own Pangu forecast field wins when the NWP pipeline has produced one
    (pangu-tempgrid.json, synced from the GPU box); otherwise the Open-Meteo
    nowcast cache. The grid carries its own `source`, so the site labels it
    honestly either way.
    """
    home = Path(home)
    try:
        return json.loads((home / "pangu-tempgrid.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass
    try:
        return json.loads((home / "tempgrid.json").read_text())["grid"]
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def _fetch() -> list[float | None]:
    lats = list(range(-90, 91, DLAT))
    lons = list(range(-180, 180, DLON))
    pts = [(a, b) for a in lats for b in lons]
    out: list[float | None] = []
    for i in range(0, len(pts), 100):  # small batches keep us under the rate limit
        batch = pts[i:i + 100]
        la = ",".join(str(a) for a, _ in batch)
        lo = ",".join(str(b) for _, b in batch)
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={la}"
               f"&longitude={lo}&current=temperature_2m")
        req = urllib.request.Request(url, headers={"User-Agent": "openthomas/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.load(resp)
        if isinstance(data, dict):
            data = [data]
        out += [(x.get("current") or {}).get("temperature_2m") for x in data]
        time.sleep(0.6)
    return out


def refresh(home: Path | str, max_age_s: int = 1800) -> dict | None:
    """Fetch the grid if the cache is older than `max_age_s`; else leave it.

    Returns the current grid (new or cached). Any network failure keeps the last
    good grid — the globe never blanks because Open-Meteo hiccuped.
    """
    cache = _cache_path(home)
    prev = None
    try:
        prev = json.loads(cache.read_text())
        if time.time() - prev.get("_t", 0) < max_age_s:
            return prev["grid"]
    except (OSError, json.JSONDecodeError, KeyError):
        prev = None

    lats = list(range(-90, 91, DLAT))
    lons = list(range(-180, 180, DLON))
    try:
        temps = _fetch()
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return prev["grid"] if prev else None

    grid = {
        "lat0": lats[0], "lon0": lons[0], "dlat": DLAT, "dlon": DLON,
        "ny": len(lats), "nx": len(lons), "as_of": now(), "source": "Open-Meteo",
        "temps": [round(t, 1) if t is not None else None for t in temps],
    }
    try:
        cache.write_text(json.dumps({"_t": time.time(), "grid": grid}))
    except OSError:
        pass
    return grid
