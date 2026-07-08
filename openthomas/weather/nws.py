"""NWS (api.weather.gov) client: station observations plus the Climatological
Report (CLI) that temperature markets settle on. Free and keyless; the API
requires a User-Agent identifying the caller.
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import httpx

from .stations import Station

API = "https://api.weather.gov"
USER_AGENT = os.environ.get(
    "OPENTHOMAS_NWS_UA", "openthomas (github.com/autotradingagent/openthomas)"
)

_MONTHS = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY",
           "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]


def c_to_f(celsius: float) -> float:
    return celsius * 9 / 5 + 32


class NWSClient:
    def __init__(self, client: httpx.Client | None = None):
        self.http = client or httpx.Client(
            base_url=API, timeout=20, follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
        )

    def observed_extreme_today(self, station: Station, kind: str = "high") -> float | None:
        """Max/min °F among today's observations (station-local day) so far.

        A hard bound while the day is in progress: the official high can only
        end at or above the max already observed.
        """
        tz = ZoneInfo(station.timezone)
        midnight = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
        resp = self.http.get(
            f"/stations/{station.obs_id}/observations",
            params={"start": midnight.astimezone(timezone.utc).isoformat(), "limit": 500},
        )
        resp.raise_for_status()
        temps = [
            f["properties"]["temperature"]["value"]
            for f in resp.json().get("features", [])
            if f.get("properties", {}).get("temperature", {}).get("value") is not None
        ]
        if not temps:
            return None
        return c_to_f(max(temps) if kind == "high" else min(temps))

    def forecast_discussion(self, station: Station, max_chars: int = 1200) -> str:
        """The near/short-term section of the latest Area Forecast Discussion —
        the human forecaster's reasoning and stated uncertainty, which is
        exactly the 'information the baseline cannot see' the LLM adjusts for."""
        if not station.wfo:
            return ""
        resp = self.http.get(f"/products/types/AFD/locations/{station.wfo}")
        resp.raise_for_status()
        graph = resp.json().get("@graph", [])
        if not graph:
            return ""
        text = self.http.get(graph[0]["@id"]).json().get("productText", "")
        # AFD sections start ".NEAR TERM..." / ".SHORT TERM..." and end at "&&".
        match = re.search(r"\.(NEAR TERM|SHORT TERM|DISCUSSION)[^\n]*\n(.*?)\n\s*&&",
                          text, re.DOTALL)
        section = match.group(2).strip() if match else text.strip()
        return section[:max_chars]

    def climate_extreme(self, station: Station, day: date, kind: str = "high") -> float | None:
        """Official high/low °F for `day` from the CLI report — the settlement
        value. None until a report covering that date is issued; a same-day
        report is preliminary (the final one arrives after local midnight).
        """
        # This endpoint 400s on any query params; trim the list client-side.
        resp = self.http.get(f"/products/types/CLI/locations/{station.cli_location}")
        resp.raise_for_status()
        wanted = re.compile(
            rf"CLIMATE SUMMARY FOR\s+{_MONTHS[day.month - 1]}\s+{day.day}\s+{day.year}"
        )
        field = re.compile(r"MAXIMUM\s+(-?\d+)" if kind == "high" else r"MINIMUM\s+(-?\d+)")
        for item in resp.json().get("@graph", [])[:8]:
            product = self.http.get(item["@id"]).json()
            text = product.get("productText", "")
            if not wanted.search(text):
                continue
            match = field.search(text)
            if match:
                return float(match.group(1))
        return None
