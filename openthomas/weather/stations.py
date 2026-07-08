"""Settlement stations for weather markets.

Kalshi temperature markets settle on the NWS Climatological Report (Daily)
for one specific station — trading NYC on a LaGuardia forecast when the
market settles on Central Park is an unforced error. Station assignments
below were verified against each series' rules_primary text on 2026-07-08.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Station:
    key: str  # registry key, e.g. "nyc"
    obs_id: str  # NWS observation station, e.g. "KNYC"
    cli_location: str  # NWS CLI product location code (settlement source)
    name: str
    lat: float
    lon: float
    timezone: str
    aliases: tuple[str, ...]  # lowercase substrings that identify it in a question
    wfo: str = ""  # forecast office issuing the AFD discussion for this station


STATIONS: dict[str, Station] = {
    s.key: s
    for s in [
        Station("nyc", "KNYC", "NYC", "Central Park, NYC", 40.783, -73.967,
                "America/New_York", ("nyc", "new york", "central park"), wfo="OKX"),
        Station("aus", "KAUS", "AUS", "Austin-Bergstrom", 30.183, -97.680,
                "America/Chicago", ("austin",), wfo="EWX"),
        Station("chi", "KMDW", "MDW", "Chicago Midway", 41.786, -87.752,
                "America/Chicago", ("chicago",), wfo="LOT"),
        Station("mia", "KMIA", "MIA", "Miami International", 25.788, -80.317,
                "America/New_York", ("miami",), wfo="MFL"),
        Station("lax", "KLAX", "LAX", "Los Angeles International", 33.938, -118.389,
                "America/Los_Angeles", ("los angeles", "lax"), wfo="LOX"),
        Station("phl", "KPHL", "PHL", "Philadelphia International", 39.873, -75.227,
                "America/New_York", ("philadelphia",), wfo="PHI"),
        Station("den", "KDEN", "DEN", "Denver International", 39.847, -104.656,
                "America/Denver", ("denver",), wfo="BOU"),
        # Rules say only "at Dallas"; NWS issues no DAL climate report, so the
        # Dallas-Fort Worth report is the settlement source until proven otherwise.
        Station("dal", "KDFW", "DFW", "Dallas-Fort Worth", 32.898, -97.019,
                "America/Chicago", ("dallas",), wfo="FWD"),
        Station("atl", "KATL", "ATL", "Atlanta Hartsfield-Jackson", 33.630, -84.442,
                "America/New_York", ("atlanta",), wfo="FFC"),
        Station("sat", "KSAT", "SAT", "San Antonio International", 29.533, -98.464,
                "America/Chicago", ("san antonio",), wfo="EWX"),
    ]
}

# Kalshi daily temperature series → (station key, which extreme settles it).
KALSHI_SERIES: dict[str, tuple[str, str]] = {
    "KXHIGHNY": ("nyc", "high"),
    "KXHIGHAUS": ("aus", "high"),
    "KXHIGHCHI": ("chi", "high"),
    "KXHIGHMIA": ("mia", "high"),
    "KXHIGHLAX": ("lax", "high"),
    "KXHIGHPHIL": ("phl", "high"),
    "KXHIGHDEN": ("den", "high"),
    "KXHIGHTDAL": ("dal", "high"),
    "KXHIGHTATL": ("atl", "high"),
    "KXLOWTMIA": ("mia", "low"),
    "KXLOWTAUS": ("aus", "low"),
    "KXLOWTSATX": ("sat", "low"),
}

_LOW_WORDS = ("low temp", "lowest temp", "minimum temp", "min temp")


def weather_series() -> list[str]:
    return list(KALSHI_SERIES)


def station_for_market(market) -> tuple[Station, str] | None:
    """(station, "high"|"low") a market settles on, or None if not a
    station-temperature market we understand."""
    if market.platform == "kalshi":
        hit = KALSHI_SERIES.get(market.id.split("-")[0])
        if hit:
            return STATIONS[hit[0]], hit[1]
    q = market.question.lower()
    if "temp" not in q:
        return None
    kind = "low" if any(w in q for w in _LOW_WORDS) else "high"
    for station in STATIONS.values():
        if any(alias in q for alias in station.aliases):
            return station, kind
    return None


def target_date(market, station: Station) -> date:
    """The calendar day (station-local) whose temperature settles the market.

    Kalshi encodes it in the ticker (KXHIGHNY-26JUL08-T83). Otherwise fall
    back to the close time in station-local terms: these markets stop trading
    at day's end but list closes shortly after local midnight, so an
    early-morning close belongs to the preceding day.
    """
    if market.platform == "kalshi":
        parts = market.id.split("-")
        if len(parts) >= 2:
            try:
                return datetime.strptime(parts[1], "%y%b%d").date()
            except ValueError:
                pass
    if market.close_time is not None:
        local = market.close_time.astimezone(ZoneInfo(station.timezone))
        if local.hour < 6:
            local -= timedelta(days=1)
        return local.date()
    return datetime.now(ZoneInfo(station.timezone)).date()
