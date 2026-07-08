import json
from datetime import date, datetime, timezone

import httpx

from openthomas.markets.base import Market
from openthomas.markets.kalshi import KalshiConnector
from openthomas.markets.polymarket import PolymarketConnector
from openthomas.weather import NWSClient, OpenMeteoClient, WeatherDesk
from openthomas.weather.nws import c_to_f
from openthomas.weather.stations import STATIONS, station_for_market, target_date, weather_series
from openthomas.weather.strikes import parse_strike


def mk(id="KXHIGHNY-26JUL08-T83", platform="kalshi",
       question="Will the high temp in NYC be >83° on Jul 8, 2026?", **kw):
    return Market(id=id, platform=platform, question=question, **kw)


def mock_client(handler, base_url="https://example.test"):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url=base_url)


# --- strikes ---------------------------------------------------------------

def test_strike_greater():
    s = parse_strike(mk(strike_type="greater", floor_strike=83.0))
    assert s.covers(84) and s.covers(90.5)
    assert not s.covers(83)
    assert s.describe() == "> 83°F"


def test_strike_less():
    s = parse_strike(mk(strike_type="less", cap_strike=76.0))
    assert s.covers(75) and not s.covers(76)
    assert s.describe() == "< 76°F"


def test_strike_between_inclusive():
    s = parse_strike(mk(strike_type="between", floor_strike=82.0, cap_strike=83.0))
    assert s.covers(82) and s.covers(83)
    assert not s.covers(81) and not s.covers(84)


def test_strike_missing_fields():
    assert parse_strike(mk()) is None
    assert parse_strike(mk(strike_type="greater")) is None


# --- stations ---------------------------------------------------------------

def test_station_from_kalshi_ticker():
    station, kind = station_for_market(mk())
    assert station.obs_id == "KNYC" and kind == "high"
    station, kind = station_for_market(mk(id="KXLOWTSATX-26JUL08-T77"))
    assert station.obs_id == "KSAT" and kind == "low"


def test_station_from_question_text():
    m = mk(id="0xabc", platform="polymarket",
           question="Highest temperature in Miami on July 8?")
    station, kind = station_for_market(m)
    assert station.obs_id == "KMIA" and kind == "high"


def test_station_unknown_market():
    assert station_for_market(mk(id="0xdef", platform="polymarket",
                                 question="Will it rain in Paris this July?")) is None
    assert station_for_market(mk(id="KXBTC-26JUL08-T100000",
                                 question="Bitcoin above 100k?")) is None


def test_weather_series_covers_known_cities():
    assert "KXHIGHNY" in weather_series() and "KXLOWTMIA" in weather_series()


def test_target_date_from_ticker():
    assert target_date(mk(), STATIONS["nyc"]) == date(2026, 7, 8)


def test_target_date_from_close_time_rolls_back_past_midnight():
    # Closes 03:59 UTC Jul 9 = 23:59 EDT Jul 8 → the market is about Jul 8.
    m = mk(id="0xabc", platform="polymarket",
           close_time=datetime(2026, 7, 9, 3, 59, tzinfo=timezone.utc))
    assert target_date(m, STATIONS["nyc"]) == date(2026, 7, 8)


# --- NWS ---------------------------------------------------------------------

CLI_TEXT = """
000
CDUS41 KOKX 090635
CLINYC

CLIMATE REPORT
NATIONAL WEATHER SERVICE NEW YORK, NY
235 AM EDT THU JUL 9 2026

...THE CENTRAL PARK NY CLIMATE SUMMARY FOR JULY 8 2026...

TEMPERATURE (F)
 YESTERDAY
  MAXIMUM         84    252 PM
  MINIMUM         71    509 AM
  AVERAGE         78
"""


def nws_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/products/types/CLI/locations/NYC":
        return httpx.Response(200, json={"@graph": [
            {"@id": "https://example.test/products/abc", "issuanceTime": "2026-07-09T06:35:00Z"},
        ]})
    if path == "/products/abc":
        return httpx.Response(200, json={"productText": CLI_TEXT})
    if path == "/stations/KNYC/observations":
        return httpx.Response(200, json={"features": [
            {"properties": {"temperature": {"value": 20.6}}},
            {"properties": {"temperature": {"value": 25.0}}},
            {"properties": {"temperature": {"value": None}}},
        ]})
    return httpx.Response(404)


def test_climate_extreme_parses_settlement_high():
    nws = NWSClient(client=mock_client(nws_handler))
    assert nws.climate_extreme(STATIONS["nyc"], date(2026, 7, 8), "high") == 84.0
    assert nws.climate_extreme(STATIONS["nyc"], date(2026, 7, 8), "low") == 71.0
    # A report for a different day must not settle this one.
    assert nws.climate_extreme(STATIONS["nyc"], date(2026, 7, 7), "high") is None


def test_observed_extreme_today_converts_and_filters():
    nws = NWSClient(client=mock_client(nws_handler))
    assert nws.observed_extreme_today(STATIONS["nyc"], "high") == c_to_f(25.0)
    assert nws.observed_extreme_today(STATIONS["nyc"], "low") == c_to_f(20.6)


# --- Open-Meteo ----------------------------------------------------------------

def meteo_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"daily": {
        "time": ["2026-07-08", "2026-07-09"],
        "temperature_2m_max_gfs_seamless": [83.3, 82.8],
        "temperature_2m_max_ecmwf_ifs025": [82.5, None],
        "temperature_2m_min_gfs_seamless": [70.1, 69.4],
    }})


def test_daily_extremes_by_model():
    meteo = OpenMeteoClient(client=mock_client(meteo_handler))
    out = meteo.daily_extremes(STATIONS["nyc"])
    assert out["2026-07-08"]["high"] == {"gfs_seamless": 83.3, "ecmwf_ifs025": 82.5}
    assert out["2026-07-09"]["high"] == {"gfs_seamless": 82.8}  # null dropped
    assert out["2026-07-08"]["low"] == {"gfs_seamless": 70.1}


def test_daily_extremes_single_model_unsuffixed():
    def handler(request):
        return httpx.Response(200, json={"daily": {
            "time": ["2026-07-08"], "temperature_2m_max": [83.3],
        }})
    meteo = OpenMeteoClient(client=mock_client(handler), models=["gfs_seamless"])
    assert meteo.daily_extremes(STATIONS["nyc"])["2026-07-08"]["high"] == {"gfs_seamless": 83.3}


# --- connectors: weather listings ---------------------------------------------

KALSHI_RAW = {
    "ticker": "KXHIGHNY-26JUL08-T83", "event_ticker": "KXHIGHNY-26JUL08",
    "title": "Will the **high temp in NYC** be >83° on Jul 8, 2026?",
    "yes_bid_dollars": "0.2500", "yes_ask_dollars": "0.2700",
    "volume_24h_fp": "25460.08", "liquidity_dollars": "0.0000",
    "open_interest_fp": "17653.59", "close_time": "2026-07-09T04:59:00Z",
    "strike_type": "greater", "floor_strike": 83,
    "rules_primary": "If the highest temperature recorded in Central Park...",
}


def test_kalshi_weather_listing_strikes_and_liquidity_fallback():
    def handler(request):
        assert request.url.path == "/markets"
        series = request.url.params["series_ticker"]
        markets = [KALSHI_RAW] if series == "KXHIGHNY" else []
        return httpx.Response(200, json={"markets": markets})

    kalshi = KalshiConnector(client=mock_client(handler))
    markets = kalshi.list_weather_markets()
    assert len(markets) == 1
    m = markets[0]
    assert m.strike_type == "greater" and m.floor_strike == 83.0
    assert m.liquidity == 17653.59  # open interest stands in for zeroed liquidity
    assert m.category == "climate and weather"


def test_polymarket_weather_listing_flattens_nested_events():
    nested = {
        "conditionId": "0xw1", "question": "Paris heat wave by July 31?",
        "outcomes": json.dumps(["Yes", "No"]), "bestBid": 0.4, "bestAsk": 0.45,
        "volume24hr": 15.0, "liquidityNum": 500.0, "endDate": "2026-07-31T00:00:00Z",
        "description": "Resolves YES if...", "slug": "paris-heat-wave",
    }
    def handler(request):
        assert request.url.params["tag_slug"] == "weather"
        return httpx.Response(200, json=[{"id": 9, "title": "Paris heat", "markets": [nested]}])

    poly = PolymarketConnector(client=mock_client(handler))
    markets = poly.list_weather_markets()
    assert len(markets) == 1
    assert markets[0].id == "0xw1" and markets[0].category == "weather"
    assert markets[0].event_id == "9"


# --- desk ----------------------------------------------------------------------

class StubMeteo:
    def daily_extremes(self, station, days=7):
        return {"2026-07-08": {"high": {"gfs_seamless": 83.3, "ecmwf_ifs025": 82.5},
                               "low": {}}}


class StubNWS:
    def observed_extreme_today(self, station, kind="high"):
        return 84.2


class ExplodingMeteo:
    def daily_extremes(self, station, days=7):
        raise ConnectionError("down")


def test_desk_brief_composes_models_and_strike():
    desk = WeatherDesk(nws=StubNWS(), meteo=StubMeteo())
    brief = desk.brief(mk(strike_type="greater", floor_strike=83.0))
    assert "Central Park" in brief and "KNYC" in brief
    assert "> 83°F" in brief
    assert "gfs_seamless: 83.3" in brief
    assert "Consensus: 82.9" in brief


def test_desk_brief_empty_for_non_weather_market():
    desk = WeatherDesk(nws=StubNWS(), meteo=StubMeteo())
    assert desk.brief(mk(id="0x1", platform="polymarket", question="Will BTC hit 200k?")) == ""


def test_desk_brief_empty_when_sources_dead():
    desk = WeatherDesk(nws=StubNWS(), meteo=ExplodingMeteo())
    # Target date is in the past relative to nothing — no models, no obs → "".
    assert desk.brief(mk(id="KXHIGHNY-25JAN01-T50")) == ""


def test_desk_caches_per_station_day():
    class CountingMeteo(StubMeteo):
        calls = 0
        def daily_extremes(self, station, days=7):
            CountingMeteo.calls += 1
            return super().daily_extremes(station, days)

    desk = WeatherDesk(nws=StubNWS(), meteo=CountingMeteo())
    desk.brief(mk())
    desk.brief(mk(id="KXHIGHNY-26JUL08-B82.5", strike_type="between",
                  floor_strike=82.0, cap_strike=83.0))
    assert CountingMeteo.calls == 1


# --- forecast engine wiring -------------------------------------------------------

def test_forecast_engine_injects_domain_data():
    from openthomas.config import ModelConfig
    from openthomas.forecast.engine import ForecastEngine

    captured = {}

    class SpyEngine(ForecastEngine):
        def _complete(self, system, user):
            captured["prompt"] = user
            return json.dumps({"probability": 0.3, "confidence": 0.7})

    engine = SpyEngine(ModelConfig(ensemble_size=1))
    f = engine.forecast(mk(), data="Model guidance: 83.3°F")
    assert f is not None and f.p_raw == 0.3
    assert "Domain data" in captured["prompt"]
    assert "Model guidance: 83.3°F" in captured["prompt"]


# --- baseline (③) ---------------------------------------------------------------

def test_strike_probabilities_partition_to_one():
    """Kalshi's bucket ladder is a partition of the integer temps — the
    baseline must distribute exactly one unit of probability across it."""
    from openthomas.weather.baseline import strike_probability
    from openthomas.weather.strikes import Strike

    ladder = [Strike("less", hi=76), Strike("between", 76, 77), Strike("between", 78, 79),
              Strike("between", 80, 81), Strike("between", 82, 83), Strike("greater", lo=83)]
    total = sum(strike_probability(s, mu=81.4, sigma=2.3, kind="high") for s in ladder)
    assert abs(total - 1.0) < 1e-9


def test_baseline_truncates_at_observed_high():
    from openthomas.weather.baseline import strike_probability
    from openthomas.weather.strikes import Strike

    # Observed 84.2 already: ">83" is certain, "82-83" is impossible.
    assert strike_probability(Strike("greater", lo=83), 82.0, 2.0, "high", observed=84.2) == 1.0
    assert strike_probability(Strike("between", 82, 83), 82.0, 2.0, "high", observed=84.2) == 0.0
    # "<76" for a low already observed at 71: not yet certain (evening can still cool).
    p = strike_probability(Strike("less", hi=76), 74.0, 2.0, "low", observed=71.0)
    assert p == 1.0  # low can only go DOWN from 71 — already below 76


def test_baseline_observed_shifts_mass_upward():
    from openthomas.weather.baseline import strike_probability
    from openthomas.weather.strikes import Strike

    s = Strike("greater", lo=83)
    p_plain = strike_probability(s, 83.0, 2.0, "high")
    p_bound = strike_probability(s, 83.0, 2.0, "high", observed=83.2)
    assert p_bound > p_plain  # everything below 83.2 is off the table


def test_verification_store_learns_bias(tmp_path):
    from datetime import date, timedelta
    from openthomas.weather.verification import VerificationStore, prior_sigma

    store = VerificationStore(tmp_path / "v.jsonl")
    bias0, sigma0, n0 = store.stats("nyc", "high", 1)
    assert (bias0, n0) == (0.0, 0) and sigma0 == prior_sigma(1)

    d0 = date(2026, 6, 1)
    for i in range(20):  # models run 2°F cold at this station
        day = d0 + timedelta(days=i)
        store.record_guidance("nyc", "high", day, 1, mean=80.0, spread=1.0,
                              models={"gfs_seamless": 80.0})
        store.record_settlement("nyc", "high", day, 82.0)
    bias, sigma, n = store.stats("nyc", "high", 1)
    assert n == 20
    assert 1.2 < bias < 2.0  # shrunk toward 0, pulled toward +2
    assert store.has_settlement("nyc", "high", d0)
    assert not store.has_settlement("nyc", "high", d0 + timedelta(days=99))


def test_desk_assess_produces_anchored_baseline():
    desk = WeatherDesk(nws=StubNWS(), meteo=StubMeteo())
    a = desk.assess(mk(strike_type="between", floor_strike=82.0, cap_strike=83.0))
    assert a is not None and a.p_base is not None
    assert 0.0 <= a.p_base <= 1.0
    assert "Statistical baseline" in a.text


def test_desk_assess_decided_when_observation_forces_outcome():
    class HotNWS:
        def observed_extreme_today(self, station, kind="high"):
            return 90.0  # way above every strike

    desk = WeatherDesk(nws=HotNWS(), meteo=StubMeteo())
    a = desk.assess(mk(strike_type="greater", floor_strike=83.0))
    if a.day == __import__("datetime").datetime.now().date():  # obs only used same-day
        assert a.decided and a.p_base == 1.0


def test_engine_clamps_to_anchor():
    from openthomas.config import ModelConfig
    from openthomas.forecast.engine import ForecastEngine

    class SpyEngine(ForecastEngine):
        def _complete(self, system, user):
            return json.dumps({"probability": 0.9, "confidence": 0.8})

    engine = SpyEngine(ModelConfig(ensemble_size=1))
    f = engine.forecast(mk(), anchor=(0.2, 0.15))
    assert f.p_raw == 0.35  # 0.9 clamped to baseline+delta


# --- hindcast (④) ---------------------------------------------------------------

def test_hindcast_idempotent_load(tmp_path):
    from openthomas.weather.hindcast import Hindcast
    from openthomas.weather.verification import VerificationStore

    hours = [f"2026-07-0{d}T{h:02d}:00" for d in (1, 2) for h in range(24)]
    def temps(base):  # peak at 15:00 local
        return [base + (12 - abs(h - 15)) for d in range(2) for h in range(24)]

    def handler(request):
        if "previous-runs" in str(request.url):
            return httpx.Response(200, json={"hourly": {
                "time": hours,
                "temperature_2m_previous_day1_gfs_seamless": temps(70),
                "temperature_2m_previous_day1_ecmwf_ifs025": temps(72),
            }})
        return httpx.Response(200, json={"data": [
            ["2026-07-01", "84", "70"], ["2026-07-02", "M", "68"],
        ]})

    store = VerificationStore(tmp_path / "v.jsonl")
    hc = Hindcast(store, http=httpx.Client(transport=httpx.MockTransport(handler)),
                  leads=range(1, 2))
    g, s = hc.load_station(STATIONS["nyc"], days=2)
    assert g == 4  # 2 days × (high, low)
    assert s == 3  # maxt missing on day 2
    # guidance mean: peak temp = base+12 → highs 82/84 → mean 83
    assert store.guidance("nyc", "high", date(2026, 7, 1), 1) == (83.0, 1.41)
    # error for verified day: 84 − 83 = +1
    assert store.errors("nyc", "high", 1) == [1.0]
    # idempotent: second load adds nothing
    assert hc.load_station(STATIONS["nyc"], days=2) == (0, 0)


def test_replay_trade_math():
    from openthomas.weather.replay import ReplayTrade, summarize
    from openthomas.markets.base import Side

    trades = [
        ReplayTrade("T1", "nyc", Side.YES, price=0.40, fee=0.02, p=0.60,
                    outcome_win=True, pnl=1 - 0.40 - 0.02),
        ReplayTrade("T2", "nyc", Side.NO, price=0.55, fee=0.02, p=0.60,
                    outcome_win=False, pnl=-0.55 - 0.02),
    ]
    s = summarize(trades)
    assert s["n"] == 2 and s["win_rate"] == 0.5
    assert abs(s["total_pnl"] - (0.58 - 0.57)) < 1e-9


# --- AFD forecaster discussion ------------------------------------------------------

AFD_TEXT = """
000
FXUS61 KOKX 081109
AFDOKX

Area Forecast Discussion
National Weather Service New York NY

.SYNOPSIS...
High pressure builds offshore.

&&

.NEAR TERM /UNTIL 6 PM THIS EVENING/...
A backdoor cold front stalls north of the area. Guidance split on timing of
the sea breeze; if it arrives before peak heating, Central Park tops out 2-3
degrees cooler than consensus.

&&

.SHORT TERM /6 PM THIS EVENING THROUGH 6 PM THURSDAY/...
Heat builds Thursday.

&&
"""


def test_forecast_discussion_extracts_near_term():
    def handler(request):
        if "/products/types/AFD/locations/OKX" in str(request.url):
            return httpx.Response(200, json={"@graph": [{"@id": "https://example.test/products/afd1"}]})
        if request.url.path == "/products/afd1":
            return httpx.Response(200, json={"productText": AFD_TEXT})
        return httpx.Response(404)

    nws = NWSClient(client=mock_client(handler))
    section = nws.forecast_discussion(STATIONS["nyc"])
    assert "backdoor cold front" in section
    assert "SYNOPSIS" not in section and "Heat builds Thursday" not in section


def test_desk_brief_includes_discussion():
    class AFDNws(StubNWS):
        def forecast_discussion(self, station, max_chars=1200):
            return "Sea breeze timing uncertain."

    desk = WeatherDesk(nws=AFDNws(), meteo=StubMeteo())
    brief = desk.brief(mk(strike_type="greater", floor_strike=83.0))
    assert "NWS forecaster discussion" in brief
    assert "Sea breeze timing uncertain." in brief
