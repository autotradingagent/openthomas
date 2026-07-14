"""The public feed: what it publishes, and what it must never publish."""

from __future__ import annotations

import json

import httpx
import pytest

from openthomas.config import ModelConfig, Settings
from openthomas.llm import CompletionClient
from openthomas.markets.base import Action, Fill, Market, Order, Side
from openthomas.memory.journal import Journal
from openthomas.memory.usage import Usage, UsageLedger, summarize
from openthomas.site.feed import build_feed, publish
from tests.test_llm import http_client


def market(mid: float = 0.40, market_id: str = "M1") -> Market:
    return Market(id=market_id, platform="kalshi", question="Will the high be >80F?",
                  category="climate/weather", yes_bid=mid - 0.01, yes_ask=mid + 0.01,
                  volume_24h=50_000, liquidity=50_000)


class Forecast:
    market_id = "M1"
    p_raw = 0.72
    p_calibrated = 0.70
    confidence = 0.8
    base_rate = 0.5
    market_gap_reason = "crowd anchors on the observed high"
    invalidation = "a marine layer forms overnight"
    reasoning = "R" * 5000
    model = "glm-5.2"


@pytest.fixture()
def settings(tmp_path) -> Settings:
    s = Settings(bankroll=1000.0, home=tmp_path)
    s.risk.min_edge = 0.08
    return s


def test_feed_reports_a_pending_edge_with_its_reasoning_truncated(settings):
    j = Journal(settings.db_path)
    j.record_forecast(Forecast(), market(mid=0.40), data="secret prompt", news="secret news")
    j.record_cycle(account_value=1100.0, cash=1100.0, n_positions=0)

    feed = build_feed(settings, j)
    (thesis,) = feed["theses"]
    assert thesis["status"] == "pending"  # forecast, no fill yet
    assert thesis["side"] == "yes"  # model 0.70 over market 0.40
    assert thesis["edge"] == pytest.approx(0.30)
    assert thesis["why"] == "crowd anchors on the observed high"
    assert thesis["invalidation"] == "a marine layer forms overnight"
    assert len(thesis["reasoning"]) == settings.site.max_reasoning_chars + 1  # + ellipsis
    assert feed["performance"]["account_value"] == 1100.0
    assert feed["performance"]["return_pct"] == pytest.approx(0.10)


def test_feed_never_leaks_prompt_inputs_or_venue_ids(settings):
    """data_text/news_text are prompt provenance, and market_id is an order
    handle. A `SELECT *` reaching the feed would ship all three."""
    j = Journal(settings.db_path)
    j.record_forecast(Forecast(), market(), data="GFS grid dump", news="paywalled article")

    blob = json.dumps(build_feed(settings, j))
    assert "GFS grid dump" not in blob
    assert "paywalled article" not in blob
    assert "M1" not in blob


def test_a_traded_market_leaves_the_outlook_and_a_held_one_stays(settings):
    j = Journal(settings.db_path)
    j.record_forecast(Forecast(), market())
    order = Order(market_id="M1", platform="kalshi", side=Side.YES, action=Action.BUY,
                  qty=10, limit_price=0.41, reason="edge")
    j.record_fill(Fill(order=order, qty=10, price=0.41, fee=0.05), market())

    (thesis,) = build_feed(settings, j)["theses"]
    assert thesis["status"] == "held"

    j.record_settlement(j.positions()[0], Side.YES)
    feed = build_feed(settings, j)
    assert feed["theses"] == []  # settled: it is track record now, not outlook
    assert feed["track_record"][0]["outcome"] == "yes"


def test_a_stale_untaken_edge_is_not_advertised_as_a_pending_bet(settings):
    j = Journal(settings.db_path)
    j.record_forecast(Forecast(), market())
    j.db.execute("UPDATE forecasts SET ts = '2020-01-01T00:00:00+00:00'")
    j.db.commit()
    assert build_feed(settings, j)["theses"] == []


def test_an_edge_under_the_bar_is_not_a_thesis(settings):
    j = Journal(settings.db_path)
    j.record_forecast(Forecast(), market(mid=0.68))  # 0.70 vs 0.68 = 0.02 < min_edge
    assert build_feed(settings, j)["theses"] == []


def test_a_serving_alias_is_not_published_as_the_model_name(settings):
    """vLLM's --served-model-name can be anything. Unset, the feed falls back to
    whatever the endpoint is called; set, it names the model and links weights."""
    settings.forecaster.model = "og-coding"
    assert build_feed(settings, Journal(settings.db_path))["agent"]["forecaster"] == {
        "label": "og-coding", "url": ""}

    settings.site.model_label = "GLM-5.2 (NVFP4)"
    settings.site.model_url = "https://huggingface.co/nvidia/GLM-5.2-NVFP4"
    assert build_feed(settings, Journal(settings.db_path))["agent"]["forecaster"] == {
        "label": "GLM-5.2 (NVFP4)", "url": "https://huggingface.co/nvidia/GLM-5.2-NVFP4"}


def test_links_are_omitted_rather_than_rendered_empty(settings):
    links = build_feed(settings, Journal(settings.db_path))["links"]
    assert links["x"] == "" and links["huggingface"] == ""  # the page hides both

    settings.site.x_handle = "openthomas"
    settings.site.huggingface = "https://huggingface.co/openthomas"
    links = build_feed(settings, Journal(settings.db_path))["links"]
    assert links["x"] == "https://x.com/openthomas"
    assert links["huggingface"] == "https://huggingface.co/openthomas"


def test_publish_writes_feed_json_atomically(settings, tmp_path):
    Journal(settings.db_path)
    path = publish(settings, tmp_path / "site")
    assert path.name == "feed.json"
    assert json.loads(path.read_text())["schema_version"] == 2
    assert not list(path.parent.glob("*.tmp"))


def test_compute_dates_the_ledger_so_zero_tokens_is_not_read_as_cheap(settings):
    j = Journal(settings.db_path)
    assert build_feed(settings, j)["compute"]["ledger_started"] is None

    UsageLedger(settings.home).record(
        Usage(ts="2026-07-10T00:00:00+00:00", node="forecast", provider="openai",
              model="glm-5.2", prompt_tokens=1000, completion_tokens=200))
    compute = build_feed(settings, j)["compute"]
    assert compute["ledger_started"] == "2026-07-10T00:00:00+00:00"
    assert compute["total"]["total_tokens"] == 1200


def test_session_spend_counts_only_this_run(settings):
    """'This run' is spend since the process started (heartbeat.run_started).
    The all-time total keeps every call; the session total keeps only the ones
    stamped after the run began."""
    from openthomas.memory.heartbeat import Heartbeat

    ledger = UsageLedger(settings.home)
    ledger.record(Usage(ts="2026-07-10T00:00:00+00:00", node="forecast", provider="openai",
                        model="glm-5.2", prompt_tokens=1000, completion_tokens=100))  # last run
    hb = Heartbeat(settings.home)
    hb.start()  # run_started = now; the row above predates it
    ledger.record(Usage(ts=hb.run_started, node="forecast", provider="openai",
                        model="glm-5.2", prompt_tokens=40, completion_tokens=60))  # this run

    compute = build_feed(settings, Journal(settings.db_path))["compute"]
    assert compute["total"]["total_tokens"] == 1200  # both calls
    assert compute["session"]["total"]["total_tokens"] == 100  # only this run
    assert compute["session"]["since"] == hb.run_started


def test_theses_carry_a_location_and_keep_the_ticker_private(settings):
    """Each weather edge is placed where its weather is — the Kalshi series
    ticker resolves to a settlement station — so the globe can pin it. The
    ticker is used to look up coords but never ships."""
    j = Journal(settings.db_path)
    f = Forecast(); f.market_id = "KXHIGHCHI-26JUL14-T94"  # Chicago Midway series
    m = Market(id=f.market_id, platform="kalshi",
               question="Will the high be 94-95° tomorrow?",
               category="climate/weather", yes_bid=0.39, yes_ask=0.41)
    j.record_forecast(f, m)
    feed = build_feed(settings, j)
    (thesis,) = feed["theses"]
    assert thesis["loc"]["place"].startswith("Chicago")
    assert thesis["loc"]["lat"] == pytest.approx(41.786, abs=0.01)
    assert "KXHIGHCHI" not in json.dumps(feed)  # the ticker stayed private


def test_a_market_we_cannot_place_gets_no_pin(settings):
    from openthomas.weather.geo import locate
    assert locate("X", "kalshi", "Will the S&P close above 6000?") is None
    assert locate("Y", "polymarket", "Will the highest temperature in Atlantis be 30°C?") is None
    assert locate("Z", "polymarket", "highest temperature in Paris be 35°C")["place"] == "Paris"


def test_board_plots_the_whole_book_with_our_view_joined(settings):
    """The globe board carries every located market. A market we forecast an
    edge on is 'pending' and carries our analysis; a plain book market carries
    price only; neither ships the market id."""
    from openthomas.memory.board import Board

    m = market(mid=0.40, market_id="KXHIGHCHI-26JUL14-T94")
    m.question = "Will the high be 94-95°?"
    Board(settings.home).write([m])  # live snapshot: one Chicago market, no view yet
    board = build_feed(settings, Journal(settings.db_path))["board"]
    assert board["from_snapshot"] is True
    (row,) = board["markets"]
    assert row["state"] == "market" and row["place"].startswith("Chicago")
    assert row["mid"] == pytest.approx(0.40) and "reasoning" not in row  # lean until we opine

    # now record a forecast with a big edge → it becomes a 'pending' pin with analysis
    j = Journal(settings.db_path)
    f = Forecast(); f.market_id = m.id
    j.record_forecast(f, m)
    row = next(r for r in build_feed(settings, j)["board"]["markets"] if r["state"] == "pending")
    assert row["reasoning"] and row["edge"] is not None
    assert "KXHIGHCHI" not in json.dumps(build_feed(settings, j))  # id stays private


def test_feed_carries_the_cached_temperature_grid_without_fetching(settings):
    """The globe's heatmap reads a cached grid; building the feed never touches
    the network (the refresh runs publish-side). No cache → no grid, no error."""
    import json as _json
    assert build_feed(settings, Journal(settings.db_path))["temperature"] is None

    grid = {"lat0": -90, "lon0": -180, "dlat": 10, "dlon": 10, "ny": 19, "nx": 36,
            "temps": [12.0] * (19 * 36), "source": "Open-Meteo", "as_of": "2026-07-14T03:00:00+00:00"}
    (settings.home / "tempgrid.json").write_text(_json.dumps({"_t": 1, "grid": grid}))
    got = build_feed(settings, Journal(settings.db_path))["temperature"]
    assert got["nx"] == 36 and got["source"] == "Open-Meteo" and len(got["temps"]) == 684

    # our own Pangu forecast field wins over the Open-Meteo nowcast when present
    (settings.home / "pangu-tempgrid.json").write_text(_json.dumps(
        {"lat0": -90, "lon0": -180, "dlat": 2.5, "dlon": 2.5, "ny": 73, "nx": 144,
         "temps": [15.0] * (73 * 144), "source": "OpenThomas · Pangu-Weather",
         "as_of": "2026-07-14T12:00:00+00:00"}))
    got = build_feed(settings, Journal(settings.db_path))["temperature"]
    assert got["source"] == "OpenThomas · Pangu-Weather" and got["nx"] == 144


def test_status_reports_liveness_from_the_heartbeat(settings):
    from openthomas.memory.heartbeat import Heartbeat

    assert build_feed(settings, Journal(settings.db_path))["status"]["last_cycle"] is None

    hb = Heartbeat(settings.home)
    hb.start()
    hb.beat()
    status = build_feed(settings, Journal(settings.db_path))["status"]
    assert status["cycles_this_run"] == 1
    assert status["run_started"] and status["last_cycle"]


# --- token ledger ----------------------------------------------------------------
def test_openai_usage_is_recorded_per_call():
    seen: list[Usage] = []
    body = {"choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 100,
                      "prompt_tokens_details": {"cached_tokens": 800}}}
    client = CompletionClient(
        ModelConfig(provider="openai", model="glm-5.2", base_url="http://x/v1"),
        http=http_client(lambda req: httpx.Response(200, json=body)),
        usage_sink=seen.append, node="forecast")

    assert client.complete("s", "u") == "ok"
    assert seen == [Usage(ts=seen[0].ts, node="forecast", provider="openai", model="glm-5.2",
                          prompt_tokens=900, completion_tokens=100, cached_tokens=800)]


def test_a_subscription_cli_call_is_counted_but_its_tokens_are_not_invented():
    """`claude -p` bills a flat rate and reports nothing. Recording 0 tokens
    would understate training cost; the summary keeps those calls apart."""
    seen: list[Usage] = []

    class Proc:
        returncode, stdout, stderr = 0, "answer", ""

    client = CompletionClient(ModelConfig(provider="claude-cli", model="sonnet"),
                              run=lambda *a, **k: Proc(), usage_sink=seen.append,
                              node="propose")
    client.complete("s", "u")

    assert seen[0].total_tokens is None
    summary = summarize(seen)
    assert summary["total"] == {"calls": 1, "prompt_tokens": 0, "completion_tokens": 0,
                                "total_tokens": 0, "calls_without_usage": 1}


def test_summarize_cuts_spend_by_node_model_and_day():
    rows = [
        Usage(ts="2026-07-09T10:00:00+00:00", node="forecast", provider="openai",
              model="glm-5.2", prompt_tokens=100, completion_tokens=10),
        Usage(ts="2026-07-10T10:00:00+00:00", node="replay", provider="openai",
              model="glm-5.2", prompt_tokens=500, completion_tokens=50),
    ]
    s = summarize(rows)
    assert s["total"]["total_tokens"] == 660
    assert [n["node"] for n in s["by_node"]] == ["replay", "forecast"]  # ranked by spend
    assert [d["day"] for d in s["by_day"]] == ["2026-07-09", "2026-07-10"]  # chronological
    assert s["by_model"][0] == {"model": "glm-5.2", "calls": 2, "prompt_tokens": 600,
                                "completion_tokens": 60, "total_tokens": 660,
                                "calls_without_usage": 0}


def test_a_torn_ledger_line_never_breaks_the_feed(settings):
    ledger = UsageLedger(settings.home)
    ledger.record(Usage(ts="2026-07-10T00:00:00+00:00", node="forecast", provider="openai",
                        model="m", prompt_tokens=5, completion_tokens=5))
    with ledger.path.open("a") as fh:
        fh.write('{"ts": "2026-07-10T01:00')  # process died mid-write
    assert summarize(ledger.read())["total"]["calls"] == 1
