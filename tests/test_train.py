"""The training set and the Hugging Face push: the rails, not the plumbing.

Each rail here, forgotten once, produces a model that looks brilliant on paper
and is worthless in a market. They are tested because they are load-bearing.
"""

from __future__ import annotations

import json
import math

import pytest

from openthomas.config import Settings
from openthomas.markets.base import Market, Position, Side
from openthomas.memory.journal import Journal
from openthomas.train import hub
from openthomas.train.dataset import REWARD_LAMBDA, build, summary, trainable, write_jsonl


def market(market_id: str = "M1") -> Market:
    return Market(id=market_id, platform="kalshi", question="Will the high be >80F?",
                  category="climate/weather", yes_bid=0.39, yes_ask=0.41)


class Forecast:
    def __init__(self, market_id="M1", p=0.7, model="glm-5.2"):
        self.market_id, self.p_raw, self.p_calibrated = market_id, p, p
        self.confidence, self.base_rate = 0.8, 0.5
        self.market_gap_reason, self.invalidation = "crowd anchors", "marine layer forms"
        self.reasoning, self.model = "because", model


@pytest.fixture()
def settings(tmp_path) -> Settings:
    s = Settings(home=tmp_path)
    s.hub.org = "openthomas"
    return s


def settle(journal: Journal, market_id: str, outcome: Side = Side.YES) -> None:
    journal.record_settlement(
        Position(market_id=market_id, platform="kalshi", side=Side.YES, qty=10,
                 avg_cost=0.4, question="Will the high be >80F?", category="climate/weather"),
        outcome)


def test_only_settled_markets_become_training_rows(settings):
    """An open position has no label — and publishing our live view of one
    would hand away the trade."""
    j = Journal(settings.db_path)
    j.record_forecast(Forecast("SETTLED"), market("SETTLED"), data="baseline 0.62")
    j.record_forecast(Forecast("OPEN"), market("OPEN"), data="baseline 0.51")
    settle(j, "SETTLED")

    rows = build(j)
    assert [r["market_id"] for r in rows] == ["SETTLED"]


def test_only_the_first_forecast_per_market_survives(settings):
    """Later forecasts watched the price move and the day advance. Training on
    them teaches hindsight, and hindsight backtests beautifully."""
    j = Journal(settings.db_path)
    j.record_forecast(Forecast(p=0.55), market(), data="morning guidance")
    j.record_forecast(Forecast(p=0.95), market(), data="afternoon, already 82F")
    settle(j, "M1")

    (row,) = build(j)
    assert row["p_forecast"] == 0.55
    assert row["data"] == "morning guidance"


def test_the_split_is_temporal_and_travels_with_the_rows(settings):
    """A random split lets a model validate on days it trained on. Assigning
    it at export means it cannot be reshuffled downstream."""
    j = Journal(settings.db_path)
    for i in range(10):
        j.record_forecast(Forecast(f"M{i}"), market(f"M{i}"), data="d")
        settle(j, f"M{i}")
    # record_forecast stamps ts at insert, so id order is time order.
    rows = build(j, valid_fraction=0.2)

    assert [r["split"] for r in rows] == ["train"] * 8 + ["validation"] * 2
    assert max(r["ts"] for r in rows if r["split"] == "train") \
        <= min(r["ts"] for r in rows if r["split"] == "validation")


def test_news_text_never_leaves_the_box_but_its_presence_is_recorded(settings):
    """Third-party headlines are not ours to redistribute. Dropping them
    silently would make these rows look like a faithful prompt replay."""
    j = Journal(settings.db_path)
    j.record_forecast(Forecast(), market(), data="ours", news="PAYWALLED HEADLINE")
    settle(j, "M1")

    (row,) = build(j)
    assert "PAYWALLED HEADLINE" not in json.dumps(row)
    assert row["had_news"] is True
    assert row["data"] == "ours"


def test_a_label_without_features_is_not_counted_as_trainable(settings):
    """Rows forecast before the journal archived prompt inputs carry an outcome
    and nothing to learn it from. Counting them flatters the dataset."""
    j = Journal(settings.db_path)
    j.record_forecast(Forecast("WITH"), market("WITH"), data="baseline 0.62")
    j.record_forecast(Forecast("WITHOUT"), market("WITHOUT"))  # no data archived
    settle(j, "WITH")
    settle(j, "WITHOUT")

    rows = build(j)
    stats = summary(rows)
    assert stats["rows"] == 2 and stats["trainable_rows"] == 1
    assert [r["market_id"] for r in trainable(rows)] == ["WITH"]


def test_reward_discounts_a_call_made_close_to_resolution(settings):
    j = Journal(settings.db_path)
    j.record_forecast(Forecast(), market(), data="d")
    settle(j, "M1")
    j.db.execute("UPDATE forecasts SET ts = '2026-07-01T00:00:00+00:00'")
    j.db.execute("UPDATE settlements SET ts = '2026-07-11T00:00:00+00:00'")  # 10 days
    j.db.commit()

    (row,) = build(j)
    assert row["days_to_close"] == pytest.approx(10.0)
    assert row["reward"] == pytest.approx(row["pnl"] * math.exp(-REWARD_LAMBDA * 10), abs=1e-3)


def test_write_jsonl_round_trips(settings, tmp_path):
    j = Journal(settings.db_path)
    j.record_forecast(Forecast(), market(), data="d")
    settle(j, "M1")
    path = write_jsonl(build(j), tmp_path / "sub" / "journal.jsonl")
    assert json.loads(path.read_text().splitlines()[0])["market_id"] == "M1"


def test_a_serving_alias_never_escapes_through_the_dataset(settings):
    """The journal records what the endpoint answered to. Under vLLM that is
    --served-model-name, which can be anything: "og-coding" names no model.
    The website already resolves this; the dataset must resolve it identically."""
    settings.forecaster.model = "og-coding"
    j = Journal(settings.db_path)
    j.record_forecast(Forecast(model="og-coding"), market(), data="d")
    settle(j, "M1")

    (bare,) = build(j)
    assert bare["forecaster"] == "og-coding"  # no mapping configured: verbatim

    settings.site.model_label = "GLM-5.2 (NVFP4)"
    (row,) = build(j, aliases=hub.aliases(settings))
    assert row["forecaster"] == "GLM-5.2 (NVFP4)"
    assert "og-coding" not in json.dumps(row)


def test_the_card_reports_how_many_rows_carry_a_market_price(settings):
    """`mid` landed in the journal after the first settlements. A card that
    hides that lets a reader assume every row can be scored against a price."""
    j = Journal(settings.db_path)
    j.record_forecast(Forecast("WITH"), market("WITH"), data="d")
    j.record_forecast(Forecast("NONE"), market("NONE"), data="d")
    j.db.execute("UPDATE forecasts SET mid = NULL WHERE market_id = 'NONE'")
    j.db.commit()
    settle(j, "WITH")
    settle(j, "NONE")

    stats = summary(build(j))
    assert stats["rows"] == 2 and stats["rows_with_market_price"] == 1
    assert "| rows with a market price at forecast time | 1 |" in hub.dataset_card(stats, settings)


# --- the hub: no claim without evidence -------------------------------------------

def _eval(**over) -> dict:
    return {"brier_base": 0.25, "brier_tuned": 0.19, "n_validation": 120, **over}


def test_weights_are_never_published_without_held_out_numbers(settings, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    for missing in hub.REQUIRED_EVAL:
        broken = _eval()
        del broken[missing]
        with pytest.raises(hub.HubError, match=missing):
            hub.push_adapter(settings, "openthomas-lora", adapter, "google/gemma-3-12b-it",
                             "abc123", broken)


def test_weights_are_never_published_without_naming_their_dataset(settings, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(hub.HubError, match="dataset revision"):
        hub.push_adapter(settings, "openthomas-lora", adapter, "google/gemma-3-12b-it",
                         "", _eval())


def test_nothing_is_pushed_until_an_org_is_configured(tmp_path):
    s = Settings(home=tmp_path)  # hub.org unset: the safe default
    with pytest.raises(hub.HubError, match="hub.org"):
        hub.dataset_repo(s)


def test_dataset_repo_defaults_to_the_orgs_journal(settings):
    assert hub.dataset_repo(settings) == "openthomas/journal"
    settings.hub.dataset = "someone/else"
    assert hub.dataset_repo(settings) == "someone/else"


def test_the_model_card_carries_the_evidence_and_the_dataset_pin(settings):
    card = hub.model_card("openthomas-lora-12b", "google/gemma-3-12b-it",
                          "openthomas/journal", "a" * 40, _eval(), settings)
    assert "0.2500" in card and "0.1900" in card and "+0.0600" in card
    assert "120 held-out settlements" in card
    assert f"tree/{'a' * 40}" in card
    assert "beats" in card
    # The risk engine is never learned; the card must say so.
    assert "never learned" in card


def test_a_model_that_lost_says_so_on_its_own_card(settings):
    card = hub.model_card("dud", "base", "openthomas/journal", "b" * 40,
                          _eval(brier_tuned=0.31), settings)
    assert "does not beat" in card and "-0.0600" in card


def test_a_thin_dataset_card_says_it_is_not_ready(settings):
    thin = hub.dataset_card({"rows": 10, "trainable_rows": 4, "rows_with_market_price": 4,
                             "train": 8, "validation": 2, "forecasters": ["GLM-5.2"],
                             "span": ("2026-07-08", "2026-07-09"), "base_rate": 0.4,
                             "platforms": ["kalshi"]}, settings)
    assert "Not yet enough" in thin
    assert "memorizes" in thin

    fat = hub.dataset_card({"rows": 900, "trainable_rows": 900, "rows_with_market_price": 900,
                            "train": 720, "validation": 180, "forecasters": ["GLM-5.2"],
                            "span": ("2026-01-01", "2026-07-09"),
                            "base_rate": 0.5, "platforms": ["kalshi"]}, settings)
    assert "Not yet enough" not in fat
