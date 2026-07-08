import json

from openthomas.memory.journal import Journal
from openthomas.memory.lessons import (
    MAX_ACTIVE_RULES, MAX_ADDS_PER_REFLECTION, LessonBook,
)


def book(tmp_path):
    return LessonBook(tmp_path / "lessons")


def test_add_revise_deprecate_cycle(tmp_path):
    b = book(tmp_path)
    audit = b.apply_ops([
        {"op": "add", "text": "Miami consensus runs 2°F cold — shade highs up",
         "scope": "Miami", "reason": "hindcast bias +1.8..2.1"},
        {"op": "add", "text": "Don't fade LAX afternoon marine layer", "scope": "lax"},
    ])
    assert len(audit) == 2
    rules = b.active_rules()
    assert [r["id"] for r in rules] == [1, 2]
    assert rules[0]["scope"] == "miami"  # normalized lowercase

    b.apply_ops([{"op": "revise", "id": 1, "text": "Miami: shade highs +2°F"}])
    assert b.active_rules()[0]["text"] == "Miami: shade highs +2°F"
    assert "revised" in b.active_rules()[0]

    b.apply_ops([{"op": "deprecate", "id": 2, "reason": "edge decayed"}])
    active = b.active_rules()
    assert len(active) == 1 and active[0]["id"] == 1
    # Deprecated rules stay on file with their reason — audit, not amnesia.
    dead = [r for r in b._load()["rules"] if r["status"] == "deprecated"]
    assert dead[0]["deprecate_reason"] == "edge decayed"


def test_caps_enforced(tmp_path):
    b = book(tmp_path)
    b.apply_ops([{"op": "add", "text": f"seed {i}", "scope": "x"} for i in range(9)])
    assert len(b.active_rules()) == MAX_ADDS_PER_REFLECTION  # per-reflection cap
    for i in range(5):  # fill to the active cap over several reflections
        b.apply_ops([{"op": "add", "text": f"more {i}a", "scope": "x"},
                     {"op": "add", "text": f"more {i}b", "scope": "x"}])
    assert len(b.active_rules()) == MAX_ACTIVE_RULES


def test_bad_ops_ignored(tmp_path):
    b = book(tmp_path)
    audit = b.apply_ops([
        {"op": "revise", "id": 99, "text": "ghost"},
        {"op": "deprecate", "id": 99},
        {"op": "add", "text": ""},
        {"op": "explode"},
    ])
    assert audit == [] and b.active_rules() == []


def test_parse_ops_from_noisy_output():
    noisy = 'Thinking...\n```json\n{"ops": [{"op": "add", "text": "x", "scope": "y"}]}\n```'
    assert LessonBook._parse_ops(noisy)[0]["op"] == "add"
    assert LessonBook._parse_ops("no json here") == []
    assert LessonBook._parse_ops('{"ops": "not-a-list"}') == []


def test_reflect_applies_ops_and_renders(tmp_path):
    j = Journal(tmp_path / "j.db")
    # seed 6 settlements so reflection engages
    for i in range(6):
        j.db.execute(
            "INSERT INTO settlements VALUES (?, ?, 'kalshi', ?, 'climate and weather', "
            "'yes', 1.0, 0.6, 0.4)",
            (f"M{i}", f"2026-07-0{i + 1}T00:00:00+00:00", f"Will Miami high be >9{i}°?"),
        )
    j.db.commit()

    def fake_llm(system, user):
        assert "Active rules" in user and "Recent settlements" in user
        return json.dumps({"ops": [{"op": "add", "scope": "miami",
                                    "text": "Miami runs hot — shade up",
                                    "reason": "6 green settlements"}]})

    b = book(tmp_path)
    rendered = b.reflect(j, fake_llm)
    assert "R1 [miami]: Miami runs hot — shade up" in rendered
    assert "Track record: 6 settled" in rendered


def test_rule_track_flags_negative_scope(tmp_path):
    j = Journal(tmp_path / "j.db")
    b = book(tmp_path)
    b.apply_ops([{"op": "add", "text": "fade denver", "scope": "denver"}])
    for i in range(12):
        j.db.execute(
            "INSERT INTO settlements VALUES (?, '2099-01-01T00:00:00+00:00', 'kalshi', "
            "'Will the high temp in Denver be >90°?', 'climate and weather', 'no', "
            "0, 0.5, -0.5)", (f"D{i}",))
    j.db.commit()
    track = b._rules_with_track(j)
    assert "NEGATIVE since adoption" in track


# --- bucketed Brier (⑥) -----------------------------------------------------------

def _seed(j, market_id, ts, p, mid, outcome):
    j.db.execute(
        "INSERT INTO forecasts (ts, market_id, platform, question, category, p_raw,"
        " p_calibrated, confidence, mid) VALUES (?,?,?,?,?,?,?,?,?)",
        (ts, market_id, "kalshi", "q", "climate and weather", p, p, 0.7, mid))
    j.db.execute(
        "INSERT INTO settlements VALUES (?, ?, 'kalshi', 'q', 'climate and weather',"
        " ?, 0, 0, 0)", (market_id, ts, outcome))
    j.db.commit()


def test_weather_skill_buckets_by_station_and_lead(tmp_path):
    from openthomas.report.brier import summarize_skill, weather_skill

    j = Journal(tmp_path / "j.db")
    # same-day NYC: model 0.8 (right), market 0.5
    _seed(j, "KXHIGHNY-26JUL08-T83", "2026-07-08T14:00:00+00:00", 0.8, 0.5, "yes")
    # lead-1 Miami: model 0.3 (wrong side), market 0.6
    _seed(j, "KXHIGHMIA-26JUL08-B82.5", "2026-07-07T14:00:00+00:00", 0.3, 0.6, "yes")
    # non-weather market must be ignored
    _seed(j, "KXBTC-26JUL08-T100000", "2026-07-08T14:00:00+00:00", 0.5, 0.5, "no")

    buckets = weather_skill(j)
    assert {(b["station"], b["lead"]) for b in buckets} == {("nyc", "0"), ("mia", "1")}
    nyc = next(b for b in buckets if b["station"] == "nyc")
    assert abs(nyc["brier_model"] - 0.04) < 1e-9
    assert abs(nyc["brier_market"] - 0.25) < 1e-9
    assert nyc["skill"] > 0.8

    total = summarize_skill(buckets)
    assert total["n"] == 2
    # model: (0.04 + 0.49)/2 ; market: (0.25 + 0.16)/2
    assert abs(total["brier_model"] - 0.265) < 1e-9
    assert abs(total["brier_market"] - 0.205) < 1e-9


def test_forecast_mid_recorded(tmp_path):
    from openthomas.forecast.engine import Forecast
    from openthomas.markets.base import Market

    j = Journal(tmp_path / "j.db")
    m = Market(id="KXHIGHNY-26JUL09-T83", platform="kalshi", question="q",
               yes_bid=0.40, yes_ask=0.44)
    j.record_forecast(Forecast(market_id=m.id, p_raw=0.6, p_calibrated=0.62,
                               confidence=0.7), m)
    row = j.db.execute("SELECT mid FROM forecasts").fetchone()
    assert abs(row["mid"] - 0.42) < 1e-9
