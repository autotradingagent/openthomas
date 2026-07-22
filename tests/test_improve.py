"""The evolution loop: kernel gate, genome lineage, proposers, meta-cycle.

The gate tests fabricate ReplayRows directly — the whole point of the
collect/decide split is that the decision rule is scoreable offline.
"""

import json
import threading
from datetime import date, timedelta

from openthomas.config import RiskProfile, Settings
from openthomas.improve.genome import (BASELINE_ID, Generation, GenerationStore,
                                       active_params, apply_params,
                                       params_from_settings)
from openthomas.improve.loop import Improver, improve_due
from openthomas.improve.proposer import (dedupe, llm_candidates,
                                         random_candidates)
from openthomas.kernel import gate
from openthomas.kernel.bounds import PARAM_SPACE, clamp_params
from openthomas.memory.journal import Journal
from openthomas.weather.replay import ReplayRow, decide

FLAT_FEE = lambda price, qty: 0.01  # noqa: E731


def row(day: str, p_model: float, bid: float, ask: float, outcome: bool,
        ticker: str = "T") -> ReplayRow:
    return ReplayRow(ticker=ticker, station="knyc", kind="high", day=day,
                     p_model=p_model, yes_bid=bid, yes_ask=ask, outcome_yes=outcome)


def make_rows(n: int, p_model=0.9, bid=0.55, ask=0.60, outcome=True,
              start="2026-06-01") -> list[ReplayRow]:
    d0 = date.fromisoformat(start)
    return [row((d0 + timedelta(days=i)).isoformat(), p_model, bid, ask, outcome,
                ticker=f"T{i}") for i in range(n)]


# --- kernel bounds ---------------------------------------------------------------

def test_clamp_params_drops_unknown_and_clamps():
    params = clamp_params({
        "risk.min_edge": 0.001,           # below lo → clamped
        "risk.market_prior_weight": 2.0,  # above hi → clamped
        "risk.max_drawdown": 0.99,        # NOT in PARAM_SPACE → dropped
        "risk.kelly_fraction": 1.0,       # kernel policy → dropped
    })
    assert params == {"risk.min_edge": PARAM_SPACE["risk.min_edge"].lo,
                      "risk.market_prior_weight": PARAM_SPACE["risk.market_prior_weight"].hi}


def test_clamp_params_rejects_garbage_values():
    assert clamp_params({"risk.min_edge": "not a number"}) == {}
    assert clamp_params(None) == {}


# --- kernel gate -----------------------------------------------------------------

def test_decide_pure_rule_takes_the_priced_side():
    trades = decide(make_rows(1), FLAT_FEE, min_edge=0.05, market_prior_weight=0.5)
    assert len(trades) == 1
    t = trades[0]
    # p_yes = 0.5·0.575 + 0.5·0.9 = 0.7375; edge = 0.7375 − 0.60 − 0.01 > 0.05
    assert t.side.value == "yes" and t.outcome_win and t.day == "2026-06-01"


def test_split_rows_rolls_with_the_data():
    rows = make_rows(20)
    held_in, held_out = gate.split_rows(rows, held_out_days=7)
    assert len(held_out) == 7 and len(held_in) == 13
    assert max(r.day for r in held_in) < min(r.day for r in held_out)


def test_gate_rejects_thin_sample():
    rows = make_rows(30)
    ok_params = {"risk.min_edge": 0.05, "risk.market_prior_weight": 0.5}
    champion = gate.score(*gate.split_rows(rows), ok_params,
                          lambda r: decide(r, FLAT_FEE, 0.05, 0.5))
    # A challenger that trades almost never, however profitable per contract.
    lucky = gate.Score(params=ok_params,
                       held_in={"n": 2, "total_pnl": 0.8},
                       held_out={"n": 1, "total_pnl": 0.4})
    ok, why = gate.beats(lucky, champion)
    assert not ok and "not enough trades" in why


def test_gate_promotes_on_gain_and_vetoes_on_heldout_regression():
    champion = gate.Score(params={}, held_in={"n": 50, "total_pnl": 1.0},
                          held_out={"n": 10, "total_pnl": 0.5})
    better = gate.Score(params={}, held_in={"n": 40, "total_pnl": 2.0},
                        held_out={"n": 9, "total_pnl": 0.5})
    ok, why = gate.beats(better, champion)
    assert ok

    overfit = gate.Score(params={}, held_in={"n": 40, "total_pnl": 2.0},
                         held_out={"n": 9, "total_pnl": -0.5})
    ok, why = gate.beats(overfit, champion)
    assert not ok and "held-out regression" in why


def test_rollback_on_fresh_window_regret():
    active = gate.Score(params={}, held_in={"n": 30, "total_pnl": 0.2},
                        held_out={"n": 5, "total_pnl": -0.3})
    parent = gate.Score(params={}, held_in={"n": 35, "total_pnl": 0.6},
                        held_out={"n": 6, "total_pnl": 0.1})
    roll, why = gate.should_rollback(active, parent)
    assert roll and "beats active" in why
    hold, _ = gate.should_rollback(parent, active)
    assert not hold


# --- genome lineage ----------------------------------------------------------------

def test_generation_store_lifecycle(tmp_path):
    store = GenerationStore(tmp_path)
    seed = {"risk.min_edge": 0.08, "risk.market_prior_weight": 0.5}
    store.ensure_baseline(seed)
    # Baseline never overrides the operator's config.
    assert active_params(tmp_path) == {}

    gen = store.add(Generation(id=-1, parent=BASELINE_ID,
                               params={"risk.min_edge": 0.06,
                                       "risk.market_prior_weight": 0.55},
                               proposer="llm", rationale="test"))
    store.promote(gen.id, note="gate pass")
    assert store.active().id == gen.id
    assert store.get(BASELINE_ID).status == "retired"
    assert active_params(tmp_path) == {"risk.min_edge": 0.06,
                                       "risk.market_prior_weight": 0.55}

    parent = store.rollback("fresh-window regret")
    assert parent.id == BASELINE_ID and store.active().id == BASELINE_ID
    assert store.get(gen.id).status == "rolled_back"
    assert active_params(tmp_path) == {}  # back to operator config


def test_active_params_survives_corruption(tmp_path):
    (tmp_path / "generations.json").write_text("{ not json")
    assert active_params(tmp_path) == {}


def test_apply_and_read_dotted_params():
    s = Settings(risk=RiskProfile())
    apply_params(s, {"risk.min_edge": 0.06, "risk.market_prior_weight": 0.7})
    assert s.risk.min_edge == 0.06 and s.risk.market_prior_weight == 0.7
    params = params_from_settings(s)
    assert params["risk.min_edge"] == 0.06
    assert params["risk.market_prior_weight"] == 0.7
    assert params["weather_anchor_delta"] == 0.15
    # Unevolved prompt stays None = "track the built-in default". Storing
    # resolved text would freeze the prompt as of the promotion day.
    assert params["forecast_prompt"] is None
    # And None round-trips through the kernel clamp (a rollback must be able
    # to restore "default").
    assert clamp_params({"forecast_prompt": None}) == {"forecast_prompt": None}


def test_param_space_never_touches_safety_rails():
    """Kernel policy: sizing, caps, and the kill-switch are never evolvable.
    This test IS the enforcement AGENTS.md points at — do not weaken it."""
    forbidden = {"kelly_fraction", "max_position_frac", "max_event_frac",
                 "max_category_frac", "max_open_risk_frac", "max_drawdown",
                 "min_liquidity", "min_price", "max_price",
                 "max_trades_per_cycle", "max_forecasts_per_cycle"}
    assert not any(key.split(".")[-1] in forbidden for key in PARAM_SPACE)


# --- proposers ---------------------------------------------------------------------

def test_llm_candidates_clamped_and_deduped():
    champion = {"risk.min_edge": 0.08, "risk.market_prior_weight": 0.5}
    reply = '''Sure! ```json
    {"candidates": [
      {"params": {"risk.min_edge": 0.30, "risk.kelly_fraction": 0.9}, "rationale": "wild"},
      {"params": {"risk.min_edge": 0.08}, "rationale": "same as champion"},
      {"params": {"risk.min_edge": 0.06}, "rationale": "more trades, edge holds"}
    ]}```'''
    cands = llm_candidates(lambda sys, usr: reply, champion, "evidence", "summary")
    # Wild one clamped to hi bound; kelly_fraction (not in PARAM_SPACE) dropped;
    # champion-identical one deduped away.
    assert [c["params"]["risk.min_edge"] for c in cands] == [0.15, 0.06]
    assert all("risk.kelly_fraction" not in c["params"] for c in cands)
    assert all(c["params"]["risk.market_prior_weight"] == 0.5 for c in cands)


def test_llm_candidates_garbage_response():
    champion = {"risk.min_edge": 0.08, "risk.market_prior_weight": 0.5}
    assert llm_candidates(lambda s, u: "I cannot help.", champion, "", "") == []


def test_random_candidates_stay_in_bounds():
    import random

    from openthomas.kernel.bounds import params_for
    champion = {"risk.min_edge": 0.08, "risk.market_prior_weight": 0.5}
    parents = [{"risk.min_edge": 0.05, "risk.market_prior_weight": 0.4}]
    cands = random_candidates(champion, parents, k=20, rng=random.Random(7))
    for c in cands:
        assert c["proposer"] == "random"
        for key, bound in params_for("decision").items():
            assert bound.lo <= c["params"][key] <= bound.hi


def test_dedupe_against_champion():
    champion = {"risk.min_edge": 0.08, "risk.market_prior_weight": 0.5}
    cands = [{"params": dict(champion), "proposer": "llm", "rationale": ""},
             {"params": {"risk.min_edge": 0.07, "risk.market_prior_weight": 0.5},
              "proposer": "llm", "rationale": ""},
             {"params": {"risk.min_edge": 0.07, "risk.market_prior_weight": 0.5},
              "proposer": "random", "rationale": ""}]
    assert len(dedupe(cands, champion)) == 1


# --- kernel: prompt template slot ---------------------------------------------------

def test_textspec_accepts_the_default_prompt():
    from openthomas.forecast.engine import PROMPT
    from openthomas.kernel.bounds import PARAM_SPACE
    assert PARAM_SPACE["forecast_prompt"].clamp(PROMPT) == PROMPT


def test_textspec_rejects_broken_templates():
    from openthomas.kernel.bounds import PARAM_SPACE
    spec = PARAM_SPACE["forecast_prompt"]
    ok = 'Q: {question}\n{data}\nReturn JSON with "probability".'
    assert spec.clamp(ok) == ok
    assert spec.clamp('{data} "probability"') is None  # missing {question}
    assert spec.clamp('{question} {data} no contract') is None  # missing literal
    assert spec.clamp('{question} {data} {outcome} "probability"') is None  # unknown slot
    assert spec.clamp('{question} {data} "probability" {broken') is None  # bad braces
    assert spec.clamp("x" * 7000) is None  # too long
    assert spec.clamp(None) is None and spec.clamp(42) is None


def test_textspec_rejects_format_injection():
    """str.format attribute/index navigation must never reach a live prompt:
    an evolved `{question.__class__}` would splice Python internals in."""
    from openthomas.kernel.bounds import PARAM_SPACE
    spec = PARAM_SPACE["forecast_prompt"]
    base = '{question} {data} "probability" '
    assert spec.clamp(base + "{question.__class__}") is None
    assert spec.clamp(base + "{data.__init__.__globals__}") is None
    assert spec.clamp(base + "{news[0]}") is None
    assert spec.clamp(base + "{0}") is None  # positional fields too
    # Plain format specs on whitelisted names are harmless and allowed.
    assert spec.clamp(base + "{category:>10}") is not None


def test_sample_rows_deterministic_and_capped():
    from openthomas.kernel.gate import sample_rows
    rows = make_rows(50)
    a = sample_rows(rows, 10)
    b = sample_rows(list(reversed(rows)), 10)
    assert a == b and len(a) == 10  # order-independent, capped
    assert sample_rows(rows, 100) == sorted(rows, key=lambda r: (r.day, r.ticker))


def test_beats_forecast_brier_veto():
    champion = gate.Score(params={},
                          held_in={"n": 50, "total_pnl": 1.0, "brier": 0.10, "n_pairs": 100},
                          held_out={"n": 10, "total_pnl": 0.5, "brier": 0.10, "n_pairs": 40})
    lucky = gate.Score(params={},
                       held_in={"n": 40, "total_pnl": 2.5, "brier": 0.15, "n_pairs": 100},
                       held_out={"n": 9, "total_pnl": 0.6, "brier": 0.15, "n_pairs": 40})
    ok, why = gate.beats_forecast(lucky, champion)
    assert not ok and "Brier worsens" in why

    skilled = gate.Score(params={},
                         held_in={"n": 40, "total_pnl": 2.5, "brier": 0.09, "n_pairs": 100},
                         held_out={"n": 9, "total_pnl": 0.6, "brier": 0.09, "n_pairs": 40})
    ok, why = gate.beats_forecast(skilled, champion)
    assert ok and "Brier" in why


# --- LLM-in-replay -------------------------------------------------------------------

class FakeClient:
    def __init__(self, p=0.95):
        self.calls = 0
        self.p = p

    def complete(self, system, user):
        self.calls += 1
        return f'{{"probability": {self.p}, "confidence": 0.8}}'


def replayer(tmp_path, p=0.95):
    from openthomas.config import ModelConfig
    from openthomas.improve.forecast_replay import ForecastReplayer
    r = ForecastReplayer(ModelConfig(provider="openai"), tmp_path / "cache.jsonl",
                         {"risk.min_edge": 0.05, "risk.market_prior_weight": 0.5},
                         FLAT_FEE)
    r.client = FakeClient(p)
    return r


def test_forecast_replay_anchor_clip_and_pairs(tmp_path):
    r = replayer(tmp_path, p=0.95)
    rows = make_rows(3, p_model=0.5, bid=0.55, ask=0.60)
    trades, pairs = r.strategy("{question} {data} \"probability\"", 0.10)(rows)
    # 0.95 clipped to 0.5 + 0.10 = 0.60 — the LLM adjusts, never replaces.
    assert all(p == 0.60 for p, _ in pairs) and len(pairs) == 3


def test_forecast_replay_cache_makes_delta_mutations_free(tmp_path):
    r = replayer(tmp_path)
    rows = make_rows(5)
    template = "{question} {data} \"probability\""
    r.strategy(template, 0.10)(rows)
    assert r.client.calls == 5
    r.strategy(template, 0.25)(rows)  # delta-only mutation: all cache hits
    assert r.client.calls == 5
    # A fresh replayer instance reloads the cache from disk.
    r2 = replayer(tmp_path)
    r2.strategy(template, 0.10)(rows)
    assert r2.client.calls == 0
    # A different template misses (different candidate = different calls).
    r2.strategy("New: {question} {data} \"probability\"", 0.10)(rows)
    assert r2.client.calls == 5


def test_forecast_replay_cache_busts_on_revised_row_data(tmp_path):
    # The key is the exact prompt: if a row's archived guidance is revised,
    # the cached answer must not be reused for a prompt the model never saw.
    from dataclasses import replace as dc_replace
    r = replayer(tmp_path)
    rows = make_rows(3, p_model=0.6)
    template = "{question} {data} \"probability\""
    r.strategy(template, 0.1)(rows)
    assert r.client.calls == 3
    revised = [dc_replace(row, p_model=0.7) for row in rows]
    r.strategy(template, 0.1)(revised)
    assert r.client.calls == 6  # all misses again


def test_forecast_replay_compact_evicts_untouched(tmp_path):
    r = replayer(tmp_path)
    rows = make_rows(4)
    dead = "Dead: {question} {data} \"probability\""
    live = "Live: {question} {data} \"probability\""
    r.strategy(dead, 0.1)(rows)
    # New run (fresh instance): only the live template is touched.
    r2 = replayer(tmp_path)
    r2.strategy(live, 0.1)(rows)
    r2.compact()
    r3 = replayer(tmp_path)
    r3.strategy(live, 0.1)(rows)
    assert r3.client.calls == 0  # live entries survived compaction
    r3.strategy(dead, 0.1)(rows)
    assert r3.client.calls == 4  # dead template's entries were evicted


def test_forecast_replay_falls_back_to_baseline_on_failure(tmp_path):
    r = replayer(tmp_path)
    r.client.complete = lambda s, u: "not json at all"
    rows = make_rows(2, p_model=0.7)
    trades, pairs = r.strategy("{question} {data} \"probability\"", 0.1)(rows)
    assert [p for p, _ in pairs] == [0.7, 0.7]  # baseline, not dropped rows
    assert not (tmp_path / "cache.jsonl").exists()  # failures never cached


def test_llm_candidates_forecast_operator_no_riders():
    champion = {"risk.min_edge": 0.08, "risk.market_prior_weight": 0.5,
                "weather_anchor_delta": 0.15, "forecast_prompt": "old {question} {data} \"probability\""}
    reply = ('{"candidates": [{"params": {'
             '"forecast_prompt": "new {question} {data} \\"probability\\"", '
             '"weather_anchor_delta": 0.2, '
             '"risk.min_edge": 0.04}, "rationale": "r"}]}')
    cands = llm_candidates(lambda s, u: reply, champion, "", "", operator="forecast")
    assert len(cands) == 1
    # The prompt and delta mutate; the decision-param rider is stripped —
    # a forecast cycle can never smuggle in an unscored decision change.
    assert cands[0]["params"]["forecast_prompt"].startswith("new")
    assert cands[0]["params"]["weather_anchor_delta"] == 0.2
    assert cands[0]["params"]["risk.min_edge"] == 0.08


# --- meta-cycle --------------------------------------------------------------------

def improver(tmp_path, llm_reply="no"):
    s = Settings(home=tmp_path, risk=RiskProfile())
    imp = Improver(s, journal=Journal(tmp_path / "journal.db"),
                   complete_fn=lambda sys, usr: llm_reply)
    imp.fee = FLAT_FEE
    return imp, s


def test_meta_cycle_promotes_a_better_candidate(tmp_path, monkeypatch):
    # Champion min_edge=0.08 forgoes trades whose true edge is 0.05 at the
    # default blend; a lower bar takes them and wins. Rows are all winners at
    # p_model=0.9, so more trades = more PnL on both windows. Random mutations
    # are silenced so the promoted params are deterministic.
    rows = make_rows(60, p_model=0.9, bid=0.72, ask=0.76)
    monkeypatch.setattr("openthomas.improve.loop.collect_all", lambda store, days: rows)
    monkeypatch.setattr("openthomas.improve.loop.random_candidates", lambda *a, **k: [])
    reply = ('{"candidates": [{"params": {"risk.min_edge": 0.05}, '
             '"rationale": "edge bar above realized edge distribution"}]}')
    imp, s = improver(tmp_path, llm_reply=reply)

    report = imp.meta_cycle()
    assert report.promoted is not None
    assert s.risk.min_edge == 0.05  # live settings mutated in place
    # A decision promotion never pins the prompt: the slot stays None so the
    # built-in default keeps tracking upstream fixes.
    assert s.forecast_prompt is None
    active = imp.store.active()
    assert active.params["forecast_prompt"] is None
    assert active.proposer in ("llm", "random") and active.parent == BASELINE_ID
    assert (tmp_path / "improve-log.jsonl").exists()
    assert not improve_due(imp.journal)  # timestamp recorded

    # A fresh Settings.load-equivalent picks the promotion up via active_params.
    assert active_params(tmp_path)["risk.min_edge"] == 0.05


def test_meta_cycle_dry_run_writes_nothing(tmp_path, monkeypatch):
    rows = make_rows(60, p_model=0.9, bid=0.72, ask=0.76)
    monkeypatch.setattr("openthomas.improve.loop.collect_all", lambda store, days: rows)
    monkeypatch.setattr("openthomas.improve.loop.random_candidates", lambda *a, **k: [])
    reply = '{"candidates": [{"params": {"risk.min_edge": 0.05}, "rationale": "r"}]}'
    imp, s = improver(tmp_path, llm_reply=reply)

    report = imp.meta_cycle(dry_run=True)
    assert report.promoted is None and "promote:" in report.reason
    assert s.risk.min_edge == 0.08  # untouched
    assert imp.store.active() is None  # not even the baseline seed
    assert not (tmp_path / "improve-log.jsonl").exists()
    assert not (tmp_path / "generations.json").exists()


def test_meta_cycle_champion_holds_when_nothing_clears_gate(tmp_path, monkeypatch):
    # Marginal rows: the market already prices the model view; no candidate
    # should find promotable edge.
    rows = make_rows(60, p_model=0.60, bid=0.55, ask=0.60)
    monkeypatch.setattr("openthomas.improve.loop.collect_all", lambda store, days: rows)
    imp, s = improver(tmp_path, llm_reply="{}")
    report = imp.meta_cycle()
    assert report.promoted is None and "champion holds" in report.reason
    assert imp.store.active().id == BASELINE_ID


def test_meta_cycle_survives_dead_llm(tmp_path, monkeypatch):
    rows = make_rows(60)
    monkeypatch.setattr("openthomas.improve.loop.collect_all", lambda store, days: rows)

    def boom(sys, usr):
        raise RuntimeError("endpoint down")

    imp, s = improver(tmp_path)
    imp._complete = boom
    report = imp.meta_cycle()  # random mutations still run; no crash
    assert "llm proposer unavailable" in report.reason or report.promoted is not None


def test_meta_cycle_insufficient_data(tmp_path, monkeypatch):
    monkeypatch.setattr("openthomas.improve.loop.collect_all", lambda store, days: [])
    imp, s = improver(tmp_path)
    report = imp.meta_cycle()
    assert "insufficient replay data" in report.reason and report.promoted is None


def test_meta_cycle_forecast_operator_promotes(tmp_path, monkeypatch):
    # Rows where the baseline finds no edge (market ≈ baseline) but the true
    # outcome is YES: a mocked model that says 0.95 gets clipped to
    # baseline + delta and clears the edge bar, winning every trade.
    rows = make_rows(60, p_model=0.62, bid=0.45, ask=0.50)
    monkeypatch.setattr("openthomas.improve.loop.collect_all", lambda store, days: rows)
    monkeypatch.setattr("openthomas.improve.loop.random_candidates", lambda *a, **k: [])

    import openthomas.improve.loop as loop_mod

    class StubReplayer:
        def __init__(self, config, cache_path, decision_params, fee_fn, usage_sink=None):
            self.decision_params = decision_params
            self.fee_fn = fee_fn

        def compact(self):
            pass

        def strategy(self, template, delta):
            # An "evolved" template makes the model bullish; the stock one
            # (None = built-in default) mirrors the baseline. No model calls.
            p_llm = 0.95 if (template or "").startswith("evolved") else 0.62

            def run(rows_):
                from dataclasses import replace as dc_replace
                adjusted, pairs = [], []
                for row in rows_:
                    p = min(max(p_llm, row.p_model - delta), row.p_model + delta)
                    pairs.append((p, 1 if row.outcome_yes else 0))
                    adjusted.append(dc_replace(row, p_model=p))
                trades = decide(adjusted, self.fee_fn,
                                min_edge=self.decision_params["risk.min_edge"],
                                market_prior_weight=self.decision_params["risk.market_prior_weight"])
                return trades, pairs
            return run

    monkeypatch.setattr(loop_mod, "ForecastReplayer", StubReplayer)
    reply = ('{"candidates": [{"params": {"forecast_prompt": '
             '"evolved {question} {data} \\"probability\\""}, '
             '"rationale": "weigh guidance spread"}]}')
    imp, s = improver(tmp_path, llm_reply=reply)

    report = imp.meta_cycle(operator="forecast")
    assert report.promoted is not None
    active = imp.store.active()
    assert active.operator == "forecast"
    assert s.forecast_prompt.startswith("evolved")  # live engine picks it up
    # Decision params rode along unchanged.
    assert s.risk.min_edge == 0.08
    # Cadences are independent: the forecast run doesn't silence decision runs.
    assert improve_due(imp.journal, "decision")
    assert not improve_due(imp.journal, "forecast")


def test_engine_uses_promoted_template():
    from openthomas.config import ModelConfig
    from openthomas.forecast.engine import ForecastEngine
    from openthomas.markets.base import Market

    seen = {}

    class Holder:
        prompt = "PROMOTED: {question} {data} \"probability\""

    engine = ForecastEngine(ModelConfig(provider="openai", ensemble_size=1),
                            prompt_fn=lambda: Holder.prompt)
    engine.client = type("C", (), {
        "complete": staticmethod(
            lambda system, user: seen.update(user=user) or '{"probability": 0.5}'),
        "status": {"active": "primary", "model": "stub"},
    })()
    market = Market(id="m", platform="kalshi", question="Will it rain?",
                    category="weather")
    engine.forecast(market)
    assert seen["user"].startswith("PROMOTED: Will it rain?")
    # Swap the template mid-run — next forecast uses it (in-place promotion).
    Holder.prompt = "V2: {question} {data} \"probability\""
    engine.forecast(market)
    assert seen["user"].startswith("V2:")


# --- the slow loop must never make the fast loop wait -------------------------------

def _agent(tmp_path):
    from openthomas.agent.loop import Agent
    return Agent(Settings(home=tmp_path, news_enabled=False))


def test_a_slow_meta_cycle_does_not_stall_the_trading_loop(tmp_path, monkeypatch):
    """The forecast operator costs a model call per sampled row — hours on a
    local reasoning model. Run inline it would leave open positions unmarked
    and the drawdown kill-switch unevaluated for a dozen trading cycles."""
    started, release = threading.Event(), threading.Event()

    def slow_improve(self, operator="decision"):
        started.set()
        release.wait(5)

    monkeypatch.setattr("openthomas.agent.loop.Agent.improve", slow_improve)
    agent = _agent(tmp_path)

    agent._start_improvements()  # returns immediately; the fast loop walks away
    assert started.wait(2), "the improvement worker never started"
    assert agent._improving.is_alive()

    release.set()
    agent._improving.join(5)
    assert not agent._improving.is_alive()


def test_a_second_settlement_does_not_stack_a_second_meta_cycle(tmp_path, monkeypatch):
    calls, entered, release = [], threading.Event(), threading.Event()

    def slow_improve(self, operator="decision"):
        calls.append(operator)
        entered.set()
        release.wait(5)

    monkeypatch.setattr("openthomas.agent.loop.Agent.improve", slow_improve)
    agent = _agent(tmp_path)

    agent._start_improvements()
    assert entered.wait(2)
    agent._start_improvements()  # a settlement lands while the first still runs
    release.set()
    agent._improving.join(5)
    assert calls == ["decision", "forecast"]  # one worker drained both, no duplicates


def test_the_meta_cycle_opens_its_own_journal_off_the_trading_thread(tmp_path):
    """A sqlite connection belongs to the thread that opened it. Handing the
    trading loop's journal to the worker raises ProgrammingError on first use."""
    import openthomas.improve.loop as loop_mod

    agent = _agent(tmp_path)
    result = {}

    class StubImprover:
        def __init__(self, settings, journal=None, complete_fn=None):
            result["borrowed"] = journal is not None
            self.journal = journal or Journal(settings.db_path)

        def meta_cycle(self, operator="decision"):
            self.journal.get_kv("peak_value")  # raises on a foreign connection
            result["read_ok"] = True

    original, loop_mod.Improver = loop_mod.Improver, StubImprover
    try:
        worker = threading.Thread(target=agent._improve_worker, args=(["decision"],))
        worker.start()
        worker.join(5)
    finally:
        loop_mod.Improver = original

    assert result["borrowed"] is False, "the worker borrowed the trading loop's journal"
    assert result.get("read_ok"), "the worker's journal read raised off-thread"


def test_a_killed_meta_cycle_never_leaves_the_lineage_truncated(tmp_path):
    """A worker that runs for hours will eventually be killed mid-write, and a
    half-written generations.json reads as "no overrides" — every promotion
    silently discarded. The write is atomic, and leaves no debris."""
    store = GenerationStore(tmp_path)
    store.ensure_baseline({"risk.min_edge": 0.08})
    gen = store.add(Generation(id=-1, parent=BASELINE_ID,
                               params={"risk.min_edge": 0.05}, proposer="llm"))
    store.promote(gen.id)

    assert not list(tmp_path.glob("*.tmp"))
    assert len(json.loads(store.path.read_text())["generations"]) == 2
    assert active_params(tmp_path) == {"risk.min_edge": 0.05}
