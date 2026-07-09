"""The trading cycle: sync → settle → scan → forecast → risk-check → execute → learn.

Modeled on the Prediction Arena cycle, with the guardrails its losing agents
lacked. Every step logs to the journal; every decision carries its reason.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from ..config import Settings
from ..edge.scanner import EdgeScanner, ScanResult
from ..forecast.calibration import PlattScaler
from ..forecast.engine import Forecast, ForecastEngine
from ..llm import CompletionClient
from ..markets.base import Action, Market, MarketConnector, Order, Side
from ..markets.kalshi import KalshiConnector
from ..markets.paper import InsufficientLiquidity, PaperBroker
from ..markets.polymarket import PolymarketConnector
from ..memory.journal import Journal
from ..memory.lessons import LessonBook
from ..research.news import NewsDesk
from ..risk.engine import PortfolioState, RiskEngine
from ..weather.desk import WeatherDesk
from ..weather.verification import VerificationStore


@dataclass
class CycleReport:
    markets_seen: int = 0
    candidates: int = 0
    forecasts: int = 0
    trades: list[str] = field(default_factory=list)
    settlements: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    arbs: list[str] = field(default_factory=list)
    account_value: float = 0.0
    cash: float = 0.0
    halted: bool = False


def build_connectors(platforms: list[str]) -> dict[str, MarketConnector]:
    registry = {"polymarket": PolymarketConnector, "kalshi": KalshiConnector}
    return {name: registry[name]() for name in platforms if name in registry}


class Agent:
    def __init__(self, settings: Settings):
        self.s = settings
        self.journal = Journal(settings.db_path)
        self.connectors = build_connectors(settings.platforms)
        self.broker = PaperBroker(self.connectors)
        self.scanner = EdgeScanner(settings.risk)
        self.risk = RiskEngine(settings.risk)
        self.lessons = LessonBook(settings.lessons_dir)
        self._scalers: dict[str, PlattScaler] = {}
        self.forecaster = ForecastEngine(settings.forecaster, calibrate=self._calibrate)
        self.reflector = CompletionClient(settings.reflector or settings.forecaster)
        self.news = NewsDesk() if settings.news_enabled else None
        from ..weather.localmodels import LocalModelSource
        self.weather = WeatherDesk(
            store=VerificationStore(settings.home / "weather-verification.jsonl"),
            local_models=LocalModelSource(settings.home / "local-models.jsonl"),
        )

    # --- calibration -----------------------------------------------------------
    def _calibrate(self, p_raw: float, category: str) -> float:
        key = category or "_all"
        if key not in self._scalers:
            pairs = self.journal.forecast_outcome_pairs(category or None)
            self._scalers[key] = PlattScaler.fit(pairs)
        return self._scalers[key].apply(p_raw)

    # --- portfolio -------------------------------------------------------------
    def portfolio_state(self, marks: dict[str, Market]) -> PortfolioState:
        positions = self.journal.positions()
        cash = self.journal.cash(self.s.bankroll)
        value = cash + sum(
            pos.mark_to_market(marks[pos.market_id]) if pos.market_id in marks else pos.cost_basis
            for pos in positions
        )
        return PortfolioState(
            bankroll=self.s.bankroll, cash=cash, positions=positions,
            peak_value=max(self.journal.peak_value(), self.s.bankroll),
            account_value=value,
        )

    def _settle(self, report: CycleReport) -> None:
        for pos in self.journal.positions():
            connector = self.connectors.get(pos.platform)
            if connector is None:
                continue
            try:
                outcome = connector.resolved_outcome(pos.market_id)
            except Exception:
                continue
            if outcome is not None:
                pnl = self.journal.record_settlement(pos, outcome)
                report.settlements.append(
                    f"{pos.question[:60]} → {outcome.value} (${pnl:+.2f})"
                )
        if report.settlements:
            self._scalers.clear()  # refit calibration with the new outcomes

    # --- the cycle ---------------------------------------------------------------
    def cycle(self, markets_per_platform: int = 150) -> CycleReport:
        report = CycleReport()

        markets: list[Market] = []
        for connector in self.connectors.values():
            try:
                if self.s.focus == "weather":
                    markets += connector.list_weather_markets()
                else:
                    markets += connector.list_markets(limit=markets_per_platform)
            except Exception as e:  # a dead venue must not kill the loop
                report.rejections.append(f"{connector.platform}: sync failed ({e})")
        report.markets_seen = len(markets)
        marks = {m.id: m for m in markets}

        self._settle(report)
        try:
            self.weather.update_verification()
        except Exception:
            pass  # verification learning must never block trading
        state = self.portfolio_state(marks)
        report.account_value, report.cash = state.account_value, state.cash

        if self.risk.drawdown_halted(state):
            report.halted = True
            self.journal.record_cycle(state.account_value, state.cash, len(state.positions),
                                      "HALTED: max drawdown kill-switch")
            return report

        scan: ScanResult = self.scanner.scan(markets)
        report.candidates = len(scan.candidates)
        report.arbs = [a.describe() for a in scan.arbs[:5]]

        lessons_text = self.lessons.render_for_prompt(self.journal)
        events = {m.id: m.event_id for m in markets}
        positioned = {p.market_id for p in state.positions}
        trades_done = 0

        for market in scan.candidates:
            if report.forecasts >= self.s.risk.max_forecasts_per_cycle:
                break
            if trades_done >= self.s.risk.max_trades_per_cycle:
                break
            if market.id in positioned or self.journal.has_recent_forecast(market.id):
                continue

            assessment = None
            try:
                assessment = self.weather.assess(market)
            except Exception:
                pass

            if assessment is not None and assessment.decided:
                # The observation already forces the outcome — no LLM needed.
                # 0.98/0.02, not 1/0: station obs and the CLI report disagree
                # by a degree often enough to keep some humility.
                p = 0.98 if assessment.p_base >= 0.5 else 0.02
                forecast = Forecast(
                    market_id=market.id, p_raw=p, p_calibrated=p, confidence=0.9,
                    reasoning=f"Observation-determined: {assessment.kind} already "
                              f"{assessment.observed:.1f}°F at {assessment.station.obs_id}",
                    model="baseline-observation",
                )
            else:
                news = ""
                if self.news:
                    try:
                        news = self.news.brief(market.question, self.s.news_max_articles)
                    except Exception:
                        pass
                anchor = None
                if assessment is not None and assessment.p_base is not None:
                    anchor = (assessment.p_base, self.s.weather_anchor_delta)
                forecast = self.forecaster.forecast(
                    market, lessons_text, news,
                    data=assessment.text if assessment else "", anchor=anchor,
                )
            report.forecasts += 1
            if forecast is None:
                continue
            self.journal.record_forecast(forecast, market)

            if forecast.confidence < self.s.risk.min_confidence:
                report.rejections.append(f"{market.question[:50]}: confidence {forecast.confidence:.2f} too low")
                continue

            # Blend toward the market price: the crowd is information, not noise.
            w = self.s.risk.market_prior_weight
            p_trade = w * (market.mid or 0.5) + (1 - w) * forecast.p_calibrated

            side = Side.YES if p_trade > (market.mid or 0.5) else Side.NO
            price = market.price_to_buy(side)
            if price is None:
                continue
            fee = self.connectors[market.platform].fee(price, 1, market.category)
            verdict = self.risk.size_entry(state, market, side, p_trade,
                                           fee_per_contract=fee, events=events)
            if not verdict.approved:
                report.rejections.append(f"{market.question[:50]}: {verdict.reason}")
                continue

            order = Order(
                market_id=market.id, platform=market.platform, side=side,
                action=Action.BUY, qty=verdict.qty, limit_price=price,
                reason=f"p={forecast.p_calibrated:.2f} vs price {price:.2f}; {verdict.reason}",
            )
            try:
                if self.s.mode == "live":
                    # Stale-quote guard: prices were fetched at cycle start and the
                    # LLM call took seconds — long enough for a faster agent to move
                    # the book. Re-validate the edge at execution time.
                    fresh = self.connectors[market.platform].get_market(market.id) or market
                    fresh_price = fresh.price_to_buy(side)
                    p_side = p_trade if side is Side.YES else 1 - p_trade
                    if fresh_price is None or p_side - fresh_price - fee < self.s.risk.min_edge:
                        report.rejections.append(
                            f"{market.question[:50]}: quote moved, edge gone at execution"
                        )
                        continue
                    order.limit_price = min(price, fresh_price)
                    fill = self.connectors[market.platform].place_order(order)
                else:
                    fill = self.broker.execute(order, market)
            except (InsufficientLiquidity, NotImplementedError) as e:
                report.rejections.append(f"{market.question[:50]}: {e}")
                continue
            self.journal.record_fill(fill, market)
            trades_done += 1
            state = self.portfolio_state(marks)  # refresh caps/cash after each fill
            report.trades.append(
                f"BUY {fill.qty} {side.value.upper()} @ {fill.price:.2f} "
                f"[{market.platform}] {market.question[:60]}"
            )

        self.journal.record_cycle(state.account_value, state.cash, len(state.positions))
        return report

    def reflect(self) -> str:
        return self.lessons.reflect(self.journal, self.reflector.complete)

    def improve(self):
        """One evolution meta-cycle (docs/RSI.md). Mutates settings in place,
        so promoted parameters take effect from the next trading cycle."""
        from ..improve.loop import Improver
        return Improver(self.s, journal=self.journal,
                        complete_fn=self.reflector.complete).meta_cycle()

    def run_forever(self, on_report=None) -> CycleReport:
        from ..improve.loop import improve_due

        while True:
            report = self.cycle()
            if on_report:
                on_report(report)
            if report.settlements:
                try:
                    self.reflect()
                except Exception:
                    pass
                try:  # the slow loop must never take down the fast loop
                    if improve_due(self.journal):
                        self.improve()
                except Exception:
                    pass
            if report.halted:
                return report
            # ±30% jitter: a fixed cadence is a signature other agents can time
            # (e.g. quoting wide just before our cycle and picking us off).
            time.sleep(self.s.cycle_minutes * 60 * random.uniform(0.7, 1.3))
