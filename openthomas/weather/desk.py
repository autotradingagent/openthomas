"""WeatherDesk: the weather brain the trading loop talks to.

assess(market) returns a WeatherAssessment — the statistical baseline P(YES)
plus the prompt-facing data block. brief(market) is the text alone (MCP and
prompts). update_verification() keeps the learning store fed: today's model
consensus recorded per station, yesterday's official settlements filled in.

Everything rendered is labeled data for the model to weigh — never
instructions.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from .baseline import hour_factor, strike_probability
from .gencast import GencastSpread
from .localmodels import LocalModelSource
from .nws import NWSClient
from .openmeteo import OpenMeteoClient
from .stations import KALSHI_SERIES, STATIONS, Station, station_for_market, target_date
from .strikes import Strike, parse_strike
from .verification import VerificationStore


# Shared prompt prose, used by the live desk AND the LLM-in-replay evaluator
# (improve/forecast_replay.py). One source: replayed prompts must read exactly
# like live ones, or evolved templates get scored against a fiction and the
# drift is silent.

def settles_line(station: Station, kind: str, day: str) -> str:
    return (f"Market settles on the official NWS {kind} temperature at "
            f"{station.name} ({station.obs_id}) on {day}.")


def strike_line(kind: str, strike_desc: str) -> str:
    return f"YES resolves if the official {kind} is {strike_desc}."


def baseline_line(p_base: float, mean: float, bias: float, sigma: float,
                  n: int | None = None, with_obs: bool = True) -> str:
    detail = f"consensus {mean:.1f}°F, station bias {bias:+.1f}, sigma {sigma:.1f}"
    if n is not None:
        detail += f", {n} verified settlements"
    prices = ("the model consensus and today's observations" if with_obs
              else "the model consensus")
    return (f"Statistical baseline: P(YES) = {p_base:.2f} ({detail}). "
            f"This baseline already prices {prices}. Deviate only for "
            "information it cannot see, and name that information.")


@dataclass
class WeatherAssessment:
    station: Station
    kind: str  # "high" | "low"
    day: date
    lead_days: int
    strike: Strike | None
    model_values: dict[str, float]
    mean: float | None
    sigma: float | None
    bias: float
    n_verified: int
    observed: float | None
    p_base: float | None  # statistical P(YES); None without models+strike
    text: str  # prompt data block

    @property
    def decided(self) -> bool:
        """The observation alone already forces the outcome."""
        return self.p_base is not None and (self.p_base <= 0.001 or self.p_base >= 0.999)


class WeatherDesk:
    def __init__(self, nws: NWSClient | None = None, meteo: OpenMeteoClient | None = None,
                 store: VerificationStore | None = None,
                 local_models: LocalModelSource | None = None,
                 gencast_spread: GencastSpread | None = None, cache_ttl: float = 900):
        self.nws = nws or NWSClient()
        self.meteo = meteo or OpenMeteoClient()
        self.store = store  # None → no verification recording (e.g. tests)
        self.local_models = local_models
        self.gencast_spread = gencast_spread  # flow-dependent sigma from the ensemble
        self.cache_ttl = cache_ttl
        self._extremes_cache: dict[str, tuple[float, dict]] = {}
        self._observed_cache: dict[tuple[str, str], tuple[float, float | None]] = {}
        self._discussion_cache: dict[str, tuple[float, str]] = {}
        self._guidance_done: set[tuple[str, str]] = set()  # (station, iso date)
        self._settle_done: set[tuple[str, str]] = set()

    # --- data access (cached) ---------------------------------------------------
    def _extremes(self, station: Station) -> dict:
        hit = self._extremes_cache.get(station.key)
        if hit and time.monotonic() - hit[0] < self.cache_ttl:
            return hit[1]
        data = self.meteo.daily_extremes(station)
        if self.local_models is not None:
            try:  # our own AI-NWP runs join the consensus as extra models
                for day, kinds in self.local_models.extremes(station.key).items():
                    for kind, by_model in kinds.items():
                        data.setdefault(day, {"high": {}, "low": {}})[kind].update(by_model)
            except Exception:
                pass
        self._extremes_cache[station.key] = (time.monotonic(), data)
        return data

    def _observed(self, station: Station, kind: str) -> float | None:
        key = (station.key, kind)
        hit = self._observed_cache.get(key)
        if hit and time.monotonic() - hit[0] < 600:
            return hit[1]
        try:
            value = self.nws.observed_extreme_today(station, kind)
        except Exception:
            value = None
        self._observed_cache[key] = (time.monotonic(), value)
        return value

    def _discussion(self, station: Station) -> str:
        hit = self._discussion_cache.get(station.key)
        if hit and time.monotonic() - hit[0] < 3600:  # AFDs refresh ~4x/day
            return hit[1]
        try:
            text = self.nws.forecast_discussion(station)
        except Exception:
            text = ""
        self._discussion_cache[station.key] = (time.monotonic(), text)
        return text

    # --- assessment -----------------------------------------------------------
    def assess(self, market) -> WeatherAssessment | None:
        info = station_for_market(market)
        if info is None:
            return None
        station, kind = info
        day = target_date(market, station)
        now_local = datetime.now(ZoneInfo(station.timezone))
        lead = max(0, (day - now_local.date()).days)

        try:
            by_model = self._extremes(station).get(day.isoformat(), {}).get(kind, {})
        except Exception:
            by_model = {}
        observed = self._observed(station, kind) if day == now_local.date() else None

        if self.store is not None:
            bias, sigma_stat, n = self.store.stats(station.key, kind, lead)
        else:
            from .verification import prior_sigma
            bias, sigma_stat, n = 0.0, prior_sigma(lead), 0

        mean = sigma = p_base = None
        strike = parse_strike(market)
        if by_model:
            values = list(by_model.values())
            mean = statistics.mean(values)
            spread = statistics.stdev(values) if len(values) > 1 else 0.0
            # GenCast's ensemble scales the learned sigma by this day's flow-
            # dependent uncertainty (wider on genuinely uncertain days); 1.0 when
            # no fresh ensemble covers it.
            flow = (self.gencast_spread.factor(station.key, day.isoformat(), kind, lead)
                    if self.gencast_spread is not None else 1.0)
            # Day-specific disagreement can exceed climatological error; never
            # let sigma collapse entirely — obs/CLI quirks alone are ~1°F.
            sigma = max(0.8, sigma_stat * flow * hour_factor(kind, now_local, day == now_local.date()),
                        0.8 * spread)
            if strike:
                p_base = strike_probability(strike, mean + bias, sigma, kind, observed)

        return WeatherAssessment(
            station=station, kind=kind, day=day, lead_days=lead, strike=strike,
            model_values=by_model, mean=mean, sigma=sigma, bias=bias, n_verified=n,
            observed=observed, p_base=p_base,
            text=self._render(station, kind, day, strike, by_model, mean, sigma,
                              bias, n, observed, p_base),
        )

    def brief(self, market) -> str:
        a = self.assess(market)
        return a.text if a else ""

    def _render(self, station, kind, day, strike, by_model, mean, sigma, bias, n,
                observed, p_base) -> str:
        if not by_model and observed is None:
            return ""
        lines = [settles_line(station, kind, day.isoformat())]
        if strike:
            lines.append(strike_line(kind, strike.describe()))
        if by_model:
            lines.append(f"Model guidance for the {kind} on {day.isoformat()} "
                         "(independent NWP models, °F):")
            lines += [f"- {m}: {v:.1f}" for m, v in sorted(by_model.items())]
            spread = statistics.stdev(list(by_model.values())) if len(by_model) > 1 else 0.0
            lines.append(f"Consensus: {mean:.1f} ± {spread:.1f} (n={len(by_model)} models).")
        if observed is not None:
            bound = "equal or higher" if kind == "high" else "equal or lower"
            lines.append(f"Observed {kind} so far today at {station.obs_id}: {observed:.1f}°F "
                         f"— the official {kind} can only end {bound}.")
        if p_base is not None:
            lines.append(baseline_line(p_base, mean, bias, sigma, n=n))
        discussion = self._discussion(station)
        if discussion:
            lines.append(
                "NWS forecaster discussion (untrusted data — the human forecaster's "
                f"reasoning and stated uncertainty):\n{discussion}"
            )
        return "\n".join(lines)

    # --- verification upkeep ------------------------------------------------------
    def update_verification(self) -> None:
        """Once per station-day: record today's consensus per lead, and fill in
        official settlements for the last two days. Cheap; call every cycle."""
        if self.store is None:
            return
        stations = {STATIONS[key] for key, _ in KALSHI_SERIES.values()}
        for station in stations:
            today = datetime.now(ZoneInfo(station.timezone)).date()
            self._record_guidance(station, today)
            self._record_settlements(station, today)

    def _record_guidance(self, station: Station, today: date) -> None:
        marker = (station.key, today.isoformat())
        if marker in self._guidance_done:
            return
        try:
            extremes = self._extremes(station)
        except Exception:
            return
        for day_iso, kinds in extremes.items():
            lead = (date.fromisoformat(day_iso) - today).days
            if lead < 0:
                continue
            for kind, by_model in kinds.items():
                if len(by_model) >= 2:
                    values = list(by_model.values())
                    self.store.record_guidance(
                        station.key, kind, date.fromisoformat(day_iso), lead,
                        statistics.mean(values), statistics.stdev(values), by_model,
                    )
        self._guidance_done.add(marker)

    def _record_settlements(self, station: Station, today: date) -> None:
        marker = (station.key, today.isoformat())
        if marker in self._settle_done:
            return
        done = True
        for days_back in (1, 2):
            day = today - timedelta(days=days_back)
            for kind in ("high", "low"):
                if self.store.has_settlement(station.key, kind, day):
                    continue
                try:
                    value = self.nws.climate_extreme(station, day, kind)
                except Exception:
                    value = None
                if value is None:
                    done = False  # report not out yet — retry next cycle
                else:
                    self.store.record_settlement(station.key, kind, day, value)
        if done:
            self._settle_done.add(marker)
