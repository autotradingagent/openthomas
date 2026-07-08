"""User configuration: bankroll, goal, risk profile, model endpoints.

Loaded from ~/.openthomas/config.yaml (created by `openthomas init`), overridable
with environment variables for secrets.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

HOME = Path(os.environ.get("OPENTHOMAS_HOME", Path.home() / ".openthomas"))


class RiskProfile(BaseModel):
    """Deterministic risk parameters. Presets: conservative / moderate / aggressive."""

    name: str = "conservative"
    kelly_fraction: float = 0.15  # fraction of full-Kelly to size
    max_position_frac: float = 0.05  # of bankroll, at cost basis, per market
    max_event_frac: float = 0.10  # per correlated event group
    max_category_frac: float = 0.25  # per category
    max_open_risk_frac: float = 0.50  # total cost basis of open positions
    max_drawdown: float = 0.15  # kill-switch: halt trading past this peak-to-trough
    min_edge: float = 0.08  # |calibrated forecast − price| after fees, else no trade
    # Weight on the market price when forming the tradable probability:
    # p_trade = w·market + (1−w)·model. Bridgewater's AIA Forecaster showed pure
    # LLM forecasts lose to market consensus but a market-heavy blend beats it
    # (arXiv:2511.07678). 0 = trust the model outright.
    market_prior_weight: float = 0.5
    min_confidence: float = 0.55  # forecaster self-reported confidence gate
    min_liquidity: float = 1000.0  # USD book depth to consider a market
    min_price: float = 0.05  # avoid longshot-bias zones
    max_price: float = 0.95
    max_trades_per_cycle: int = 3
    max_forecasts_per_cycle: int = 12  # LLM budget per cycle
    categories_blocked: list[str] = Field(default_factory=list)

    @staticmethod
    def preset(name: str) -> "RiskProfile":
        presets = {
            "conservative": {},
            "moderate": {
                "kelly_fraction": 0.25, "max_position_frac": 0.08, "max_drawdown": 0.20,
                "min_edge": 0.06, "max_trades_per_cycle": 5,
            },
            "aggressive": {
                "kelly_fraction": 0.33, "max_position_frac": 0.12, "max_event_frac": 0.15,
                "max_drawdown": 0.30, "min_edge": 0.05, "max_trades_per_cycle": 8,
            },
        }
        if name not in presets:
            raise ValueError(f"unknown risk profile {name!r}; use one of {list(presets)}")
        return RiskProfile(name=name, **presets[name])


class ModelConfig(BaseModel):
    """LLM endpoint for forecasting. OpenAI-compatible or Anthropic.

    For local models (Ollama, vLLM, llama.cpp server) set provider="openai" and
    base_url to the local endpoint, e.g. http://localhost:11434/v1 for Ollama.
    """

    provider: str = "anthropic"  # "anthropic" | "openai" | "claude-cli" | "codex-cli"
    model: str = "claude-sonnet-5"
    base_url: str | None = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    command: str | None = None  # CLI providers: override the binary path
    ensemble_size: int = 3  # independent forecast samples aggregated per question
    temperature: float = 0.7
    timeout_s: float = 600.0  # reasoning models on local GPUs can take minutes
    max_tokens: int = 4096
    # Extra JSON merged into OpenAI-compatible request bodies — e.g. vLLM's
    # chat_template_kwargs to toggle a model's thinking mode.
    extra_body: dict = Field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


class Settings(BaseModel):
    bankroll: float = 1000.0  # USD the agent may deploy; it can never exceed this
    goal: str = "Grow the bankroll steadily; protecting capital beats chasing returns."
    mode: str = "paper"  # "paper" | "live"
    # "weather": only station-temperature and weather-tagged markets (the
    # product focus); "all": scan every market like a generalist.
    focus: str = "weather"
    platforms: list[str] = Field(default_factory=lambda: ["polymarket", "kalshi"])
    risk: RiskProfile = Field(default_factory=RiskProfile)
    forecaster: ModelConfig = Field(default_factory=ModelConfig)
    # Lesson distillation runs high-token but low-difficulty work; point it at
    # a cheap/local endpoint independently of the forecaster. None = same.
    reflector: ModelConfig | None = None
    cycle_minutes: int = 30
    # How far the LLM may move a weather market's statistical baseline: the
    # blend thesis again — the model adjusts the statistics, never replaces them.
    weather_anchor_delta: float = 0.15
    news_enabled: bool = True  # free keyless retrieval (GDELT + Google News RSS)
    news_max_articles: int = 6
    home: Path = HOME

    @property
    def db_path(self) -> Path:
        return self.home / "journal.db"

    @property
    def skills_dir(self) -> Path:
        return self.home / "skills"

    @property
    def lessons_dir(self) -> Path:
        return self.home / "lessons"

    def save(self) -> Path:
        self.home.mkdir(parents=True, exist_ok=True)
        path = self.home / "config.yaml"
        data = self.model_dump(mode="json", exclude={"home"})
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        return path

    @classmethod
    def load(cls) -> "Settings":
        path = HOME / "config.yaml"
        if path.exists():
            data = yaml.safe_load(path.read_text()) or {}
            return cls(**data)
        return cls()
