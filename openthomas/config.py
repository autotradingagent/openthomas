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
    # Ride through a transient endpoint outage instead of dropping the sample.
    # A local vLLM server restarting (model reload, OOM recovery) refuses
    # connections or answers 503 for tens of seconds; retrying with backoff
    # lets a forecast wait it out rather than skipping the whole cycle. Only
    # connection errors and 5xx/429 are retried — a 400 will never fix itself.
    retries: int = 4
    retry_backoff_s: float = 1.5  # exponential: 1.5, 3, 6, 12 … capped at retry_max_s
    retry_max_s: float = 20.0
    # Extra JSON merged into OpenAI-compatible request bodies — e.g. vLLM's
    # chat_template_kwargs to toggle a model's thinking mode.
    extra_body: dict = Field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


class SiteConfig(BaseModel):
    """The build-in-public feed rendered by `openthomas publish`.

    Everything the agent knows is not everything the agent should broadcast:
    the feed builder whitelists fields, and these knobs bound how much of the
    tail it carries.
    """

    x_handle: str = ""  # without the "@"; empty hides X links entirely
    github: str = "https://github.com/PredictionMarketTrader/openthomas"
    huggingface: str = ""  # org page where models trained on the journal are released
    # A serving alias is not a model name: vLLM's --served-model-name can be
    # anything ("og-coding"), and publishing it tells a reader nothing about
    # what the agent thinks with. Name the model, and link its weights.
    model_label: str = ""  # "" falls back to the forecaster's configured model id
    model_url: str = ""  # where the weights live, e.g. a Hugging Face repo
    max_theses: int = 12  # open positions + live edges shown
    max_board: int = 500  # weather markets plotted on the globe
    max_curve_points: int = 500  # equity curve is downsampled to this
    max_reasoning_chars: int = 700  # per-thesis excerpt of model reasoning


class HubConfig(BaseModel):
    """Where the model plane of RSI lives (docs/RSI.md).

    GitHub carries the harness, because a code change is reviewable as a diff.
    Hugging Face carries the weights and the data they were fit on, because a
    weight change is not — the only way to judge it is to see its training set
    and its held-out score.
    """

    org: str = ""  # "" disables every push; nothing leaves the box by default
    dataset: str = ""  # defaults to "<org>/journal"
    private: bool = False


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
    # Forecast prompt template override. None = the built-in default
    # (forecast/engine.py PROMPT). Set by the self-improvement loop when an
    # evolved template clears the kernel gate; hand-editing works too.
    forecast_prompt: str | None = None
    news_enabled: bool = True  # free keyless retrieval (GDELT + Google News RSS)
    news_max_articles: int = 6
    site: SiteConfig = Field(default_factory=SiteConfig)
    hub: HubConfig = Field(default_factory=HubConfig)
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
        data = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
        settings = cls(**data)
        # Layer on the self-improvement loop's active generation — parameter
        # overrides that cleared the kernel gate on replay (docs/RSI.md).
        # Bounds-validated at read; any problem at all means "no overrides",
        # because trading must never fail to start over improvement state.
        try:
            from .improve.genome import active_params, apply_params
            apply_params(settings, active_params(settings.home))
        except Exception:
            pass
        return settings
