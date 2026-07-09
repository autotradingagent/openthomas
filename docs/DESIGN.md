# OpenThomas Architecture

> An autonomous AI agent that finds edge, manages risk, and trades prediction markets
> (Polymarket, Kalshi) on your behalf. You give it capital, a risk profile, and a goal.
> It does the rest — and learns from every settled trade.

## Design thesis

The Prediction Arena benchmark (arXiv:2604.07355) ran six frontier LLMs as autonomous
traders with real capital for 57 days. **Every one of them lost money** (-16% to -30.8%
on Kalshi). The single profitable run on record (+10.9%) had a distinct signature:

| Trait | Losing models | The one profitable run |
|---|---|---|
| Settlement win rate | 15–52% | 55.2% |
| Win/loss asymmetry | roughly symmetric | +$63.89 avg win vs −$3.23 avg loss |
| Max drawdown | up to 30.9% | 4.1% |
| Behavior | overtrading, early exits, concentration in unfamiliar categories | selective, held winners, cut losers |

The paper's success-factor hierarchy: (1) initial prediction accuracy, (2) sizing up when
correct, (3) sizing down when uncertain, (4) exit quality. Research *volume* had zero
correlation with returns.

**Conclusion: the LLM is not the edge. The harness is the edge.** OpenThomas is designed
as a disciplined trading harness around a forecasting brain:

1. **Selectivity over activity.** Only trade when `|forecast − price|` clears an
   explicit expected-value threshold after fees and spread. "No trade" is the default
   action, not a failure mode.
2. **Hard risk constraints outside the LLM.** Position sizing (fractional Kelly),
   per-market concentration caps, category exposure caps, drawdown kill-switch, and
   solvency checks are deterministic code. The model proposes; the risk engine disposes.
3. **Asymmetry by construction.** Prefer holding winners to settlement (the paper shows
   early exits systematically underperform), cut losers on thesis invalidation, never
   average down without a re-forecast.
4. **Calibration as a first-class metric.** Every forecast is logged; Brier score and
   calibration curves are tracked per category; a calibration layer (Platt scaling)
   corrects the model's systematic biases before sizing.
5. **Learning loop.** Settled trades become training data: post-mortems distill into
   lessons (memory), strategy parameters re-tune from the journal, and — for users with
   GPUs — local models fine-tune on the accumulated forecast/outcome pairs.
6. **Self-improvement as a gated loop.** OpenThomas updates OpenThomas: an evolution
   loop mines its own journal, proposes bounded parameter mutations, and must clear a
   kernel-owned promotion gate on leak-free replay before anything takes effect —
   with lineage, audit log, and automatic rollback. See [RSI.md](RSI.md).

## Layers

```
┌─────────────────────────────────────────────────────────┐
│ CLI / Dashboard          openthomas run|scan|report|vital│
├─────────────────────────────────────────────────────────┤
│ Agent Orchestrator       trading cycle loop, scheduling  │
├───────────────┬──────────────────┬──────────────────────┤
│ Edge Scanner  │ Forecast Engine  │ Memory & Skills      │
│ mispricing,   │ LLM ensemble +   │ journal, lessons,    │
│ cross-market  │ calibration      │ skill files, post-   │
│ arb, filters  │ layer            │ mortems              │
├───────────────┴──────────────────┴──────────────────────┤
│ Strategy Layer           pluggable: fundamental, arb,   │
│                          market-making (skills = files) │
├─────────────────────────────────────────────────────────┤
│ Risk Engine              Kelly sizing, caps, kill-switch│
│                          (deterministic, non-LLM)       │
├─────────────────────────────────────────────────────────┤
│ Execution                limit orders, liquidity checks │
├─────────────────────────────────────────────────────────┤
│ Market Connectors        Polymarket │ Kalshi │ Paper    │
└─────────────────────────────────────────────────────────┘
```

### Market connectors (`openthomas/markets/`)

A unified `Market` / `Order` / `Position` model over:

- **Polymarket** — Gamma API for market discovery/metadata, CLOB API for books and
  orders (py-clob-client), USDC on Polygon.
- **Kalshi** — trade-api/v2 REST + WebSocket, RSA-signed requests, CFTC-regulated.
- **Paper** — wraps either real connector's *market data* but simulates fills locally
  with a conservative slippage model (fills at bid/ask, not mid). Default mode.

Mark-to-market always uses **bid prices** (liquidation value), following the paper.

### Forecast engine (`openthomas/forecast/`)

Pipeline per market question: retrieve news/context → extract base rates → independent
probability estimates from N model calls (different framings) → aggregate (trimmed
mean) → calibration correction from the user's own logged history. Provider-agnostic:
Anthropic/OpenAI-compatible APIs, including local endpoints (Ollama, vLLM) for
self-hosted models such as Gemma-class open weights.

### Edge scanner (`openthomas/edge/`)

Finds candidate trades *before* spending forecast tokens:

- **Mispricing candidates**: liquidity/volume/time-to-resolution filters, category
  whitelist from the user's risk profile, skip markets within the no-edge band.
- **Cross-platform arbitrage**: same real-world event listed on both Polymarket and
  Kalshi with a price gap > combined fees/spread.
- **Multi-outcome coherence**: outcome sets whose prices sum ≠ 1 beyond fee bounds.
- **Structural filters** from the literature: longshot-bias zones, resolution-rule
  traps (flag markets whose rules text diverges from the headline question).

### Risk engine (`openthomas/risk/`)

Deterministic. Inputs: bankroll, user risk profile (conservative / moderate /
aggressive → parameter set), forecast probability + confidence, market price/liquidity.
Outputs: approved size or rejection, with reason.

- Fractional Kelly (profile-scaled: 0.1×–0.33×) capped by per-market concentration
  (default 5–15% of bankroll at cost basis), per-category exposure, and open-risk total.
- Solvency check including fees before every order.
- Drawdown kill-switch: trading halts at max-drawdown threshold; requires human resume.
- Correlated-settlement guard: cap aggregate exposure to positions that resolve on the
  same underlying event/date (the paper's biggest single-session losses came from
  correlated positions settling together).

### Memory & self-improvement (`openthomas/memory/`, `skills/`)

- **Journal**: every forecast, order, fill, and settlement in SQLite. The source of
  truth for PnL, calibration, and training data.
- **Lessons**: after each settlement batch, a reflection pass writes/updates lesson
  files (markdown) that are injected into future forecast prompts — the same
  memory-file pattern as OpenClaw/Hermes-style personal agents.
- **Skills**: strategies and domain playbooks are markdown files with YAML frontmatter
  (`skills/`). Users can add, edit, share them; the agent can draft new ones from its
  own post-mortems (human-approved before activation).
- **Calibration store**: forecast/outcome pairs per category → Platt scaling params.

### Local training (`scripts/train/`)

For users with GPUs: LoRA fine-tuning recipes that turn the journal into a
calibration-tuned local forecaster (e.g. Gemma-class 12B/27B+ models), evaluated
against the user's own held-out settled markets before being allowed to drive sizing.

## Safety rails (non-negotiable defaults)

- Paper trading is the default; live mode requires explicit `--live` plus a funded-cap
  config value.
- The agent can never deposit, withdraw, or move funds — only trade within the
  allocated bankroll.
- Kill-switch on drawdown; heartbeat file so a crashed loop never leaves naked orders.
- Full audit log of every decision with the forecast and risk-check that justified it.
