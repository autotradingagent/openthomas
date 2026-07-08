# OpenThomas — the weather trader for prediction markets

[![PyPI](https://img.shields.io/pypi/v/openthomas)](https://pypi.org/project/openthomas/) [![CI](https://github.com/autotradingagent/openthomas/actions/workflows/ci.yml/badge.svg)](https://github.com/autotradingagent/openthomas/actions) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://pypi.org/project/openthomas/)

**An autonomous AI agent that trades weather markets on Kalshi and Polymarket. It knows the exact NWS station each market settles on, builds a probability for every strike from a seven-model forecast consensus, learns each station's systematic bias from months of leak-free hindcasts, lets an LLM adjust — within hard bounds — for what the statistics can't see, and sizes every position with fractional Kelly under risk limits no model can override. It learns from every settled degree.**

```bash
pip install openthomas
openthomas init --bankroll 1000 --risk conservative
openthomas hindcast   # teach the baseline each station's bias — before the first trade
openthomas run        # paper trading on live market prices — the default
```

No wallet, no exchange API keys needed to start: paper mode simulates fills against real Polymarket + Kalshi order books. You only connect real money after you've watched it trade.

> ⚠️ **Prediction market trading can lose all the money you allocate.** OpenThomas ships with paper trading as the default, hard position limits, and a drawdown kill-switch — but no software makes trading safe. Roughly 70% of Polymarket addresses have lost money. This is not financial advice.

---

## Why another trading bot? Because the LLM is not the edge.

The [Prediction Arena benchmark](https://arxiv.org/abs/2604.07355) gave six frontier AI models $10,000 each and let them trade real prediction markets autonomously for 57 days. **Every single one lost money** — between −16% and −30.8%. Research volume didn't matter. Token spend didn't matter. What separated the least-bad from the worst:

1. **Initial prediction accuracy** — being right, early
2. **Sizing up when correct, down when uncertain**
3. **Exit discipline** — early exits systematically underperformed holding to settlement
4. **Not trading at all** when there's no edge

The only profitable run on record (+10.9%) was selective (112 trades), asymmetric (avg win $63.89 vs avg loss $3.23), and low-drawdown (4.1%). That profile is a *harness* property, not a model property. OpenThomas is that harness — pointed at the one market category where the harness can actually be measured.

## Why weather?

- **Settlement is daily, objective, and public.** Every temperature market resolves against the NWS Climatological Report for one named station. No ambiguous resolution rules, no months of waiting: dozens of ground-truth data points per day. It's the fastest learning loop in prediction markets.
- **The edge is calibration, not news.** Markets settle on a *specific station* (NYC = Central Park, Chicago = Midway). Global forecast models have systematic, learnable biases at each one — our first 90-day hindcast found Miami consensus running **1.8–2.1°F cold** (σ 1.7), Philadelphia **+2.3°F**, LAX **1°F warm**. That's a moat you train from public data, per station, on your machine.
- **The market lags the physics.** Forecast models update every 6–12 hours; order books take time to digest each run. A baseline that reprices on fresh guidance — truncated by what today's thermometer has already locked in — is structurally ahead of casual flow.

And the receipts, because build-in-public means the losing numbers too: our replay backtest (21 days, real Kalshi order books, fees included) started at **−$0.038/contract** for a naked model-vs-market strategy. Adding learned station bias and the market-prior blend: −$0.018. Fixing a timing leak (never let the replayed market know the morning low before the model does): **−$0.000 — breakeven** — for a baseline deliberately stripped of every live advantage (same-day model runs, intraday observations, LLM adjustment, calibration). Each layer of discipline was worth ~2¢ a contract. The live agent trades with all four advantages switched on; whether they clear the bar is what the public paper run is for.

| Layer | What it does | Who's in control |
|---|---|---|
| **Edge scanner** | Filters markets down to plausible mispricings; flags cross-platform arbitrage (Polymarket vs Kalshi price gaps) | deterministic code |
| **Weather baseline** | Seven independent NWP models → per-strike probability at the exact settlement station, with per-(station, lead) bias/σ learned from leak-free hindcasts and hard truncation by today's observed extreme | deterministic code |
| **Forecast engine** | LLM ensemble (median of N independent estimates), grounded in resolution rules and base rates — and clamped to baseline ± 0.15 on weather markets: the model adjusts the statistics, it never replaces them | your choice of model |
| **Calibration layer** | Platt-scales raw forecasts against *your own* settled-trade history; blends with the market price ([pure LLM forecasts lose to market consensus; a blend beats it](https://arxiv.org/abs/2511.07678)) | deterministic code |
| **Risk engine** | Fractional Kelly sizing, per-market / per-event / per-category caps, fee-aware EV threshold, longshot-zone filter, drawdown kill-switch | deterministic code — **the model proposes, the risk engine disposes** |
| **Memory** | Every forecast and fill journaled to SQLite; post-settlement reflection distills lessons that feed back into future prompts | agent, human-auditable |

## What it looks like

```
$ openthomas run --once
──────────── cycle · account $99,962.22 · cash $99,711.90 ────────────
markets 414 → candidates 64 → forecasts 8 → trades 1
  TRADE BUY 610 YES @ 0.41 [kalshi] Will the high temp in NYC be >83° on Jul 8?
  skip: Will the high temp in Philadelphia be 87-88°?: edge +0.049 below threshold 0.080
  skip: Will the high temp in Chicago be 88-89°?: confidence 0.45 too low
```

```
$ openthomas hindcast --days 90
✓ KMIA: +900 guidance, +178 settlements
✓ KNYC: +900 guidance, +178 settlements
...
KMIA  high L1  bias +1.8 σ 1.7   ← Miami models run 2°F cold; the baseline prices it in
KPHL  high L1  bias +2.3 σ 2.1
```

`openthomas report` scores every settled forecast against the market price it traded at, bucketed by station × lead — Brier skill > 0 in a bucket is the only license to size up there. `openthomas replay` backtests the decision rule against real historical order books, with bias learned strictly out-of-window — no peeking. (The example trade above is illustrative; the hindcast and replay numbers in this README are real runs from 2026-07-08.)

`openthomas vital` renders a shareable performance card (equity curve, win rate, Brier score, max drawdown) — post your track record, good or bad.

## Quickstart

```bash
pip install openthomas

# 1. Configure: bankroll, risk appetite, goal, forecasting model
openthomas init --bankroll 1000 --risk conservative \
  --goal "Grow steadily; protecting capital beats chasing returns"
export ANTHROPIC_API_KEY=sk-ant-...   # or any OpenAI-compatible endpoint

# 2. Teach the baseline: 90 days of leak-free per-station history in one command
openthomas hindcast

# 3. See what it sees, then backtest the decision rule on real order books
openthomas scan
openthomas replay --days 21

# 4. Let it trade (paper mode: real prices, simulated fills)
openthomas run

# 5. Check in
openthomas report
openthomas vital
```

OpenThomas is weather-first, not weather-only: set `focus: all` in `~/.openthomas/config.yaml` to scan every market like a generalist.

**Local / self-hosted models** (no API bill, full privacy):

```bash
openthomas init --provider openai --base-url http://localhost:11434/v1 --model gemma3:12b
```

Any OpenAI-compatible server works — Ollama, vLLM, llama.cpp. If you have GPUs, [docs/TRAINING.md](docs/TRAINING.md) covers fine-tuning a local model on your own trade journal (calibration LoRA) so the agent's forecaster improves on *your* markets.

**Subscriptions instead of API credits**: set `provider: claude-cli` (Claude Code) or `provider: codex-cli` (OpenAI Codex) in `config.yaml` and OpenThomas bills your existing Claude/ChatGPT subscription. Every LLM node is configured independently — run the high-token reflection pass on a local server and route only the hardest forecasts to a frontier model:

```yaml
forecaster: { provider: openai, model: glm-5.2, base_url: "http://localhost:8000/v1" }
reflector:  { provider: claude-cli, model: sonnet }
```

## Use it from Claude (MCP)

Your agent brings the market view; OpenThomas enforces the discipline:

```bash
pip install 'openthomas[mcp]'
claude mcp add openthomas -- openthomas-mcp
```

Now Claude (or OpenClaw, Hermes, any MCP client) can scan markets, pull news,
and `propose_trade` — every proposal is blended with the market price, sized
with fractional Kelly, and checked against hard exposure caps before a paper
fill. Rejections name the binding constraint. See [docs/MCP.md](docs/MCP.md).

## Risk profiles

You hand OpenThomas a mandate, not a suggestion — these are enforced in deterministic code, outside the LLM:

| | conservative | moderate | aggressive |
|---|---|---|---|
| Kelly fraction | 0.15× | 0.25× | 0.33× |
| Max per market | 5% | 8% | 12% |
| Min edge after fees | 8¢ | 6¢ | 5¢ |
| Drawdown kill-switch | 15% | 20% | 30% |
| Max trades/cycle | 3 | 5 | 8 |

Plus, in every profile: per-event correlation caps (correlated settlements caused the largest single-session losses in Prediction Arena), longshot-zone filter (contracts under 10¢ lose >60% of stake on average — [Whelan et al. 2026](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5502658)), solvency checks including fees, and mark-to-market at bid prices (liquidation value, not hope).

## How it learns

- **Journal** — every forecast (with the market price it disagreed with), fill, and settlement in SQLite (`~/.openthomas/journal.db`). Your data, on your machine.
- **Verification store** — every cycle records the model consensus per station and lead; official settlements are backfilled from the NWS report. Station bias and spread are re-estimated continuously; `openthomas hindcast` seeds it with 90 days on day one.
- **Calibration** — once ≥30 markets settle, OpenThomas fits Platt scaling per category to correct your model's systematic bias.
- **Playbook** — after settlements, a reflection pass proposes add/revise/deprecate *operations* on structured rules, each carrying scope, evidence, and a post-adoption track record. Code enforces the caps (8 active rules max); rules that go negative get flagged and retired with the reason kept on file. No wholesale rewrites, no silently vanishing lessons.
- **Skills** — strategies and domain playbooks are markdown files (`skills/`), the same pattern as OpenClaw / Hermes / Claude skills. Add your own, or let the agent draft one from its post-mortems for you to approve.

Everything the agent learns — journal, verification history, calibration, playbook, any LoRA you train — lives in `~/.openthomas/`, not in this repo. The code is open source; **your trained strategy and its track record are yours.**

## FAQ

**Can an AI bot actually make money on Polymarket or Kalshi?**
Unproven for pure LLM forecasting — every frontier model tested in live benchmarks lost money. Our own replay backtest says the same thing in miniature: a naked model-vs-market strategy lost $0.038/contract; only after station-bias learning, market-prior blending, and timing hygiene did it reach breakeven — *before* the live-only advantages (fresh model runs, intraday observations, calibration) are counted. The honest claim: the edges are structural and small, discipline is what keeps them, and weather is where they're most measurable. Paper-trade first. Expect losses.

**Which markets does it trade by default?**
Daily station-temperature markets on Kalshi (NYC, LA, Miami, Chicago, Philadelphia, Denver, Austin, Dallas, Atlanta, San Antonio — highs and lows) plus weather-tagged Polymarket markets. `focus: all` restores generalist scanning.

**Do I need API keys to try it?**
No exchange keys. Paper mode uses public market data from both venues. You need an LLM (API key or a local model via Ollama) for forecasting.

**Which venues?**
Polymarket (global, crypto-settled in pUSD) and Kalshi (US, CFTC-regulated). Kalshi's demo exchange is supported (`KALSHI_DEMO=1`). Live Polymarket trading from the US is geoblocked to close-only on the offshore CLOB; Polymarket US is a separate venue (integration planned).

**Can it withdraw or move my funds?**
No. The agent only trades within the bankroll you allocate. It has no withdrawal, transfer, or deposit capability — and live mode requires two independent explicit switches.

**What models work best?**
On weather markets the statistical baseline does the heavy lifting, so the LLM matters less than you'd think — its job is bounded adjustment for discontinuities (blown fronts, stale consensus), which mid-size local reasoning models handle. Any Anthropic / OpenAI-compatible endpoint or a Claude/ChatGPT subscription CLI works; see [docs/TRAINING.md](docs/TRAINING.md) for fine-tuning on your own journal.

## Documentation

- [Architecture & design rationale](docs/DESIGN.md)
- [The edge playbook — documented inefficiencies with sources](docs/EDGE.md)
- [Trading against agents — the adversarial playbook](docs/ADVERSARIAL.md)
- [Live trading setup (Kalshi, Polymarket)](docs/LIVE_TRADING.md)
- [Training a local forecaster on your journal](docs/TRAINING.md)

## Roadmap

- [x] Paper trading loop on live Polymarket + Kalshi data
- [x] LLM ensemble forecasting, calibration, market-prior blending
- [x] Deterministic risk engine (Kelly, caps, kill-switch)
- [x] Weather focus: settlement-station registry, NWS report parsing, 7-model baseline
- [x] Leak-free hindcast — 90 days of per-station bias/σ in one command
- [x] Replay backtests against real historical order books
- [x] Structured playbook (curated rule operations with track records)
- [x] Bucketed Brier skill reporting (station × lead, model vs market)
- [x] Model backends: Anthropic / OpenAI-compatible / Claude & ChatGPT subscription CLIs
- [x] Cross-platform arbitrage scanner
- [x] News retrieval pipeline for forecasts (GDELT + Google News, keyless)
- [x] MCP server — Claude/any agent brings the view, OpenThomas enforces risk ([docs](docs/MCP.md))
- [ ] AI-NWP ensemble inference on local GPUs (GraphCast / Aurora / AIFS open weights)
- [ ] Station post-processing models trained on hindcast data
- [ ] Live Kalshi execution hardening (order amend/cancel, WebSocket fills)
- [ ] Polymarket live execution via official SDK (pUSD)
- [ ] Market-making on temperature ladders (maker rebates, no taker fees)
- [ ] Journal → LoRA fine-tuning recipes for local models
- [ ] Web dashboard, trade-in-public feed (vitals live, fills T+1)
- [ ] Public community leaderboard of (opt-in) vitals

## Contributing

The interesting problems are open: retrieval quality, resolution-rule matching for cross-venue arbitrage, calibration under small samples, market-making without adverse selection. PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). If you run OpenThomas, share your `vital` card in [Discussions](https://github.com/autotradingagent/openthomas/discussions) — including the losses; honest track records are how this gets better.

## License

MIT. Trade at your own risk; comply with your local laws and each venue's terms.
