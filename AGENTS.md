# OpenThomas — guide for coding agents

Autonomous AI trading agent for prediction markets (Polymarket, Kalshi).
Python 3.10+, package in `openthomas/`, tests in `tests/` (pytest), deps via
`pip install -e ".[dev]"`, lint with `ruff check`.

## Map

- `openthomas/agent/loop.py` — the trading cycle orchestrator
- `openthomas/markets/` — connectors (`base.py` = unified model; `paper.py` =
  simulated fills at bid/ask on real data)
- `openthomas/forecast/` — LLM ensemble engine + Platt calibration
- `openthomas/research/news.py` — keyless news retrieval (GDELT, Google News RSS)
- `openthomas/edge/scanner.py` — pre-LLM filters + cross-platform arb detection
- `openthomas/risk/engine.py` — Kelly sizing, caps, drawdown kill-switch
- `openthomas/memory/` — SQLite journal + lesson distillation + LLM token ledger
- `openthomas/kernel/` — parameter bounds + promotion gate for self-improvement
  (kernel plane: operator-owned, see docs/RSI.md)
- `openthomas/improve/` — the evolution loop: propose → gate → promote/rollback
- `openthomas/site/` — the public build-in-public feed (`feed.json`); the
  static page that renders it is `site/`, deployed by `deploy/` (docs/SITE.md)
- `openthomas/train/` — the journal as a leak-free training set, and the
  Hugging Face push for datasets and adapters (docs/TRAINING.md). The harness
  ships to GitHub, the model plane ships to HF (docs/RSI.md).
- `openthomas/cli.py` — typer CLI
  (`init/scan/run/report/vital/publish/improve/dataset/push-model`)
- `openthomas/mcp_server.py` — MCP server (`openthomas-mcp`); paper-only by design
- `docs/DESIGN.md` — architecture rationale; `docs/EDGE.md` — strategy basis

## Hard rules

- `openthomas/risk/` must stay deterministic: no LLM calls, no learned
  parameters, full unit coverage. The model proposes; the risk engine disposes.
  Precisely: the engine code and every safety rail (Kelly fraction, exposure
  caps, drawdown kill-switch, price-zone and liquidity floors, trade-rate
  caps) are never evolved. The two entry-*selectivity* knobs that happen to
  live on RiskProfile — `min_edge` and `market_prior_weight` — are strategy
  parameters, gate-evolvable inside operator-set bounds (docs/RSI.md);
  `tests/test_improve.py::test_param_space_never_touches_safety_rails`
  enforces the line.
- Paper mode stays the default; live trading keeps requiring both
  `mode: live` in config AND the `--live` flag.
- Prices are probabilities in [0,1] for the YES side everywhere; Kalshi's
  dollar-string fields are converted at the connector boundary.
- Mark-to-market uses bid (liquidation value), never mid.
- Don't touch `openthomas/risk/` and connector order paths in the same PR.
- `openthomas/kernel/`, `weather/verification.py`, and
  `weather/replay.py::collect_rows` are the kernel plane (docs/RSI.md): the
  self-improvement loop must never gain a write path to them, directly or
  indirectly. The LLM proposer returns JSON only; all file writes stay in
  deterministic gate-checked code.
- Nothing public is published without its evidence. `train/hub.py::push_adapter`
  refuses weights that lack a held-out score and the dataset revision they were
  fit on; `site/feed.py` whitelists every field it emits. Both are tested — the
  rails are the product, not the plumbing.

## Verify

`pytest -q` must pass; `openthomas scan` (no keys needed) is the live
end-to-end smoke test against real venue APIs.
