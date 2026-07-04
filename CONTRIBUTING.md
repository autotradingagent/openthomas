# Contributing to OpenThomas

## Setup

```bash
git clone https://github.com/autotradingagent/openthomas
cd openthomas
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Ground rules

- **The risk engine is sacred.** Anything under `openthomas/risk/` must stay
  deterministic, fully unit-tested, and free of LLM calls. PRs that let the
  model influence sizing or caps will be rejected regardless of backtests.
- **Paper mode is the default.** No change may make live trading easier to
  reach than two explicit steps.
- **Honesty in docs.** No profitability claims without a reproducible journal.
- Match the existing style (ruff, line length 100). Add tests for new logic.

## Where help is most valuable

- News retrieval for the forecast engine (the #1 accuracy lever in the research)
- Resolution-rule matching for cross-venue arbitrage (the hard 20%)
- Kalshi live execution hardening; Polymarket official-SDK integration
- Market-making strategy with adverse-selection controls
- Training scripts (`scripts/train/`): LoRA fine-tune + honest eval

## Releasing (maintainers)

Publishing is automated via PyPI trusted publishing — no tokens anywhere:

```bash
# 1. bump `version` in pyproject.toml, commit, push
# 2. tag and push the tag — CI does the rest (test → build → PyPI → GH release)
git tag v0.2.0 && git push origin v0.2.0
```

The workflow refuses to publish if the tag doesn't match the pyproject version.

## AI-agent contributors

PRs written with coding agents are welcome. Keep diffs focused, include the
reasoning in the PR body, and never touch `openthomas/risk/` and connector
order paths in the same PR.
