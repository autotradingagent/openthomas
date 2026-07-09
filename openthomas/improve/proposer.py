"""Mutation operators: how the evolution loop generates candidate genomes.

Two operators today, one planned (docs/RSI.md):
- **llm**: mine evidence from the journal, ask the reflector endpoint (a
  local model — the improvement loop must run on the operator's own compute)
  for directed parameter mutations. The LLM supplies judgment as JSON; it
  never touches files and its output is clamped into kernel bounds.
- **random**: Gaussian jitter around a parent, drawn from the whole archive,
  not just the champion — directed search exploits, random search keeps the
  population from collapsing onto one hill.
- (planned) **code**: diffs against agent-plane source, same gate.
"""

from __future__ import annotations

import json
import random
import re

from ..kernel.bounds import PARAM_SPACE, clamp_params
from ..memory.journal import Journal
from ..memory.lessons import stats_block

PROPOSE_PROMPT = """You tune the decision-rule parameters of OpenThomas, a \
prediction-market weather-trading agent. The rule: blend the model probability \
with the market price (weight = market_prior_weight on the price), then trade \
only when the blended edge after fees exceeds min_edge.

Current champion parameters:
{params}

Tunable parameters and HARD bounds (values outside are clamped):
{bounds}

Track record (live journal):
{evidence}

Champion on replay (held-in / held-out total PnL over settled weather markets):
{champion}

Propose up to {k} alternative parameter sets worth testing against the champion \
on replay. Move parameters for a stated reason, not for variety's sake; if the \
evidence is thin, propose nothing. Respond with ONLY JSON:
{{"candidates": [{{"params": {{"risk.min_edge": 0.07, \
"risk.market_prior_weight": 0.5}}, "rationale": "<one line of evidence>"}}]}}"""


def mine_evidence(journal: Journal) -> str:
    """Failure-pattern summary from the live journal. No LLM required."""
    parts = [stats_block(journal) or "(no settled trades yet)"]
    recent = journal.recent_settlements(15)
    losers = [s for s in recent if s["pnl"] < 0]
    if losers:
        parts.append("Recent losing settlements:")
        parts += [f"- {s['question'][:70]} (${s['pnl']:+.2f})" for s in losers[:8]]
    return "\n".join(parts)


def _parse_candidates(text: str) -> list[dict]:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return []
    try:
        cands = json.loads(match.group()).get("candidates", [])
        return cands if isinstance(cands, list) else []
    except (json.JSONDecodeError, AttributeError):
        return []


def llm_candidates(complete_fn, champion_params: dict, evidence: str,
                   champion_summary: str, k: int = 3) -> list[dict]:
    """[{params, rationale, proposer}] — full param dicts, clamped, deduped."""
    bounds = "\n".join(f"- {key}: [{b.lo}, {b.hi}]" for key, b in PARAM_SPACE.items())
    response = complete_fn(
        "You tune trading-rule parameters. Be terse, specific, evidence-driven.",
        PROPOSE_PROMPT.format(
            params=json.dumps(champion_params), bounds=bounds,
            evidence=evidence, champion=champion_summary, k=k,
        ),
    )
    out = []
    for cand in _parse_candidates(response)[:k]:
        if not isinstance(cand, dict):
            continue
        params = {**champion_params, **clamp_params(cand.get("params") or {})}
        out.append({"params": params, "proposer": "llm",
                    "rationale": str(cand.get("rationale", ""))[:200]})
    return dedupe(out, champion_params)


def random_candidates(parents: list[dict], k: int = 2,
                      jitter: float = 0.10, rng: random.Random | None = None) -> list[dict]:
    """Gaussian mutations, sigma = `jitter` × the bound's width, clamped."""
    rng = rng or random.Random()
    out = []
    for _ in range(k):
        parent = rng.choice(parents)
        params = {}
        for key, bound in PARAM_SPACE.items():
            base = float(parent.get(key, (bound.lo + bound.hi) / 2))
            params[key] = round(bound.clamp(
                rng.gauss(base, jitter * (bound.hi - bound.lo))), 4)
        out.append({"params": params, "proposer": "random",
                    "rationale": "archive mutation"})
    return out


def dedupe(candidates: list[dict], champion_params: dict) -> list[dict]:
    seen = {_key(champion_params)}
    out = []
    for cand in candidates:
        key = _key(cand["params"])
        if key not in seen:
            seen.add(key)
            out.append(cand)
    return out


def _key(params: dict) -> tuple:
    return tuple(sorted((k, round(float(v), 4)) for k, v in params.items()))
