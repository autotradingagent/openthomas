"""The journal as a training set.

One row per settled market: the *first* forecast made on it, the market price
it was formed against, and how the world actually resolved. This is the model
plane's raw material (docs/RSI.md) — the harness lives on GitHub, the weights
and the data they were fit on live on Hugging Face.

Three rails are baked into the artifact rather than left to whoever trains on
it, because each of them, forgotten once, produces a model that looks brilliant
and is worthless:

1. **Settled only.** An open position has no label, and publishing our live
   view of one would hand away the trade. A settled market has no alpha left;
   it is evidence.
2. **First forecast per market.** Later forecasts on the same market saw the
   price move and the day advance. Training on them teaches hindsight.
3. **Temporal split, carried in the data.** `split` is assigned by time, not
   sampled — the most recent slice is validation. A random split lets a model
   validate on days it trained on and will lie to you about its Brier score.

`news` is deliberately absent. The live prompt carried third-party headlines we
have no right to redistribute; `had_news` records that the model saw them, so
nobody mistakes this for a faithful prompt replay.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from ..memory.journal import Journal

SCHEMA_VERSION = 1

# Time-discounted reward, R·exp(-λ·days_to_close) (docs/RSI.md). A call made
# far from resolution is made under more uncertainty; λ says how much less a
# late, easy win is worth. 0.05/day ≈ a two-week half-life. Weather markets
# resolve within days, so this barely bites there — it matters when the same
# pipeline is pointed at slower categories.
REWARD_LAMBDA = 0.05

VALID_FRACTION = 0.2

QUERY = """
SELECT f.ts AS forecast_ts, f.market_id, f.platform, f.question, f.category,
       f.p_raw, f.p_calibrated, f.confidence, f.base_rate, f.mid,
       f.market_gap_reason, f.invalidation, f.reasoning, f.model,
       f.data_text, f.news_text,
       s.ts AS settled_ts, s.outcome, s.pnl, s.cost_basis
FROM settlements s
JOIN forecasts f ON f.id = (
  SELECT id FROM forecasts WHERE market_id = s.market_id ORDER BY ts LIMIT 1)
ORDER BY f.ts
"""


def _days(a: str, b: str) -> float:
    return max((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds(), 0) / 86400


def build(journal: Journal, lam: float = REWARD_LAMBDA,
          valid_fraction: float = VALID_FRACTION,
          aliases: dict[str, str] | None = None) -> list[dict]:
    """Ordered oldest-first, with `split` already assigned by time.

    `aliases` maps a serving alias to the model's public name. The journal
    records whatever the endpoint answered to — under vLLM that is
    `--served-model-name`, which can be anything — and shipping "og-coding" to
    a public dataset names nothing. Callers pass `{endpoint_id: model_label}`.
    """
    aliases = aliases or {}
    raw = journal.db.execute(QUERY).fetchall()
    rows = []
    for r in raw:
        held = _days(r["forecast_ts"], r["settled_ts"])
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "ts": r["forecast_ts"],
            "settled_ts": r["settled_ts"],
            "market_id": r["market_id"],  # settled: verifiable against the venue
            "platform": r["platform"],
            "question": r["question"],
            "category": r["category"] or "",
            # The prompt's data block as the model saw it. Empty for markets
            # forecast before the journal archived prompt inputs — those rows
            # carry a label but no features, and `trainable` excludes them.
            "data": r["data_text"] or "",
            "had_news": bool(r["news_text"]),
            "p_market": r["mid"],
            "p_forecast": r["p_raw"],
            "p_calibrated": r["p_calibrated"],
            "confidence": r["confidence"],
            "base_rate": r["base_rate"],
            "why": r["market_gap_reason"] or "",
            "invalidation": r["invalidation"] or "",
            "reasoning": r["reasoning"] or "",
            "forecaster": aliases.get(r["model"] or "", r["model"] or ""),
            "outcome": 1 if r["outcome"] == "yes" else 0,
            "pnl": r["pnl"],
            "cost_basis": r["cost_basis"],
            "days_to_close": round(held, 4),
            "reward": round(r["pnl"] * math.exp(-lam * held), 4),
        })

    # Temporal split: the tail in time is validation. Assigned here so the
    # artifact cannot be reshuffled downstream into a leak.
    cut = len(rows) - int(len(rows) * valid_fraction)
    for i, row in enumerate(rows):
        row["split"] = "train" if i < cut else "validation"
    return rows


def trainable(rows: list[dict]) -> list[dict]:
    """Rows whose prompt can actually be rebuilt. A label without features
    teaches nothing; counting them as training data flatters the dataset."""
    return [r for r in rows if r["data"]]


def write_jsonl(rows: list[dict], out: str | Path) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def summary(rows: list[dict]) -> dict:
    usable = trainable(rows)
    return {
        "rows": len(rows),
        "trainable_rows": len(usable),
        # `mid` was added to the journal after the first settlements, so early
        # rows have no price to score an edge against. The card says so rather
        # than letting a reader assume every row carries one.
        "rows_with_market_price": sum(r["p_market"] is not None for r in rows),
        "train": sum(r["split"] == "train" for r in rows),
        "validation": sum(r["split"] == "validation" for r in rows),
        "span": (rows[0]["ts"][:10], rows[-1]["ts"][:10]) if rows else None,
        "base_rate": round(sum(r["outcome"] for r in rows) / len(rows), 4) if rows else None,
        "platforms": sorted({r["platform"] for r in rows}),
        "forecasters": sorted({r["forecaster"] for r in rows if r["forecaster"]}),
    }
