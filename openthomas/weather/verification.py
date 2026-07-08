"""Forecast verification store: model guidance recorded at forecast time,
settlements filled in from the CLI report, per-(station, kind, lead) error
statistics out.

This is the learning substrate for the temperature baseline: bias and spread
are estimated from *our own* recorded guidance vs official settlements, with
shrinkage toward climatological priors while the sample is small. The
hindcast harness bulk-loads years of rows through the same interface.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

# Consensus-error priors (°F std dev) by lead day for daily extremes at a
# US station — deliberately a touch pessimistic; real stats take over as
# settlements accumulate.
_PRIOR_SIGMA = {0: 2.0, 1: 2.3, 2: 2.8, 3: 3.4, 4: 4.0, 5: 4.6}


def prior_sigma(lead_days: int) -> float:
    lead = max(0, lead_days)
    return _PRIOR_SIGMA.get(lead, 4.6 + 0.6 * (lead - 5))


class VerificationStore:
    """Append-only JSONL; guidance rows are deduped on read (last wins) by
    (station, kind, target_date, lead)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, row: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def _rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def record_guidance(self, station: str, kind: str, target_date: date,
                        lead_days: int, mean: float, spread: float,
                        models: dict[str, float]) -> None:
        self._append({
            "type": "guidance", "station": station, "kind": kind,
            "target_date": target_date.isoformat(), "lead": lead_days,
            "mean": round(mean, 2), "spread": round(spread, 2), "models": models,
        })

    def record_settlement(self, station: str, kind: str, target_date: date,
                          value: float) -> None:
        self._append({
            "type": "settlement", "station": station, "kind": kind,
            "target_date": target_date.isoformat(), "value": value,
        })

    def has_settlement(self, station: str, kind: str, target_date: date) -> bool:
        want = target_date.isoformat()
        return any(
            r["type"] == "settlement" and r["station"] == station
            and r["kind"] == kind and r["target_date"] == want
            for r in self._rows()
        )

    def errors(self, station: str, kind: str, lead_days: int) -> list[float]:
        """settled − consensus, one per verified target date at this lead."""
        guidance: dict[str, float] = {}
        settled: dict[str, float] = {}
        for r in self._rows():
            if r["station"] != station or r["kind"] != kind:
                continue
            if r["type"] == "guidance" and r["lead"] == lead_days:
                guidance[r["target_date"]] = r["mean"]  # last write wins
            elif r["type"] == "settlement":
                settled[r["target_date"]] = r["value"]
        return [settled[d] - guidance[d] for d in guidance.keys() & settled.keys()]

    def stats(self, station: str, kind: str, lead_days: int,
              shrink: float = 10.0) -> tuple[float, float, int]:
        """(bias, sigma, n_verified) with shrinkage toward the prior: an empty
        store returns (0, prior, 0); each verified day earns the local stats
        more weight."""
        errs = self.errors(station, kind, lead_days)
        n = len(errs)
        prior = prior_sigma(lead_days)
        bias = sum(errs) / (n + shrink) if n else 0.0
        var = (sum((e - bias) ** 2 for e in errs) + shrink * prior * prior) / (n + shrink)
        return bias, math.sqrt(var), n
