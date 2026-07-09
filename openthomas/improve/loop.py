"""The evolution loop (slow loop): OpenThomas updating OpenThomas.

One meta-cycle = rollback-check → mine → propose → gate → promote → log.
Runs daily-ish after settlements, or on demand via `openthomas improve`.
The trading loop (fast loop) never waits on it, and it can never kill
trading: every promotion is bounds-clamped, gate-approved, journaled to
improve-log.jsonl, and reversible via lineage rollback.

Division of labor: this module (agent plane) decides what to TRY; the kernel
decides what COUNTS. All file writes happen in this deterministic code after
the gate has ruled — the proposer LLM only ever returns JSON suggestions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial

from ..config import Settings
from ..kernel import gate
from ..llm import CompletionClient
from ..markets.kalshi import KalshiConnector
from ..memory.journal import Journal
from ..weather.replay import collect_all, decide
from ..weather.verification import VerificationStore
from .genome import (BASELINE_ID, Generation, GenerationStore, apply_params,
                     params_from_settings)
from .proposer import dedupe, llm_candidates, mine_evidence, random_candidates

REPLAY_DAYS = 45
LLM_CANDIDATES = 3
RANDOM_CANDIDATES = 2
IMPROVE_EVERY_HOURS = 20  # daily-ish; jittered by the trading loop's cadence
LAST_RUN_KEY = "improve_last_ts"


@dataclass
class MetaReport:
    rows: int = 0
    rollback: str = ""
    candidates: list[dict] = field(default_factory=list)
    promoted: int | None = None
    reason: str = ""

    def as_dict(self) -> dict:
        return {"ts": _now(), "rows": self.rows, "rollback": self.rollback,
                "candidates": self.candidates, "promoted": self.promoted,
                "reason": self.reason}


def improve_due(journal: Journal) -> bool:
    last = journal.get_kv(LAST_RUN_KEY)
    if not last:
        return True
    age = datetime.now(timezone.utc) - datetime.fromisoformat(last)
    return age > timedelta(hours=IMPROVE_EVERY_HOURS)


class Improver:
    def __init__(self, settings: Settings, journal: Journal | None = None,
                 complete_fn=None):
        self.s = settings
        self.journal = journal or Journal(settings.db_path)
        self.store = GenerationStore(settings.home)
        self._complete = complete_fn
        self.fee = KalshiConnector().fee
        self.log_path = settings.home / "improve-log.jsonl"

    def complete(self, system: str, user: str) -> str:
        if self._complete is None:
            self._complete = CompletionClient(self.s.reflector or self.s.forecaster).complete
        return self._complete(system, user)

    def _strategy(self, params: dict):
        return partial(decide, fee_fn=self.fee,
                       min_edge=params["risk.min_edge"],
                       market_prior_weight=params["risk.market_prior_weight"])

    def _score(self, rows_in, rows_out, params: dict) -> gate.Score:
        return gate.score(rows_in, rows_out, params, self._strategy(params))

    # --- the meta-cycle ----------------------------------------------------------
    def meta_cycle(self, days: int = REPLAY_DAYS, dry_run: bool = False) -> MetaReport:
        report = MetaReport()
        champion_params = params_from_settings(self.s)
        if not dry_run:  # dry run means zero writes, including the seed
            self.store.ensure_baseline(champion_params)

        store = VerificationStore(self.s.home / "weather-verification.jsonl")
        rows = collect_all(store, days)
        report.rows = len(rows)
        rows_in, rows_out = gate.split_rows(rows)
        if not rows_in:
            report.reason = "insufficient replay data — run `openthomas hindcast`?"
            self._finish(report, dry_run)
            return report

        champion_score = self._score(rows_in, rows_out, champion_params)

        # 1. Out-of-sample regret: has the fresh window turned against the
        # active promotion? Checked before proposing anything new.
        active = self.store.active()
        if active and active.id != BASELINE_ID and active.parent is not None:
            parent = self.store.get(active.parent)
            if parent:
                parent_score = self._score(rows_in, rows_out, parent.params)
                roll, why = gate.should_rollback(champion_score, parent_score)
                if roll:
                    report.rollback = why
                    if not dry_run:
                        self.store.rollback(why)
                        apply_params(self.s, parent.params)
                    champion_params = parent.params
                    champion_score = parent_score

        # 2. Propose: directed mutations from evidence + random archive mutations.
        evidence = mine_evidence(self.journal)
        champion_summary = (f"held-in ${champion_score.pnl_in:+.2f} "
                            f"({champion_score.held_in.get('n', 0)} trades), "
                            f"held-out ${champion_score.pnl_out:+.2f} "
                            f"({champion_score.held_out.get('n', 0)} trades)")
        candidates: list[dict] = []
        try:
            candidates += llm_candidates(self.complete, champion_params,
                                         evidence, champion_summary, LLM_CANDIDATES)
        except Exception as e:  # a dead LLM endpoint must not stop evolution
            report.reason = f"llm proposer unavailable ({e}); random mutations only. "
        archive = [g.params for g in self.store.all()] or [champion_params]
        candidates += random_candidates(archive, RANDOM_CANDIDATES)
        candidates = dedupe(candidates, champion_params)

        # 3. Gate every candidate; keep the best qualifying by held-in PnL.
        best: tuple[dict, gate.Score, str] | None = None
        for cand in candidates:
            sc = self._score(rows_in, rows_out, cand["params"])
            ok, why = gate.beats(sc, champion_score)
            report.candidates.append({
                "params": cand["params"], "proposer": cand["proposer"],
                "rationale": cand["rationale"], "held_in": sc.held_in,
                "held_out": sc.held_out, "verdict": "pass" if ok else why,
            })
            if ok and (best is None or sc.pnl_in > best[1].pnl_in):
                best = (cand, sc, why)

        # 4. Promote through the store; the running loop picks it up in place.
        if best:
            cand, sc, why = best
            report.reason += f"promote: {why}"
            if not dry_run:
                active = self.store.active()
                gen = self.store.add(Generation(
                    id=-1, parent=active.id if active else BASELINE_ID,
                    params=cand["params"], proposer=cand["proposer"],
                    rationale=cand["rationale"], evidence=evidence[:600],
                    scores=sc.as_dict(),
                ))
                self.store.promote(gen.id, note=why)
                apply_params(self.s, cand["params"])
                report.promoted = gen.id
        elif not report.rollback:
            report.reason += "no candidate cleared the gate; champion holds"

        self._finish(report, dry_run)
        return report

    def _finish(self, report: MetaReport, dry_run: bool) -> None:
        if dry_run:
            return
        self.journal.set_kv(LAST_RUN_KEY, _now())
        with self.log_path.open("a") as f:
            f.write(json.dumps(report.as_dict()) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
