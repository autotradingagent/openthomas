"""Generation lineage: every parameter set the evolution loop has tried.

Generations form an archive, not just a champion/challenger pair: rejected
and rolled-back generations stay on file with their scores, both for audit
and as parents for future mutation (diversity against local optima). Every
promoted generation records its parent, the evidence that motivated it, and
the gate scores that justified it, so any promotion can be audited and any
regression rolled back to its parent.

Generation 0 is the operator's config — the seed. It never overrides
anything: the human's config wins until the loop has promoted
evidence-backed changes past it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..kernel.bounds import PARAM_SPACE, clamp_params

BASELINE_ID = 0


def params_from_settings(settings) -> dict:
    """Read the current effective value of every genome parameter."""
    return {key: _get_dotted(settings, key) for key in PARAM_SPACE}


def apply_params(settings, params: dict) -> None:
    for key, value in params.items():
        obj = settings
        *path, attr = key.split(".")
        for part in path:
            obj = getattr(obj, part)
        setattr(obj, attr, value)


def _get_dotted(obj, key: str):
    for part in key.split("."):
        obj = getattr(obj, part)
    return obj


@dataclass
class Generation:
    id: int
    parent: int | None
    params: dict
    proposer: str = ""  # "baseline" | "llm" | "random" | "human"
    rationale: str = ""
    evidence: str = ""
    scores: dict = field(default_factory=dict)
    status: str = "candidate"  # candidate | active | retired | rejected | rolled_back
    created: str = ""
    note: str = ""


class GenerationStore:
    """Lineage archive in ~/.openthomas/generations.json.

    Written only by the meta-cycle's deterministic code paths — the LLM
    proposes JSON that the gate validates; model output never touches files.
    Corruption or absence must never break trading — readers treat any
    problem as "no overrides".
    """

    def __init__(self, home: Path):
        self.path = Path(home) / "generations.json"

    # --- storage ---------------------------------------------------------------
    def _load(self) -> list[Generation]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text())
        return [Generation(**g) for g in data.get("generations", [])]

    def _save(self, gens: list[Generation]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"generations": [asdict(g) for g in gens]}, indent=1) + "\n"
        )

    # --- reads -----------------------------------------------------------------
    def all(self) -> list[Generation]:
        return self._load()

    def get(self, gen_id: int) -> Generation | None:
        return next((g for g in self._load() if g.id == gen_id), None)

    def active(self) -> Generation | None:
        return next((g for g in self._load() if g.status == "active"), None)

    # --- writes ----------------------------------------------------------------
    def ensure_baseline(self, params: dict) -> Generation:
        gens = self._load()
        if gens:
            return gens[0]
        baseline = Generation(
            id=BASELINE_ID, parent=None, params=dict(params), proposer="baseline",
            rationale="operator config at first meta-cycle", status="active",
            created=_now(),
        )
        self._save([baseline])
        return baseline

    def add(self, gen: Generation) -> Generation:
        gens = self._load()
        gen.id = max((g.id for g in gens), default=-1) + 1
        gen.created = _now()
        gens.append(gen)
        self._save(gens)
        return gen

    def promote(self, gen_id: int, note: str = "") -> Generation:
        gens = self._load()
        target = next(g for g in gens if g.id == gen_id)
        for g in gens:
            if g.status == "active":
                g.status = "retired"
        target.status = "active"
        if note:
            target.note = note
        self._save(gens)
        return target

    def rollback(self, reason: str) -> Generation | None:
        """Demote the active generation and reactivate its parent."""
        gens = self._load()
        active = next((g for g in gens if g.status == "active"), None)
        if active is None or active.parent is None:
            return None
        parent = next((g for g in gens if g.id == active.parent), None)
        if parent is None:
            return None
        active.status = "rolled_back"
        active.note = reason
        parent.status = "active"
        self._save(gens)
        return parent


def active_params(home: Path) -> dict:
    """Bounds-validated overrides from the active generation, {} on any
    problem — trading must never die because improvement state is corrupt.
    The baseline generation returns {} by construction: see module docstring.
    """
    try:
        gen = GenerationStore(home).active()
        if gen is None or gen.id == BASELINE_ID:
            return {}
        return clamp_params(gen.params)
    except Exception:
        return {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
