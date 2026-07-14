"""Liveness beacon: proof the trading loop is running, and when it last acted.

Two readers depend on this file. The public feed turns `last_cycle` into the
"live" indicator on openthomas.com — a page that claims to show a *running*
agent has to show that it is, in fact, still running. And `run_started` dates
the current process, which is what "this run" means when the site reports token
spend for the session next to the all-time total: a restart (a GPU reboot, a
model swap, a deploy) begins a new run, and the session counter resets with it.

Written like the token ledger — best-effort, never in the hot path's way. A
read-only home costs us a liveness dot, not a cycle. The write is small and
whole-file, so a reader either sees the previous good beat or the new one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .usage import now


class Heartbeat:
    def __init__(self, home: Path | str):
        self.path = Path(home) / "heartbeat.json"
        self.run_started: str | None = None
        self.cycles = 0

    def start(self) -> None:
        """Mark the beginning of this run. Resets the session cycle counter and
        stamps `run_started`, which the feed uses to scope 'this run' spend."""
        self.run_started = now()
        self.cycles = 0
        self._write(last_cycle=None)

    def beat(self) -> None:
        """One completed cycle. Bumps the counter and stamps `last_cycle`."""
        if self.run_started is None:  # beat() before start(): treat as run start
            self.start()
        self.cycles += 1
        self._write(last_cycle=now())

    def _write(self, last_cycle: str | None) -> None:
        payload = {
            "run_started": self.run_started,
            "last_cycle": last_cycle,
            "cycles_this_run": self.cycles,
            "pid": os.getpid(),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(self.path)  # atomic: a reader never sees a half-written beat
        except OSError:
            pass


def read(home: Path | str) -> dict | None:
    """The last recorded beat, or None if the loop has never run here."""
    path = Path(home) / "heartbeat.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
