"""The live weather board: a snapshot of every market the agent is watching.

The public globe shows the whole book — not just the edges we took — so it needs
the markets the trading loop sees each cycle, with their live prices. The loop
already holds them; it dumps a lean snapshot here and the feed builder joins it
against the journal to mark which we forecast, hold, or have settled.

Written best-effort like the heartbeat and the token ledger: a snapshot we fail
to write costs the globe some dots for one cycle, never a trade. Only market
facts (question, prices, close) are stored — the same fields a reader sees on the
venue — plus the id, which stays server-side as the join key and never ships.
"""

from __future__ import annotations

import json
from pathlib import Path

from .usage import now


class Board:
    def __init__(self, home: Path | str):
        self.path = Path(home) / "board.json"

    def write(self, markets) -> None:
        rows = [
            {
                "id": m.id, "platform": m.platform, "question": m.question,
                "category": m.category or "", "yes_bid": m.yes_bid, "yes_ask": m.yes_ask,
                "volume_24h": round(m.volume_24h or 0.0, 2),
                "close_time": m.close_time.isoformat() if m.close_time else None,
            }
            for m in markets
        ]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"ts": now(), "markets": rows}))
            tmp.replace(self.path)
        except OSError:
            pass


def read(home: Path | str) -> dict | None:
    try:
        return json.loads((Path(home) / "board.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
