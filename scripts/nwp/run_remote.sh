#!/usr/bin/env bash
# Run one Pangu inference on a remote GPU box and pull the station rows back.
#
#   scripts/nwp/run_remote.sh [host]     # default host: gpu-host2
#
# The remote needs: this repo at ~/openthomas, the nwp venv-gpu, and the
# model assets (see docs comments in run_pangu.sh). Compute servers produce
# artifacts; the trading box consumes files — ssh is the whole protocol.
set -euo pipefail

HOST="${1:-gpu-host2}"
OUT="$HOME/.openthomas/local-models.jsonl"

echo "[$(date -Is)] remote pangu run on $HOST"
ssh "$HOST" 'cd ~/openthomas && git pull -q && bash scripts/nwp/run_pangu.sh --gpu'

RUN=$(ssh "$HOST" 'ls -dt ~/.openthomas/nwp/runs/* | head -1')
mkdir -p "$(dirname "$OUT")"
ssh "$HOST" "cat '$RUN/rows.jsonl'" >> "$OUT"
echo "[$(date -Is)] merged $(ssh "$HOST" "wc -l < '$RUN/rows.jsonl'") rows from $HOST → $OUT"
