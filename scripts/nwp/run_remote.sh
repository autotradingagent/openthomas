#!/usr/bin/env bash
# Run one Pangu inference on a remote GPU host and pull the station rows back.
#
#   scripts/nwp/run_remote.sh [host] [mode]
#
# host: ssh target of a box with a spare GPU (default: $OPENTHOMAS_NWP_HOST).
# mode: "--gpu" (default) or "" for a CPU-only box.
#
# The remote needs this repo at ~/openthomas, the nwp venv-gpu, and the model
# assets (see run_pangu.sh). Compute hosts produce artifacts; the trading box
# consumes files — ssh is the whole protocol.
set -euo pipefail

# Machine-specific settings (ssh host, GPU UUID, vLLM container name) live in a
# private file outside the repo and are never committed — keep the fleet's
# topology out of the open-source tree.
[ -f "$HOME/.openthomas/nwp.env" ] && . "$HOME/.openthomas/nwp.env"

HOST="${1:-${OPENTHOMAS_NWP_HOST:?pass a host as arg 1 or set OPENTHOMAS_NWP_HOST}}"
MODE="${2:---gpu}"   # GPU by default; pass "" for a CPU-only box. A ~48GB card
                     # fits Pangu fp32 under ORT; a 24GB one does not.
# Pin the remote run to one GPU when the box is shared (CUDA_VISIBLE_DEVICES
# honours GPU-UUID strings, which survive index reshuffles and other tenants).
# Set OPENTHOMAS_GPU_UUID in the private env above; empty means "use device 0".
GPU_UUID="${OPENTHOMAS_GPU_UUID:-}"
OUT="$HOME/.openthomas/local-models.jsonl"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

# The GPU box may reach this box only over a flaky link; keep sessions alive and
# tolerate a slow connect rather than aborting a whole run on one hiccup.
SSH_OPTS=(-o ConnectTimeout=25 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)

# The GPU box may have no clean route to GitHub — push the minimal file set over
# ssh instead of git-pulling there. Layout is preserved so extract_stations.py
# finds the station registry relative to itself.
echo "[$(date -Is)] syncing pipeline to $HOST"
rsync -a --relative -e "ssh ${SSH_OPTS[*]}" \
  "$REPO/./scripts/nwp/run_pangu.sh" \
  "$REPO/./scripts/nwp/extract_stations.py" \
  "$REPO/./openthomas/weather/stations.py" \
  "$HOST:openthomas/"

echo "[$(date -Is)] remote pangu run on $HOST (mode: ${MODE:-cpu})"
# The GPU box may have no clean route to ECMWF — reverse-SOCKS the downloads
# through this box (needs pysocks in the remote venv; OpenSSH ≥7.6).
ssh "${SSH_OPTS[@]}" -R 18080 -o ExitOnForwardFailure=yes "$HOST" \
  "cd ~/openthomas && OPENTHOMAS_GPU_UUID='$GPU_UUID' \
   ALL_PROXY=socks5h://127.0.0.1:18080 HTTPS_PROXY=socks5h://127.0.0.1:18080 \
   bash scripts/nwp/run_pangu.sh $MODE"

RUN=$(ssh "${SSH_OPTS[@]}" "$HOST" 'ls -dt ~/.openthomas/nwp/runs/* | head -1')
mkdir -p "$(dirname "$OUT")"
ssh "${SSH_OPTS[@]}" "$HOST" "cat '$RUN/rows.jsonl'" >> "$OUT"
echo "[$(date -Is)] merged $(ssh "${SSH_OPTS[@]}" "$HOST" "wc -l < '$RUN/rows.jsonl'") rows from $HOST → $OUT"

# Pull the global temperature grid too — our own forecast field for the site's
# heatmap. Best-effort: an older run may predate it, and a missing grid just
# leaves the globe on the Open-Meteo nowcast.
GRID="$HOME/.openthomas/pangu-tempgrid.json"
if ssh "${SSH_OPTS[@]}" "$HOST" "test -f '$RUN/tempgrid.json'"; then
  ssh "${SSH_OPTS[@]}" "$HOST" "cat '$RUN/tempgrid.json'" > "$GRID"
  echo "[$(date -Is)] pulled Pangu temperature grid → $GRID"
else
  echo "[$(date -Is)] no tempgrid.json in $RUN (older run?) — globe stays on Open-Meteo"
fi
