#!/usr/bin/env bash
# One Pangu-Weather inference run → OpenThomas local model source.
#
# CPU by default (the vLLM server owns the GPUs); a 10-day forecast takes
# tens of minutes — fine for 2 runs/day. Initial conditions come from ECMWF
# open data (free, no account). Cron it or run by hand:
#   scripts/nwp/run_pangu.sh
set -euo pipefail

NWP_HOME="${OPENTHOMAS_NWP_HOME:-$HOME/.openthomas/nwp}"
VENV="$NWP_HOME/venv"
WORK="$NWP_HOME/runs/$(date -u +%Y%m%dT%H%M)"
mkdir -p "$WORK"
cd "$WORK"

echo "[$(date -Is)] pangu run starting in $WORK"
"$VENV/bin/ai-models" --input ecmwf-open-data \
  --assets "$NWP_HOME/assets" --download-assets \
  --lead-time 168 --path pangu.grib panguweather

"$VENV/bin/python" "$(dirname "$0")/extract_stations.py" pangu.grib --model pangu_local
echo "[$(date -Is)] pangu run done"

# Keep the last few runs only; GRIBs are ~2GB each.
ls -dt "$NWP_HOME"/runs/* | tail -n +4 | xargs -r rm -rf
