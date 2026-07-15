#!/usr/bin/env bash
# One GenCast (0.25° Operational) ENSEMBLE inference → the probabilistic forecast
# that prediction markets price against. GenCast is a score-based diffusion model:
# each ensemble member is a full sampling trajectory (num_noise_levels = 20 model
# forwards per 12 h step), so it is GPU-only and wants a big card (the 0.25° model
# fits an 80 GB A800; CPU is impractical — hours per member).
#
# Input: ERA5 via CDS (--input cds), NOT ECMWF open-data. GenCast needs the sst
# (sea-surface temperature) surface field, which open-data omits but ERA5 carries
# (along with all 37 levels). Requires ~/.cdsapirc with the ERA5 single-levels AND
# pressure-levels licences accepted. NB: ERA5 is a reanalysis ~5 days behind real
# time, so this is the hindcast / backtest / skill-comparison path — for a live
# real-time GenCast you'd instead inject a free sst field (e.g. NOAA OISST) into
# the open-data path.
#
# Variant is chosen by the ai-models MODEL NAME (a separate entry point each), NOT
# by --model-version (which ai-models silently ignores here):
#   gencast-1.0       1.0° (default) — ~a few GB, fits one GPU with the ensemble
#   gencast-1.0-Mini  1.0° mini
#   gencast-0.25-Oper 0.25° operational — needs >80GB PER MEMBER (TPU/model-sharding
#                     territory), does NOT fit a single 80GB A800; avoid on GPU.
#
#   run_gencast.sh [members] [date YYYYMMDD] [time HHMM] [lead_h] [model]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"  # resolve before any cd
GCX_HOME="${OPENTHOMAS_GENCAST_HOME:-$HOME/.openthomas/gencast}"
VENV="$GCX_HOME/venv-gc"
MEMBERS="${1:-8}"; DATE="${2:-}"; TIME="${3:-0000}"; LEAD="${4:-168}"; MODEL="${5:-gencast-1.0}"
[ -z "$DATE" ] && { echo "usage: run_gencast.sh <members> <YYYYMMDD (>=~5 days back)> [HHMM] [lead_h] [model]"; exit 2; }

# Pin to the configured GPU (UUID survives index reshuffles on a shared box).
if [ -n "${OPENTHOMAS_GENCAST_GPU_UUID:-}" ]; then export CUDA_VISIBLE_DEVICES="$OPENTHOMAS_GENCAST_GPU_UUID"; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # allocate on demand, use the whole card

WORK="$GCX_HOME/runs/$(date -u +%Y%m%dT%H%M%S)"; mkdir -p "$WORK"; cd "$WORK"
echo "[$(date -Is)] $MODEL, $MEMBERS members, init ${DATE} ${TIME}z, lead ${LEAD}h in $WORK"
# --download-assets fetches GenCast's static masks (land-sea, sst, orography) into
# --assets on first run; idempotent afterwards.
"$VENV/bin/ai-models" --input cds --date "$DATE" --time "$TIME" \
  --download-assets --assets "$GCX_HOME" --lead-time "$LEAD" --path gencast.grib \
  "$MODEL" --num-ensemble-members "$MEMBERS"
echo "[$(date -Is)] gencast run done → $WORK/gencast.grib ($MEMBERS members)"

# Distil the ensemble into per-station flow-dependent spread for the desk.
"$VENV/bin/python" "$SCRIPT_DIR/extract_gencast.py" gencast.grib
echo "[$(date -Is)] wrote gencast-spread.json"

# Keep the last few runs only; ensemble GRIBs are large.
ls -dt "$GCX_HOME"/runs/* | tail -n +4 | xargs -r rm -rf
