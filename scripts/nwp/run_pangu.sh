#!/usr/bin/env bash
# One Pangu-Weather inference run → OpenThomas local model source.
#
# Default: CPU (the vLLM server owns the GPUs); ~1h per run, fine for the
# 2x/day cadence — ECMWF open-data initial conditions are ~7h stale anyway,
# so faster inference buys no real latency on this input.
#
# --gpu: stop the vLLM container, infer on GPU (~2min), restart it — Tom
# authorized the stop (2026-07-08). The trading loop is blind to GLM while
# the container is down, so this mode is for hand-run experiments and the
# future fast-IC path (NOAA AWS GFS, ~3.5h), not for cron.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"  # resolve before any cd
NWP_HOME="${OPENTHOMAS_NWP_HOME:-$HOME/.openthomas/nwp}"
VENV="$NWP_HOME/venv"
VLLM_CONTAINER="vllm-container"

if [ "${1:-}" = "--gpu" ]; then
  VENV="$NWP_HOME/venv-gpu"
  # On the trading box the vLLM server owns the VRAM — evict it for the run
  # and guarantee it comes back. On a dedicated GPU box (gpu-host2) there is no
  # such container and the GPU is simply free.
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$VLLM_CONTAINER"; then
    echo "[$(date -Is)] stopping $VLLM_CONTAINER for GPU inference"
    docker stop "$VLLM_CONTAINER"
    trap 'echo "[$(date -Is)] restarting $VLLM_CONTAINER"; docker start "$VLLM_CONTAINER"' EXIT
  fi
else
  # onnxruntime-gpu ignores CUDA_VISIBLE_DEVICES; the CPU venv carries the
  # CPU build, this is just belt-and-braces.
  export CUDA_VISIBLE_DEVICES=""
fi

WORK="$NWP_HOME/runs/$(date -u +%Y%m%dT%H%M)"
mkdir -p "$WORK"
cd "$WORK"

echo "[$(date -Is)] pangu run starting in $WORK (venv: $VENV)"
"$VENV/bin/ai-models" --input ecmwf-open-data \
  --assets "$NWP_HOME/assets" --download-assets \
  --lead-time 168 --path pangu.grib panguweather

"$VENV/bin/python" "$SCRIPT_DIR/extract_stations.py" pangu.grib --model pangu_local
echo "[$(date -Is)] pangu run done"

# Keep the last few runs only; GRIBs are ~2GB each.
ls -dt "$NWP_HOME"/runs/* | tail -n +4 | xargs -r rm -rf
