#!/bin/bash
# RTX 4090 (24 GB, Ada sm_89) launcher: run_server.sh with 24 GB defaults.
# Every value below is only a default -- export a variable to override it, e.g.
#   JOYOMNI_4090_PRESET=fast bash deploy/run_server_4090.sh
#   JOYOMNI_DIT_OFFLOAD_BLOCKS=12 bash deploy/run_server_4090.sh
#
# Presets (JOYOMNI_4090_PRESET):
#   balanced (default) 840x480 @ 18 FPS, 8 DiT blocks streamed from host RAM
#   fast               672x384 @ 24 FPS, 4 DiT blocks streamed
#   safe               672x384 @ 15 FPS, 10 DiT blocks streamed (most VRAM headroom)
# These are starting points, not measured numbers: run tools/probe_rtx4090.py
# first and see DEPLOYMENT.md "RTX 4090" for how to tune them.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PRESET="${JOYOMNI_4090_PRESET:-balanced}"
case "$PRESET" in
  balanced) _W=840; _H=480; _FPS=18; _OFF=8 ;;
  fast)     _W=672; _H=384; _FPS=24; _OFF=4 ;;
  safe)     _W=672; _H=384; _FPS=15; _OFF=10 ;;
  *) echo "unknown JOYOMNI_4090_PRESET=$PRESET (balanced|fast|safe)" >&2; exit 2 ;;
esac

export JOYOMNI_WIDTH="${JOYOMNI_WIDTH:-$_W}"
export JOYOMNI_HEIGHT="${JOYOMNI_HEIGHT:-$_H}"
export JOYOMNI_FPS="${JOYOMNI_FPS:-$_FPS}"

# Same consumer-card switches as the RTX 5090 profile: Ada GeForce also runs
# fp32-accumulate FP8 MMAs at half rate, and SageAttention2 ships sm_89 kernels.
export JOYOMNI_LOW_VRAM="${JOYOMNI_LOW_VRAM:-1}"
export JOYOMNI_SAGE_ATTN="${JOYOMNI_SAGE_ATTN:-1}"
export JOYOMNI_FP8_FAST_ACCUM="${JOYOMNI_FP8_FAST_ACCUM:-1}"
export JOYOMNI_FP8_IMG="${JOYOMNI_FP8_IMG:-1}"
export JOYOMNI_FP8_TXT="${JOYOMNI_FP8_TXT:-1}"

# 4090-specific: stream N FP8 DiT blocks from pinned host RAM (xvideo/blockswap.py)
# and build the DiT on the meta device so loading does not need ~55 GB of RAM.
export JOYOMNI_DIT_OFFLOAD_BLOCKS="${JOYOMNI_DIT_OFFLOAD_BLOCKS:-$_OFF}"
export JOYOMNI_DIT_OFFLOAD_SLOTS="${JOYOMNI_DIT_OFFLOAD_SLOTS:-2}"
export JOYOMNI_DIT_META_INIT="${JOYOMNI_DIT_META_INIT:-1}"

export JOYOMNI_CACHE_ROOT="${JOYOMNI_CACHE_ROOT:-$HERE/deps/cache_rtx4090}"

if command -v nvidia-smi >/dev/null 2>&1; then
  _dev="${JOYOMNI_DEVICE:-cuda:0}"; _idx="${_dev#cuda:}"; [ "$_idx" = "$_dev" ] && _idx=0
  _info="$(nvidia-smi -i "$_idx" --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
  echo "[run_server_4090] GPU $_idx: ${_info:-unknown} (MiB total, MiB used)"
  _used="$(echo "$_info" | awk -F', ' '{print $3}')"
  if [ -n "${_used:-}" ] && [ "${_used%.*}" -gt 1024 ] 2>/dev/null; then
    echo "[run_server_4090] WARNING: ${_used} MiB already in use on this GPU (desktop/other processes)." >&2
    echo "[run_server_4090]          Every MiB counts on 24 GB: drive the display from another GPU/iGPU or" >&2
    echo "[run_server_4090]          raise JOYOMNI_DIT_OFFLOAD_BLOCKS / use JOYOMNI_4090_PRESET=safe." >&2
  fi
fi

echo "[run_server_4090] preset=$PRESET ${JOYOMNI_WIDTH}x${JOYOMNI_HEIGHT}@${JOYOMNI_FPS} " \
     "offload_blocks=$JOYOMNI_DIT_OFFLOAD_BLOCKS sage=$JOYOMNI_SAGE_ATTN fast_accum=$JOYOMNI_FP8_FAST_ACCUM"

exec bash "$HERE/run_server.sh" "$@"
