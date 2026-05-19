#!/usr/bin/env bash
# start.sh — Start MagicMirror + camera pipeline with the standard profile.
# MagicMirror runs in the foreground (so Electron can open on your display).
# Camera runs in the background.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MM_DIR="$ROOT/MagicMirror"
CAM_DIR="$ROOT/camera"
LOGS_DIR="$ROOT/logs"
mkdir -p "$LOGS_DIR"

# ── Start camera pipeline in background ────────────────────────────────────
# Uses bash run.sh which sets up critical env vars: MEDIAPIPE_DISABLE_GPU, OMP_NUM_THREADS, etc.
echo "[start] Starting camera pipeline (background)..."
cd "$CAM_DIR"
nohup bash run.sh \
  --device "${CAMERA_DEVICE:-/dev/video0}" \
  --bridge-port "${BRIDGE_PORT:-8082}" \
  > "$LOGS_DIR/camera.log" 2>&1 &
echo $! > "$LOGS_DIR/camera.pid"
echo "[start] Camera PID: $(cat $LOGS_DIR/camera.pid)"
cd "$ROOT"

# Kill camera on exit
trap 'bash "$ROOT/scripts/stop.sh" 2>/dev/null || true' EXIT

cd "$MM_DIR"

# Use the standard profile unless MM_CONFIG_FILE points to a child/custom config.
MM_CFG="${MM_CONFIG_FILE:-config/config.js}"
if [ ! -f "$MM_CFG" ]; then
  echo "[start] $MM_CFG not found, falling back to config/config.js"
  MM_CFG="config/config.js"
fi

# ── Start MagicMirror in foreground (Electron needs display access) ─────────
echo "[start] Starting MagicMirror..."
echo "  Config: $MM_CFG"
echo "  Logs:   $LOGS_DIR/"
echo "  Stop:   Ctrl+C (stops both MM and camera)"
echo ""

MM_CONFIG_FILE="$MM_CFG" npm run start:default
