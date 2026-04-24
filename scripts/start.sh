#!/usr/bin/env bash
# start.sh — Start MagicMirror + camera pipeline with the child profile.
# MagicMirror runs in the foreground (so Electron can open on your display).
# Camera runs in the background.
# Override profile: MM_CONFIG_FILE=config/config.js npm run start:mm3
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MM_DIR="$ROOT/MagicMirror"
CAM_DIR="$ROOT/camera"
LOGS_DIR="$ROOT/logs"
mkdir -p "$LOGS_DIR"

# ── Python venv setup (first run only) ─────────────────────────────────────
VENV="$CAM_DIR/.venv"
if [ ! -d "$VENV" ]; then
  echo "[start] First run — setting up Python environment..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet -r "$CAM_DIR/requirements.txt"
  echo "[start] Python environment ready."
fi

# ── Start camera pipeline in background ────────────────────────────────────
echo "[start] Starting camera pipeline (background)..."
nohup "$VENV/bin/python3" "$CAM_DIR/main.py" \
  --device "${CAMERA_DEVICE:-/dev/video0}" \
  --bridge-port "${BRIDGE_PORT:-8082}" \
  > "$LOGS_DIR/camera.log" 2>&1 &
echo $! > "$LOGS_DIR/camera.pid"
echo "[start] Camera PID: $(cat $LOGS_DIR/camera.pid)"

# Kill camera on exit
trap 'bash "$ROOT/scripts/stop.sh" 2>/dev/null || true' EXIT

cd "$MM_DIR"

# Default to child profile; override by setting MM_CONFIG_FILE before calling.
MM_CFG="${MM_CONFIG_FILE:-config/config.child.js}"
if [ ! -f "$MM_DIR/$MM_CFG" ] && [ ! -f "$MM_CFG" ]; then
  echo "[start] $MM_CFG not found, falling back to config/config.child.js"
  MM_CFG="config/config.child.js"
fi

# Raspberry Pi/ARM systems can be more stable with GPU acceleration disabled.
ARCH="$(uname -m)"
if [[ "$ARCH" == arm* || "$ARCH" == aarch64 ]]; then
  export ELECTRON_DISABLE_GPU="${ELECTRON_DISABLE_GPU:-1}"
  export ELECTRON_ENABLE_GPU="${ELECTRON_ENABLE_GPU:-0}"
  echo "[start] ARM mode detected ($ARCH). GPU acceleration disabled by default."
fi

# ── Start MagicMirror in foreground (Electron needs display access) ─────────
echo "[start] Starting MagicMirror..."
echo "  Config: $MM_CFG"
echo "  Logs:   $LOGS_DIR/"
echo "  Stop:   Ctrl+C (stops both MM and camera)"
echo ""

MM_CONFIG_FILE="$MM_CFG" npm run start:default
