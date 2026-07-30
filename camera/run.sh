#!/usr/bin/env bash
# run.sh — Start the camera pipeline (Phase 1: swipe + hand tracking)
# RTX 4090 rewrite — Python 3.10+ with venv
# Usage: bash run.sh [--device /dev/video0] [--bridge-port 8082] [--debug]

cd "$(dirname "$0")"

# Use venv if it exists, else system Python 3.10+
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Camera / display
export CAMERA_WIDTH=${CAMERA_WIDTH:-640}
export CAMERA_HEIGHT=${CAMERA_HEIGHT:-480}
export CAMERA_FPS=${CAMERA_FPS:-15}
export MIRROR_FLIP=${MIRROR_FLIP:-false}

# Debug server
export DEBUG_PORT=${DEBUG_PORT:-8083}

exec python3 main.py "$@"
