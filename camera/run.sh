#!/usr/bin/env bash
# run.sh — Start the camera pipeline on Jetson Nano
# Uses system Python 3.6 + apt/pip3 --user packages (no venv)
# Usage: bash run.sh [--device /dev/video0] [--bridge-port 8082] [--debug]

cd "$(dirname "$0")"

# Packages installed with pip3 --user land here
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.6/dist-packages:${PYTHONPATH:-}"

# Prevent MediaPipe from crashing with EGL/OpenGL segfault on Jetson Nano ARM64
export DISPLAY=:0
export MEDIAPIPE_DISABLE_GPU=1
export MESA_GL_VERSION_OVERRIDE=3.3

# RAM / CPU limits
export OMP_NUM_THREADS=2

# Camera / AI defaults (can be overridden per-environment)
export CAMERA_WIDTH=${CAMERA_WIDTH:-640}
export CAMERA_HEIGHT=${CAMERA_HEIGHT:-480}
export AI_SCALE=${AI_SCALE:-0.25}
export FACE_DETECT_EVERY=${FACE_DETECT_EVERY:-10}
export CAMERA_FPS=${CAMERA_FPS:-10}

exec python3 main.py "$@"
