#!/usr/bin/env bash
# run.sh — Start the camera pipeline on Jetson Nano
# Uses system Python 3.6 + apt/pip3 --user packages (no venv)
# Usage: bash run.sh [--device /dev/video0] [--bridge-port 8082] [--debug]

cd "$(dirname "$0")"

# Packages installed with pip3 --user land here
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.6/dist-packages:${PYTHONPATH:-}"

exec python3 main.py "$@"
