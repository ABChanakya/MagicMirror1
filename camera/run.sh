#!/usr/bin/env bash
# run.sh — Start the camera pipeline with correct CUDA library paths
# Usage: bash run.sh [--device /dev/video1] [--bridge-port 8082] [--debug]

cd "$(dirname "$0")"
source .venv/bin/activate

# Add pip-installed CUDA libs so onnxruntime-gpu can find them
SITE=.venv/lib/python3.12/site-packages
CUDA_LIBS=$(find "$SITE/nvidia" -maxdepth 2 -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH}"

exec python3 main.py "$@"
