#!/usr/bin/env bash
# train.sh — Train face model with correct CUDA library paths
# Usage: bash train.sh

cd "$(dirname "$0")"
source .venv/bin/activate

SITE=.venv/lib/python3.12/site-packages
CUDA_LIBS=$(find "$SITE/nvidia" -maxdepth 2 -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH}"

exec python3 train.py "$@"
