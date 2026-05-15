#!/usr/bin/env bash
# setup.sh — Install camera pipeline on Jetson Nano (JetPack 4.6.x)
# Python 3.6, Ubuntu 18.04, OpenCV 3.2 via apt
#
# Run from inside the camera/ directory:
#   bash setup.sh

set -euo pipefail
cd "$(dirname "$0")"
CAMERA_DIR="$(pwd)"
MODEL_DIR="$CAMERA_DIR/model"

echo "=== MagicMirror3 Camera Pipeline — Jetson Nano Setup ==="
echo "Python: $(python3 --version 2>&1)"
echo "pip:    $(pip3 --version 2>&1)"
echo ""

# ── 1. System packages ──────────────────────────────────────────────────────
echo "[1/5] System packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3-opencv \
    python3-pip \
    python3-pil \
    libatlas-base-dev \
    libopenblas-dev \
    cmake \
    libboost-python-dev \
    libboost-thread-dev \
    git wget

# ── 2. Pure-Python pip packages ─────────────────────────────────────────────
echo "[2/5] pip packages..."
pip3 install --user -r "$CAMERA_DIR/requirements.txt"

# ── 3. MediaPipe for Jetson Nano (PINTO0309 prebuilt wheel) ─────────────────
echo "[3/5] MediaPipe..."
if python3 -c "import mediapipe" 2>/dev/null; then
    echo "  MediaPipe already installed."
else
    MP_CLONE_DIR="/tmp/mediapipe-bin"
    if [ ! -d "$MP_CLONE_DIR" ]; then
        git clone --depth 1 https://github.com/PINTO0309/mediapipe-bin "$MP_CLONE_DIR"
    fi
    cd "$MP_CLONE_DIR"
    bash v0.8.5/download.sh
    # cp36 = Python 3.6, cuda102 = JetPack 4.6.x CUDA 10.2, numpy119x
    pip3 install --user \
        v0.8.5/numpy119x/py36/mediapipe-0.8.5_cuda102-cp36-cp36m-linux_aarch64.whl
    cd "$CAMERA_DIR"
    echo "  MediaPipe installed."
fi

# ── 4. dlib + face_recognition (replaces ONNX Runtime) ─────────────────────
# dlib 19.21.1 is the last version supporting Python 3.6.
# It compiles from source — this takes ~30 minutes on Jetson Nano.
echo "[4/5] dlib + face_recognition..."
if python3 -c "import face_recognition" 2>/dev/null; then
    echo "  face_recognition already installed."
else
    echo "  Installing dlib (compiles from source, ~30 min on Jetson Nano)..."
    pip3 install --user dlib==19.21.1
    pip3 install --user face_recognition==1.3.0
    echo "  face_recognition installed."
fi

# ── 5. WebSocket bridge package ─────────────────────────────────────────────
echo "[5/5] websockets..."
if python3 -c "import websockets" 2>/dev/null; then
    echo "  websockets already installed."
else
    pip3 install --user 'websockets>=8.0,<10'
fi

# ── Model directory ──────────────────────────────────────────────────────────
mkdir -p "$MODEL_DIR"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Add face photos to camera/dataset/<your_name>/"
echo "  2. Run: python3 train.py        (generates model/encodings.pkl)"
echo "  3. Run: bash run.sh --device /dev/video0 --bridge-port 8082 --debug"
echo ""
echo "To train face recognition:"
echo "  bash train.sh"
