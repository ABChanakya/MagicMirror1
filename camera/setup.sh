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
echo "[1/4] System packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3-opencv \
    python3-pip \
    python3-pil \
    libatlas-base-dev \
    libopenblas-dev \
    git wget

# ── 2. Pure-Python pip packages ─────────────────────────────────────────────
echo "[2/4] pip packages..."
pip3 install --user -r "$CAMERA_DIR/requirements.txt"

# ── 3. MediaPipe for Jetson Nano (PINTO0309 prebuilt wheel) ─────────────────
echo "[3/4] MediaPipe..."
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

# ── 4. ONNX Runtime (CPU, needed for face recognition) ──────────────────────
echo "[4/4] ONNX Runtime..."
if python3 -c "import onnxruntime" 2>/dev/null; then
    echo "  onnxruntime already installed."
else
    if pip3 install --user onnxruntime 2>/dev/null; then
        echo "  onnxruntime installed via pip."
    else
        echo ""
        echo "  WARNING: pip onnxruntime failed on this platform."
        echo "  Face recognition will be disabled."
        echo "  To enable, get the Jetson-compatible build from:"
        echo "    https://elinux.org/Jetson_Zoo#ONNX_Runtime"
        echo ""
    fi
fi

# ── Face embedding model (optional, enables face recognition) ────────────────
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_DIR/face_embedding.onnx" ]; then
    echo ""
    echo "========================================================"
    echo "OPTIONAL: Face recognition setup"
    echo ""
    echo "To enable face recognition (identify people by name),"
    echo "place a 112x112 ONNX face embedding model at:"
    echo "  $MODEL_DIR/face_embedding.onnx"
    echo ""
    echo "Compatible models: MobileFaceNet, ArcFace-MobileNet"
    echo "Example source: https://github.com/deepinsight/insightface"
    echo "                (model zoo -> buffalo_sc -> det_500m.onnx is NOT this)"
    echo "                Look for w600k_mbf.onnx or similar MobileFaceNet"
    echo ""
    echo "Without this file, the pipeline still detects PRESENCE"
    echo "(someone is at the mirror) but cannot identify WHO."
    echo "========================================================"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To start camera pipeline:"
echo "  bash run.sh --device /dev/video0 --bridge-port 8082 --debug"
echo ""
echo "To train face recognition (after placing face_embedding.onnx):"
echo "  bash train.sh"
