#!/usr/bin/env bash
# train-faces.sh — Add a face profile and rebuild the recognition model
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAM_DIR="$ROOT/camera"
VENV="$CAM_DIR/.venv"

if [ $# -lt 1 ]; then
  echo "Usage: ./scripts/train-faces.sh <profile-name>"
  echo ""
  echo "  Creates camera/dataset/<profile-name>/ if it doesn't exist."
  echo "  Add face photos (JPG/PNG) to that folder, then run this script again."
  echo ""
  echo "  Example profiles: kind1  kind2  mama  papa"
  exit 1
fi

PROFILE="$1"
DATASET_DIR="$CAM_DIR/dataset/$PROFILE"

mkdir -p "$DATASET_DIR"

# Count images
IMAGE_COUNT=$(find "$DATASET_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)

if [ "$IMAGE_COUNT" -eq 0 ]; then
  echo "[train] Created folder: $DATASET_DIR"
  echo ""
  echo "  Add at least 5-10 face photos to that folder, then run:"
  echo "  ./scripts/train-faces.sh $PROFILE"
  exit 0
fi

echo "[train] Found $IMAGE_COUNT image(s) for profile '$PROFILE'"

# Set up venv if needed
if [ ! -d "$VENV" ]; then
  echo "[train] Setting up Python environment..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet -r "$CAM_DIR/requirements.txt"
fi

echo "[train] Building face encodings..."
"$VENV/bin/python3" "$CAM_DIR/train.py"
echo "[train] Done. Restart the camera pipeline to apply changes."
