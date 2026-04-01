#!/usr/bin/env bash
# setup.sh — Create venv and install all camera pipeline dependencies using uv
# Usage: bash setup.sh

set -e
cd "$(dirname "$0")"

# Install uv if not present
if ! command -v uv &>/dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Creating virtual environment..."
uv venv .venv --python 3.12

echo "Installing dependencies..."
uv pip install --python .venv/bin/python -r requirements.txt

echo ""
echo "Done. To activate:"
echo "  source .venv/bin/activate"
echo ""
echo "To train face model:"
echo "  bash train.sh"
echo ""
echo "To start camera pipeline:"
echo "  bash run.sh --device /dev/video1 --bridge-port 8082 --debug"
