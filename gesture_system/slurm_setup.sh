#!/usr/bin/env bash
#
# slurm_setup.sh — run ONCE, on the SLURM submit node, to prepare the project.
#
# This does not need a GPU, so run it directly on sl-sn01/sl-sn02, not inside a
# job. It creates the venv, installs everything, and copies the code + clips
# across. Model weights are NOT downloaded here — that happens inside the job,
# onto local /scratch, because the shared NFS home is the wrong place for a
# 46 GB download (see the README's section 4).
#
# Usage (on the submit node, after `ssh username@10.215.44.154`):
#   mkdir -p ~/magicmirror && cd ~/magicmirror
#   # copy this repo in — see the rsync command printed by prep_transfer.sh
#   bash slurm_setup.sh

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root (this script lives in gesture_system/)

echo "== venv =="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "== core deps =="
# torchvision must match torch's build (both from the cu121 index) — dataset.py
# imports torchvision.transforms.functional, so animate_variants.py won't even
# start without it.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers accelerate transformers bitsandbytes \
            imageio imageio-ffmpeg opencv-python-headless mediapipe pyyaml tqdm

mkdir -p logs
echo
echo "Done. Submit the generation job with:"
echo "  sbatch gesture_system/slurm_animate.sh"
echo
echo "Or grab an interactive H100 to test first:"
echo "  salloc --partition=dgx_01 --qos=research_qos --gres=gpu:h100:1 \\"
echo "         --cpus-per-task=8 --mem=64G --job-name=animate-test"
echo "  srun --pty bash"
