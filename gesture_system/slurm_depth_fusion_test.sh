#!/bin/bash
#SBATCH --partition=dgx_01
#SBATCH --qos=research_qos
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=depth-fusion-test
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
cd ~/magicmirror/gesture_system
source ~/magicmirror/.venv/bin/activate
pip install -q transformers 2>&1 | tail -5
python3 test_depth_fusion.py ../camera/gesture_training_data/swipe_right/swipe_right_chanakya_20260821-205247_142.mp4
echo "Done."
