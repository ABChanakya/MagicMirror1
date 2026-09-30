#!/bin/bash
#SBATCH --partition=dgx_01
#SBATCH --qos=research_qos
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=real-ik
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
cd ~/magicmirror/gesture_system
source ~/magicmirror/kimodo_venv/bin/activate
python3 kimodo_pure_ik_retarget.py --target real472_full.npz --out real472_pure_ik.npz
python3 kimodo_export_bvh.py --npz real472_pure_ik.npz --out real472_pure_ik.bvh
echo "Done."
