#!/bin/bash
#SBATCH --partition=dgx_01
#SBATCH --qos=research_qos
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=twostream-train
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=CHANGE_ME@haw-landshut.de
#
# Trains landmark-only, video-only, and fusion (train_twostream.py --mode all)
# on the same session-held-out split, so the video branch's contribution is
# measured like-for-like rather than assumed.
#
# Data footprint is small (landmarks_holistic.npz ~16 MB, gesture_training_data
# ~630 MB) — nowhere near the 46 GB threshold that forces /scratch use for the
# Animate weights (README section 4), so this reads/writes the NFS home
# directly and needs no scratch copy-back step.
#
# First run downloads Swin3D-B Kinetics-400 weights (~350 MB) into
# ~/.cache/torch/hub — on NFS home, so it persists and later jobs don't
# re-download it.
#
# Before submitting:
#   1. sed -i 's/CHANGE_ME@haw-landshut.de/your_actual_address/' slurm_train_twostream.sh
#   2. Make sure data/landmarks_holistic.npz and ../camera/gesture_training_data/
#      are present under $HOME/magicmirror/gesture_system (see rsync command
#      below if they aren't yet).
#
# Try a 2-epoch smoke test first (~a few minutes, catches broken paths/imports
# before burning a 3-hour allocation):
#   SMOKE=1 sbatch slurm_train_twostream.sh
#
# Full run:
#   sbatch slurm_train_twostream.sh
#
# Check on it:
#   squeue -u $USER
#   tail -f twostream-train_<jobid>.out

set -euo pipefail

PROJDIR="$HOME/magicmirror"
GSDIR="$PROJDIR/gesture_system"

cd "$GSDIR"
source "$PROJDIR/.venv/bin/activate"

echo "== node/GPU =="
hostname
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

SMOKE_FLAG=""
if [ "${SMOKE:-0}" = "1" ]; then
    SMOKE_FLAG="--smoke-test"
    echo "== SMOKE TEST: 2 epochs per mode, sanity check only =="
fi

echo "== training: landmark, video, fusion =="
python train_twostream.py \
    --mode all \
    --epochs "${EPOCHS:-30}" \
    --split session \
    --out-dir checkpoints \
    $SMOKE_FLAG

echo "Done. Checkpoints in $GSDIR/checkpoints/twostream_*.pt"
echo "Summary in $GSDIR/twostream_results.json"
