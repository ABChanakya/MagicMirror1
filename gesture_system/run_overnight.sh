#!/usr/bin/env bash
# run_overnight.sh — wait for preprocessing, then train and verify unattended.
#
# Runs two trainings so the class-imbalance question (350/347/35) can be judged
# on evidence rather than assumption:
#   A) with inverse-frequency class weights   -> checkpoints/landmark_best.pt
#   B) without them                            -> checkpoints/landmark_noweight.pt
# Then replays held-out clips through the live inference path for A.
#
# Everything lands in training_run.log.

set -uo pipefail
cd "$(dirname "$0")"

LOG=training_run.log
: > "$LOG"

say() { echo -e "\n########## $* ##########\n" | tee -a "$LOG"; }
# Strip the TF/MediaPipe startup noise and tqdm redraws
clean() { grep -vE "tensorflow|TF-TRT|cuda_|oneDNN|cpu_feature|external/local|absl::|SymbolDatabase|warnings.warn|XNNPACK|Feedback manager|WARNING: All log|AVX2|gl_context|libEGL|pci id|it/s\]$"; }

say "waiting for preprocess_landmarks.py"
while pgrep -f preprocess_landmarks.py >/dev/null 2>&1; do sleep 10; done

if [ ! -f data/landmarks.npz ]; then
  say "FAILED — data/landmarks.npz was never produced; nothing to train on"
  exit 1
fi

say "dataset"
python - <<'PY' 2>&1 | clean | tee -a "$LOG"
import numpy as np
d = np.load('data/landmarks.npz', allow_pickle=True)
classes = [str(c) for c in d['classes']]
print(f"X={d['X'].shape}  classes={classes}")
for i, c in enumerate(classes):
    print(f"  {c:12} {int((d['y']==i).sum()):4} clips")
PY

say "A) training WITH class weights"
python train_landmark.py --epochs 80 2>&1 | clean | tee -a "$LOG"

say "B) training WITHOUT class weights (comparison)"
python train_landmark.py --epochs 80 --no-class-weights \
    --out checkpoints/landmark_noweight.pt 2>&1 | clean | tee -a "$LOG"

say "regression tests"
python test_pipeline.py 2>&1 | clean | tee -a "$LOG"

say "replaying held-out clips through the live inference path (model A)"
python verify_inference.py --n-per-class 10 2>&1 | clean | tee -a "$LOG"

say "summary"
python - <<'PY' 2>&1 | clean | tee -a "$LOG"
import torch
for name, path in [("with weights   ", "checkpoints/landmark_best.pt"),
                   ("without weights", "checkpoints/landmark_noweight.pt")]:
    try:
        c = torch.load(path, map_location='cpu', weights_only=False)
        print(f"{name}  test acc {c['test_acc']*100:5.1f}%   "
              f"balanced {c.get('test_balanced_acc', float('nan'))*100:5.1f}%   ({path})")
    except Exception as e:
        print(f"{name}  FAILED to load: {e}")
PY

say "done — $(date)"
