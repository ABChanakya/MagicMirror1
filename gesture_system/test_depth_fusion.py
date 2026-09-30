#!/usr/bin/env python3
"""
test_depth_fusion.py -- proof of concept: replace MediaPipe's noisy Z with
depth sampled (at the same pixel MediaPipe already found for X/Y) from a
dedicated monocular depth model (Depth Anything V2 -- Apache-2.0, not
gated, no wait). Needs mediapipe (real Holistic extraction) + transformers
(depth model) in the same env -- run in gesture_system/.venv on the
cluster, not kimodo_venv (mediapipe is broken in this repo's local venv).

Usage
-----
python test_depth_fusion.py camera/gesture_training_data/swipe_right/swipe_right_chanakya_20260821-205247_142.mp4
"""

import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '.')
from preprocess_holistic import _init_holistic, extract_frame, N_POINTS, L_WRIST, R_WRIST
from dataset import _extract_frames

VIDEO = sys.argv[1]
NUM_FRAMES = 30

print(f"extracting frames from {VIDEO}")
frames = _extract_frames(VIDEO, num_frames=NUM_FRAMES)  # (T,H,W,3) RGB uint8
if frames is None:
    raise SystemExit(f"could not read {VIDEO}")

print("loading MediaPipe Holistic")
hol = _init_holistic()

print("loading Depth Anything V2 (small)")
from transformers import pipeline
depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
                       device=0 if torch.cuda.is_available() else -1)

old_z = np.full((NUM_FRAMES, N_POINTS), np.nan, dtype=np.float32)
new_z = np.full((NUM_FRAMES, N_POINTS), np.nan, dtype=np.float32)
xy = np.full((NUM_FRAMES, N_POINTS, 2), np.nan, dtype=np.float32)

for t, frame in enumerate(frames):
    lm = extract_frame(hol, frame)  # (56,3): x,y normalized [0,1], z = MediaPipe's own noisy depth
    old_z[t] = lm[:, 2]
    xy[t] = lm[:, :2]

    depth_map = depth_pipe(Image.fromarray(frame))['depth']
    depth_arr = np.array(depth_map).astype(np.float32)
    dh, dw = depth_arr.shape[:2]

    for i in range(N_POINTS):
        x, y = lm[i, 0], lm[i, 1]
        if np.isnan(x):
            continue
        px = int(np.clip(x * dw, 0, dw - 1))
        py = int(np.clip(y * dh, 0, dh - 1))
        new_z[t, i] = depth_arr[py, px]
    print(f"  frame {t+1}/{NUM_FRAMES} done")

for name, idx in (('left wrist', L_WRIST), ('right wrist', R_WRIST)):
    old_j = np.abs(np.diff(old_z[:, idx]))
    new_j = np.abs(np.diff(new_z[:, idx]))
    print(f"{name}: MediaPipe Z jitter max={np.nanmax(old_j):.3f} mean={np.nanmean(old_j):.3f}"
          f"   |   DepthAnything Z jitter max={np.nanmax(new_j):.3f} mean={np.nanmean(new_j):.3f}")

np.savez('depth_fusion_test.npz', old_z=old_z, new_z=new_z, xy=xy)
print("wrote depth_fusion_test.npz")
