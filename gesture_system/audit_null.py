#!/usr/bin/env python3
"""
audit_null.py — measure whether `null` training data covers the case that
actually causes false positives.

The `null` class is doing two different jobs, and only one of them is real:

  1. "nobody is there"          — the inference hand-presence gate already
                                  handles this deterministically, before the
                                  model runs. The model does not need to learn it.
  2. "a hand is visible and
      moving, but this is not
      a deliberate swipe"       — scratching your head, reaching past the
                                  mirror, gesturing while talking. This is the
                                  only case that can produce a false fire, and
                                  it is what `null` must actually teach.

This script counts how many `null` clips fall in each bucket, and evaluates a
trained checkpoint on bucket 2 alone. A high headline `null` recall that comes
entirely from bucket 1 is measuring the gate, not the model.

Usage: python audit_null.py [--checkpoint checkpoints/landmark_best.pt]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset import (_extract_frames, _extract_landmarks_from_frame,
                     _init_mediapipe)
from inference import _normalise_landmark_window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/landmark_best.pt')
    ap.add_argument('--config', default='config.yaml')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    classes = [str(c) for c in cfg['data']['classes']]
    raw = Path(cfg['data']['raw_dir'])
    num_frames = cfg['data']['num_frames']
    gate = int(round(num_frames * cfg['inference'].get('min_hand_frames_frac', 0.5)))
    conf_threshold = cfg['inference']['confidence_threshold']
    null_label = classes.index('null')

    hands = _init_mediapipe()

    print(f"Auditing `null` clips (gate = {gate}/{num_frames} frames with a hand)\n")

    clips = []
    for vid in sorted((raw / 'null').glob('*.mp4')):
        frames = _extract_frames(str(vid), num_frames=num_frames)
        if frames is None:
            continue
        window = np.zeros((num_frames, 63), dtype=np.float32)
        n_hand = 0
        for t, f in enumerate(frames):
            lm = _extract_landmarks_from_frame(hands, f)
            if lm is not None:
                window[t] = lm
                n_hand += 1
        clips.append((vid.stem, n_hand, window))

    empty = [c for c in clips if c[1] == 0]
    sparse = [c for c in clips if 0 < c[1] < gate]
    real = [c for c in clips if c[1] >= gate]

    print(f"  {len(clips)} null clips total")
    print(f"    {len(empty):3}  no hand at all        -> dropped at preprocessing")
    print(f"    {len(sparse):3}  hand in 1..{gate-1} frames    -> gated at inference, model never sees them")
    print(f"    {len(real):3}  hand in >={gate} frames     -> the only clips that TEST THE MODEL")
    print(f"\n  median frames with a hand: {np.median([c[1] for c in clips]):.0f}/{num_frames}")

    if not Path(args.checkpoint).exists():
        print(f"\n  (no checkpoint at {args.checkpoint} — skipping model evaluation)")
        return

    from models.fusion_head import LandmarkOnlyModel
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = LandmarkOnlyModel(num_classes=len(classes), input_dim=66, **ck['hparams'])
    model.load_state_dict(ck['model_state_dict'], strict=False)
    model.eval()

    if not real:
        print("\n  No null clip survives the gate — the model's `null` behaviour for the")
        print("  case that matters (hand visible, not swiping) is entirely UNTESTED.")
        return

    # Which split is each clip in? Most of them will be in TRAIN, so a correct
    # prediction there is memorisation, not evidence of generalisation.
    split_of = {}
    cache = Path('data/landmarks.npz')
    if cache.exists():
        from train_landmark import split_by_clip
        d = np.load(cache, allow_pickle=True)
        tr, va, te = split_by_clip(d['clip_ids'], d['y'], seed=42)
        for name, idxs in (('TRAIN', tr), ('val', va), ('TEST', te)):
            for cid in d['clip_ids'][idxs]:
                split_of[str(cid).split('/')[-1]] = name

    print(f"\n  Model on the {len(real)} clips that actually test it:")
    correct = would_fire = 0
    for stem, n_hand, window in real:
        enc = _normalise_landmark_window(window)
        with torch.no_grad():
            p = torch.softmax(model(torch.from_numpy(enc).unsqueeze(0)), -1)[0]
        pred, conf = int(p.argmax()), float(p.max())
        ok = pred == null_label
        correct += ok
        fires = (not ok) and conf >= conf_threshold
        would_fire += fires
        flag = "FALSE FIRE" if fires else ("ok" if ok else "wrong, below threshold")
        where = split_of.get(stem, '?')
        print(f"    [{where:5}] {stem[:38]:40} {n_hand:2}/{num_frames} hand  "
              f"-> {classes[pred]:12} {conf*100:5.1f}%  {flag}")

    held_out = [s for s, _, _ in real if split_of.get(s) in ('TEST', 'val')]
    print(f"\n    correct: {correct}/{len(real)}")
    print(f"    of these, held out from training: {len(held_out)}")
    print(f"    would fire a spurious gesture: {would_fire}/{len(real)}")
    print(f"\n  Caveat: {len(real)} clips is far too few to trust this number. It is")
    print("  reported to show the size of the gap, not to certify the model.")


if __name__ == '__main__':
    main()
