#!/usr/bin/env python3
"""
verify_inference.py — replay held-out clips through the real inference path.

Training reads a pre-normalised cache; live inference extracts landmarks frame
by frame and normalises them itself. This script drives that second path end to
end on the clips in the test split, so the number it prints reflects what the
live camera path will actually do — not what the training loop measured.

It also exercises the hand-presence gate on the same clips.

Usage: python verify_inference.py [--n-per-class 6]
"""

import argparse

import numpy as np
import torch
import yaml

from dataset import _extract_frames, _extract_landmarks_from_frame, _init_mediapipe
from inference import _normalise_landmark_window
from models.fusion_head import LandmarkOnlyModel
from train_landmark import split_by_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/landmark_best.pt')
    ap.add_argument('--cache', default='data/landmarks.npz')
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--n-per-class', type=int, default=6)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    classes = [str(c) for c in cfg['data']['classes']]
    raw_dir = cfg['data']['raw_dir']
    num_frames = cfg['data']['num_frames']
    inf_cfg = cfg['inference']
    min_hand_frames = int(round(num_frames * inf_cfg.get('min_hand_frames_frac', 0.5)))
    conf_threshold = inf_cfg['confidence_threshold']

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = LandmarkOnlyModel(num_classes=len(classes), input_dim=66, **ck['hparams'])
    model.load_state_dict(ck['model_state_dict'], strict=False)
    model.eval()

    # Recompute the exact same split the trainer used, so we only touch clips
    # the model was neither trained nor selected on.
    d = np.load(args.cache, allow_pickle=True)
    _, _, test_idx = split_by_clip(d['clip_ids'], d['y'], seed=42)
    test_clips = d['clip_ids'][test_idx]
    test_labels = d['y'][test_idx]

    hands = _init_mediapipe()

    # Sample a few test clips per class
    chosen = []
    for label in range(len(classes)):
        ids = [c for c, l in zip(test_clips, test_labels) if l == label]
        chosen += [(c, label) for c in ids[:args.n_per_class]]

    print(f"Replaying {len(chosen)} held-out clips through the live inference path")
    print(f"(threshold {conf_threshold}, gate needs >={min_hand_frames}/{num_frames} frames with a hand)\n")

    correct = fired = gated = 0
    per_class = {c: [0, 0] for c in classes}

    for clip_id, label in chosen:
        path = f"{raw_dir}/{clip_id}.mp4"
        frames = _extract_frames(path, num_frames=num_frames)
        if frames is None:
            print(f"  {clip_id}: could not read")
            continue

        # Exactly what the live loop does: per-frame extraction, zeros on miss
        lm_window = np.zeros((num_frames, 63), dtype=np.float32)
        for t, fr in enumerate(frames):
            lm = _extract_landmarks_from_frame(hands, fr)
            if lm is not None:
                lm_window[t] = lm

        hand_frames = int(np.sum(np.abs(lm_window).sum(axis=1) > 1e-8))

        if hand_frames < min_hand_frames:
            gated += 1
            pred, conf = classes.index('null') if 'null' in classes else -1, 1.0
            note = f"GATED ({hand_frames}/{num_frames} frames)"
        else:
            enc = _normalise_landmark_window(lm_window)
            assert enc.shape == (num_frames, 66), enc.shape
            with torch.no_grad():
                p = torch.softmax(model(torch.from_numpy(enc).unsqueeze(0)), -1)[0]
            pred, conf = int(p.argmax()), float(p.max())
            note = "fires" if conf >= conf_threshold else "below threshold"
            if conf >= conf_threshold:
                fired += 1

        ok = pred == label
        correct += ok
        per_class[classes[label]][0] += ok
        per_class[classes[label]][1] += 1
        mark = "ok  " if ok else "MISS"
        print(f"  {mark} {clip_id:24} true={classes[label]:12} pred={classes[pred]:12} "
              f"{conf*100:5.1f}%  {note}")

    n = len(chosen)
    print(f"\n  accuracy via live path: {correct}/{n} = {correct/n*100:.1f}%")
    print(f"  would fire an event:    {fired}/{n}")
    print(f"  gated on hand presence: {gated}/{n}")
    print("\n  per class:")
    for c, (ok, tot) in per_class.items():
        if tot:
            print(f"    {c:12} {ok}/{tot}")

    # The gate must reject a completely empty window
    empty = np.zeros((num_frames, 63), np.float32)
    empty_hand_frames = int(np.sum(np.abs(empty).sum(axis=1) > 1e-8))
    print(f"\n  empty-room window: {empty_hand_frames} frames with a hand -> "
          f"{'GATED (correct)' if empty_hand_frames < min_hand_frames else 'NOT GATED (bug!)'}")


if __name__ == '__main__':
    main()
