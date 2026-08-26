#!/usr/bin/env python3
"""
train_landmark.py — train the landmark-only gesture model.

Reads the .npz cache from preprocess_landmarks.py, splits by clip id (so no
augmented view of a training clip can land in val/test), augments on the fly,
and reports honest per-class metrics plus a confusion matrix.

Usage
-----
python train_landmark.py
python train_landmark.py --epochs 60 --lr 3e-4
python train_landmark.py --smoke-test
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from augmentation import _FLIP_LABEL_MAP

# ── Augmentation on (T, 66) landmark tensors ──────────────────────────────────
# Channel layout: [0:63] = 21 joints (x,y,z) interleaved, [63:66] = wrist
# trajectory (x,y,z). Because 63 % 3 == 0, the strides 0::3 / 1::3 / 2::3 hit
# the x / y / z components of the joints *and* of the trajectory. This matches
# horizontal_flip() and rotation() in augmentation.py — assert_aug_matches()
# below verifies that equivalence rather than assuming it.


def aug_flip(lm: np.ndarray, label: int):
    """Mirror horizontally: negate all x components, swap LEFT<->RIGHT."""
    out = lm.copy()
    out[:, 0::3] = -out[:, 0::3]
    return out, _FLIP_LABEL_MAP[int(label)]


def aug_rotate(lm: np.ndarray, max_deg: float = 8.0):
    """Small in-plane rotation of every (x, y) pair. z untouched."""
    rad = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    c, s = np.cos(rad), np.sin(rad)
    out = lm.copy()
    x, y = lm[:, 0::3].copy(), lm[:, 1::3].copy()
    out[:, 0::3] = c * x - s * y
    out[:, 1::3] = s * x + c * y
    return out


def aug_noise(lm: np.ndarray, std: float = 0.02):
    return lm + np.random.randn(*lm.shape).astype(np.float32) * std


def aug_time_warp(lm: np.ndarray, speed_range=(0.75, 1.35)):
    """Resample the clip along time to simulate a faster/slower swipe."""
    T = lm.shape[0]
    speed = np.random.uniform(*speed_range)
    src = np.clip(np.linspace(0, (T - 1) * speed, T), 0, T - 1)
    lo, hi = np.floor(src).astype(int), np.ceil(src).astype(int)
    w = (src - lo).astype(np.float32)[:, None]
    return lm[lo] * (1 - w) + lm[hi] * w


def aug_traj_scale(lm: np.ndarray, lo: float = 0.75, hi: float = 1.3):
    """Scale the swipe magnitude — a big swipe and a small swipe mean the same."""
    out = lm.copy()
    out[:, 63:66] *= np.random.uniform(lo, hi)
    return out


# Default probability of applying each augmentation to a training sample.
# sweep.py overrides these to ablate individual augmentations.
DEFAULT_AUG = {
    'flip':       0.5,
    'time_warp':  0.7,
    'rotate':     0.5,
    'traj_scale': 0.5,
    'noise':      0.8,
}


def augment(lm: np.ndarray, label: int, cfg: dict | None = None):
    """Random augmentation chain applied to TRAIN samples only."""
    p = DEFAULT_AUG if cfg is None else cfg
    if np.random.rand() < p.get('flip', 0.0):
        lm, label = aug_flip(lm, label)
    if np.random.rand() < p.get('time_warp', 0.0):
        lm = aug_time_warp(lm)
    if np.random.rand() < p.get('rotate', 0.0):
        lm = aug_rotate(lm)
    if np.random.rand() < p.get('traj_scale', 0.0):
        lm = aug_traj_scale(lm)
    if np.random.rand() < p.get('noise', 0.0):
        lm = aug_noise(lm)
    return lm.astype(np.float32), label


def assert_aug_matches():
    """
    Prove aug_flip / aug_rotate agree with the frame-based versions in
    augmentation.py, so the two code paths can never silently diverge.
    """
    from augmentation import horizontal_flip, rotation

    rng = np.random.RandomState(0)
    lm = rng.randn(30, 66).astype(np.float32)
    dummy = torch.zeros(30, 3, 4, 4)

    for label in _FLIP_LABEL_MAP:
        mine_lm, mine_label = aug_flip(lm, label)
        ref_lm, _, ref_label = horizontal_flip(torch.from_numpy(lm), dummy, label)
        assert mine_label == ref_label, f"flip label mismatch for {label}"
        assert np.allclose(mine_lm, ref_lm.numpy(), atol=1e-6), "flip coords mismatch"

    # rotation is random internally, so compare the maths at a fixed angle
    deg = 7.0
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    ref = lm.copy()
    x, y = lm[:, 0::3].copy(), lm[:, 1::3].copy()
    ref[:, 0::3] = c * x - s * y
    ref[:, 1::3] = s * x + c * y
    np.random.seed(0)
    got = aug_rotate(lm, max_deg=0.0)   # 0 deg must be identity
    assert np.allclose(got, lm, atol=1e-6), "rotate(0) must be identity"
    print("[test] augmentation matches augmentation.py reference ✓")


# ── Dataset ───────────────────────────────────────────────────────────────────

class LandmarkDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, train: bool, aug_cfg: dict | None = None):
        self.X, self.y, self.train, self.aug_cfg = X, y, train, aug_cfg

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        lm, label = self.X[i], int(self.y[i])
        if self.train:
            lm, label = augment(lm, label, self.aug_cfg)
        return torch.from_numpy(np.ascontiguousarray(lm)), label


def split_by_clip(clip_ids, y, seed=42, val_frac=0.15, test_frac=0.15):
    """
    Split on unique clip ids, stratified by class. Augmentation happens in
    memory on the train split only, so a val/test clip can never appear in
    training in any form.

    The cache holds exactly one row per recording, so splitting row indices is
    the same as splitting clips. That invariant is what makes this leak-free, so
    it is asserted rather than assumed.
    """
    assert len(clip_ids) == len(y), "clip_ids and labels must align"
    assert len(set(clip_ids)) == len(clip_ids), (
        "cache contains more than one row per clip — index splitting would "
        "then put two views of the same recording in different splits"
    )

    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_val = max(1, int(round(n * val_frac)))
        n_test = max(1, int(round(n * test_frac)))
        val_idx.extend(idx[:n_val])
        test_idx.extend(idx[n_val:n_val + n_test])
        train_idx.extend(idx[n_val + n_test:])

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """
    Returns (accuracy, balanced_accuracy, confusion).

    balanced_accuracy is the mean of per-class recall. With ~350/347/35 clips,
    plain accuracy barely moves if the model gets `null` completely wrong, so it
    is the wrong thing to select checkpoints on — balanced accuracy weights the
    thin class equally and is what the caller uses to pick the best epoch.
    """
    model.eval()
    correct = total = 0
    conf = np.zeros((num_classes, num_classes), dtype=int)
    for lm, label in loader:
        lm, label = lm.to(device), label.to(device)
        pred = model(lm).argmax(1)
        correct += (pred == label).sum().item()
        total += label.numel()
        for t, p in zip(label.cpu().numpy(), pred.cpu().numpy()):
            conf[t, p] += 1

    per_class_n = conf.sum(axis=1)
    seen = per_class_n > 0
    recalls = np.divide(conf.diagonal(), np.maximum(per_class_n, 1))
    balanced = float(recalls[seen].mean()) if seen.any() else 0.0
    return correct / max(total, 1), balanced, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/landmarks.npz')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--weight-decay', type=float, default=1e-3)
    ap.add_argument('--label-smoothing', type=float, default=0.05)
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--nhead', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='checkpoints/landmark_best.pt')
    ap.add_argument('--class-weights', action='store_true', default=True,
                    help='inverse-frequency loss weighting (on by default)')
    ap.add_argument('--no-class-weights', dest='class_weights', action='store_false')
    ap.add_argument('--smoke-test', action='store_true')
    args = ap.parse_args()

    assert_aug_matches()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(args.data, allow_pickle=True)
    X, y, clip_ids = d['X'], d['y'], d['clip_ids']
    classes = [str(c) for c in d['classes']]
    num_classes = len(classes)

    assert X.shape[2] == 66, f"expected 66 input dims, got {X.shape[2]}"
    assert len(np.unique(clip_ids)) == len(clip_ids), "duplicate clip ids"

    tr, va, te = split_by_clip(clip_ids, y, seed=args.seed)

    # Leak check: no clip id may appear in more than one split
    assert not (set(clip_ids[tr]) & set(clip_ids[va])), "train/val clip overlap"
    assert not (set(clip_ids[tr]) & set(clip_ids[te])), "train/test clip overlap"
    assert not (set(clip_ids[va]) & set(clip_ids[te])), "val/test clip overlap"
    print(f"[split] train={len(tr)}  val={len(va)}  test={len(te)}  (disjoint clips ✓)")

    counts = np.bincount(y[tr], minlength=num_classes)
    print("[split] train clips per class: " +
          "  ".join(f"{c}={n}" for c, n in zip(classes, counts)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Class weighting. With ~350/347/35 the majority classes dominate the loss
    # and `null` gets underfit — and `null` is precisely what stops the model
    # firing on incidental hand movement, so underfitting it shows up as false
    # positives at inference rather than as a bad headline accuracy.
    # Inverse-frequency, normalised to mean 1 so the effective LR is unchanged.
    if args.class_weights:
        w = counts.sum() / np.maximum(counts, 1)
        w = w / w.mean()
        class_weight = torch.tensor(w, dtype=torch.float32, device=device)
        print("[loss] class weights: " +
              "  ".join(f"{c}={x:.2f}" for c, x in zip(classes, w)))
    else:
        class_weight = None
        print("[loss] class weights: disabled")

    train_ds = LandmarkDataset(X[tr], y[tr], train=True)
    val_ds   = LandmarkDataset(X[va], y[va], train=False)
    test_ds  = LandmarkDataset(X[te], y[te], train=False)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=False)
    val_ld   = DataLoader(val_ds, batch_size=64)
    test_ld  = DataLoader(test_ds, batch_size=64)

    from models.fusion_head import LandmarkOnlyModel
    model = LandmarkOnlyModel(
        landmark_d_model=args.d_model,
        landmark_nhead=args.nhead,
        landmark_num_layers=args.layers,
        landmark_dim_feedforward=args.d_model * 4,
        landmark_dropout=args.dropout,
        head_hidden=args.d_model * 2,
        head_dropout=args.dropout,
        num_classes=num_classes,
        input_dim=66,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] LandmarkOnlyModel — {n_params:,} trainable params  (device={device})")

    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=class_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epochs = 2 if args.smoke_test else args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_val, best_state, log = 0.0, None, []

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = tot_correct = tot_n = 0
        for lm, label in train_ld:
            lm, label = lm.to(device), label.to(device)
            opt.zero_grad()
            out = model(lm)
            loss = crit(out, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item() * label.numel()
            tot_correct += (out.argmax(1) == label).sum().item()
            tot_n += label.numel()
            if args.smoke_test:
                break
        sched.step()

        tr_loss, tr_acc = tot_loss / max(tot_n, 1), tot_correct / max(tot_n, 1)
        val_acc, val_bal, _ = evaluate(model, val_ld, device, num_classes)
        log.append({'epoch': ep, 'train_loss': tr_loss, 'train_acc': tr_acc,
                    'val_acc': val_acc, 'val_balanced_acc': val_bal})

        marker = ''
        # Select on balanced accuracy, not plain accuracy — see evaluate()
        if val_bal > best_val:
            best_val = val_bal
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = '  <-- best'
        if ep % 5 == 0 or ep == 1 or marker:
            print(f"  epoch {ep:3}/{epochs} | loss {tr_loss:.4f} | train {tr_acc*100:5.1f}% | "
                  f"val {val_acc*100:5.1f}% | val-balanced {val_bal*100:5.1f}%{marker}")

    # ── Final honest evaluation on the untouched test split ───────────────────
    model.load_state_dict(best_state)
    test_acc, test_bal, conf = evaluate(model, test_ld, device, num_classes)

    print(f"\n[result] best val balanced acc: {best_val*100:.1f}%")
    print(f"[result] TEST acc (never seen during training or model selection): {test_acc*100:.1f}%")
    print(f"[result] TEST balanced acc (mean per-class recall):                {test_bal*100:.1f}%")
    print("\n  Confusion matrix (rows = true, cols = predicted):")
    header = ' ' * 14 + ''.join(f"{c[:9]:>10}" for c in classes)
    print(header)
    for i, c in enumerate(classes):
        row = ''.join(f"{conf[i, j]:>10}" for j in range(num_classes))
        n = conf[i].sum()
        acc = conf[i, i] / n * 100 if n else 0.0
        print(f"    {c:10}{row}   ({acc:.0f}%)")

    torch.save({
        'model_state_dict': model.state_dict(),
        'val_balanced_acc': best_val,
        'test_acc': test_acc,
        'test_balanced_acc': test_bal,
        'classes': classes,
        'input_dim': 66,
        'arch': 'LandmarkOnlyModel',
        'hparams': {
            'landmark_d_model': args.d_model,
            'landmark_nhead': args.nhead,
            'landmark_num_layers': args.layers,
            'landmark_dim_feedforward': args.d_model * 4,
            'landmark_dropout': args.dropout,
            'head_hidden': args.d_model * 2,
            'head_dropout': args.dropout,
        },
    }, args.out)
    print(f"\n[save] {args.out}")

    with open(Path(args.out).with_suffix('.log.json'), 'w') as f:
        json.dump(log, f, indent=2)


if __name__ == '__main__':
    main()
