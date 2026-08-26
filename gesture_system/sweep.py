#!/usr/bin/env python3
"""
sweep.py — compare training variants with k-fold cross-validation.

Why CV rather than the single split train_landmark.py uses: `null` has only 29
clips. A 70/15/15 split leaves ~4 of them in test, so one run's `null` recall
moves in 25-point steps and any comparison between variants is noise. k-fold
puts every clip in test exactly once and reports mean +/- std across folds, so a
difference between configurations means something.

Reports balanced accuracy (mean per-class recall) as the headline, because with
350/347/29 plain accuracy barely moves even if `null` is predicted 0% correctly.

Usage
-----
python sweep.py                     # full sweep, writes sweep_results.json
python sweep.py --folds 3 --quick   # faster, fewer epochs
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.fusion_head import LandmarkOnlyModel
from train_landmark import LandmarkDataset, augment, evaluate


def stratified_folds(y, k, seed=0):
    """Split indices into k stratified folds; every clip lands in exactly one."""
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, sample in enumerate(idx):
            folds[i % k].append(sample)
    return [np.array(sorted(f)) for f in folds]


def train_one(X, y, train_idx, val_idx, test_idx, cfg, num_classes, device, epochs):
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    train_ld = DataLoader(LandmarkDataset(X[train_idx], y[train_idx], train=True),
                          batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
    val_ld = DataLoader(LandmarkDataset(X[val_idx], y[val_idx], train=False), batch_size=128)
    test_ld = DataLoader(LandmarkDataset(X[test_idx], y[test_idx], train=False), batch_size=128)

    model = LandmarkOnlyModel(
        landmark_d_model=cfg['d_model'],
        landmark_nhead=cfg['nhead'],
        landmark_num_layers=cfg['layers'],
        landmark_dim_feedforward=cfg['d_model'] * 4,
        landmark_dropout=cfg['dropout'],
        head_hidden=cfg['d_model'] * 2,
        head_dropout=cfg['dropout'],
        num_classes=num_classes,
        input_dim=66,
    ).to(device)

    if cfg['class_weights']:
        counts = np.bincount(y[train_idx], minlength=num_classes)
        w = counts.sum() / np.maximum(counts, 1)
        w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    else:
        w = None

    crit = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'], weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for lm, label in train_ld:
            lm, label = lm.to(device), label.to(device)
            opt.zero_grad()
            loss = crit(model(lm), label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        _, val_bal, _ = evaluate(model, val_ld, device, num_classes)
        if val_bal > best_val:
            best_val = val_bal
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc, test_bal, conf = evaluate(model, test_ld, device, num_classes)
    n_params = sum(p.numel() for p in model.parameters())
    return test_acc, test_bal, conf, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/landmarks.npz')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='sweep_results.json')
    args = ap.parse_args()

    epochs = 20 if args.quick else args.epochs

    d = np.load(args.data, allow_pickle=True)
    X, y = d['X'], d['y']
    classes = [str(c) for c in d['classes']]
    num_classes = len(classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"data: X={X.shape}  classes={classes}")
    for i, c in enumerate(classes):
        print(f"  {c:12} {int((y == i).sum()):4}")
    print(f"device: {device} | {args.folds}-fold CV | {epochs} epochs/fold\n")

    # Variants. Model size matters because 726 clips is small — a 5M-param
    # transformer has ample capacity to memorise them.
    grid = {
        'class_weights': [True, False],
        'size': [('tiny', 64, 2, 4), ('small', 128, 3, 4), ('medium', 256, 4, 8)],
    }
    base = dict(lr=3e-4, batch_size=32, weight_decay=1e-3,
                label_smoothing=0.05, dropout=0.2, seed=0)

    folds = stratified_folds(y, args.folds, seed=0)
    results = []

    combos = list(itertools.product(grid['class_weights'], grid['size']))
    print(f"{len(combos)} configurations x {args.folds} folds = {len(combos)*args.folds} runs\n")

    for cw, (size_name, d_model, layers, nhead) in combos:
        cfg = dict(base, class_weights=cw, d_model=d_model, layers=layers, nhead=nhead)
        name = f"{size_name:6} weights={'on ' if cw else 'off'}"

        accs, bals, per_class = [], [], np.zeros((num_classes, num_classes), dtype=int)
        t0 = time.time()
        for fi in range(args.folds):
            test_idx = folds[fi]
            val_idx = folds[(fi + 1) % args.folds]
            train_idx = np.concatenate([folds[j] for j in range(args.folds)
                                        if j not in (fi, (fi + 1) % args.folds)])
            acc, bal, conf, n_params = train_one(
                X, y, train_idx, val_idx, test_idx, cfg, num_classes, device, epochs)
            accs.append(acc)
            bals.append(bal)
            per_class += conf

        recalls = per_class.diagonal() / np.maximum(per_class.sum(axis=1), 1)
        elapsed = time.time() - t0
        print(f"  {name} | {n_params/1e3:6.0f}K params | "
              f"acc {np.mean(accs)*100:5.1f}+-{np.std(accs)*100:4.1f}  "
              f"balanced {np.mean(bals)*100:5.1f}+-{np.std(bals)*100:4.1f}  | "
              f"recall " + " ".join(f"{c}={r*100:.0f}%" for c, r in zip(classes, recalls)) +
              f"  [{elapsed:.0f}s]")

        results.append({
            'size': size_name, 'class_weights': cw, 'params': int(n_params),
            'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
            'balanced_mean': float(np.mean(bals)), 'balanced_std': float(np.std(bals)),
            'per_class_recall': {c: float(r) for c, r in zip(classes, recalls)},
            'confusion_summed': per_class.tolist(),
            'cfg': {k: v for k, v in cfg.items()},
        })

    results.sort(key=lambda r: -r['balanced_mean'])
    print(f"\n{'='*78}\nRanked by balanced accuracy (mean over {args.folds} folds):")
    for r in results:
        print(f"  {r['balanced_mean']*100:5.1f}+-{r['balanced_std']*100:4.1f}%  "
              f"{r['size']:6} weights={'on' if r['class_weights'] else 'off':3} "
              f"({r['params']/1e3:.0f}K params)")

    Path(args.out).write_text(json.dumps(
        {'classes': classes, 'folds': args.folds, 'epochs': epochs, 'results': results},
        indent=2))
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
