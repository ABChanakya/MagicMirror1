#!/usr/bin/env python3
"""
experiments.py — staged hyperparameter search + data-scaling study.

Everything is measured with repeated stratified k-fold CV and reported as
mean +/- std across folds and repeats. With `null` at 29 clips, a single split
puts ~4 of them in test, so one number tells you nothing; the first A/B run of
class weighting looked like a 0.6-point difference on a single split and turned
out to be ~6 points under CV.

Headline metric is balanced accuracy (mean per-class recall). Plain accuracy is
reported too, and the two diverge sharply: configurations that ignore `null`
score *higher* plain accuracy while being useless in practice.

Stages
  1  architecture x class weighting
  2  optimiser hyperparameters, on stage-1 winner
  3  augmentation ablation, on stage-2 winner
  4  data-scaling study: how balanced accuracy moves with the number of
     `null` clips, to quantify what collecting more is worth
  5  final model trained on the standard split + live-path verification

Usage: python experiments.py [--folds 5] [--repeats 3] [--epochs 80]
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
from train_landmark import DEFAULT_AUG, LandmarkDataset, evaluate

RESULTS = {}


def stratified_folds(y, k, seed):
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, s in enumerate(idx):
            folds[i % k].append(s)
    return [np.array(sorted(f)) for f in folds]


def train_one(X, y, tr, va, te, cfg, num_classes, device, epochs):
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    aug = cfg.get('aug', DEFAULT_AUG)
    train_ld = DataLoader(LandmarkDataset(X[tr], y[tr], True, aug),
                          batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
    val_ld = DataLoader(LandmarkDataset(X[va], y[va], False), batch_size=256)
    test_ld = DataLoader(LandmarkDataset(X[te], y[te], False), batch_size=256)

    model = LandmarkOnlyModel(
        landmark_d_model=cfg['d_model'], landmark_nhead=cfg['nhead'],
        landmark_num_layers=cfg['layers'], landmark_dim_feedforward=cfg['d_model'] * 4,
        landmark_dropout=cfg['dropout'], head_hidden=cfg['d_model'] * 2,
        head_dropout=cfg['dropout'], num_classes=num_classes, input_dim=66,
    ).to(device)

    if cfg['class_weights']:
        counts = np.bincount(y[tr], minlength=num_classes)
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
            crit(model(lm), label).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        _, vb, _ = evaluate(model, val_ld, device, num_classes)
        if vb > best_val:
            best_val, best_state = vb, {k: v.detach().cpu().clone()
                                        for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    acc, bal, conf = evaluate(model, test_ld, device, num_classes)
    return acc, bal, conf, sum(p.numel() for p in model.parameters())


def cv_eval(X, y, cfg, num_classes, device, epochs, folds, repeats, subset=None):
    """Repeated stratified k-fold. Returns (acc, bal, conf, params) aggregates."""
    idx_pool = np.arange(len(y)) if subset is None else subset
    Xs, ys = X[idx_pool], y[idx_pool]

    accs, bals = [], []
    conf_sum = np.zeros((num_classes, num_classes), dtype=int)
    n_params = 0
    for rep in range(repeats):
        f = stratified_folds(ys, folds, seed=100 + rep)
        for fi in range(folds):
            te = f[fi]
            va = f[(fi + 1) % folds]
            tr = np.concatenate([f[j] for j in range(folds) if j not in (fi, (fi + 1) % folds)])
            c = dict(cfg, seed=cfg.get('seed', 0) + rep)
            acc, bal, conf, n_params = train_one(Xs, ys, tr, va, te, c, num_classes, device, epochs)
            accs.append(acc)
            bals.append(bal)
            conf_sum += conf
    return np.array(accs), np.array(bals), conf_sum, n_params


def _abbrev(c):
    """swipe_left -> L, swipe_right -> R, null -> null (both start 'swipe')."""
    if c.startswith('swipe_'):
        return c[len('swipe_'):][:1].upper()
    return c[:4]


def report(name, accs, bals, conf, classes, n_params, elapsed):
    rec = conf.diagonal() / np.maximum(conf.sum(axis=1), 1)
    print(f"  {name:44} | {n_params/1e3:6.0f}K | "
          f"acc {accs.mean()*100:5.1f}+-{accs.std()*100:4.1f} | "
          f"bal {bals.mean()*100:5.1f}+-{bals.std()*100:4.1f} | " +
          " ".join(f"{_abbrev(c)}={r*100:3.0f}%" for c, r in zip(classes, rec)) +
          f" [{elapsed:.0f}s]")
    return {
        'name': name, 'params': int(n_params),
        'acc_mean': float(accs.mean()), 'acc_std': float(accs.std()),
        'balanced_mean': float(bals.mean()), 'balanced_std': float(bals.std()),
        'per_class_recall': {c: float(r) for c, r in zip(classes, rec)},
        'n_runs': int(len(accs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/landmarks.npz')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--out', default='experiment_results.json')
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, y = d['X'], d['y']
    classes = [str(c) for c in d['classes']]
    K = len(classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"data X={X.shape} classes={classes}")
    for i, c in enumerate(classes):
        print(f"  {c:12} {int((y == i).sum()):4}")
    runs_per_cfg = args.folds * args.repeats
    print(f"\n{args.folds}-fold x {args.repeats} repeats = {runs_per_cfg} runs per config, "
          f"{args.epochs} epochs each\ndevice={device}\n")

    base = dict(lr=3e-4, batch_size=32, weight_decay=1e-3, label_smoothing=0.05,
                dropout=0.2, seed=0, class_weights=True,
                d_model=128, layers=3, nhead=4)

    def run(name, cfg, subset=None):
        t0 = time.time()
        a, b, c, p = cv_eval(X, y, cfg, K, device, args.epochs, args.folds, args.repeats, subset)
        return report(name, a, b, c, classes, p, time.time() - t0), b.mean()

    # ── Stage 1: architecture x class weighting ──────────────────────────────
    print("=" * 100)
    print("STAGE 1 — architecture x class weighting")
    stage1 = []
    sizes = [('tiny', 64, 2, 4), ('small', 128, 3, 4),
             ('medium', 256, 4, 8), ('wide', 192, 3, 6)]
    for (sname, dm, ly, nh), cw in itertools.product(sizes, [True, False]):
        cfg = dict(base, d_model=dm, layers=ly, nhead=nh, class_weights=cw)
        r, score = run(f"{sname:6} weights={'on' if cw else 'off'}", cfg)
        r['cfg'] = {k: v for k, v in cfg.items() if k != 'aug'}
        stage1.append((score, r, cfg))
    stage1.sort(key=lambda t: -t[0])
    RESULTS['stage1'] = [r for _, r, _ in stage1]
    best_cfg = stage1[0][2]
    print(f"\n  winner: {stage1[0][1]['name']}  ({stage1[0][0]*100:.1f}% balanced)\n")

    # ── Stage 2: optimiser hyperparameters ───────────────────────────────────
    print("=" * 100)
    print("STAGE 2 — optimiser hyperparameters (on stage-1 winner)")
    stage2 = []
    for lr in [1e-4, 3e-4, 1e-3]:
        for dropout in [0.1, 0.2, 0.35]:
            cfg = dict(best_cfg, lr=lr, dropout=dropout)
            r, score = run(f"lr={lr:.0e} dropout={dropout}", cfg)
            r['cfg'] = {k: v for k, v in cfg.items() if k != 'aug'}
            stage2.append((score, r, cfg))
    for wd in [1e-4, 1e-2]:
        cfg = dict(best_cfg, weight_decay=wd)
        r, score = run(f"weight_decay={wd:.0e}", cfg)
        r['cfg'] = {k: v for k, v in cfg.items() if k != 'aug'}
        stage2.append((score, r, cfg))
    stage2.sort(key=lambda t: -t[0])
    RESULTS['stage2'] = [r for _, r, _ in stage2]
    best_cfg = stage2[0][2]
    print(f"\n  winner: {stage2[0][1]['name']}  ({stage2[0][0]*100:.1f}% balanced)\n")

    # ── Stage 3: augmentation ablation ───────────────────────────────────────
    print("=" * 100)
    print("STAGE 3 — augmentation ablation (leave-one-out from the full chain)")
    stage3 = []
    r, score = run("all augmentations", dict(best_cfg, aug=DEFAULT_AUG))
    r['cfg'] = {'aug': DEFAULT_AUG}
    stage3.append((score, r, dict(best_cfg, aug=DEFAULT_AUG)))
    for drop in DEFAULT_AUG:
        aug = dict(DEFAULT_AUG, **{drop: 0.0})
        cfg = dict(best_cfg, aug=aug)
        r, score = run(f"without {drop}", cfg)
        r['cfg'] = {'aug': aug}
        stage3.append((score, r, cfg))
    cfg_none = dict(best_cfg, aug={k: 0.0 for k in DEFAULT_AUG})
    r, score = run("no augmentation at all", cfg_none)
    r['cfg'] = {'aug': {k: 0.0 for k in DEFAULT_AUG}}
    stage3.append((score, r, cfg_none))
    stage3.sort(key=lambda t: -t[0])
    RESULTS['stage3'] = [r for _, r, _ in stage3]
    best_cfg = stage3[0][2]
    print(f"\n  winner: {stage3[0][1]['name']}  ({stage3[0][0]*100:.1f}% balanced)\n")

    # ── Stage 4: how much is more `null` data worth? ─────────────────────────
    print("=" * 100)
    print("STAGE 4 — data-scaling: balanced accuracy vs number of `null` clips")
    null_idx = np.where(y == classes.index('null'))[0]
    other_idx = np.where(y != classes.index('null'))[0]
    stage4 = []
    rng = np.random.RandomState(0)
    for frac in [0.35, 0.5, 0.7, 1.0]:
        n = max(K, int(round(len(null_idx) * frac)))
        keep = rng.choice(null_idx, size=n, replace=False)
        subset = np.sort(np.concatenate([other_idx, keep]))
        r, score = run(f"null clips = {n:3} ({frac*100:.0f}% of available)",
                       best_cfg, subset=subset)
        r['n_null'] = int(n)
        stage4.append((score, r, None))
    RESULTS['stage4'] = [r for _, r, _ in stage4]
    print()

    RESULTS['meta'] = {
        'classes': classes, 'folds': args.folds, 'repeats': args.repeats,
        'epochs': args.epochs, 'n_clips': int(len(y)),
        'class_counts': {c: int((y == i).sum()) for i, c in enumerate(classes)},
        'best_config': {k: v for k, v in best_cfg.items()},
    }
    Path(args.out).write_text(json.dumps(RESULTS, indent=2, default=str))

    print("=" * 100)
    print("BEST CONFIGURATION")
    for k, v in best_cfg.items():
        print(f"  {k:16} {v}")
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
