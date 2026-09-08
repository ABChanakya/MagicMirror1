#!/usr/bin/env python3
"""
scaling_study.py — where is more data actually worth collecting?

Varies the number of `null` clips and the number of swipe clips independently
and measures balanced accuracy under repeated k-fold CV. The point is to answer
a collection question with evidence: swipes already have 350+347 clips while
`null` has 29, so the marginal clip is almost certainly worth far more in one
place than the other, and this quantifies by how much.

Two curves:
  A. null clips varied, swipe clips held at full
  B. swipe clips varied, null clips held at full

If curve B is flat, collecting more swipes is wasted effort. If curve A is still
climbing at the right-hand end, `null` is the binding constraint.

Usage: python scaling_study.py [--folds 5] [--repeats 5] [--epochs 100]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments import cv_eval
from train_landmark import DEFAULT_AUG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/landmarks.npz')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--config', default='experiment_results.json',
                    help='use the winning config from experiments.py if present')
    ap.add_argument('--d-model', type=int, default=None)
    ap.add_argument('--layers', type=int, default=None)
    ap.add_argument('--nhead', type=int, default=None)
    ap.add_argument('--out', default='scaling_results.json')
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, y = d['X'], d['y']
    classes = [str(c) for c in d['classes']]
    K = len(classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    null_lbl = classes.index('null')

    cfg = dict(lr=3e-4, batch_size=32, weight_decay=1e-3, label_smoothing=0.05,
               dropout=0.2, seed=0, class_weights=True,
               d_model=128, layers=3, nhead=4, aug=DEFAULT_AUG)
    if Path(args.config).exists():
        try:
            best = json.loads(Path(args.config).read_text())['meta']['best_config']
            for k in ('lr', 'dropout', 'weight_decay', 'd_model', 'layers',
                      'nhead', 'class_weights', 'label_smoothing', 'batch_size'):
                if k in best:
                    cfg[k] = type(cfg[k])(best[k]) if not isinstance(best[k], bool) else best[k]
            print(f"using winning config from {args.config}")
        except Exception as e:
            print(f"could not read {args.config} ({e}); using defaults")

    for k, v in (('d_model', args.d_model), ('layers', args.layers), ('nhead', args.nhead)):
        if v is not None:
            cfg[k] = v
    print(f"config: d_model={cfg['d_model']} layers={cfg['layers']} lr={cfg['lr']} "
          f"dropout={cfg['dropout']} class_weights={cfg['class_weights']}")
    print(f"{args.folds}-fold x {args.repeats} repeats, {args.epochs} epochs\n")

    null_idx = np.where(y == null_lbl)[0]
    swipe_idx = np.where(y != null_lbl)[0]
    rng = np.random.RandomState(0)
    out = {'classes': classes, 'A_vary_null': [], 'B_vary_swipe': []}

    def run(subset):
        a, b, conf, _ = cv_eval(X, y, cfg, K, device, args.epochs,
                                args.folds, args.repeats, subset=subset)
        rec = conf.diagonal() / np.maximum(conf.sum(axis=1), 1)
        return a.mean(), b.mean(), b.std(), rec

    # Sample points across whatever null data exists, so the curve stays
    # informative as the dataset grows.
    n_null = len(null_idx)
    pts = sorted({max(K, int(round(n_null * f))) for f in (0.12, 0.25, 0.45, 0.7, 1.0)})
    print(f"A. varying `null` clips (swipes held at full {len(swipe_idx)})")
    for n in pts:
        keep = rng.choice(null_idx, size=min(n, len(null_idx)), replace=False)
        subset = np.sort(np.concatenate([swipe_idx, keep]))
        acc, bal, std, rec = run(subset)
        print(f"   null={n:3}  balanced {bal*100:5.1f}+-{std*100:4.1f}%   "
              f"null recall {rec[null_lbl]*100:5.1f}%   acc {acc*100:5.1f}%")
        out['A_vary_null'].append({'n_null': int(n), 'balanced': float(bal),
                                   'balanced_std': float(std),
                                   'null_recall': float(rec[null_lbl])})

    print("\nB. varying swipe clips (`null` held at full 29)")
    for frac in [0.15, 0.3, 0.5, 0.75, 1.0]:
        keep = []
        for lbl in range(K):
            if lbl == null_lbl:
                continue
            idx = np.where(y == lbl)[0]
            keep.append(rng.choice(idx, size=max(3, int(len(idx) * frac)), replace=False))
        subset = np.sort(np.concatenate(keep + [null_idx]))
        n_swipe = sum(len(k) for k in keep)
        acc, bal, std, rec = run(subset)
        print(f"   swipes={n_swipe:4} ({frac*100:3.0f}%)  balanced {bal*100:5.1f}+-{std*100:4.1f}%   "
              f"null recall {rec[null_lbl]*100:5.1f}%   acc {acc*100:5.1f}%")
        out['B_vary_swipe'].append({'n_swipe': int(n_swipe), 'frac': float(frac),
                                    'balanced': float(bal), 'balanced_std': float(std),
                                    'null_recall': float(rec[null_lbl])})

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")

    a0 = out['A_vary_null'][0]['balanced']
    a1 = out['A_vary_null'][-1]['balanced']
    b0 = out['B_vary_swipe'][0]['balanced']
    b1 = out['B_vary_swipe'][-1]['balanced']
    print("\nInterpretation")
    an0, an1 = out['A_vary_null'][0]['n_null'], out['A_vary_null'][-1]['n_null']
    bn0, bn1 = out['B_vary_swipe'][0]['n_swipe'], out['B_vary_swipe'][-1]['n_swipe']
    print(f"  null  {an0} -> {an1} clips : {a0*100:.1f}% -> {a1*100:.1f}%  "
          f"({(a1-a0)*100:+.1f} pts for +{an1-an0} clips, {(a1-a0)/max(an1-an0,1)*100:.3f} pts/clip)")
    print(f"  swipe {bn0} -> {bn1} clips: {b0*100:.1f}% -> {b1*100:.1f}%  "
          f"({(b1-b0)*100:+.1f} pts for +{bn1-bn0} clips, {(b1-b0)/max(bn1-bn0,1)*100:.3f} pts/clip)")
    # Has the null curve flattened? Compare the last step against its own spread.
    last = out['A_vary_null'][-1]; prev = out['A_vary_null'][-2]
    step = (last['balanced'] - prev['balanced']) * 100
    print(f"\n  last null step ({prev['n_null']} -> {last['n_null']}): {step:+.1f} pts "
          f"against +-{last['balanced_std']*100:.1f} spread -> "
          f"{'SATURATED, stop collecting' if abs(step) < last['balanced_std']*100 else 'still climbing'}")


if __name__ == '__main__':
    main()
