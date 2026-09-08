#!/usr/bin/env python3
"""
train_twostream.py — landmarks + video together.

Trains three models on the same folds so the video branch has to earn its place:

    landmark-only   the 174-dim body+hands skeleton alone
    video-only      Video Swin alone
    fusion          both streams

Reporting all three matters because a fusion model that beats landmark-only is
not evidence the pixels helped — it may simply be a bigger model. The comparison
here is like-for-like: same splits, same schedule, same metric.

Video Swin stays frozen except its last stage. It is 88M parameters against ~900
clips, and unfrozen it memorises appearance rather than motion — which is how it
scored ~100% on held-out clips of one person and then failed on everyone else.

Usage
-----
python train_twostream.py --mode fusion
python train_twostream.py --mode all --epochs 30
python train_twostream.py --mode landmark --epochs 60      # fast baseline
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_twostream import TwoStreamDataset


def split_by_session(clip_ids, y, classes, test_session='20260816'):
    """
    Hold out an entire recording session.

    Random splitting hides a shortcut here: session 20260827 is 200 `null` clips
    and nothing else, so a model that sees pixels can recognise that day's room,
    lighting and clothing and get 85% of `null` right without learning anything
    about hand motion. Same-session clips landing on both sides of a random
    split make that look like real accuracy.

    Holding out a whole session removes the shortcut: the test session's
    appearance was never seen in training, so only transferable signal helps.
    20260816 is the default because it is the one session containing all three
    classes.
    """
    import re
    sess = np.array([re.search(r'(\d{8})-\d{6}', c).group(1)
                     if re.search(r'(\d{8})-\d{6}', c) else 'unknown'
                     for c in clip_ids])
    te = np.where(sess == test_session)[0]
    rest = np.where(sess != test_session)[0]
    rng = np.random.RandomState(0)
    rng.shuffle(rest)
    n_val = max(1, int(len(rest) * 0.12))
    return rest[n_val:], rest[:n_val], te


def split_by_clip(clip_ids, y, seed=42, val_frac=0.15, test_frac=0.15):
    assert len(set(clip_ids)) == len(clip_ids), "one row per clip expected"
    rng = np.random.RandomState(seed)
    tr, va, te = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        nv, nt = max(1, round(n * val_frac)), max(1, round(n * test_frac))
        va += list(idx[:nv])
        te += list(idx[nv:nv + nt])
        tr += list(idx[nv + nt:])
    return np.array(tr), np.array(va), np.array(te)


@torch.no_grad()
def evaluate(model, loader, device, K, mode):
    model.eval()
    conf = np.zeros((K, K), int)
    for lm, fr, label in loader:
        lm, label = lm.to(device), label.to(device)
        if mode == 'landmark':
            out = model(lm)
        elif mode == 'video':
            out = model(fr.to(device))
        else:
            out = model(lm, fr.to(device))
        for t, p in zip(label.cpu().numpy(), out.argmax(1).cpu().numpy()):
            conf[t, p] += 1
    acc = conf.diagonal().sum() / max(conf.sum(), 1)
    per = conf.sum(1)
    rec = conf.diagonal() / np.maximum(per, 1)
    bal = float(rec[per > 0].mean()) if (per > 0).any() else 0.0
    return acc, bal, conf, rec


def build(mode, K, input_dim, args, device):
    from models.fusion_head import GestureRecognitionModel, LandmarkOnlyModel

    if mode == 'landmark':
        m = LandmarkOnlyModel(
            landmark_d_model=args.d_model, landmark_nhead=args.nhead,
            landmark_num_layers=args.layers, landmark_dim_feedforward=args.d_model * 4,
            landmark_dropout=args.dropout, head_hidden=args.d_model * 2,
            head_dropout=args.dropout, num_classes=K, input_dim=input_dim)
        # list(), not the generator: the optimiser consumes a generator, leaving
        # clip_grad_norm_ with nothing and silently disabling gradient clipping.
        return m.to(device), list(m.parameters())

    full = GestureRecognitionModel(
        landmark_d_model=args.d_model, landmark_nhead=args.nhead,
        landmark_num_layers=args.layers, landmark_dim_feedforward=args.d_model * 4,
        landmark_dropout=args.dropout, fusion_hidden=args.d_model * 2,
        fusion_dropout=args.dropout, num_classes=K,
        pretrained_swin=True, input_dim=input_dim).to(device)

    # Kinetics features are the only reason the video branch is usable at this
    # data scale; keep all but the last stage frozen.
    full.video_encoder.freeze_except_last_stage()

    if mode == 'video':
        class VideoOnly(nn.Module):
            def __init__(self, enc, dim, hidden, K, p):
                super().__init__()
                self.video_encoder = enc
                self.head = nn.Sequential(
                    nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                    nn.Dropout(p), nn.Linear(hidden, K))

            def forward(self, frames):
                return self.head(self.video_encoder(frames))

        m = VideoOnly(full.video_encoder, 1024, args.d_model * 2, K, args.dropout).to(device)
        return m, [p for p in m.parameters() if p.requires_grad]

    return full, [p for p in full.parameters() if p.requires_grad]


def run_one(mode, data, args, device):
    X, y, rel, clip_ids, classes, flip_map, root = data
    K = len(classes)
    if args.split == 'session':
        tr, va, te = split_by_session(clip_ids, y, classes, args.test_session)
    else:
        tr, va, te = split_by_clip(clip_ids, y, seed=args.seed)
    use_video = mode in ('video', 'fusion')

    def mk(idx, train):
        return TwoStreamDataset(X[idx], y[idx], rel[idx], root, flip_map,
                                train=train, num_frames=args.num_frames,
                                size=args.image_size, use_video=use_video,
                                appearance=args.appearance, seed=args.seed)

    nw = args.workers if use_video else 0
    ld_tr = DataLoader(mk(tr, True), batch_size=args.batch_size, shuffle=True,
                       num_workers=nw, pin_memory=use_video, drop_last=False,
                       persistent_workers=bool(nw))
    ld_va = DataLoader(mk(va, False), batch_size=args.batch_size, num_workers=nw)
    ld_te = DataLoader(mk(te, False), batch_size=args.batch_size, num_workers=nw)

    model, params = build(mode, K, X.shape[2], args, device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"\n=== {mode.upper()} ===  trainable {n_train:,} / {n_all:,}")

    counts = np.bincount(y[tr], minlength=K)
    w = counts.sum() / np.maximum(counts, 1)
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing,
                               weight=w if args.class_weights else None)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    epochs = 2 if args.smoke_test else args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_bal, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        tot = cor = 0
        loss_sum = 0.0
        for lm, fr, label in ld_tr:
            lm, label = lm.to(device), label.to(device)
            opt.zero_grad()
            if mode == 'landmark':
                out = model(lm)
            elif mode == 'video':
                out = model(fr.to(device, non_blocking=True))
            else:
                out = model(lm, fr.to(device, non_blocking=True))
            loss = crit(out, label)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            loss_sum += loss.item() * label.numel()
            cor += (out.argmax(1) == label).sum().item()
            tot += label.numel()
            if args.smoke_test:
                break
        sched.step()

        _, vbal, _, _ = evaluate(model, ld_va, device, K, mode)
        mark = ''
        if vbal > best_bal:
            best_bal = vbal
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = '  <-- best'
        if ep % 5 == 0 or ep == 1 or mark:
            print(f"  epoch {ep:3}/{epochs} | loss {loss_sum/max(tot,1):.4f} | "
                  f"train {cor/max(tot,1)*100:5.1f}% | val-balanced {vbal*100:5.1f}%{mark}")

    model.load_state_dict(best_state)
    acc, bal, conf, rec = evaluate(model, ld_te, device, K, mode)
    print(f"  TEST acc {acc*100:.1f}%  balanced {bal*100:.1f}%   " +
          "  ".join(f"{c}={r*100:.0f}%" for c, r in zip(classes, rec)))

    ck = Path(args.out_dir) / f"twostream_{mode}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'mode': mode,
                'classes': classes, 'input_dim': int(X.shape[2]),
                'test_acc': acc, 'test_balanced_acc': bal,
                'hparams': {'landmark_d_model': args.d_model,
                            'landmark_nhead': args.nhead,
                            'landmark_num_layers': args.layers,
                            'landmark_dim_feedforward': args.d_model * 4,
                            'landmark_dropout': args.dropout,
                            'head_hidden': args.d_model * 2,
                            'head_dropout': args.dropout}}, ck)
    return {'mode': mode, 'test_acc': acc, 'test_balanced': bal,
            'per_class': {c: float(r) for c, r in zip(classes, rec)},
            'trainable': int(n_train)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/landmarks_holistic.npz')
    ap.add_argument('--video-root', default='../camera/gesture_training_data')
    ap.add_argument('--mode', default='all',
                    choices=['landmark', 'video', 'fusion', 'all'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--weight-decay', type=float, default=1e-3)
    ap.add_argument('--label-smoothing', type=float, default=0.05)
    ap.add_argument('--d-model', type=int, default=192)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--nhead', type=int, default=6)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--num-frames', type=int, default=30)
    ap.add_argument('--image-size', type=int, default=112)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--class-weights', action='store_true', default=True)
    ap.add_argument('--no-class-weights', dest='class_weights', action='store_false')
    ap.add_argument('--out-dir', default='checkpoints')
    ap.add_argument('--split', default='clip', choices=['clip', 'session'],
                    help="'session' holds out a whole recording day, which "
                         "removes the appearance shortcut a pixel model can use")
    ap.add_argument('--test-session', default='20260816')
    ap.add_argument('--appearance', default='light', choices=['light', 'strong'],
                    help="'strong' adds crop/colour/grayscale/cutout to break "
                         "the appearance-to-session shortcut")
    ap.add_argument('--smoke-test', action='store_true')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(args.data, allow_pickle=True)
    X, y = d['X'], d['y']
    classes = [str(c) for c in d['classes']]
    rel = d['rel_paths']
    clip_ids = d['clip_ids']

    # flip map by class name, so it follows config rather than fixed indices
    name_flip = {'swipe_left': 'swipe_right', 'swipe_right': 'swipe_left'}
    flip_map = {i: classes.index(name_flip.get(c, c)) for i, c in enumerate(classes)}

    print(f"data X={X.shape}  classes={classes}")
    for i, c in enumerate(classes):
        print(f"  {c:12} {int((y == i).sum()):4}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}  input_dim={X.shape[2]}  flip_map={flip_map}")

    data = (X, y, rel, clip_ids, classes, flip_map, args.video_root)
    modes = ['landmark', 'video', 'fusion'] if args.mode == 'all' else [args.mode]
    results = [run_one(m, data, args, device) for m in modes]

    print("\n" + "=" * 68)
    print(f"{'model':12} {'trainable':>12} {'test acc':>10} {'balanced':>10}")
    for r in results:
        print(f"{r['mode']:12} {r['trainable']:>12,} {r['test_acc']*100:>9.1f}% "
              f"{r['test_balanced']*100:>9.1f}%")

    if len(results) > 1:
        lm = next((r for r in results if r['mode'] == 'landmark'), None)
        fu = next((r for r in results if r['mode'] == 'fusion'), None)
        if lm and fu:
            delta = (fu['test_balanced'] - lm['test_balanced']) * 100
            print(f"\nfusion vs landmark-only: {delta:+.1f} points balanced accuracy")
            print("  -> video branch earns its place" if delta > 2 else
                  "  -> video adds little; landmark-only is the cheaper model")

    Path('twostream_results.json').write_text(json.dumps(results, indent=2))
    print("\nwrote twostream_results.json")


if __name__ == '__main__':
    main()
