#!/usr/bin/env python3
"""
hparam_search.py — Automatic hyperparameter optimization using Optuna.

Optuna uses Bayesian optimization (TPE) to find the best hyperparameters
by intelligently sampling the search space, not random grid search.

Usage:
    python hparam_search.py               # 30 trials, ~2-3 hours
    python hparam_search.py --trials 10   # quick search
    python hparam_search.py --show        # show best params from previous run
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import yaml
import torch
import torch.nn as nn
import numpy as np
import optuna
from optuna.samplers import TPESampler
from pathlib import Path
from torch.utils.data import DataLoader

optuna.logging.set_verbosity(optuna.logging.WARNING)

CONFIG_PATH  = Path("config.yaml")
STUDY_PATH   = Path("checkpoints/hparam_study.db")
RESULTS_PATH = Path("checkpoints/best_hparams.yaml")

TRAIN_EPOCHS_PER_TRIAL = 8   # short run per trial — enough to judge quality


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def run_trial(trial: optuna.Trial, cfg: dict, device: torch.device) -> float:
    """Train for TRAIN_EPOCHS_PER_TRIAL and return best val accuracy."""
    from models.fusion_head import GestureRecognitionModel
    from dataset import build_split_datasets
    from augmentation import AugmentationPipeline

    # ── Suggest hyperparameters ───────────────────────────────────────────────
    lr           = trial.suggest_float("lr",           1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.1,  log=True)
    dropout      = trial.suggest_float("dropout",      0.1,  0.5)
    label_smooth = trial.suggest_float("label_smoothing", 0.0, 0.2)
    grad_clip    = trial.suggest_float("gradient_clip", 0.1, 2.0)
    batch_size   = trial.suggest_categorical("batch_size", [8, 16, 32])
    d_model      = trial.suggest_categorical("d_model", [128, 256])
    num_layers   = trial.suggest_int("num_layers", 2, 6)

    # ── Build datasets ────────────────────────────────────────────────────────
    aug_pipeline = AugmentationPipeline(cfg['augmentation'])
    try:
        train_ds, val_ds, _ = build_split_datasets(cfg, aug_pipeline)
    except Exception as e:
        raise optuna.exceptions.TrialPruned(f"Dataset error: {e}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Build model ────────────────────────────────────────────────────────────
    model = GestureRecognitionModel(
        landmark_d_model=d_model,
        landmark_nhead=8 if d_model == 256 else 4,
        landmark_num_layers=num_layers,
        landmark_dim_feedforward=d_model * 4,
        landmark_dropout=dropout,
        fusion_hidden=512,
        fusion_dropout=dropout,
        num_classes=cfg['model']['num_classes'],
        pretrained_swin=False,  # skip pretrained download for speed in trials
    ).to(device)

    # Freeze Swin for trial (landmark stream only — faster, still indicative)
    for p in model.video_encoder.parameters():
        p.requires_grad_(False)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_EPOCHS_PER_TRIAL, eta_min=lr * 0.01
    )

    best_val_acc = 0.0

    for epoch in range(TRAIN_EPOCHS_PER_TRIAL):
        # Train
        model.train()
        for lm, fr, lb in train_loader:
            lm, fr, lb = lm.to(device), fr.to(device), lb.to(device)
            logits = model(lm, fr)
            loss   = criterion(logits, lb)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for lm, fr, lb in val_loader:
                lm, fr, lb = lm.to(device), fr.to(device), lb.to(device)
                preds = model(lm, fr).argmax(dim=-1)
                correct += (preds == lb).sum().item()
                total   += lb.size(0)

        val_acc = correct / max(total, 1)
        best_val_acc = max(best_val_acc, val_acc)

        # Pruning — kill unpromising trials early
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=30, help='Number of trials')
    parser.add_argument('--show',   action='store_true',  help='Show best params from previous run')
    args = parser.parse_args()

    if args.show:
        if RESULTS_PATH.exists():
            with open(RESULTS_PATH) as f:
                print(yaml.safe_load(f))
        else:
            print("No results yet — run without --show first")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg    = load_config()

    STUDY_PATH.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name="gesture_hparam_search",
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        storage=f"sqlite:///{STUDY_PATH}",
        load_if_exists=True,
    )

    print(f"\n{'='*60}")
    print(f"  Hyperparameter Search — {args.trials} trials")
    print(f"  {TRAIN_EPOCHS_PER_TRIAL} epochs per trial (landmark stream only)")
    print(f"  Best params saved to: {RESULTS_PATH}")
    print(f"{'='*60}\n")

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"  Resuming from trial {completed + 1}/{args.trials}\n")

    def progress_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"  Trial {trial.number+1:3d}/{args.trials} | "
                  f"val_acc={trial.value*100:.1f}% | "
                  f"lr={trial.params.get('lr', 0):.2e} | "
                  f"Best so far: {study.best_value*100:.1f}%")

    study.optimize(
        lambda trial: run_trial(trial, cfg, device),
        n_trials=args.trials,
        callbacks=[progress_callback],
        catch=(Exception,),
    )

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"  BEST TRIAL: #{best.number} — val_acc={best.value*100:.1f}%")
    print(f"{'='*60}")
    for k, v in best.params.items():
        print(f"  {k:25s} = {v}")

    # Save best params to yaml
    best_params = {
        'val_acc': round(best.value * 100, 2),
        **best.params
    }
    with open(RESULTS_PATH, 'w') as f:
        yaml.dump(best_params, f, default_flow_style=False)
    print(f"\n  Saved to {RESULTS_PATH}")
    print(f"\n  To train with best params:")
    p = best.params
    print(f"  python train.py \\")
    print(f"    --phase1_lr {p['lr']:.2e} \\")
    print(f"    --phase2_lr {p['lr']/5:.2e} \\")
    print(f"    --batch_size {p['batch_size']} \\")
    print(f"    --gradient_clip {p['gradient_clip']:.2f} \\")
    print(f"    --weight_decay {p['weight_decay']:.2e} \\")
    print(f"    --label_smoothing {p['label_smoothing']:.2f} \\")
    print(f"    --dropout {p['dropout']:.2f}")


if __name__ == '__main__':
    main()
