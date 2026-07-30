"""
Three-phase training script for the GestureRecognitionModel.

Phase 1 (5 epochs):  Freeze VideoSwin. Train landmark encoder + heads only.
Phase 2 (3 epochs):  Freeze everything except fusion Linear layers.
Phase 3 (15 epochs): Unfreeze all. Differential LRs + cosine annealing.

Usage
-----
python train.py                         # full training
python train.py --smoke-test            # 2 batches per phase, quick sanity check
python train.py --config my_config.yaml
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_vram_gb(model: nn.Module, batch_size: int, precision: str) -> float:
    """
    Very rough VRAM estimate: parameters + activations heuristic.
    params_bytes = num_params * bytes_per_param
    activation_heuristic = params_bytes * batch_size * 0.1   (very rough)
    """
    bytes_per_param = 2 if precision == 'bf16' else 4
    param_bytes = count_parameters(model) * bytes_per_param
    activation_bytes = param_bytes * batch_size * 0.1  # rough multiplier
    total_gb = (param_bytes + activation_bytes) / (1024 ** 3)
    return total_gb


def set_requires_grad(module: nn.Module, value: bool):
    for p in module.parameters():
        p.requires_grad = value


# ──────────────────────────────────────────────────────────────────────────────
# Per-class accuracy
# ──────────────────────────────────────────────────────────────────────────────

def compute_per_class_accuracy(
    all_preds: List[int],
    all_labels: List[int],
    num_classes: int,
) -> Dict[int, float]:
    correct = [0] * num_classes
    total   = [0] * num_classes
    for p, l in zip(all_preds, all_labels):
        total[l] += 1
        if p == l:
            correct[l] += 1
    return {
        c: (correct[c] / total[c]) if total[c] > 0 else float('nan')
        for c in range(num_classes)
    }


# ──────────────────────────────────────────────────────────────────────────────
# Single epoch
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[GradScaler],
    device: torch.device,
    use_bf16: bool,
    gradient_clip: float,
    is_train: bool,
    smoke_test: bool = False,
    num_classes: int = 5,
):
    model.train(is_train)
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    ctx = torch.no_grad() if not is_train else torch.enable_grad()

    with ctx:
        for batch_idx, (landmarks, frames, labels) in enumerate(
            tqdm(loader, desc="train" if is_train else "val ", leave=False)
        ):
            if smoke_test and batch_idx >= 2:
                break

            landmarks = landmarks.to(device)  # (B, 30, 63)
            frames    = frames.to(device)      # (B, 3, 30, 112, 112)
            labels    = labels.to(device)      # (B,)

            amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_bf16):
                logits = model(landmarks, frames)      # (B, num_classes)
                loss   = criterion(logits, labels)

            if is_train:
                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    continue  # skip corrupt batch, don't update weights
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    n_batches = min(batch_idx + 1, 2) if smoke_test else len(loader)
    avg_loss  = total_loss / max(n_batches, 1)
    overall_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_preds), 1)
    per_class   = compute_per_class_accuracy(all_preds, all_labels, num_classes)

    return avg_loss, overall_acc, per_class


# ──────────────────────────────────────────────────────────────────────────────
# Phase execution
# ──────────────────────────────────────────────────────────────────────────────

def train_phase(
    phase: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    cfg: Dict[str, Any],
    epochs: int,
    smoke_test: bool,
    log: List[Dict],
    checkpoint_dir: Path,
    best_val_acc: float,
) -> float:
    """Run one training phase, return updated best_val_acc."""

    train_cfg   = cfg['training']
    use_bf16    = train_cfg.get('precision', 'fp32') == 'bf16'
    grad_clip   = train_cfg['gradient_clip']
    num_classes = cfg['model']['num_classes']
    classes     = cfg['data']['classes']

    print(f"\n{'='*60}")
    print(f"  Phase {phase}  |  {epochs} epochs  |  trainable: "
          f"{count_trainable_parameters(model):,} params")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc, train_pc = run_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, use_bf16, grad_clip,
            is_train=True, smoke_test=smoke_test, num_classes=num_classes,
        )

        val_loss, val_acc, val_pc = run_epoch(
            model, val_loader, criterion, None,
            None, device, use_bf16, grad_clip,
            is_train=False, smoke_test=smoke_test, num_classes=num_classes,
        )

        elapsed = time.time() - t0

        # Per-class accuracy string
        pc_str = "  ".join(
            f"{classes[c]}={val_pc[c]*100:.1f}%" for c in range(num_classes)
        )

        print(
            f"Phase {phase} | Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_acc={train_acc*100:.1f}% val_acc={val_acc*100:.1f}% | "
            f"{elapsed:.1f}s"
        )
        print(f"  Per-class val acc: {pc_str}")

        entry = {
            'phase': phase,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'per_class_val_acc': {classes[c]: val_pc[c] for c in range(num_classes)},
        }
        log.append(entry)

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                'phase': phase,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }
            torch.save(ckpt, checkpoint_dir / 'best_model.pt')
            print(f"  Saved new best checkpoint (val_acc={val_acc*100:.1f}%)")

        if scheduler is not None:
            scheduler.step()

        if smoke_test:
            print("  [smoke-test] stopping phase early")
            break

    return best_val_acc


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train gesture recognition model")
    parser.add_argument('--config',              default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--smoke-test',          action='store_true',   help='Run 2 batches only to verify pipeline')
    parser.add_argument('--preprocess',          action='store_true',   help='Run preprocessing before training')
    parser.add_argument('--resume',              default=None,          help='Load weights from checkpoint before training')
    # Hyperparameter overrides — any of these override config.yaml
    parser.add_argument('--batch_size',          type=int,   default=None)
    parser.add_argument('--phase1_lr',           type=float, default=None)
    parser.add_argument('--phase2_lr',           type=float, default=None)
    parser.add_argument('--phase3_lr_backbone',  type=float, default=None)
    parser.add_argument('--phase3_lr_head',      type=float, default=None)
    parser.add_argument('--phase1_epochs',       type=int,   default=None)
    parser.add_argument('--phase2_epochs',       type=int,   default=None)
    parser.add_argument('--phase3_epochs',       type=int,   default=None)
    parser.add_argument('--gradient_clip',       type=float, default=None)
    parser.add_argument('--weight_decay',        type=float, default=None)
    parser.add_argument('--label_smoothing',     type=float, default=None)
    parser.add_argument('--dropout',             type=float, default=None, help='Fusion head dropout')
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg['training']

    # Apply CLI overrides to config
    _overrides = {
        'batch_size': args.batch_size, 'phase1_lr': args.phase1_lr,
        'phase2_lr': args.phase2_lr, 'phase3_lr_backbone': args.phase3_lr_backbone,
        'phase3_lr_head': args.phase3_lr_head, 'phase1_epochs': args.phase1_epochs,
        'phase2_epochs': args.phase2_epochs, 'phase3_epochs': args.phase3_epochs,
        'gradient_clip': args.gradient_clip, 'weight_decay': args.weight_decay,
        'label_smoothing': args.label_smoothing,
    }
    for key, val in _overrides.items():
        if val is not None:
            print(f"[train] Override: {key} = {val} (was {train_cfg.get(key)})")
            train_cfg[key] = val
    if args.dropout is not None:
        print(f"[train] Override: fusion_dropout = {args.dropout}")
        cfg['model']['fusion_dropout'] = args.dropout
    model_cfg = cfg['model']

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] Device: {device}")
    if device.type == 'cuda':
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[train] Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Optional preprocessing ────────────────────────────────────────────────
    if args.preprocess:
        from dataset import preprocess_all
        print("[train] Running preprocessing...")
        preprocess_all(cfg)

    # ── Data ──────────────────────────────────────────────────────────────────
    from dataset import build_split_datasets

    # No online augmentation — all augmented samples are pre-computed on disk
    aug_pipeline = None

    batch_size = train_cfg['batch_size']

    if args.smoke_test:
        # Smoke-test: skip dataset loading entirely, use random tensors
        print("[smoke-test] Using random tensors — no data required")
        from torch.utils.data import TensorDataset
        B, T, C_lm, C, H, W = batch_size, 30, 63, 3, 112, 112
        dummy_lm = torch.randn(B * 4, T, C_lm)
        dummy_fr = torch.randn(B * 4, C, T, H, W)
        dummy_lb = torch.randint(0, model_cfg['num_classes'], (B * 4,))
        dummy_ds = TensorDataset(dummy_lm, dummy_fr, dummy_lb)
        train_loader = DataLoader(dummy_ds, batch_size=B, shuffle=True)
        val_loader   = DataLoader(dummy_ds, batch_size=B, shuffle=False)
        print(f"[smoke-test] {len(dummy_ds)} synthetic samples, batch_size={B}")
    else:
        print("[train] Building datasets...")
        try:
            train_ds, val_ds, test_ds = build_split_datasets(cfg, aug_pipeline)
        except Exception as e:
            print(f"[train] Dataset error: {e}")
            print("[train] Tip: run with --preprocess flag first, or add videos to data/raw/<class>/")
            sys.exit(1)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )
        print(f"[train] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    from models.fusion_head import GestureRecognitionModel

    model = GestureRecognitionModel(
        landmark_d_model=model_cfg['landmark_d_model'],
        landmark_nhead=model_cfg['landmark_nhead'],
        landmark_num_layers=model_cfg['landmark_num_layers'],
        landmark_dim_feedforward=model_cfg['landmark_dim_feedforward'],
        landmark_dropout=model_cfg['landmark_dropout'],
        fusion_hidden=model_cfg['fusion_hidden'],
        fusion_dropout=model_cfg['fusion_dropout'],
        num_classes=model_cfg['num_classes'],
        pretrained_swin=True,
    ).to(device)

    total_params = count_parameters(model)
    est_vram = estimate_vram_gb(model, batch_size, train_cfg.get('precision', 'bf16'))
    print(f"[train] Total parameters:    {total_params:,}")
    print(f"[train] Est. VRAM usage:     ~{est_vram:.1f} GB (rough, at batch_size={batch_size})")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[train] Resumed from {args.resume} (val_acc={ckpt.get('val_acc', '?')})")

    # ── Loss & scaler ─────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg['label_smoothing'])
    use_bf16  = train_cfg.get('precision', 'fp32') == 'bf16'
    scaler    = None  # fp32 needs no scaler; set to GradScaler only if using fp16

    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_log: List[Dict] = []
    best_val_acc = 0.0

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 1: Swin fully frozen — train landmark encoder + heads only
    # ═════════════════════════════════════════════════════════════════════════
    model.video_encoder.freeze_all()
    set_requires_grad(model.landmark_encoder, True)
    set_requires_grad(model.fusion_head, True)
    set_requires_grad(model.lm_only_head, True)

    trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[phase1] Trainable: {trainable_p1:,} params (Swin fully frozen)")

    optimizer_p1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg['phase1_lr'],
        weight_decay=train_cfg['weight_decay'],
    )

    best_val_acc = train_phase(
        phase=1,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_p1,
        scheduler=None,
        criterion=criterion,
        scaler=scaler,
        device=device,
        cfg=cfg,
        epochs=train_cfg['phase1_epochs'],
        smoke_test=args.smoke_test,
        log=training_log,
        checkpoint_dir=checkpoint_dir,
        best_val_acc=best_val_acc,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 2: Unfreeze Swin Stage 4 + norm only — differential LRs
    # Stages 1-3 stay frozen (Kinetics-400 features preserved)
    # Stage 4 gets 10× smaller LR to nudge without overwriting
    # ═════════════════════════════════════════════════════════════════════════
    torch.cuda.empty_cache()
    model.video_encoder.freeze_except_last_stage()
    set_requires_grad(model.landmark_encoder, True)
    set_requires_grad(model.fusion_head, True)
    set_requires_grad(model.lm_only_head, True)

    swin_stage4_params = [p for p in model.video_encoder.parameters() if p.requires_grad]
    other_params       = [p for p in list(model.landmark_encoder.parameters()) +
                          list(model.fusion_head.parameters()) +
                          list(model.lm_only_head.parameters()) if p.requires_grad]

    trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[phase2] Trainable: {trainable_p2:,} params (Swin Stage 4 + norm + heads)")
    print(f"[phase2] LR: Stage4={train_cfg['phase2_lr']/10:.2e}  Heads={train_cfg['phase2_lr']:.2e}")

    optimizer_p2 = torch.optim.AdamW(
        [
            {"params": swin_stage4_params, "lr": train_cfg['phase2_lr'] / 10},  # 10× smaller
            {"params": other_params,       "lr": train_cfg['phase2_lr']},
        ],
        weight_decay=train_cfg['weight_decay'],
    )

    best_val_acc = train_phase(
        phase=2,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_p2,
        scheduler=None,
        criterion=criterion,
        scaler=scaler,
        device=device,
        cfg=cfg,
        epochs=train_cfg['phase2_epochs'],
        smoke_test=args.smoke_test,
        log=training_log,
        checkpoint_dir=checkpoint_dir,
        best_val_acc=best_val_acc,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 3: Unfreeze all, differential LRs, cosine annealing
    # ═════════════════════════════════════════════════════════════════════════
    torch.cuda.empty_cache()
    model.video_encoder.enable_gradient_checkpointing()  # recompute activations → ~50% less VRAM
    set_requires_grad(model, True)

    # Phase 3: batch_size=2 — Swin fully unfrozen needs maximum memory headroom
    p3_batch = 2
    if args.smoke_test:
        from torch.utils.data import TensorDataset
        B, T, C_lm, C, H, W = p3_batch, 30, 63, 3, 112, 112
        dummy_lm = torch.randn(p3_batch * 4, T, C_lm)
        dummy_fr = torch.randn(p3_batch * 4, C, T, H, W)
        dummy_lb = torch.randint(0, model_cfg['num_classes'], (p3_batch * 4,))
        dummy_ds = TensorDataset(dummy_lm, dummy_fr, dummy_lb)
        train_loader = DataLoader(dummy_ds, batch_size=p3_batch, shuffle=True)
        val_loader   = DataLoader(dummy_ds, batch_size=p3_batch, shuffle=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=p3_batch, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=p3_batch, shuffle=False,
                                  num_workers=4, pin_memory=True)
    print(f"[phase3] batch_size={p3_batch} (reduced for Swin memory)")

    swin_params     = model.swin_parameters()
    non_swin_params = model.non_swin_parameters()

    optimizer_p3 = torch.optim.AdamW(
        [
            {'params': swin_params,     'lr': train_cfg['phase3_lr_backbone']},
            {'params': non_swin_params, 'lr': train_cfg['phase3_lr_head']},
        ],
        weight_decay=train_cfg['weight_decay'],
    )

    scheduler_p3 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p3,
        T_max=train_cfg['phase3_epochs'],
        eta_min=1e-7,
    )

    best_val_acc = train_phase(
        phase=3,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_p3,
        scheduler=scheduler_p3,
        criterion=criterion,
        scaler=scaler,
        device=device,
        cfg=cfg,
        epochs=train_cfg['phase3_epochs'],
        smoke_test=args.smoke_test,
        log=training_log,
        checkpoint_dir=checkpoint_dir,
        best_val_acc=best_val_acc,
    )

    # ── Save training log ────────────────────────────────────────────────────
    log_path = checkpoint_dir / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"\n[train] Training log saved to {log_path}")
    print(f"[train] Best val accuracy: {best_val_acc*100:.1f}%")
    print(f"[train] Best model saved to {checkpoint_dir / 'best_model.pt'}")


if __name__ == '__main__':
    main()
