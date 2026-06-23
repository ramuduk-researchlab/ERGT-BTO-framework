"""
train.py
=========
Main training script for ERGT-BTO (Section 3.1.4, 3.3.4).

Features:
  - Optional BTO hyperparameter optimization before training
  - Patient-level stratified data splitting (Section 3.1.3)
  - Multi-run reproducibility (10 seeds, as in the paper)
  - Checkpoint saving and early stopping
  - Full metric logging

Usage:
  # Train with BTO optimization (recommended):
  python train.py --dataset bach --data_root ./datasets/BACH --use_bto

  # Train with fixed hyperparameters:
  python train.py --dataset bach --data_root ./datasets/BACH \
      --lr 3e-4 --batch_size 16 --epochs 60

  # Multi-run experiment (10 seeds, as in paper):
  python train.py --dataset bach --data_root ./datasets/BACH --n_runs 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import build_dataset
from data.patient_split import build_patient_splits, print_split_class_distribution
from models.ergt_bto import ERGTBTOModel
from optimization.bto_optimizer import BluefinTrevallyOptimizer, DEFAULT_SEARCH_SPACE
from utils.metrics import compute_all_metrics, print_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scheduler: Optional[object] = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    if scheduler is not None:
        scheduler.step()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    num_classes: int = 4,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = compute_all_metrics(y_true, y_pred, y_prob, num_classes=num_classes)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
# BTO fitness function wrapper
# ─────────────────────────────────────────────────────────────────────────────

def build_bto_fitness_fn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    num_classes: int = 4,
    n_warmup_epochs: int = 5,
):
    """
    Returns a fitness function that trains a model for n_warmup_epochs
    and returns validation F1 score. Used by the BTO optimizer.
    """

    def fitness_fn(params: Dict) -> float:
        try:
            model = ERGTBTOModel.from_bto_config(params, num_classes=num_classes).to(device)
            lr = float(params.get("learning_rate", 3e-4))
            wd = float(params.get("weight_decay", 1e-4))
            bs = int(params.get("batch_size", 16))

            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(n_warmup_epochs):
                model.train()
                for images, labels, _ in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(images), labels)
                    loss.backward()
                    optimizer.step()

            _, metrics = evaluate(model, val_loader, criterion, device, num_classes)
            return metrics["f1"]

        except Exception as e:
            print(f"[BTO] fitness_fn error: {e}")
            return 0.0

    return fitness_fn


# ─────────────────────────────────────────────────────────────────────────────
# Full training run
# ─────────────────────────────────────────────────────────────────────────────

def run_training(args, hyperparams: Dict, seed: int, run_id: int) -> Dict:
    """Single training run with given hyperparameters and seed."""
    set_seed(seed)
    device = args.device

    # ── Data ────────────────────────────────────────────────────────────────
    full_dataset = build_dataset(
        args.dataset, args.data_root, split="train", image_size=224
    )
    patient_ids = full_dataset.get_patient_ids()
    labels = full_dataset.get_labels()

    train_idx, val_idx, test_idx = build_patient_splits(
        patient_ids, labels, train_frac=0.70, val_frac=0.15, seed=seed
    )
    print_split_class_distribution(labels, train_idx, val_idx, test_idx)

    train_ds = build_dataset(args.dataset, args.data_root, split="train",  split_indices=train_idx)
    val_ds   = build_dataset(args.dataset, args.data_root, split="val",    split_indices=val_idx)
    test_ds  = build_dataset(args.dataset, args.data_root, split="test",   split_indices=test_idx)

    batch_size = int(hyperparams.get("batch_size", args.batch_size))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = ERGTBTOModel.from_bto_config(hyperparams, num_classes=4).to(device)
    model.print_summary()

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    lr = float(hyperparams.get("learning_rate", args.lr))
    wd = float(hyperparams.get("weight_decay", args.weight_decay))

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1 = 0.0
    best_ckpt_path = Path(args.checkpoint_dir) / f"run{run_id}_seed{seed}_best.pt"
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        val_f1 = val_metrics["f1"]
        print(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} "
            f"({elapsed:.1f}s)"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "hyperparams": hyperparams,
                "val_metrics": val_metrics,
            }, best_ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[EarlyStopping] No improvement for {args.patience} epochs. Stopping.")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, test_metrics = evaluate(model, test_loader, criterion, device)
    print_metrics(test_metrics, title=f"Test Set – Run {run_id} Seed {seed}")

    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ERGT-BTO")
    parser.add_argument("--dataset",        default="bach",          help="'bach' or 'breakhis'")
    parser.add_argument("--data_root",      required=True,           help="Path to dataset root")
    parser.add_argument("--epochs",         type=int, default=60)
    parser.add_argument("--batch_size",     type=int, default=16)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--dropout",        type=float, default=0.3)
    parser.add_argument("--patience",       type=int, default=15)
    parser.add_argument("--n_runs",         type=int, default=1,     help="Number of independent runs (paper uses 10)")
    parser.add_argument("--use_bto",        action="store_true",     help="Run BTO before training")
    parser.add_argument("--bto_agents",     type=int, default=20)
    parser.add_argument("--bto_iters",      type=int, default=50)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"\n[ERGT-BTO] Using device: {args.device}")

    # Default hyperparameters (paper's BTO-optimal values, Table 12)
    hyperparams = {
        "learning_rate":      3e-4,
        "weight_decay":       1e-4,
        "dropout":            args.dropout,
        "gcn_hidden_dim":     128,
        "transformer_heads":  8,
        "transformer_layers": 4,
        "batch_size":         args.batch_size,
        "delta_px":           42.0,
    }

    # ── BTO optimization (optional) ──────────────────────────────────────────
    if args.use_bto:
        print("\n[BTO] Starting hyperparameter optimization ...")
        full_dataset = build_dataset(args.dataset, args.data_root, image_size=224)
        patient_ids  = full_dataset.get_patient_ids()
        labels       = full_dataset.get_labels()
        train_idx, val_idx, _ = build_patient_splits(patient_ids, labels, seed=0)

        bto_train = build_dataset(args.dataset, args.data_root, "train", train_idx)
        bto_val   = build_dataset(args.dataset, args.data_root, "val",   val_idx)
        bto_tl = DataLoader(bto_train, batch_size=16, shuffle=True,  num_workers=2)
        bto_vl = DataLoader(bto_val,   batch_size=16, shuffle=False, num_workers=2)

        fitness_fn = build_bto_fitness_fn(bto_tl, bto_vl, args.device, n_warmup_epochs=3)
        bto = BluefinTrevallyOptimizer(
            search_space=DEFAULT_SEARCH_SPACE,
            n_agents=args.bto_agents,
            max_iterations=args.bto_iters,
        )
        hyperparams = bto.optimize(fitness_fn, verbose=True)
        bto.print_convergence_summary()

        with open(Path(args.checkpoint_dir) / "bto_best_params.json", "w") as f:
            json.dump({k: (v if not isinstance(v, np.generic) else float(v))
                       for k, v in hyperparams.items()}, f, indent=2)

    # ── Multi-run training ───────────────────────────────────────────────────
    all_results = []
    for run_id in range(args.n_runs):
        seed = run_id
        print(f"\n{'='*55}")
        print(f"  Run {run_id+1}/{args.n_runs}  |  Seed {seed}")
        print(f"{'='*55}")
        result = run_training(args, hyperparams, seed=seed, run_id=run_id)
        all_results.append(result)

    # ── Aggregate across runs ────────────────────────────────────────────────
    if args.n_runs > 1:
        print(f"\n{'='*55}")
        print("  Multi-Run Summary")
        print(f"{'='*55}")
        for key in ["accuracy", "f1", "auc", "mcc", "kappa"]:
            vals = [r[key] for r in all_results if not np.isnan(r.get(key, float("nan")))]
            if vals:
                print(f"  {key:<20}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        with open(Path(args.checkpoint_dir) / "multi_run_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    print("\n[ERGT-BTO] Training complete.")


if __name__ == "__main__":
    main()
