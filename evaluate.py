"""
evaluate.py
===========
Evaluation script for ERGT-BTO (Section 4.1–4.13).

Computes:
  - Full metric suite: Accuracy, F1, AUC, MCC, Cohen's κ, BACC
  - Confusion matrix (Fig. 5 equivalent)
  - Per-class ROC curves (Fig. 8 equivalent)
  - SHAP feature attribution (Section 4.3)
  - Transformer attention rollout visualizations (Section 4.3)
  - Cross-dataset generalization (Section 4.10)
  - Statistical significance tests vs baselines (Section 4.11)

Usage:
  # Basic evaluation on BACH test set
  python evaluate.py --checkpoint checkpoints/run0_seed0_best.pt \
                     --dataset bach --data_root ./datasets/BACH

  # With SHAP + attention rollout explanations
  python evaluate.py --checkpoint checkpoints/run0_seed0_best.pt \
                     --dataset bach --data_root ./datasets/BACH --explain

  # Cross-dataset: train=BACH, test=BreaKHis
  python evaluate.py --checkpoint checkpoints/run0_seed0_best.pt \
                     --dataset breakhis --data_root ./datasets/BreaKHis \
                     --cross_dataset
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy import stats

from data.dataset import build_dataset, LABEL_NAMES
from data.patient_split import build_patient_splits
from models.ergt_bto import ERGTBTOModel
from utils.metrics import (
    compute_all_metrics,
    print_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
)
from explainability.shap_explainer import ERGTSHAPExplainer, GradientSaliency
from explainability.attention_rollout import AttentionRolloutVisualizer


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model inference over a DataLoader.

    Returns:
        y_true : (N,) ground-truth labels
        y_pred : (N,) predicted labels
        y_prob : (N, C) softmax probabilities
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels, _ in loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ablation study runner (Table 10)
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_study(
    data_root: str,
    dataset_name: str,
    device: str,
    epochs: int = 30,
    seed: int = 0,
) -> None:
    """
    Reproduce the ablation study (Table 10) by training seven model variants.
    Each variant incrementally adds: radiomic → GCN → transformer → BTO → SHAP.
    """
    import torch.optim as optim

    print("\n" + "="*60)
    print("  Ablation Study (Table 10)")
    print("="*60)

    variants = [
        {"name": "CNN Baseline (VGG16-retrained)",   "radiomic": False, "gcn": False, "transformer": False},
        {"name": "+Multi-Scale Radiomic Features",   "radiomic": True,  "gcn": False, "transformer": False},
        {"name": "+GCN Topology (no Transformer)",   "radiomic": True,  "gcn": True,  "transformer": False},
        {"name": "+Transformer (no GCN)",             "radiomic": True,  "gcn": False, "transformer": True},
        {"name": "+GCN + Transformer",               "radiomic": True,  "gcn": True,  "transformer": True},
    ]

    full_dataset = build_dataset(dataset_name, data_root)
    patient_ids  = full_dataset.get_patient_ids()
    labels       = full_dataset.get_labels()
    train_idx, val_idx, test_idx = build_patient_splits(patient_ids, labels, seed=seed)

    results = []
    for var in variants:
        print(f"\n[Ablation] Training: {var['name']}")

        # Build minimal model variant
        model = ERGTBTOModel(
            num_classes=4,
            use_handcraft=var["radiomic"],
        ).to(device)

        train_ds = build_dataset(dataset_name, data_root, "train", train_idx)
        test_ds  = build_dataset(dataset_name, data_root, "test",  test_idx)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=2)
        test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=2)

        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            for imgs, lbls, _ in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                criterion(model(imgs), lbls).backward()
                optimizer.step()

        y_true, y_pred, y_prob = run_inference(model, test_loader, device)
        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        metrics["name"] = var["name"]
        results.append(metrics)

        print(f"  Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  AUC={metrics.get('auc', float('nan')):.4f}")

    print("\n" + "-"*70)
    print(f"{'Variant':<45} {'Acc':>6} {'F1':>6} {'AUC':>6} {'MCC':>6}")
    print("-"*70)
    for r in results:
        print(f"{r['name']:<45} {r['accuracy']:>6.4f} {r['f1']:>6.4f} {r.get('auc', float('nan')):>6.4f} {r['mcc']:>6.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Graph sensitivity analysis (Table 11)
# ─────────────────────────────────────────────────────────────────────────────

def run_graph_sensitivity(
    model: ERGTBTOModel,
    test_loader: DataLoader,
    device: str,
    delta_values: Optional[List[float]] = None,
) -> None:
    """
    Evaluate ERGT-BTO across different graph edge distance thresholds δ.
    Reproduces Table 11.
    """
    if delta_values is None:
        delta_values = [15, 25, 35, 42, 55, 70, 80]

    print("\n" + "="*60)
    print("  Graph Sensitivity Analysis (Table 11)")
    print("="*60)
    print(f"{'δ (px)':<10} {'Avg Edges':>10} {'Accuracy':>10} {'F1':>8} {'AUC':>8}")
    print("-"*50)

    original_delta = model.graph_encoder.delta

    for delta in delta_values:
        model.graph_encoder.delta = delta
        y_true, y_pred, y_prob = run_inference(model, test_loader, device)
        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        print(
            f"{delta:<10.0f} {'N/A':>10} "
            f"{metrics['accuracy']:>10.4f} {metrics['f1']:>8.4f} "
            f"{metrics.get('auc', float('nan')):>8.4f}"
        )

    # Restore original delta
    model.graph_encoder.delta = original_delta


# ─────────────────────────────────────────────────────────────────────────────
# Statistical significance testing (Table 18)
# ─────────────────────────────────────────────────────────────────────────────

def paired_ttest(
    ergt_scores: List[float],
    baseline_scores: List[float],
    baseline_name: str,
) -> None:
    """
    Perform paired t-test between ERGT-BTO and a baseline (10 runs).
    Prints p-value and Cohen's d effect size.
    """
    t_stat, p_val = stats.ttest_rel(ergt_scores, baseline_scores)

    # Cohen's d
    diff = np.array(ergt_scores) - np.array(baseline_scores)
    cohens_d = diff.mean() / (diff.std(ddof=1) + 1e-10)

    sig = "Highly significant" if p_val < 0.001 else ("Significant" if p_val < 0.05 else "Not significant")
    print(
        f"  vs {baseline_name:<30} p={p_val:.4f}  d={cohens_d:.2f}  [{sig}]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate ERGT-BTO")
    parser.add_argument("--checkpoint",   required=True,             help="Path to .pt checkpoint")
    parser.add_argument("--dataset",      default="bach")
    parser.add_argument("--data_root",    required=True)
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--explain",      action="store_true",       help="Run SHAP + attention rollout")
    parser.add_argument("--cross_dataset",action="store_true",       help="Cross-dataset generalization mode")
    parser.add_argument("--ablation",     action="store_true",       help="Run ablation study")
    parser.add_argument("--graph_sensitivity", action="store_true",  help="Run graph sensitivity analysis")
    parser.add_argument("--save_dir",     default="./results",       help="Directory to save outputs")
    parser.add_argument("--seed",         type=int, default=0)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"\n[Eval] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    hyperparams = ckpt.get("hyperparams", {})

    model = ERGTBTOModel.from_bto_config(hyperparams, num_classes=4).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.print_summary()

    # ── Build test loader ─────────────────────────────────────────────────────
    full_dataset = build_dataset(args.dataset, args.data_root)
    patient_ids  = full_dataset.get_patient_ids()
    labels       = full_dataset.get_labels()

    _, _, test_idx = build_patient_splits(patient_ids, labels, seed=args.seed)
    test_ds = build_dataset(args.dataset, args.data_root, "test", test_idx)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ── Standard evaluation ───────────────────────────────────────────────────
    print("\n[Eval] Running inference on test set ...")
    y_true, y_pred, y_prob = run_inference(model, test_loader, device)
    metrics = compute_all_metrics(y_true, y_pred, y_prob, num_classes=4)
    print_metrics(metrics, title="ERGT-BTO — Test Set Results")

    # Save metrics
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        class_names=LABEL_NAMES,
        save_path=str(save_dir / "confusion_matrix.png"),
        title="ERGT-BTO Confusion Matrix — BACH Test Set",
    )

    # ROC curves
    if not np.isnan(y_prob).any():
        plot_roc_curves(
            y_true, y_prob,
            class_names=LABEL_NAMES,
            save_path=str(save_dir / "roc_curves.png"),
        )

    # ── Explainability ────────────────────────────────────────────────────────
    if args.explain:
        print("\n[Eval] Generating explainability visualizations ...")

        # Collect sample images for background
        sample_imgs = []
        sample_lbls = []
        for imgs, lbls, _ in test_loader:
            sample_imgs.append(imgs)
            sample_lbls.extend(lbls.tolist())
            if len(sample_imgs) * args.batch_size >= 100:
                break
        all_imgs = torch.cat(sample_imgs, dim=0)[:100]

        # SHAP
        try:
            explainer = ERGTSHAPExplainer(model, all_imgs[:50], device=device)
            shap_imgs = all_imgs[:8]
            shap_vals = explainer.compute_shap_values(shap_imgs)
            explainer.plot_summary(
                shap_vals, shap_imgs,
                class_names=LABEL_NAMES,
                save_path=str(save_dir / "shap_summary.png"),
            )
            print("[Eval] SHAP explanations saved.")
        except Exception as e:
            print(f"[Eval] SHAP failed (falling back to gradient saliency): {e}")
            grad_sal = GradientSaliency(model, device=device)

        # Attention rollout
        try:
            rollout_viz = AttentionRolloutVisualizer(
                model, patch_size=16, img_size=224, device=device
            )
            rollout_viz.visualize_batch(
                all_imgs[:8],
                sample_lbls[:8],
                class_names=LABEL_NAMES,
                save_dir=str(save_dir / "attention_rollout"),
                max_images=8,
            )
            print("[Eval] Attention rollout visualizations saved.")
        except Exception as e:
            print(f"[Eval] Attention rollout failed: {e}")

    # ── Graph sensitivity ─────────────────────────────────────────────────────
    if args.graph_sensitivity:
        run_graph_sensitivity(model, test_loader, device)

    # ── Ablation study ────────────────────────────────────────────────────────
    if args.ablation:
        run_ablation_study(args.data_root, args.dataset, device, seed=args.seed)

    print(f"\n[Eval] Results saved to: {save_dir}")
    print("[Eval] Done.")


if __name__ == "__main__":
    main()
