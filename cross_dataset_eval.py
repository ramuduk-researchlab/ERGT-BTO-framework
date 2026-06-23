"""
cross_dataset_eval.py
======================
Cross-dataset generalization and statistical significance testing
(Section 4.10 – 4.11).

Experiments:
  1. Train on BACH → Evaluate on BreaKHis (cross-dataset generalization)
  2. Train on BreaKHis → Evaluate on BACH
  3. Domain adaptation fine-tuning (10% target data)
  4. Paired t-test vs all baselines (Table 18)
  5. Wilcoxon signed-rank test (non-parametric alternative)

Usage:
  python cross_dataset_eval.py \
      --bach_root ./datasets/BACH \
      --breakhis_root ./datasets/BreaKHis \
      --checkpoint checkpoints/run0_seed0_best.pt \
      --mode cross_eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from torch.utils.data import DataLoader, Subset

from data.dataset import build_dataset, LABEL_NAMES
from data.patient_split import build_patient_splits
from models.ergt_bto import ERGTBTOModel
from models.baselines import build_baseline, BASELINE_REGISTRY
from utils.metrics import compute_all_metrics, print_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Cross-dataset evaluation (Section 4.10, Table 14)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_cross_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int = 4,
) -> Dict[str, float]:
    """Run inference on a different-domain dataset and return metrics."""
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

    return compute_all_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        num_classes=num_classes,
    )


def run_cross_dataset_experiment(
    bach_root: str,
    breakhis_root: str,
    checkpoint_path: str,
    device: str,
    batch_size: int = 16,
) -> None:
    """
    BACH-trained ERGT-BTO evaluated on BreaKHis (Section 4.10).
    Measures domain generalization without any fine-tuning.
    """
    print("\n" + "="*60)
    print("  Cross-Dataset Generalization: BACH → BreaKHis")
    print("="*60)

    # Load BACH-trained model
    ckpt = torch.load(checkpoint_path, map_location=device)
    hyperparams = ckpt.get("hyperparams", {})
    model = ERGTBTOModel.from_bto_config(hyperparams, num_classes=4).to(device)
    model.load_state_dict(ckpt["model_state"])

    # BreaKHis test set (all magnifications combined)
    for mag in [40, 100, 200, 400, None]:
        mag_label = f"{mag}×" if mag else "All"
        try:
            ds = build_dataset(
                "breakhis", breakhis_root,
                magnification=mag, map_to_bach=True,
            )
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
            metrics = evaluate_cross_dataset(model, loader, device)
            print(f"\n  Magnification: {mag_label}  (n={len(ds)})")
            print(f"  Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
                  f"AUC={metrics.get('auc', float('nan')):.4f}  MCC={metrics['mcc']:.4f}")
        except Exception as e:
            print(f"  Magnification {mag_label}: skipped ({e})")


def run_domain_adaptation(
    source_root: str,
    target_root: str,
    source_dataset: str,
    target_dataset: str,
    checkpoint_path: str,
    device: str,
    finetune_fraction: float = 0.10,
    finetune_epochs: int = 10,
    batch_size: int = 16,
) -> Dict[str, float]:
    """
    Few-shot domain adaptation: fine-tune on finetune_fraction of target data.
    Section 4.10 — Table 15.
    """
    print(f"\n[DomainAdaptation] {source_dataset.upper()} → {target_dataset.upper()}")
    print(f"  Fine-tune fraction: {finetune_fraction*100:.0f}%  Epochs: {finetune_epochs}")

    # Load pretrained model
    ckpt = torch.load(checkpoint_path, map_location=device)
    hyperparams = ckpt.get("hyperparams", {})
    model = ERGTBTOModel.from_bto_config(hyperparams, num_classes=4).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Freeze all except classifier (fast adaptation)
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    # Target dataset
    target_ds = build_dataset(target_dataset, target_root)
    patient_ids = target_ds.get_patient_ids()
    labels_list = target_ds.get_labels()
    train_idx, _, test_idx = build_patient_splits(patient_ids, labels_list, train_frac=0.6)

    # Use only finetune_fraction of training data
    n_finetune = max(1, int(len(train_idx) * finetune_fraction))
    rng = np.random.default_rng(42)
    finetune_idx = rng.choice(train_idx, n_finetune, replace=False).tolist()

    finetune_ds = build_dataset(target_dataset, target_root, "train", finetune_idx)
    test_ds     = build_dataset(target_dataset, target_root, "test",  test_idx)

    finetune_loader = DataLoader(finetune_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader     = DataLoader(test_ds,     batch_size=batch_size, shuffle=False, num_workers=4)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(finetune_epochs):
        model.train()
        for imgs, lbls, _ in finetune_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            criterion(model(imgs), lbls).backward()
            optimizer.step()

    metrics = evaluate_cross_dataset(model, test_loader, device)
    print_metrics(metrics, title=f"Domain Adaptation Result ({finetune_fraction*100:.0f}% target data)")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Statistical significance testing (Section 4.11, Table 18)
# ─────────────────────────────────────────────────────────────────────────────

def run_significance_tests(
    ergt_scores: List[float],
    baseline_results: Dict[str, List[float]],
    metric_name: str = "F1-Score",
    alpha: float = 0.05,
) -> None:
    """
    Paired t-test and Wilcoxon signed-rank test for ERGT-BTO vs baselines.
    Reproduces Table 18.

    Args:
        ergt_scores     : List of 10 ERGT-BTO scores (one per run)
        baseline_results: {baseline_name: [10 scores]}
        metric_name     : Which metric the scores correspond to
        alpha           : Significance threshold
    """
    print(f"\n{'='*70}")
    print(f"  Statistical Significance Testing — {metric_name}")
    print(f"  ERGT-BTO: mean={np.mean(ergt_scores):.4f} ± {np.std(ergt_scores):.4f}")
    print(f"{'='*70}")
    print(f"{'Baseline':<32} {'t-stat':>8} {'p-value':>9} {'W-stat':>8} {'p (Wilcoxon)':>13} {'Sig?':>6}")
    print(f"{'-'*78}")

    for name, scores in baseline_results.items():
        if len(scores) != len(ergt_scores):
            print(f"  {name}: skipped (length mismatch)")
            continue

        t_stat, p_ttest = stats.ttest_rel(ergt_scores, scores)
        w_stat, p_wilcox = stats.wilcoxon(ergt_scores, scores, alternative="greater")

        sig = "✅" if p_ttest < alpha else "❌"
        diff = np.mean(ergt_scores) - np.mean(scores)
        print(
            f"{name:<32} {t_stat:>8.3f} {p_ttest:>9.4f} {w_stat:>8.1f} {p_wilcox:>13.4f} {sig:>6}"
            f"  (Δ={diff:+.4f})"
        )

    print(f"\n  α threshold = {alpha}.  ✅ = ERGT-BTO significantly better.")

    # Effect size (Cohen's d)
    print(f"\n  Cohen's d effect sizes:")
    for name, scores in baseline_results.items():
        diff = np.array(ergt_scores) - np.array(scores)
        if diff.std(ddof=1) > 0:
            d = diff.mean() / diff.std(ddof=1)
            magnitude = "large" if abs(d) > 0.8 else ("medium" if abs(d) > 0.5 else "small")
            print(f"  {name:<32} d={d:.3f}  ({magnitude})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-dataset & significance testing")
    parser.add_argument("--bach_root",     required=True)
    parser.add_argument("--breakhis_root", required=True)
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--mode", choices=["cross_eval", "domain_adapt", "significance"],
                        default="cross_eval")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    if args.mode == "cross_eval":
        run_cross_dataset_experiment(
            args.bach_root, args.breakhis_root,
            args.checkpoint, args.device, args.batch_size,
        )

    elif args.mode == "domain_adapt":
        run_domain_adaptation(
            source_root=args.bach_root,
            target_root=args.breakhis_root,
            source_dataset="bach",
            target_dataset="breakhis",
            checkpoint_path=args.checkpoint,
            device=args.device,
            finetune_fraction=0.10,
        )

    elif args.mode == "significance":
        # Example with synthetic multi-run data (replace with actual run results)
        print("\n[Note] Replace these with your actual 10-run results:")
        ergt_f1_scores = [0.9762, 0.9748, 0.9771, 0.9759, 0.9780,
                          0.9755, 0.9768, 0.9774, 0.9763, 0.9766]
        baseline_f1s = {
            "VGG-16":            [0.9110, 0.9098, 0.9125, 0.9103, 0.9118, 0.9095, 0.9130, 0.9112, 0.9108, 0.9122],
            "ResNet-50":         [0.9300, 0.9285, 0.9312, 0.9295, 0.9308, 0.9288, 0.9320, 0.9303, 0.9297, 0.9310],
            "EfficientNet-B4":   [0.9498, 0.9485, 0.9512, 0.9495, 0.9508, 0.9490, 0.9520, 0.9502, 0.9495, 0.9510],
            "ViT-B/16":          [0.9575, 0.9561, 0.9589, 0.9572, 0.9585, 0.9564, 0.9597, 0.9580, 0.9573, 0.9587],
            "RadiomicNet":       [0.9465, 0.9451, 0.9478, 0.9461, 0.9474, 0.9455, 0.9485, 0.9468, 0.9461, 0.9475],
            "GCN-only":          [0.9388, 0.9374, 0.9401, 0.9385, 0.9398, 0.9378, 0.9410, 0.9392, 0.9385, 0.9399],
        }
        run_significance_tests(ergt_f1_scores, baseline_f1s, metric_name="F1-Score")


if __name__ == "__main__":
    main()
