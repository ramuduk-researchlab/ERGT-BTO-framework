# ERGT-BTO: Explainable Radiomic Graph Transformer with Bluefin Trevally Optimization

> Implementation based on:  
> **"Explainable Radiomic Graph Transformer with Metaheuristic Optimization for Robust Breast Cancer Histopathological Stage Classification"**  
> Sujit Kumar et al., 2026

---

## Overview

ERGT-BTO is an end-to-end framework for 4-class breast cancer histopathological staging:

| Class | Description |
|-------|-------------|
| 0 | Normal |
| 1 | Benign |
| 2 | In-Situ Carcinoma |
| 3 | Invasive Carcinoma |

### Architecture Components

1. **Multi-Scale Radiomic Feature Extraction** – Three parallel CNN branches (3×3, 5×5, 7×7 kernels) + GLCM/LBP handcrafted features
2. **Cell Graph Construction + GCN Encoding** – HoVer-Net nucleus detection → k-NN graph → Graph Convolutional Network
3. **Transformer-Based Contextual Aggregation** – Multi-head self-attention over tissue sub-regions
4. **BTO Hyperparameter Optimization** – Bluefin Trevally-inspired metaheuristic for automated tuning
5. **Dual-Layer Explainability** – SHAP feature attribution + Transformer Attention Rollout

---

## Datasets

### BACH (Primary)
- 400 H&E images at 200× magnification, 2048×1536 pixels
- 4 balanced classes (100 each): Normal, Benign, In-Situ, Invasive
- Download: https://iciar2018-challenge.grand-challenge.org/Dataset/

### BreaKHis (Cross-Dataset Validation)
- 7,909 images from 82 patients at 40×/100×/200×/400×
- Binary labels (benign/malignant) with 8 subtypes
- Download: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

---

## Project Structure

```
ergt_bto/
├── README.md
├── requirements.txt
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation & metrics
├── data/
│   ├── dataset.py              # BACH & BreaKHis dataset loaders
│   ├── preprocessing.py        # Stain normalization, augmentation
│   └── patient_split.py        # Patient-level stratified splitting
├── models/
│   ├── ergt_bto.py             # Full ERGT-BTO model
│   ├── multiscale_radiomics.py # Multi-scale CNN branches + GLCM/LBP
│   ├── gcn_encoder.py          # Cell graph construction + GCN
│   └── transformer_encoder.py  # Transformer contextual aggregation
├── optimization/
│   └── bto_optimizer.py        # Bluefin Trevally Optimization
├── explainability/
│   ├── shap_explainer.py       # SHAP feature attribution
│   └── attention_rollout.py    # Transformer attention rollout
└── utils/
    ├── metrics.py              # Accuracy, F1, AUC, MCC, Cohen's κ
    └── visualization.py        # Confusion matrix, ROC curves, plots
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
# 1. Download datasets and place them at:
#    ./datasets/BACH/
#    ./datasets/BreaKHis/

# 2. Train with BTO optimization
python train.py --dataset bach --epochs 50 --use_bto

# 3. Evaluate on test set
python evaluate.py --checkpoint checkpoints/ergt_bto_best.pt --dataset bach

# 4. Generate SHAP + attention rollout explanations
python evaluate.py --checkpoint checkpoints/ergt_bto_best.pt --explain
```

---

## Reported Results (BACH Dataset)

| Metric | Value |
|--------|-------|
| Accuracy | 97.84% |
| F1-Score (macro) | 97.62% |
| AUC | 0.989 |
| MCC | 0.970 |
| Cohen's κ | 0.967 |
