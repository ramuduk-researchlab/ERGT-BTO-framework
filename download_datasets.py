"""
download_datasets.py
====================
Helper script with instructions and automated download for:
  1. BACH Dataset  — https://iciar2018-challenge.grand-challenge.org/Dataset/
  2. BreaKHis Dataset — https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

Note: Both datasets require free registration / agreement to terms of use.
This script provides download instructions and verifies directory structure
after manual download.
"""

from __future__ import annotations

import os
import sys
import zipfile
import hashlib
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# BACH Dataset
# ─────────────────────────────────────────────────────────────────────────────

BACH_INFO = """
╔══════════════════════════════════════════════════════════════════╗
║                    BACH Dataset                                  ║
║  Breast Cancer Histology (ICIAR 2018 Grand Challenge)            ║
╠══════════════════════════════════════════════════════════════════╣
║  URL    : https://iciar2018-challenge.grand-challenge.org/       ║
║           Dataset/                                               ║
║  Format : 400 H&E tif images (2048×1536 px, 200× magnification) ║
║  Classes: Normal (100), Benign (100), InSitu (100),              ║
║           Invasive (100)                                          ║
║  License: CC BY-NC-SA 4.0 (free for research)                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Download Steps:                                                 ║
║  1. Register at grand-challenge.org                              ║
║  2. Visit the BACH challenge page (link above)                   ║
║  3. Download ICIAR2018_BACH_Challenge.zip (~3.8 GB)              ║
║  4. Extract to ./datasets/BACH/                                  ║
║                                                                  ║
║  Expected structure:                                             ║
║    datasets/BACH/                                                ║
║      Normal/    *.tif                                            ║
║      Benign/    *.tif                                            ║
║      InSitu/    *.tif                                            ║
║      Invasive/  *.tif                                            ║
╚══════════════════════════════════════════════════════════════════╝
"""

BREAKHIS_INFO = """
╔══════════════════════════════════════════════════════════════════╗
║                  BreaKHis Dataset                                ║
║  Breast Cancer Histopathological Image Classification             ║
╠══════════════════════════════════════════════════════════════════╣
║  URL    : https://web.inf.ufpr.br/vri/databases/                 ║
║           breast-cancer-histopathological-database-breakhis/     ║
║  Format : 7,909 PNG images from 82 patients                      ║
║           Magnifications: 40×, 100×, 200×, 400×                 ║
║  Classes: Benign (2,480) / Malignant (5,429)                     ║
║           8 subtypes total                                        ║
║  License: Academic use (contact authors)                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Download Steps:                                                 ║
║  1. Visit the dataset page (link above)                          ║
║  2. Fill the request form                                        ║
║  3. Download BreaKHis_v1.tar.gz (~2.8 GB)                        ║
║  4. Extract to ./datasets/BreaKHis/                              ║
║                                                                  ║
║  Expected structure:                                             ║
║    datasets/BreaKHis/                                            ║
║      breast_cancer_organized/                                    ║
║        benign/SOB/                                               ║
║          adenosis/ fibroadenoma/ phyllodes_tumor/ tubular_adenoma║
║        malignant/SOB/                                            ║
║          ductal_carcinoma/ lobular_carcinoma/                    ║
║          mucinous_carcinoma/ papillary_carcinoma/                ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
# Verification helpers
# ─────────────────────────────────────────────────────────────────────────────

def verify_bach(root: str = "./datasets/BACH") -> bool:
    """Verify BACH dataset structure and count images."""
    root = Path(root)
    expected = {"Normal": 100, "Benign": 100, "InSitu": 100, "Invasive": 100}
    all_ok = True

    print(f"\n[BACH] Verifying dataset at: {root}")
    print(f"{'Class':<15} {'Expected':>10} {'Found':>10} {'Status':>10}")
    print("-" * 50)

    for cls_name, expected_count in expected.items():
        cls_path = root / cls_name
        if not cls_path.exists():
            print(f"{cls_name:<15} {expected_count:>10} {'MISSING DIR':>10} {'❌':>10}")
            all_ok = False
            continue
        found = len(list(cls_path.glob("*.tif")))
        status = "✅" if found >= expected_count else f"❌ ({found})"
        print(f"{cls_name:<15} {expected_count:>10} {found:>10} {status:>10}")
        if found < expected_count:
            all_ok = False

    if all_ok:
        print("\n✅ BACH dataset verified successfully (400 images across 4 classes)")
    else:
        print("\n❌ BACH dataset verification failed. Check the structure above.")
    return all_ok


def verify_breakhis(root: str = "./datasets/BreaKHis") -> bool:
    """Verify BreaKHis dataset structure and count images."""
    root = Path(root)
    subtypes = [
        ("benign", ["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma"]),
        ("malignant", ["ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"]),
    ]
    total_found = 0
    all_ok = True

    print(f"\n[BreaKHis] Verifying dataset at: {root}")
    for category, subtypes_list in subtypes:
        print(f"\n  {category.upper()}:")
        for subtype in subtypes_list:
            # Search for the subtype directory recursively
            matches = list(root.rglob(f"*{subtype}*"))
            dirs = [m for m in matches if m.is_dir()]
            count = sum(len(list(d.rglob("*.png"))) for d in dirs)
            total_found += count
            status = "✅" if count > 0 else "❌ NOT FOUND"
            print(f"    {subtype:<25} {count:>6} images  {status}")
            if count == 0:
                all_ok = False

    print(f"\n  Total images found: {total_found}")
    expected_min = 7000
    if total_found >= expected_min:
        print(f"✅ BreaKHis dataset verified successfully ({total_found} images)")
    else:
        print(f"❌ Expected ~7909 images, found {total_found}. Check structure.")
        all_ok = False
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Dataset statistics
# ─────────────────────────────────────────────────────────────────────────────

def print_dataset_statistics(root: str, dataset: str = "bach") -> None:
    """Print detailed dataset statistics."""
    if dataset.lower() == "bach":
        root = Path(root)
        print("\n[BACH] Dataset Statistics:")
        print(f"{'Class':<15} {'Images':>8} {'% Total':>10}")
        print("-" * 35)
        total = 0
        counts = {}
        for cls in ["Normal", "Benign", "InSitu", "Invasive"]:
            cls_path = root / cls
            if cls_path.exists():
                n = len(list(cls_path.glob("*.tif")))
                counts[cls] = n
                total += n
        for cls, n in counts.items():
            print(f"{cls:<15} {n:>8} {100*n/max(total,1):>9.1f}%")
        print(f"{'Total':<15} {total:>8}")

    elif dataset.lower() == "breakhis":
        print("\n[BreaKHis] Multi-magnification dataset statistics:")
        print("  400 images per magnification × 4 magnifications")
        print("  (40×, 100×, 200×, 400×)")
        print("  Benign subtypes: adenosis, fibroadenoma, phyllodes_tumor, tubular_adenoma")
        print("  Malignant subtypes: ductal_carcinoma, lobular_carcinoma, mucinous_carcinoma, papillary_carcinoma")
        print("  Total: ~7,909 images from 82 patients")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dataset setup helper for ERGT-BTO")
    parser.add_argument("--action", choices=["info", "verify", "stats"],
                        default="info", help="Action to perform")
    parser.add_argument("--dataset", choices=["bach", "breakhis", "both"],
                        default="both")
    parser.add_argument("--bach_root",     default="./datasets/BACH")
    parser.add_argument("--breakhis_root", default="./datasets/BreaKHis")
    args = parser.parse_args()

    if args.action == "info":
        if args.dataset in ("bach", "both"):
            print(BACH_INFO)
        if args.dataset in ("breakhis", "both"):
            print(BREAKHIS_INFO)

    elif args.action == "verify":
        if args.dataset in ("bach", "both"):
            verify_bach(args.bach_root)
        if args.dataset in ("breakhis", "both"):
            verify_breakhis(args.breakhis_root)

    elif args.action == "stats":
        if args.dataset in ("bach", "both"):
            print_dataset_statistics(args.bach_root, "bach")
        if args.dataset in ("breakhis", "both"):
            print_dataset_statistics(args.breakhis_root, "breakhis")


if __name__ == "__main__":
    main()
