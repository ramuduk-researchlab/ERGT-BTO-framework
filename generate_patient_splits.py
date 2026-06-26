"""
generate_patient_splits.py
---------------------------------------------------------------------
Generates the patient-ID-to-partition (train/validation/test) manifest
CSVs described in Section 3.1.3 of the ERGT-BTO manuscript.

This script must be run against your ACTUAL local copies of the BACH
and BreaKHis datasets. It will not invent patient IDs -- it reads them
directly from the real filenames/annotation files, then applies the
exact stratified 70/15/15 patient-level split described in the paper.

Usage:
    python generate_patient_splits.py \
        --bach_dir /path/to/BACH/ICIAR2018_BACH_Challenge/Photos \
        --breakhis_dir /path/to/BreaKHis_v1/histology_slides/breast \
        --seed 0 \
        --out_dir ./splits

Output:
    splits/bach_patient_splits.csv
    splits/breakhis_patient_splits.csv

    Each CSV has columns: patient_id, class_label, split, n_images
---------------------------------------------------------------------
"""

import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np


def stratified_patient_split(patient_to_class, seed, train_frac=0.70, val_frac=0.15):
    """
    Patient-level stratified split. Splitting happens on unique patient
    IDs (never on images), so every image from a given patient lands
    in exactly one partition -- matching Section 3.1.3 of the manuscript.
    """
    rng = np.random.RandomState(seed)

    class_to_patients = defaultdict(list)
    for pid, cls in patient_to_class.items():
        class_to_patients[cls].append(pid)

    split_assignment = {}
    for cls, patients in class_to_patients.items():
        patients = sorted(patients)  # deterministic ordering before shuffle
        rng.shuffle(patients)
        n = len(patients)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        for i, pid in enumerate(patients):
            if i < n_train:
                split_assignment[pid] = "train"
            elif i < n_train + n_val:
                split_assignment[pid] = "validation"
            else:
                split_assignment[pid] = "test"
    return split_assignment


def parse_breakhis_patient_id(filename):
    """
    BreaKHis filenames follow the convention:
        SOB_<B|M>_<TUMOR>-<YEAR>-<BIOPSY_ID>-<MAG>-<SEQ>.png
    e.g. SOB_B_TA-14-4659-40-001.png  ->  patient/biopsy ID = "14-4659"
    """
    m = re.search(r"SOB_[BM]_[A-Z]+-(\d+-\d+)-\d+-\d+", filename)
    if not m:
        raise ValueError(f"Could not parse patient ID from: {filename}")
    return m.group(1)


def collect_breakhis(breakhis_dir):
    patient_to_class = {}
    patient_image_count = defaultdict(int)
    for root, _, files in os.walk(breakhis_dir):
        for f in files:
            if not f.lower().endswith((".png", ".tif", ".jpg")):
                continue
            pid = parse_breakhis_patient_id(f)
            # class derived from folder path (benign/malignant subtype)
            cls = "malignant" if "_M_" in f else "benign"
            patient_to_class[pid] = cls
            patient_image_count[pid] += 1
    return patient_to_class, patient_image_count


def collect_bach(bach_dir):
    """
    The public ICIAR2018 BACH release does not embed a multi-image
    patient/case identifier in its filenames -- each photograph
    corresponds to one annotated case. If your copy of BACH has a
    supplementary case/patient annotation file (as referenced in your
    Methods, 'BACH dataset annotation information'), point this
    function at that file instead. Edit the path below.
    """
    patient_to_class = {}
    patient_image_count = defaultdict(int)
    classes = ["Normal", "Benign", "InSitu", "Invasive"]
    for cls in classes:
        cls_dir = os.path.join(bach_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for f in sorted(os.listdir(cls_dir)):
            if not f.lower().endswith((".tif", ".png", ".jpg")):
                continue
            # Default: one image = one case ID (filename stem).
            # Replace with real case-ID lookup if you have the
            # annotation file mapping multiple images to one patient.
            pid = os.path.splitext(f)[0]
            patient_to_class[pid] = cls
            patient_image_count[pid] += 1
    return patient_to_class, patient_image_count


def write_csv(path, patient_to_class, split_assignment, image_counts):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient_id", "class_label", "split", "n_images"])
        for pid in sorted(patient_to_class):
            writer.writerow([
                pid,
                patient_to_class[pid],
                split_assignment[pid],
                image_counts[pid],
            ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bach_dir", required=False)
    ap.add_argument("--breakhis_dir", required=False)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="./splits")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.bach_dir:
        p2c, counts = collect_bach(args.bach_dir)
        split = stratified_patient_split(p2c, seed=args.seed)
        out = os.path.join(args.out_dir, "bach_patient_splits.csv")
        write_csv(out, p2c, split, counts)
        print(f"Wrote {out}  ({len(p2c)} unique cases)")

    if args.breakhis_dir:
        p2c, counts = collect_breakhis(args.breakhis_dir)
        split = stratified_patient_split(p2c, seed=args.seed)
        out = os.path.join(args.out_dir, "breakhis_patient_splits.csv")
        write_csv(out, p2c, split, counts)
        print(f"Wrote {out}  ({len(p2c)} unique patients)")

    if not args.bach_dir and not args.breakhis_dir:
        print("Provide --bach_dir and/or --breakhis_dir pointing at your local dataset copies.")


if __name__ == "__main__":
    main()
