"""Compute the real-brain HF energy reference used by diagnose_mean_blur.

Mirrors compute_hf_energy_ratio() in src/medgen/scripts/diagnose_mean_blur.py
(FFT energy ratio above radial frequency 0.25, optionally masked by the brain
or segmentation mask). Walks all bravo.nii.gz files under one or more split
directories of brainmetshare-3 and reports the mean/std/n across the set.

The resulting reference anchors the "data-scale ceiling" and "FFL inert" claims
in the thesis Discussion. Replaces the prior throw-away /tmp/compute_real_hf.py.

Usage:
    python misc/analysis/compute_real_hf_reference.py \\
        --data-root /cluster/work/$USER/MedicalDataSets/brainmetshare-3 \\
        --splits test_new \\
        --mask brain
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def compute_hf_energy_ratio(volume: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Fraction of FFT energy above radial freq 0.25 (Nyquist/4).

    Identical to compute_hf_energy_ratio in src/medgen/scripts/diagnose_mean_blur.py.
    Kept here as a copy so this script can run standalone without importing the
    diffusion stack (which pulls torch and the rest of the repo).
    """
    if mask is not None:
        volume = volume * mask
    fft = np.fft.fftn(volume)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2
    d, h, w = volume.shape
    cd, ch, cw = d // 2, h // 2, w // 2
    dz, dy, dx = np.ogrid[-cd:d - cd, -ch:h - ch, -cw:w - cw]
    radius = np.sqrt((dz / d) ** 2 + (dy / h) ** 2 + (dx / w) ** 2)
    high = radius > 0.25
    total = power.sum()
    if total <= 0:
        return 0.0
    return float(power[high].sum() / total)


def list_subjects(data_root: Path, splits: list[str]) -> list[Path]:
    """Return all subject directories across the given splits, sorted."""
    out: list[Path] = []
    for split in splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            raise SystemExit(f"Split directory not found: {split_dir}")
        for d in sorted(split_dir.iterdir()):
            if (d / "bravo.nii.gz").exists():
                out.append(d)
    return out


def load_volume(path: Path) -> np.ndarray:
    """Load NIfTI as float32, min-max-normalised to [0, 1]."""
    arr = nib.load(str(path)).get_fdata().astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True,
                   help="Root of brainmetshare-3 (must contain split subdirs).")
    p.add_argument("--splits", nargs="+", default=["test_new"],
                   help="Split(s) to scan. Default: test_new (n=51).")
    p.add_argument("--mask", choices=["none", "brain", "seg"], default="brain",
                   help="Mask applied before HF computation. brain = bravo > 0.02. "
                        "seg = use seg.nii.gz if present (Boolean union with brain).")
    p.add_argument("--brain-threshold", type=float, default=0.02)
    p.add_argument("--max-subjects", type=int, default=None,
                   help="Limit to first N subjects (testing only).")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional output JSON path.")
    args = p.parse_args()

    subjects = list_subjects(args.data_root, args.splits)
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]
    if not subjects:
        raise SystemExit("No subjects found.")
    print(f"Scanning {len(subjects)} subjects across splits {args.splits} "
          f"with mask={args.mask}")

    values: list[float] = []
    for i, subj in enumerate(subjects, 1):
        bravo = load_volume(subj / "bravo.nii.gz")
        if args.mask == "none":
            mask = None
        elif args.mask == "brain":
            mask = (bravo > args.brain_threshold).astype(np.float32)
        elif args.mask == "seg":
            seg_path = subj / "seg.nii.gz"
            if seg_path.exists():
                seg = nib.load(str(seg_path)).get_fdata().astype(np.float32)
                mask = ((bravo > args.brain_threshold) | (seg > 0.5)).astype(np.float32)
            else:
                mask = (bravo > args.brain_threshold).astype(np.float32)
        ratio = compute_hf_energy_ratio(bravo, mask=mask)
        values.append(ratio)
        print(f"  [{i:3d}/{len(subjects)}] {subj.name}: HF={ratio:.4f}")

    arr = np.asarray(values, dtype=np.float64)
    summary = {
        "data_root": str(args.data_root.resolve()),
        "splits": args.splits,
        "mask": args.mask,
        "brain_threshold": args.brain_threshold,
        "n_subjects": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    print()
    print("=== Real-brain HF reference ===")
    print(f"  n      = {summary['n_subjects']}")
    print(f"  mean   = {summary['mean']:.4f}")
    print(f"  std    = {summary['std']:.4f}")
    print(f"  min    = {summary['min']:.4f}")
    print(f"  max    = {summary['max']:.4f}")
    print(f"  → reference: {summary['mean']:.4f} ± {summary['std']:.4f} (n={summary['n_subjects']})")

    if args.output is not None:
        import json
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
