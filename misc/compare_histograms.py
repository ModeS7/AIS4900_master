#!/usr/bin/env python3
"""Compare intensity histograms of real vs generated 3D BRAVO volumes.

Loads multiple real and generated volumes, computes per-volume statistics
and overlaid histograms to identify distribution mismatches.

Usage:
    python misc/compare_histograms.py
    python misc/compare_histograms.py --n-samples 20
    python misc/compare_histograms.py --real-dir /path/to/real --gen-dir /path/to/gen
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def load_real_bravo(patient_dir: Path) -> np.ndarray:
    """Load real BRAVO, min-max normalize to [0,1], clamp."""
    img = nib.load(str(patient_dir / "bravo.nii.gz"))
    data = img.get_fdata().astype(np.float32)
    # Min-max normalize (same as ScaleIntensity in training pipeline)
    dmin, dmax = data.min(), data.max()
    if dmax > dmin:
        data = (data - dmin) / (dmax - dmin)
    data = np.clip(data, 0.0, 1.0)
    return data


def load_generated_bravo(sample_dir: Path) -> np.ndarray:
    """Load generated BRAVO (already in [0,1] from generation pipeline)."""
    img = nib.load(str(sample_dir / "bravo.nii.gz"))
    return img.get_fdata().astype(np.float32)


def compute_stats(data: np.ndarray, label: str) -> dict:
    """Compute intensity statistics for a volume."""
    # Brain-only mask (non-zero voxels)
    brain_mask = data > 0.01
    brain_vals = data[brain_mask]

    return {
        'label': label,
        'shape': data.shape,
        'min': data.min(),
        'max': data.max(),
        'mean_all': data.mean(),
        'std_all': data.std(),
        'mean_brain': brain_vals.mean() if len(brain_vals) > 0 else 0,
        'std_brain': brain_vals.std() if len(brain_vals) > 0 else 0,
        'median_brain': np.median(brain_vals) if len(brain_vals) > 0 else 0,
        'p5_brain': np.percentile(brain_vals, 5) if len(brain_vals) > 0 else 0,
        'p95_brain': np.percentile(brain_vals, 95) if len(brain_vals) > 0 else 0,
        'pct_zero': (data < 0.01).mean() * 100,
        'pct_above_0.9': (data > 0.9).mean() * 100,
        'pct_at_1.0': (data >= 0.999).mean() * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare real vs generated 3D BRAVO histograms")
    parser.add_argument("--real-dir", type=str,
                        default="/home/mode/NTNU/MedicalDataSets/brainmetshare-3/train")
    parser.add_argument("--gen-dir", type=str,
                        default="/home/mode/NTNU/MedicalDataSets/generated/exp1_1_3d_256_525_baseline")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of volumes to load from each set")
    parser.add_argument("--output", type=str, default="misc/histogram_comparison.png",
                        help="Output image path")
    args = parser.parse_args()

    real_dir = Path(args.real_dir)
    gen_dir = Path(args.gen_dir)

    # Find patient dirs
    real_patients = sorted([d for d in real_dir.iterdir() if d.is_dir() and (d / "bravo.nii.gz").exists()])
    gen_samples = sorted([d for d in gen_dir.iterdir() if d.is_dir() and (d / "bravo.nii.gz").exists()])

    n = min(args.n_samples, len(real_patients), len(gen_samples))
    print(f"Loading {n} real and {n} generated volumes...")

    # Load volumes
    real_stats = []
    gen_stats = []
    real_brain_vals = []
    gen_brain_vals = []

    for i in range(n):
        print(f"  Real {i+1}/{n}: {real_patients[i].name}")
        data = load_real_bravo(real_patients[i])
        stats = compute_stats(data, real_patients[i].name)
        real_stats.append(stats)
        brain = data[data > 0.01]
        real_brain_vals.append(brain)

    for i in range(n):
        print(f"  Gen  {i+1}/{n}: {gen_samples[i].name}")
        data = load_generated_bravo(gen_samples[i])
        stats = compute_stats(data, gen_samples[i].name)
        gen_stats.append(stats)
        brain = data[data > 0.01]
        gen_brain_vals.append(brain)

    # Print statistics table
    print("\n" + "=" * 90)
    print(f"{'':>20} {'Mean':>8} {'Std':>8} {'Med':>8} {'P5':>8} {'P95':>8} {'%>0.9':>8} {'%=1.0':>8}")
    print("-" * 90)

    for s in real_stats:
        print(f"{'R ' + s['label']:>20} {s['mean_brain']:8.4f} {s['std_brain']:8.4f} "
              f"{s['median_brain']:8.4f} {s['p5_brain']:8.4f} {s['p95_brain']:8.4f} "
              f"{s['pct_above_0.9']:8.2f} {s['pct_at_1.0']:8.2f}")

    print("-" * 90)
    for s in gen_stats:
        print(f"{'G ' + s['label']:>20} {s['mean_brain']:8.4f} {s['std_brain']:8.4f} "
              f"{s['median_brain']:8.4f} {s['p5_brain']:8.4f} {s['p95_brain']:8.4f} "
              f"{s['pct_above_0.9']:8.2f} {s['pct_at_1.0']:8.2f}")

    # Aggregate stats
    print("\n" + "=" * 90)
    r_means = [s['mean_brain'] for s in real_stats]
    g_means = [s['mean_brain'] for s in gen_stats]
    r_stds = [s['std_brain'] for s in real_stats]
    g_stds = [s['std_brain'] for s in gen_stats]
    r_above90 = [s['pct_above_0.9'] for s in real_stats]
    g_above90 = [s['pct_above_0.9'] for s in gen_stats]
    r_at1 = [s['pct_at_1.0'] for s in real_stats]
    g_at1 = [s['pct_at_1.0'] for s in gen_stats]

    print(f"{'REAL avg':>20} mean={np.mean(r_means):.4f}  std={np.mean(r_stds):.4f}  "
          f"%>0.9={np.mean(r_above90):.2f}  %=1.0={np.mean(r_at1):.2f}")
    print(f"{'GEN avg':>20} mean={np.mean(g_means):.4f}  std={np.mean(g_stds):.4f}  "
          f"%>0.9={np.mean(g_above90):.2f}  %=1.0={np.mean(g_at1):.2f}")

    # Plot histograms
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Real vs Generated 3D BRAVO Intensity Distribution", fontsize=14)

    # 1. All voxels histogram
    ax = axes[0, 0]
    all_real = np.concatenate(real_brain_vals)
    all_gen = np.concatenate(gen_brain_vals)
    ax.hist(all_real, bins=200, alpha=0.6, label=f'Real (n={n})', color='blue', density=True)
    ax.hist(all_gen, bins=200, alpha=0.6, label=f'Generated (n={n})', color='red', density=True)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.set_title("Brain Voxels (>0.01) — All Samples Pooled")
    ax.legend()
    ax.set_xlim(0, 1.05)

    # 2. Zoomed into high-intensity region
    ax = axes[0, 1]
    ax.hist(all_real, bins=200, alpha=0.6, label='Real', color='blue', density=True, range=(0.5, 1.0))
    ax.hist(all_gen, bins=200, alpha=0.6, label='Generated', color='red', density=True, range=(0.5, 1.0))
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.set_title("High Intensity Region (0.5 – 1.0)")
    ax.legend()

    # 3. Per-volume mean comparison
    ax = axes[1, 0]
    x = np.arange(n)
    width = 0.35
    ax.bar(x - width/2, r_means, width, label='Real', color='blue', alpha=0.7)
    ax.bar(x + width/2, g_means, width, label='Generated', color='red', alpha=0.7)
    ax.set_xlabel("Volume index")
    ax.set_ylabel("Brain Mean Intensity")
    ax.set_title("Per-Volume Brain Mean")
    ax.legend()

    # 4. CDF comparison
    ax = axes[1, 1]
    real_sorted = np.sort(all_real)
    gen_sorted = np.sort(all_gen)
    real_cdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
    gen_cdf = np.arange(1, len(gen_sorted) + 1) / len(gen_sorted)
    # Subsample for plotting speed
    step_r = max(1, len(real_sorted) // 5000)
    step_g = max(1, len(gen_sorted) // 5000)
    ax.plot(real_sorted[::step_r], real_cdf[::step_r], label='Real', color='blue')
    ax.plot(gen_sorted[::step_g], gen_cdf[::step_g], label='Generated', color='red')
    ax.set_xlabel("Intensity")
    ax.set_ylabel("CDF")
    ax.set_title("Cumulative Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved histogram comparison to: {args.output}")


if __name__ == "__main__":
    main()
