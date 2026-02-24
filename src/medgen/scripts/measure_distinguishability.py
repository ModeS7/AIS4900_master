"""Measure distinguishability between training data distribution and N(0,1) noise prior.

Computes per-voxel statistics and KL divergence from N(0,1) under different normalizations:
  - [0, 1] raw (exp1)
  - [-1, 1] rescale (exp1b)
  - N(0, 1) brain-only (exp1c)

Also reports Wasserstein-2 distance (1D marginal) and overlap coefficient.
"""

import glob
import math
import os

import nibabel as nib
import numpy as np
from tqdm import tqdm

DATA_DIR = "/home/mode/NTNU/MedicalDataSets/brainmetshare-3/train"
SHIFT = 0.2033
SCALE = 0.0832


def load_bravo_volumes():
    """Load all bravo volumes, scale to [0, 1]."""
    patients = sorted(glob.glob(os.path.join(DATA_DIR, "Mets_*")))
    all_voxels = []
    brain_voxels = []
    for p in tqdm(patients, desc="Loading volumes"):
        path = os.path.join(p, "bravo.nii.gz")
        if not os.path.exists(path):
            continue
        vol = nib.load(path).get_fdata().astype(np.float32)
        # Scale to [0, 1] (same as ScaleIntensity in training)
        vmin, vmax = vol.min(), vol.max()
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
        all_voxels.append(vol.ravel())
        brain_voxels.append(vol[vol > 0.01].ravel())

    return np.concatenate(all_voxels), np.concatenate(brain_voxels)


def kl_from_standard_normal(mu: float, var: float) -> float:
    """KL(N(mu, var) || N(0, 1))."""
    return 0.5 * (var + mu**2 - 1 - math.log(var))


def wasserstein2_1d(mu1: float, var1: float, mu2: float = 0.0, var2: float = 1.0) -> float:
    """W2 distance between two 1D Gaussians."""
    return math.sqrt((mu1 - mu2) ** 2 + (math.sqrt(var1) - math.sqrt(var2)) ** 2)


def overlap_coefficient(mu1: float, std1: float, mu2: float = 0.0, std2: float = 1.0) -> float:
    """Approximate overlap coefficient between two Gaussians (Bhattacharyya-based)."""
    avg_var = (std1**2 + std2**2) / 2
    bhatt = 0.25 * (mu1 - mu2) ** 2 / avg_var + 0.5 * math.log(avg_var / (std1 * std2))
    return math.exp(-bhatt)


def report(name: str, data: np.ndarray):
    """Compute and print all metrics for a given normalization."""
    mu = float(data.mean())
    var = float(data.var())
    std = math.sqrt(var)
    kl = kl_from_standard_normal(mu, var)
    w2 = wasserstein2_1d(mu, var)
    ovl = overlap_coefficient(mu, std)

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Mean:          {mu:.4f}")
    print(f"  Std:           {std:.4f}")
    print(f"  Var:           {var:.4f}")
    print(f"  Range:         [{data.min():.4f}, {data.max():.4f}]")
    print(f"  KL(data||N01): {kl:.4f}")
    print(f"  W2(data,N01):  {w2:.4f}")
    print(f"  Overlap coef:  {ovl:.4f}  (1=identical, 0=no overlap)")
    print(f"{'=' * 60}")
    return {"name": name, "mu": mu, "std": std, "kl": kl, "w2": w2, "overlap": ovl}


def main():
    print("Loading training data...")
    all_voxels, brain_voxels = load_bravo_volumes()
    print(f"Total voxels: {len(all_voxels):,}")
    print(f"Brain voxels: {len(brain_voxels):,}")

    results = []

    # 1. Raw [0, 1] — exp1
    results.append(report("exp1: [0, 1] raw (all voxels)", all_voxels))
    results.append(report("exp1: [0, 1] raw (brain only)", brain_voxels))

    # 2. Rescale [-1, 1] — exp1b
    rescaled_all = all_voxels * 2 - 1
    rescaled_brain = brain_voxels * 2 - 1
    results.append(report("exp1b: [-1, 1] rescale (all voxels)", rescaled_all))
    results.append(report("exp1b: [-1, 1] rescale (brain only)", rescaled_brain))

    # 3. Brain-only N(0,1) — exp1c
    normed_all = (all_voxels - SHIFT) / SCALE
    normed_brain = (brain_voxels - SHIFT) / SCALE
    results.append(report("exp1c: N(0,1) brain-norm (all voxels)", normed_all))
    results.append(report("exp1c: N(0,1) brain-norm (brain only)", normed_brain))

    # Summary table
    print(f"\n{'=' * 75}")
    print(f"  {'Normalization':<40} {'KL':>8} {'W2':>8} {'Overlap':>8}")
    print(f"{'=' * 75}")
    for r in results:
        print(f"  {r['name']:<40} {r['kl']:>8.3f} {r['w2']:>8.3f} {r['overlap']:>8.4f}")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()
