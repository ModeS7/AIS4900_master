#!/usr/bin/env python3
"""Compute PCA shape model from real brain masks.

Builds a principal component model of brain shapes from training data.
At generation time, a generated brain mask is projected onto this model
and the reconstruction error is used to detect non-brain-like shapes.

Methodology (paper-defensible):
    - PCA is FIT on one set of real brains (``--fit-splits``, default ``train``).
    - k (number of components) and the accept-threshold are grounded on a
      DISJOINT held-out set of real brains (``--heldout-splits``, default
      ``test_new``) that the PCA never saw.
    - The accept-threshold is the ``--threshold-percentile`` (default 99th)
      of held-out reconstruction errors: a generated brain is accepted if its
      shape is as reconstructable as that fraction of real brains the model
      never saw. This replaces the ad-hoc ``max(error) x multiplier``.
    - The held-out reconstruction-error-vs-k curve is printed so k is chosen
      by generalization rather than by eye. ``--n-components full`` keeps all
      components (cumulative EVR = 1.0). Use validate_pca_filter.py (ROC vs
      corrupted brains) to pick k x resolution objectively.

NOTE: do NOT include overlapping splits in --fit-splits (e.g. train AND val,
where val is drawn from train) — that double-counts brains (the historical
"n_samples=181" bug when the true unique count was 156).

Usage:
    python -m medgen.scripts.compute_brain_pca \
        --data-root /path/to/brainmetshare-3 \
        --output data/brain_pca_256x256x160.npz \
        --n-components full --threshold-percentile 99
"""
import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from medgen.metrics.brain_mask import create_brain_mask

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Default downsample resolution for PCA (keeps shape info, reduces compute).
# Overridable via CLI (--pca-size, --pca-depth).
PCA_DEPTH = 80
PCA_SIZE = 128


def load_volume(path: Path, depth: int, image_size: int) -> np.ndarray:
    """Load NIfTI -> [D, H, W] numpy array in [0, 1]."""
    vol = nib.load(str(path)).get_fdata().astype(np.float32)
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)

    vol = np.transpose(vol, (2, 0, 1))  # [H, W, D] -> [D, H, W]

    d = vol.shape[0]
    if d < depth:
        pad = np.zeros((depth - d, vol.shape[1], vol.shape[2]), dtype=np.float32)
        vol = np.concatenate([vol, pad], axis=0)
    elif d > depth:
        vol = vol[:depth]

    if vol.shape[1] != image_size or vol.shape[2] != image_size:
        vol_tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
        vol_tensor = torch.nn.functional.interpolate(
            vol_tensor, size=(depth, image_size, image_size),
            mode='trilinear', align_corners=False,
        )
        vol = vol_tensor.squeeze().numpy()

    return vol


def downsample_mask(mask: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Downsample binary mask using trilinear interpolation + threshold."""
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    down = torch.nn.functional.interpolate(
        mask_tensor, size=target_shape, mode='trilinear', align_corners=False,
    )
    return (down.squeeze().numpy() > 0.5).astype(np.float32)


def load_masks_for_splits(
    data_root: Path,
    splits: list[str],
    depth: int,
    image_size: int,
    brain_threshold: float,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Load + downsample brain masks for the given splits into a [N, n_voxels] array."""
    masks: list[np.ndarray] = []
    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            logger.warning(f"Split '{split}' not found, skipping")
            continue
        files = sorted(split_dir.glob("*/bravo.nii.gz"))
        logger.info(f"  {split}: {len(files)} volumes")
        for i, path in enumerate(files):
            vol = load_volume(path, depth, image_size)
            mask = create_brain_mask(vol, threshold=brain_threshold, fill_holes=True, dilate_pixels=0)
            masks.append(downsample_mask(mask, target_shape).flatten())
            if (i + 1) % 25 == 0:
                logger.info(f"    Processed {i + 1}/{len(files)}")
    if not masks:
        raise RuntimeError(f"No brain masks loaded for splits {splits} under {data_root}")
    return np.array(masks, dtype=np.float32)


def reconstruction_errors(masks: np.ndarray, mean: np.ndarray, components: np.ndarray, k: int) -> np.ndarray:
    """Per-sample MSE reconstruction error using the top-k components."""
    centered = masks - mean
    proj = centered @ components[:k].T
    recon = proj @ components[:k] + mean
    return np.mean((masks - recon) ** 2, axis=1)


def main():
    parser = argparse.ArgumentParser(description="Compute brain PCA shape model")
    parser.add_argument('--data-root', required=True, help='Dataset root (contains train/, test_new/, ...)')
    parser.add_argument('--output', default='data/brain_pca_256x256x160.npz', help='Output .npz path')
    parser.add_argument('--image-size', type=int, default=256, help='Target H/W resolution')
    parser.add_argument('--depth', type=int, default=160, help='Target depth (D)')
    parser.add_argument('--threshold', type=float, default=0.05, help='Brain detection threshold')
    parser.add_argument('--n-components', default='full',
                        help="Number of PCA components: an int, or 'full' (all, cumEVR=1.0). Default: full")
    parser.add_argument('--pca-size', type=int, default=PCA_SIZE,
                        help='Target H/W after downsampling for PCA (default 128)')
    parser.add_argument('--pca-depth', type=int, default=PCA_DEPTH,
                        help='Target D after downsampling for PCA (default 80)')
    parser.add_argument('--threshold-percentile', type=float, default=99.0,
                        help='Accept-threshold = this percentile of HELD-OUT real reconstruction errors (default 99)')
    parser.add_argument('--fit-splits', nargs='+', default=['train'],
                        help='Splits to FIT PCA on (default: train). Must be disjoint from --heldout-splits.')
    parser.add_argument('--heldout-splits', nargs='+', default=['test1'],
                        help='Disjoint real splits to calibrate k + threshold on (default: test1, the '
                             'original 51 held-out; val/test_new are sub-splits of test1). Must be disjoint from fit.')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    target_shape = (args.pca_depth, args.pca_size, args.pca_size)
    n_voxels = args.pca_depth * args.pca_size * args.pca_size

    overlap = set(args.fit_splits) & set(args.heldout_splits)
    if overlap:
        raise ValueError(f"--fit-splits and --heldout-splits overlap ({overlap}); that leaks/double-counts brains.")

    logger.info(f"Data root: {data_root}")
    logger.info(f"Full resolution: {args.image_size}x{args.image_size}x{args.depth}")
    logger.info(f"PCA resolution: {args.pca_size}x{args.pca_size}x{args.pca_depth}")
    logger.info(f"Fit splits: {args.fit_splits} | Held-out splits: {args.heldout_splits}")

    # Load fit + held-out masks (disjoint real brains)
    logger.info("Loading FIT masks...")
    fit_masks = load_masks_for_splits(data_root, args.fit_splits, args.depth, args.image_size, args.threshold, target_shape)
    logger.info("Loading HELD-OUT masks...")
    heldout_masks = load_masks_for_splits(data_root, args.heldout_splits, args.depth, args.image_size, args.threshold, target_shape)
    n_fit, n_heldout = fit_masks.shape[0], heldout_masks.shape[0]
    logger.info(f"Fit brains: {n_fit} | Held-out brains: {n_heldout} | voxels: {n_voxels}")

    # Fit PCA on the fit set (economy SVD; N << n_voxels)
    max_k = n_fit - 1
    mean = fit_masks.mean(axis=0)
    centered = fit_masks - mean
    logger.info(f"Computing SVD ({n_fit} x {n_voxels})...")
    _U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Resolve k (int or 'full')
    if str(args.n_components).lower() == 'full':
        k = max_k
    else:
        k = min(int(args.n_components), max_k)
    components_all = Vt[:max_k]  # keep all for the k-curve
    total_variance = (S ** 2).sum()
    cum_evr = np.cumsum(S ** 2) / total_variance

    # Held-out reconstruction-error-vs-k curve (objective k selection aid)
    logger.info("\nHeld-out reconstruction error vs k (choose k by generalization, not by eye):")
    logger.info(f"{'k':>5} {'cumEVR':>8} {'heldout_err':>13} {'fit_err':>11}")
    k_grid = sorted({kk for kk in [10, 20, 30, 60, 90, 120, 150, max_k] if 1 <= kk <= max_k} | {k})
    k_curve = []
    for kk in k_grid:
        he = reconstruction_errors(heldout_masks, mean, components_all, kk).mean()
        fe = reconstruction_errors(fit_masks, mean, components_all, kk).mean()
        k_curve.append((kk, float(cum_evr[kk - 1]), float(he), float(fe)))
        mark = "  <- selected" if kk == k else ""
        logger.info(f"{kk:>5} {cum_evr[kk-1]:>8.4f} {he:>13.6f} {fe:>11.6f}{mark}")

    # Final model at chosen k
    components = components_all[:k]
    explained_variance = (S[:k] ** 2) / (n_fit - 1)
    fit_errors = reconstruction_errors(fit_masks, mean, components, k)
    heldout_errors = reconstruction_errors(heldout_masks, mean, components, k)

    # Threshold calibrated on HELD-OUT real brains (percentile), NOT max(fit)*mult
    error_threshold = float(np.percentile(heldout_errors, args.threshold_percentile))
    logger.info(f"\nChosen k={k} (cumEVR={cum_evr[k-1]:.4f})")
    logger.info("Held-out reconstruction errors (real, unseen):")
    logger.info(f"  mean={heldout_errors.mean():.6f}  p95={np.percentile(heldout_errors,95):.6f}  "
                f"p99={np.percentile(heldout_errors,99):.6f}  max={heldout_errors.max():.6f}")
    logger.info(f"  Accept-threshold = p{args.threshold_percentile:g}(held-out) = {error_threshold:.6f}")

    # Save (keeps the .npz contract consumed by the generation pipeline; adds calibration fields)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        mean=mean,
        components=components,
        explained_variance=explained_variance,
        full_singular_values=S.astype(np.float32),
        error_threshold=np.array([error_threshold]),
        pca_shape=np.array(target_shape),
        full_shape=np.array([args.depth, args.image_size, args.image_size]),
        n_samples=np.array([n_fit]),          # brains PCA was FIT on (no double-count)
        real_errors=fit_errors,               # fit-set errors (kept for back-compat)
        heldout_errors=heldout_errors,        # NEW: unseen-real errors (threshold basis)
        n_heldout=np.array([n_heldout]),
        chosen_k=np.array([k]),
        threshold_percentile=np.array([args.threshold_percentile]),
        k_curve=np.array(k_curve, dtype=np.float32),  # [ (k, cumEVR, heldout_err, fit_err), ... ]
    )
    logger.info(f"Saved PCA model to {output_path}")


if __name__ == '__main__':
    main()
