#!/usr/bin/env python3
"""Compute PCA shape model from real brain masks.

Builds a principal component model of brain shapes from training data.
At generation time, a generated brain mask is projected onto this model
and the reconstruction error is used to detect non-brain-like shapes.

Usage:
    python -m medgen.scripts.compute_brain_pca \
        --data-root /path/to/brainmetshare-3 \
        --output data/brain_pca_256x256x160.npz \
        --image-size 256 --depth 160
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


def main():
    parser = argparse.ArgumentParser(description="Compute brain PCA shape model")
    parser.add_argument('--data-root', required=True, help='Dataset root (contains train/)')
    parser.add_argument('--output', default='data/brain_pca_256x256x160.npz', help='Output .npz path')
    parser.add_argument('--image-size', type=int, default=256, help='Target H/W resolution')
    parser.add_argument('--depth', type=int, default=160, help='Target depth (D)')
    parser.add_argument('--threshold', type=float, default=0.05, help='Brain detection threshold')
    parser.add_argument('--n-components', type=int, default=30, help='Number of PCA components')
    parser.add_argument('--pca-size', type=int, default=PCA_SIZE,
                        help='Target H/W after downsampling for PCA (default 128)')
    parser.add_argument('--pca-depth', type=int, default=PCA_DEPTH,
                        help='Target D after downsampling for PCA (default 80)')
    parser.add_argument('--threshold-multiplier', type=float, default=3.0,
                        help='Threshold = max real error x multiplier (default 3x)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test1'],
                        help='Dataset splits to include')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    target_shape = (args.pca_depth, args.pca_size, args.pca_size)
    n_voxels = args.pca_depth * args.pca_size * args.pca_size

    logger.info(f"Data root: {data_root}")
    logger.info(f"Full resolution: {args.image_size}x{args.image_size}x{args.depth}")
    logger.info(f"PCA resolution: {args.pca_size}x{args.pca_size}x{args.pca_depth}")
    logger.info(f"Components: {args.n_components}")

    # Collect all brain masks
    masks = []
    for split in args.splits:
        split_dir = data_root / split
        if not split_dir.exists():
            logger.warning(f"Split '{split}' not found, skipping")
            continue

        files = sorted(split_dir.glob("*/bravo.nii.gz"))
        logger.info(f"  {split}: {len(files)} volumes")

        for i, path in enumerate(files):
            vol = load_volume(path, args.depth, args.image_size)
            mask = create_brain_mask(vol, threshold=args.threshold, fill_holes=True, dilate_pixels=0)
            mask_down = downsample_mask(mask, target_shape)
            masks.append(mask_down.flatten())

            if (i + 1) % 25 == 0:
                logger.info(f"    Processed {i + 1}/{len(files)}")

    masks = np.array(masks, dtype=np.float32)  # [N, n_voxels]
    n_samples = masks.shape[0]
    logger.info(f"Total masks: {n_samples}, shape: {masks.shape}")

    # Compute PCA
    n_components = min(args.n_components, n_samples - 1)
    mean = masks.mean(axis=0)
    centered = masks - mean

    # Economy SVD (N x n_voxels, N << n_voxels)
    logger.info(f"Computing SVD ({n_samples} x {n_voxels})...")
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]  # [n_components, n_voxels]
    explained_variance = (S[:n_components] ** 2) / (n_samples - 1)
    total_variance = (S ** 2).sum() / (n_samples - 1)
    explained_ratio = explained_variance / total_variance

    logger.info(f"Variance explained by {n_components} components: {explained_ratio.sum():.1%}")
    for i in range(min(5, n_components)):
        logger.info(f"  PC{i+1}: {explained_ratio[i]:.1%}")

    # Compute reconstruction errors for all real brains
    projections = centered @ components.T  # [N, n_components]
    reconstructed = projections @ components + mean  # [N, n_voxels]
    errors = np.mean((masks - reconstructed) ** 2, axis=1)  # MSE per sample

    error_threshold = errors.max() * args.threshold_multiplier
    logger.info("\nReconstruction errors (real data):")
    logger.info(f"  Mean: {errors.mean():.6f}")
    logger.info(f"  Std:  {errors.std():.6f}")
    logger.info(f"  Min:  {errors.min():.6f}")
    logger.info(f"  Max:  {errors.max():.6f}")
    logger.info(f"  Threshold: {error_threshold:.6f} (max x {args.threshold_multiplier})")

    # Save
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
        n_samples=np.array([n_samples]),
        real_errors=errors,
    )
    logger.info(f"Saved PCA model to {output_path}")


if __name__ == '__main__':
    main()
