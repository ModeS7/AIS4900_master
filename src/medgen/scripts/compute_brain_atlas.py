#!/usr/bin/env python3
"""Compute a brain atlas (union of all training brain masks).

The atlas represents the maximum extent of brain tissue across all
training volumes. It can be used during generation to validate that
generated tumors fall within the brain boundary.

Usage:
    python -m medgen.scripts.compute_brain_atlas \
        --data-root /path/to/brainmetshare-3 \
        --output brain_atlas.nii.gz \
        --image-size 128 --depth 160
"""
import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from medgen.data.utils import save_nifti
from medgen.metrics.brain_mask import create_brain_mask

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_volume(path: Path, depth: int, image_size: int) -> torch.Tensor:
    """Load NIfTI -> [1, 1, D, H, W] tensor in [0, 1]."""
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

    vol_tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
    if vol_tensor.shape[3] != image_size or vol_tensor.shape[4] != image_size:
        vol_tensor = torch.nn.functional.interpolate(
            vol_tensor, size=(depth, image_size, image_size),
            mode='trilinear', align_corners=False,
        )
    return vol_tensor


def compute_brain_atlas(
    data_root: Path,
    image_size: int = 128,
    depth: int = 160,
    threshold: float = 0.05,
    dilate_pixels: int = 3,
) -> np.ndarray:
    """Compute brain atlas as union of all training brain masks.

    Args:
        data_root: Dataset root containing train/ subdirectory.
        image_size: Target H/W resolution.
        depth: Target depth (D).
        threshold: Intensity threshold for brain detection.
        dilate_pixels: Dilation applied to each individual brain mask
            before taking the union.

    Returns:
        Bool array [D, H, W] representing the atlas.
    """
    train_dir = data_root / 'train'
    if not train_dir.exists():
        # Try to find a subdirectory with bravo volumes
        for d in sorted(data_root.iterdir()):
            if d.is_dir() and list(d.glob("*/bravo.nii.gz")):
                train_dir = d
                break

    files = sorted(train_dir.glob("*/bravo.nii.gz"))
    if not files:
        raise FileNotFoundError(f"No bravo.nii.gz files found in {train_dir}")

    logger.info(f"Found {len(files)} training volumes in {train_dir}")

    atlas = np.zeros((depth, image_size, image_size), dtype=bool)

    for i, path in enumerate(files):
        vol = load_volume(path, depth, image_size)
        # vol is [1, 1, D, H, W], squeeze to [D, H, W]
        vol_np = vol.squeeze().numpy()

        mask = create_brain_mask(
            vol_np,
            threshold=threshold,
            fill_holes=True,
            dilate_pixels=dilate_pixels,
        )
        atlas |= mask

        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            logger.info(f"  Processed {i + 1}/{len(files)} (coverage: {atlas.mean():.1%})")

    return atlas


def main():
    parser = argparse.ArgumentParser(description="Compute brain atlas from training data")
    parser.add_argument('--data-root', required=True, help='Dataset root (contains train/)')
    parser.add_argument('--output', default='brain_atlas.nii.gz', help='Output NIfTI path')
    parser.add_argument('--image-size', type=int, default=128, help='Target H/W resolution')
    parser.add_argument('--depth', type=int, default=160, help='Target depth (D)')
    parser.add_argument('--threshold', type=float, default=0.05, help='Brain detection threshold')
    parser.add_argument('--dilate-pixels', type=int, default=3, help='Dilation per mask before union')
    parser.add_argument('--fov-mm', type=float, default=240.0, help='Field of view in mm')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    logger.info(f"Data root: {data_root}")
    logger.info(f"Resolution: {args.image_size}x{args.image_size}x{args.depth}")
    logger.info(f"Threshold: {args.threshold}, Dilate: {args.dilate_pixels}px")

    atlas = compute_brain_atlas(
        data_root,
        image_size=args.image_size,
        depth=args.depth,
        threshold=args.threshold,
        dilate_pixels=args.dilate_pixels,
    )

    logger.info(f"Atlas coverage: {atlas.mean():.1%} of volume")

    # Transpose [D, H, W] -> [H, W, D] for NIfTI
    atlas_nifti = np.transpose(atlas.astype(np.float32), (1, 2, 0))

    xy_spacing = args.fov_mm / args.image_size
    voxel_size = (xy_spacing, xy_spacing, 1.0)
    save_nifti(atlas_nifti, args.output, voxel_size=voxel_size)
    logger.info(f"Saved atlas to {args.output}")


if __name__ == '__main__':
    main()
