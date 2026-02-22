#!/usr/bin/env python3
"""Compute per-channel pixel-space normalization stats for diffusion training.

Computes mean/std using Welford + law of total variance (same algorithm
as wavelet/latent stats). Output values can be hardcoded in Hydra config.

Usage:
    python -m medgen.scripts.compute_pixel_stats \
        --data-root /path/to/brainmetshare-3 \
        --modalities bravo seg

    # Specific resolution (if training resizes)
    python -m medgen.scripts.compute_pixel_stats \
        --data-root /path/to/brainmetshare-3 \
        --image-size 256 --depth 160 \
        --modalities bravo seg
"""
import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_volume(
    path: Path, depth: int, image_size: int, binarize: bool = False,
) -> torch.Tensor:
    """Load NIfTI → [1, 1, D, H, W] tensor in [0, 1]."""
    vol = nib.load(str(path)).get_fdata().astype(np.float32)
    if binarize:
        vol = (vol > 0.5).astype(np.float32)
    else:
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
        mode = 'nearest' if binarize else 'trilinear'
        vol_tensor = torch.nn.functional.interpolate(
            vol_tensor, size=(depth, image_size, image_size),
            mode=mode, **(dict(align_corners=False) if mode == 'trilinear' else {}),
        )
    return vol_tensor


def compute_stats(volumes: list[torch.Tensor]) -> tuple[list[float], list[float]]:
    """Welford + law of total variance → per-channel mean/std."""
    n = 0
    mean = None
    m2 = None
    var_sum = None

    for vol in volumes:
        flat = vol[0].reshape(vol.shape[1], -1).float()  # [C, N]
        sample_mean = flat.mean(dim=1)
        sample_var = flat.var(dim=1)

        n += 1
        if mean is None:
            mean = sample_mean.clone()
            m2 = torch.zeros_like(mean)
            var_sum = sample_var.clone()
        else:
            delta = sample_mean - mean
            mean += delta / n
            delta2 = sample_mean - mean
            m2 += delta * delta2
            var_sum += sample_var

    if mean is None or n < 2:
        raise ValueError("Not enough volumes")

    avg_within_var = var_sum / n
    between_var = m2 / (n - 1)
    total_var = avg_within_var + between_var
    std = torch.sqrt(total_var).clamp(min=1e-6)
    return mean.tolist(), std.tolist()


def main():
    parser = argparse.ArgumentParser(description="Compute pixel-space normalization stats")
    parser.add_argument('--data-root', required=True, help='Dataset root')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--modalities', nargs='+', default=['bravo', 'seg', 't1_pre', 't1_gd'],
                        help='Modalities to compute stats for')
    parser.add_argument('--max-volumes', type=int, default=200)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_dir = data_root / 'train'
    if not train_dir.exists():
        for d in sorted(data_root.iterdir()):
            if d.is_dir() and list(d.glob("*/bravo.nii.gz")):
                train_dir = d
                break

    logger.info(f"Data root: {data_root}")
    logger.info(f"Train dir: {train_dir}")
    logger.info(f"Resolution: {args.image_size}x{args.image_size}x{args.depth}")
    logger.info(f"Max volumes: {args.max_volumes}")
    logger.info("")

    all_stats = {}

    for modality in args.modalities:
        binarize = (modality == 'seg')
        files = sorted(train_dir.glob(f"*/{modality}.nii.gz"))[:args.max_volumes]
        if not files:
            logger.warning(f"  {modality}: no files found, skipping")
            continue

        logger.info(f"--- {modality} ({len(files)} volumes) ---")
        volumes = [load_volume(p, args.depth, args.image_size, binarize=binarize) for p in files]
        shift, scale = compute_stats(volumes)

        all_stats[modality] = {'shift': shift, 'scale': scale}

        for ch in range(len(shift)):
            logger.info(f"  ch{ch}: mean={shift[ch]:.6f}, std={scale[ch]:.6f}")

        # Also show what the data looks like after normalization
        normalized = [(v - shift[0]) / scale[0] for v in volumes[:50]]
        norm_means = [n[0].mean().item() for n in normalized]
        norm_stds = [n[0].std().item() for n in normalized]
        avg_norm_mean = sum(norm_means) / len(norm_means)
        avg_norm_std = sum(norm_stds) / len(norm_stds)
        logger.info(f"  After normalization: mean={avg_norm_mean:.4f}, std={avg_norm_std:.4f}")

        # Show range before/after
        raw_vals = torch.cat([v.flatten() for v in volumes[:50]])
        logger.info(f"  Raw range: [{raw_vals.min().item():.4f}, {raw_vals.max().item():.4f}]")
        norm_vals = (raw_vals - shift[0]) / scale[0]
        logger.info(f"  Normalized range: [{norm_vals.min().item():.4f}, {norm_vals.max().item():.4f}]")
        logger.info("")

    # Print config-ready output
    logger.info("=" * 60)
    logger.info("CONFIG VALUES (copy-paste into Hydra yaml):")
    logger.info("=" * 60)
    for modality, stats in all_stats.items():
        logger.info(f"# {modality} (N={len(sorted(train_dir.glob(f'*/{modality}.nii.gz'))[:args.max_volumes])} volumes, "
                     f"{args.image_size}x{args.image_size}x{args.depth})")
        logger.info(f"#   shift: {stats['shift']}")
        logger.info(f"#   scale: {stats['scale']}")
        logger.info("")

    # Also show [-1,1] rescaled stats for comparison
    logger.info("=" * 60)
    logger.info("COMPARISON: if training uses [-1,1] rescaling (2*x - 1):")
    logger.info("=" * 60)
    for modality, stats in all_stats.items():
        rescaled_shift = [2 * s - 1 for s in stats['shift']]
        rescaled_scale = [2 * s for s in stats['scale']]
        logger.info(f"# {modality} in [-1,1]:")
        logger.info(f"#   mean: {[f'{v:.6f}' for v in rescaled_shift]}")
        logger.info(f"#   std:  {[f'{v:.6f}' for v in rescaled_scale]}")
        logger.info("")


if __name__ == '__main__':
    main()
