#!/usr/bin/env python3
"""Measure the FID floor of a compression model (VAE/VQ-VAE/DC-AE).

Loads real NIfTI volumes, encodes them through the compression model,
decodes back to pixel space, and computes FID/KID/CMMD between originals
and reconstructions. This gives the theoretical best FID that any latent
diffusion model can achieve with this compression model.

Usage:
    # VQ-VAE reconstruction FID
    python -m medgen.scripts.eval_compression_fid \
        --compression-checkpoint runs/compression_3d/.../checkpoint_latest.pt \
        --compression-type vqvae \
        --data-root ~/MedicalDataSets/brainmetshare-3

    # Auto-detect compression type
    python -m medgen.scripts.eval_compression_fid \
        --compression-checkpoint runs/compression_3d/.../checkpoint_latest.pt \
        --data-root ~/MedicalDataSets/brainmetshare-3

    # Specify output directory
    python -m medgen.scripts.eval_compression_fid \
        --compression-checkpoint runs/compression_3d/.../checkpoint_latest.pt \
        --data-root ~/MedicalDataSets/brainmetshare-3 \
        --output-dir eval_compression_fid
"""
import argparse
import json
import logging
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
)
logger = logging.getLogger(__name__)

# Target volume dimensions (must match what diffusion models use)
DEFAULT_DEPTH = 160
DEFAULT_IMAGE_SIZE = 256
TRIM_SLICES = 4  # Padding slices to exclude from metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def discover_splits(data_root: Path, modality: str = 'bravo') -> dict[str, Path]:
    """Auto-discover dataset splits containing {modality}.nii.gz files."""
    splits = {}
    for subdir in sorted(data_root.iterdir()):
        if not subdir.is_dir():
            continue
        files = list(subdir.glob(f"*/{modality}.nii.gz"))
        if files:
            splits[subdir.name] = subdir
            logger.info(f"  Found split '{subdir.name}': {len(files)} volumes")
    if not splits:
        raise FileNotFoundError(f"No splits with {modality}.nii.gz found in {data_root}")
    return splits


def load_volume(path: Path, depth: int, image_size: int) -> np.ndarray:
    """Load a NIfTI volume, normalize to [0,1], transpose, and pad/crop.

    Returns:
        numpy array [D, H, W] in [0, 1].
    """
    vol = nib.load(str(path)).get_fdata().astype(np.float32)
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)

    # [H, W, D] -> [D, H, W]
    vol = np.transpose(vol, (2, 0, 1))

    # Pad or crop depth
    d = vol.shape[0]
    if d < depth:
        pad = np.zeros((depth - d, vol.shape[1], vol.shape[2]), dtype=np.float32)
        vol = np.concatenate([vol, pad], axis=0)
    elif d > depth:
        vol = vol[:depth]

    return vol


# ═══════════════════════════════════════════════════════════════════════════════
# Reconstruction
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def reconstruct_volume(
    model: nn.Module,
    vol_tensor: torch.Tensor,
    compression_type: str,
    device: torch.device,
) -> torch.Tensor:
    """Encode and decode a volume through the compression model.

    Args:
        model: Compression model (VAE/VQ-VAE/DC-AE).
        vol_tensor: [1, 1, D, H, W] tensor in [0, 1].
        compression_type: 'vae', 'vqvae', or 'dcae'.
        device: CUDA device.

    Returns:
        Reconstructed [1, 1, D, H, W] tensor clamped to [0, 1].
    """
    x = vol_tensor.to(device)

    with autocast('cuda', dtype=torch.bfloat16):
        if compression_type == 'vqvae':
            z = model.encode(x)
            recon = model.decode_stage_2_outputs(z)
        elif compression_type == 'vae':
            z_mu, z_sigma = model.encode(x)
            # Use mean (deterministic) for reconstruction quality measurement
            recon = model.decode(z_mu)
        elif compression_type == 'dcae':
            z = model.encode(x)
            recon = model.decode(z)
        else:
            raise ValueError(f"Unknown compression type: {compression_type}")

    return recon.float().clamp(0, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature extraction (reuse patterns from eval_ode_solvers)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features_from_volumes(
    volumes: list[np.ndarray],
    extractor: nn.Module,
    trim_slices: int,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Extract features from a list of [D, H, W] numpy volumes.

    Returns:
        Feature tensor [total_slices, feat_dim].
    """
    all_features = []
    for vol_np in volumes:
        if trim_slices > 0:
            vol_np = vol_np[:-trim_slices]
        # [D, H, W] -> [D, 1, H, W]
        slices_tensor = torch.from_numpy(vol_np).unsqueeze(1)

        for start in range(0, slices_tensor.shape[0], chunk_size):
            end = min(start + chunk_size, slices_tensor.shape[0])
            chunk = slices_tensor[start:end]
            features = extractor.extract_features(chunk)
            all_features.append(features.cpu())
            del features

    return torch.cat(all_features, dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Measure compression model reconstruction FID (encode→decode→FID vs originals).",
    )
    parser.add_argument('--compression-checkpoint', type=str, required=True,
                        help='Path to compression model checkpoint')
    parser.add_argument('--compression-type', type=str, default='auto',
                        choices=['auto', 'vae', 'vqvae', 'dcae'],
                        help='Compression model type (default: auto-detect)')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory of dataset (e.g., brainmetshare-3)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (default: auto)')
    parser.add_argument('--modality', type=str, default='bravo',
                        help='Modality to evaluate (default: bravo)')
    parser.add_argument('--depth', type=int, default=DEFAULT_DEPTH,
                        help=f'Volume depth (default: {DEFAULT_DEPTH})')
    parser.add_argument('--image-size', type=int, default=DEFAULT_IMAGE_SIZE,
                        help=f'Volume H/W (default: {DEFAULT_IMAGE_SIZE})')
    parser.add_argument('--max-volumes', type=int, default=None,
                        help='Max volumes per split (default: all)')
    parser.add_argument('--split', type=str, default=None,
                        help='Only evaluate this split (default: all splits)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data_root)
    comp_ckpt = args.compression_checkpoint

    # Auto output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ckpt_name = Path(comp_ckpt).parent.name
        output_dir = Path(f"eval_compression_fid_{ckpt_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Compression Model Reconstruction FID ===")
    logger.info(f"Checkpoint: {comp_ckpt}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Output: {output_dir}")

    # ── Load compression model ──
    from medgen.data.loaders.compression_detection import load_compression_model

    comp_model, detected_type, comp_spatial_dims, scale_factor, latent_channels = (
        load_compression_model(comp_ckpt, args.compression_type, device, spatial_dims='auto')
    )
    logger.info(f"Compression: {detected_type} (spatial_dims={comp_spatial_dims}, "
                f"scale={scale_factor}x, latent_ch={latent_channels})")

    # ── Discover splits ──
    splits = discover_splits(data_root, modality=args.modality)
    if args.split:
        if args.split not in splits:
            raise ValueError(f"Split '{args.split}' not found. Available: {list(splits.keys())}")
        splits = {args.split: splits[args.split]}

    # ── Process each split ──
    all_originals: list[np.ndarray] = []
    all_reconstructions: list[np.ndarray] = []
    per_split_results: dict[str, dict] = {}
    total_time = 0.0

    for split_name, split_dir in splits.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing split: {split_name}")
        logger.info(f"{'='*60}")

        vol_paths = sorted(split_dir.glob(f"*/{args.modality}.nii.gz"))
        if args.max_volumes:
            vol_paths = vol_paths[:args.max_volumes]
        logger.info(f"  Volumes to process: {len(vol_paths)}")

        split_originals: list[np.ndarray] = []
        split_recons: list[np.ndarray] = []
        split_psnrs: list[float] = []

        for i, path in enumerate(vol_paths):
            patient_id = path.parent.name

            # Load and normalize
            vol_np = load_volume(path, args.depth, args.image_size)
            vol_tensor = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]

            # Reconstruct
            t0 = time.time()
            recon_tensor = reconstruct_volume(comp_model, vol_tensor, detected_type, device)
            dt = time.time() - t0
            total_time += dt

            recon_np = recon_tensor[0, 0].cpu().numpy()  # [D, H, W]

            # Per-volume PSNR
            mse = np.mean((vol_np - recon_np) ** 2)
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
            split_psnrs.append(psnr)

            split_originals.append(vol_np)
            split_recons.append(recon_np)

            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"  [{i+1}/{len(vol_paths)}] {patient_id}: "
                            f"PSNR={psnr:.2f}dB ({dt:.1f}s)")

        avg_psnr = np.mean(split_psnrs)
        logger.info(f"  Split '{split_name}': avg PSNR = {avg_psnr:.2f}dB")

        per_split_results[split_name] = {
            'num_volumes': len(vol_paths),
            'avg_psnr': float(avg_psnr),
            'psnrs': [float(p) for p in split_psnrs],
        }
        all_originals.extend(split_originals)
        all_reconstructions.extend(split_recons)

    # ── Compute distributional metrics ──
    logger.info(f"\n{'='*60}")
    logger.info("Computing distributional metrics (FID, KID, CMMD)")
    logger.info(f"{'='*60}")
    logger.info(f"Total volumes: {len(all_originals)} originals, {len(all_reconstructions)} reconstructions")

    from medgen.metrics.feature_extractors import BiomedCLIPFeatures, ResNet50Features
    from medgen.metrics.generation import compute_cmmd, compute_fid, compute_kid

    trim = TRIM_SLICES

    # ResNet50 features
    logger.info("  Extracting ResNet50 features...")
    resnet = ResNet50Features(device, compile_model=False)
    orig_resnet = extract_features_from_volumes(all_originals, resnet, trim)
    recon_resnet = extract_features_from_volumes(all_reconstructions, resnet, trim)
    logger.info(f"    Originals: {orig_resnet.shape}, Reconstructions: {recon_resnet.shape}")
    resnet.unload()

    # BiomedCLIP features
    logger.info("  Extracting BiomedCLIP features...")
    clip = BiomedCLIPFeatures(device, compile_model=False)
    orig_clip = extract_features_from_volumes(all_originals, clip, trim)
    recon_clip = extract_features_from_volumes(all_reconstructions, clip, trim)
    logger.info(f"    Originals: {orig_clip.shape}, Reconstructions: {recon_clip.shape}")
    clip.unload()

    # Compute metrics
    fid = compute_fid(orig_resnet, recon_resnet)
    min_n = min(orig_resnet.shape[0], recon_resnet.shape[0])
    kid_mean, kid_std = compute_kid(orig_resnet, recon_resnet, subset_size=min(100, min_n))
    cmmd = compute_cmmd(orig_clip, recon_clip)

    logger.info(f"\n{'='*60}")
    logger.info("RESULTS: Compression Reconstruction Quality")
    logger.info(f"{'='*60}")
    logger.info(f"  Model:  {detected_type} ({scale_factor}x downscale, {latent_channels}ch)")
    logger.info(f"  Checkpoint: {comp_ckpt}")
    logger.info(f"  Volumes: {len(all_originals)}")
    logger.info(f"  Avg PSNR: {np.mean([r['avg_psnr'] for r in per_split_results.values()]):.2f} dB")
    logger.info(f"  FID:  {fid:.4f}")
    logger.info(f"  KID:  {kid_mean:.6f} ± {kid_std:.6f}")
    logger.info(f"  CMMD: {cmmd:.6f}")
    logger.info(f"  Time: {total_time:.1f}s ({total_time/len(all_originals):.2f}s/vol)")
    logger.info("")
    logger.info("  This FID is the FLOOR — no LDM using this compression model")
    logger.info("  can achieve a lower FID than this value.")

    # ── Save results ──
    results = {
        'compression_checkpoint': comp_ckpt,
        'compression_type': detected_type,
        'scale_factor': scale_factor,
        'latent_channels': latent_channels,
        'spatial_dims': comp_spatial_dims,
        'data_root': str(data_root),
        'modality': args.modality,
        'depth': args.depth,
        'image_size': args.image_size,
        'total_volumes': len(all_originals),
        'metrics': {
            'fid': float(fid),
            'kid_mean': float(kid_mean),
            'kid_std': float(kid_std),
            'cmmd': float(cmmd),
        },
        'per_split': per_split_results,
        'total_time_s': total_time,
    }

    results_path = output_dir / 'reconstruction_fid.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
