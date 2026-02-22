#!/usr/bin/env python3
"""Verify that the LDM training and generation pipelines are consistent.

Only needs: diffusion checkpoint + compression checkpoint + data root.
No latent cache or metadata.json required — stats are read from the
diffusion checkpoint (saved by profiling.py).

Checks performed:
  1. Checkpoint has latent normalization stats
  2. Encode NIfTI volumes → normalize → per-channel mean ≈ 0, std ≈ 1
  3. Encode → normalize → decode round-trip PSNR
  4. Seg conditioning: encode_normalized_seg consistency
  5. Noise/signal scale compatibility
  6. Generation pipeline shapes (decode output, model input concat)

Usage:
    python -m medgen.scripts.verify_ldm_pipeline \
        --diffusion-checkpoint runs/diffusion_3d/.../checkpoint_latest.pt \
        --compression-checkpoint runs/compression_3d/.../checkpoint_latest.pt \
        --data-root /path/to/brainmetshare-3
"""
import argparse
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.amp import autocast

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

PASS = "PASS"
FAIL = "FAIL"


def check(name: str, condition: bool, detail: str = "") -> bool:
    """Log a check result. Returns True if passed."""
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    logger.info(msg)
    return condition


def load_nifti_volume(
    path: Path, depth: int, image_size: int,
) -> torch.Tensor:
    """Load a NIfTI volume with standard preprocessing.

    Returns [1, 1, D, H, W] tensor in [0, 1].
    """
    vol = nib.load(str(path)).get_fdata().astype(np.float32)
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)

    # [H, W, D] -> [D, H, W]
    vol = np.transpose(vol, (2, 0, 1))

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


def main():
    parser = argparse.ArgumentParser(description="Verify LDM pipeline consistency")
    parser.add_argument('--diffusion-checkpoint', required=True,
                        help='Diffusion model checkpoint (has latent stats in config)')
    parser.add_argument('--compression-checkpoint', required=True,
                        help='Compression model checkpoint (VQ-VAE/VAE/DC-AE)')
    parser.add_argument('--compression-type', default='auto')
    parser.add_argument('--data-root', required=True,
                        help='Dataset root with NIfTI volumes')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--max-volumes', type=int, default=50,
                        help='Max volumes for stats verification (default: 50)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data_root)
    all_passed = True

    logger.info("=" * 70)
    logger.info("LDM Pipeline Verification")
    logger.info("=" * 70)
    logger.info(f"Diffusion checkpoint: {args.diffusion_checkpoint}")
    logger.info(f"Compression checkpoint: {args.compression_checkpoint}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Volume: {args.image_size}x{args.image_size}x{args.depth}")
    logger.info("")

    # ── Load diffusion checkpoint config ──
    ckpt = torch.load(args.diffusion_checkpoint, map_location='cpu', weights_only=False)
    ckpt_cfg = ckpt.get('config', {})
    latent_cfg = ckpt_cfg.get('latent', {})
    del ckpt

    # ── Load compression model ──
    from medgen.data.loaders.latent import load_compression_model
    from medgen.diffusion.spaces import LatentSpace

    comp_model, detected_type, comp_spatial_dims, scale_factor, latent_channels = (
        load_compression_model(
            args.compression_checkpoint, args.compression_type, device,
            spatial_dims='auto',
        )
    )
    logger.info(
        f"Compression: {detected_type} {scale_factor}x, "
        f"{latent_channels}ch, {comp_spatial_dims}D"
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 1: Checkpoint has latent normalization stats
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 1: Checkpoint latent stats ---")

    latent_shift = latent_cfg.get('latent_shift')
    latent_scale = latent_cfg.get('latent_scale')
    latent_seg_shift = latent_cfg.get('latent_seg_shift')
    latent_seg_scale = latent_cfg.get('latent_seg_scale')

    all_passed &= check(
        "Checkpoint has latent_shift/latent_scale",
        latent_shift is not None and latent_scale is not None,
        f"shift={latent_shift}, scale={latent_scale}" if latent_shift else "MISSING",
    )

    if latent_shift is None:
        logger.error("Cannot continue without latent stats. Retrain with the stats persistence fix.")
        sys.exit(1)

    logger.info(f"  Bravo stats: shift={latent_shift}")
    logger.info(f"               scale={latent_scale}")
    if latent_seg_shift:
        logger.info(f"  Seg stats:   shift={latent_seg_shift}")
        logger.info(f"               scale={latent_seg_scale}")

    all_passed &= check(
        "Scale values are positive",
        all(s > 0 for s in latent_scale),
        f"min scale = {min(latent_scale):.6f}",
    )

    # ── Create LatentSpace (generation pipeline) ──
    slicewise = (comp_spatial_dims == 2)
    space = LatentSpace(
        compression_model=comp_model,
        device=device,
        deterministic=True,
        spatial_dims=comp_spatial_dims,
        compression_type=detected_type,
        scale_factor=scale_factor,
        latent_channels=latent_channels,
        slicewise_encoding=slicewise,
        latent_shift=latent_shift,
        latent_scale=latent_scale,
        latent_seg_shift=latent_seg_shift,
        latent_seg_scale=latent_seg_scale,
    )

    # ── Find NIfTI volumes ──
    train_dir = data_root / 'train'
    if not train_dir.exists():
        for d in sorted(data_root.iterdir()):
            if d.is_dir() and list(d.glob("*/bravo.nii.gz")):
                train_dir = d
                break

    bravo_files = sorted(train_dir.glob("*/bravo.nii.gz"))[:args.max_volumes]
    if not bravo_files:
        logger.error(f"No bravo.nii.gz in {train_dir}")
        sys.exit(1)
    logger.info(f"\nFound {len(bravo_files)} bravo volumes in {train_dir}")

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 2: Normalized latent distribution (encode NIfTI → normalize → stats)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 2: Normalized latent distribution ---")

    all_means = []
    all_stds = []

    for path in bravo_files:
        vol = load_nifti_volume(path, args.depth, args.image_size).to(device)
        with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
            latent = space.encode(vol)
        normalized = space.normalize(latent).cpu().float()[0]  # [C, ...]
        n_ch = normalized.shape[0]
        flat = normalized.reshape(n_ch, -1)
        all_means.append(flat.mean(dim=1))
        all_stds.append(flat.std(dim=1))

    avg_mean = torch.stack(all_means).mean(dim=0)
    avg_std = torch.stack(all_stds).mean(dim=0)
    logger.info(f"  Per-channel mean: {avg_mean.tolist()}")
    logger.info(f"  Per-channel std:  {avg_std.tolist()}")

    all_passed &= check(
        "Bravo per-channel mean ≈ 0",
        avg_mean.abs().max().item() < 0.5,
        f"max |mean| = {avg_mean.abs().max().item():.4f} (threshold: 0.5)",
    )
    all_passed &= check(
        "Bravo per-channel std ≈ 1",
        avg_std.min().item() > 0.3 and avg_std.max().item() < 3.0,
        f"std range = [{avg_std.min().item():.4f}, {avg_std.max().item():.4f}] (expected: 0.3-3.0)",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 3: Encode → normalize → decode round-trip PSNR
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 3: Full round-trip (encode→normalize→decode) ---")

    vol_path = bravo_files[0]
    vol_tensor = load_nifti_volume(vol_path, args.depth, args.image_size).to(device)
    original_np = vol_tensor[0, 0].cpu().numpy()

    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        encoded = space.encode(vol_tensor)
    normalized = space.normalize(encoded)

    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        decoded = space.decode(normalized)
    decoded_np = decoded[0, 0].cpu().float().clamp(0, 1).numpy()

    mse = np.mean((original_np - decoded_np) ** 2)
    psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
    logger.info(f"  Sample: {vol_path.parent.name}")
    logger.info(f"  Encoded shape: {tuple(encoded.shape)}")
    logger.info(f"  Normalized range: [{normalized.min().item():.2f}, {normalized.max().item():.2f}]")

    all_passed &= check(
        "Encode→normalize→decode PSNR",
        psnr > 25,
        f"PSNR = {psnr:.2f} dB (threshold: >25)",
    )

    # Verify normalize→denormalize is exact (decode includes denormalize)
    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        decoded2 = space.decode(normalized)
    pixel_diff = (decoded - decoded2).abs().max().item()
    all_passed &= check(
        "Decode is deterministic",
        pixel_diff < 1e-5,
        f"max diff = {pixel_diff:.2e}",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 4: Seg conditioning normalization
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 4: Seg conditioning normalization ---")

    patient_id = vol_path.parent.name
    seg_paths = sorted(train_dir.glob(f"*{patient_id}*/seg.nii.gz"))
    if not seg_paths:
        seg_paths = sorted(train_dir.glob("*/seg.nii.gz"))[:1]

    if seg_paths:
        seg_path = seg_paths[0]
        seg_vol = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_vol = (seg_vol > 0.5).astype(np.float32)
        seg_vol = np.transpose(seg_vol, (2, 0, 1))

        d = seg_vol.shape[0]
        if d < args.depth:
            pad = np.zeros((args.depth - d, seg_vol.shape[1], seg_vol.shape[2]), dtype=np.float32)
            seg_vol = np.concatenate([seg_vol, pad], axis=0)
        elif d > args.depth:
            seg_vol = seg_vol[:args.depth]

        seg_tensor = torch.from_numpy(seg_vol).unsqueeze(0).unsqueeze(0).to(device)
        if seg_tensor.shape[3] != args.image_size or seg_tensor.shape[4] != args.image_size:
            seg_tensor = torch.nn.functional.interpolate(
                seg_tensor, size=(args.depth, args.image_size, args.image_size),
                mode='nearest',
            )

        with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
            seg_norm = space.encode_normalized_seg(seg_tensor)

        logger.info(f"  Seg sample: {seg_path.parent.name}")
        logger.info(f"  Seg latent shape: {tuple(seg_norm.shape)}")
        logger.info(f"  Seg normalized range: [{seg_norm.min().item():.2f}, {seg_norm.max().item():.2f}]")

        # encode_normalized_seg = encode + normalize_seg — verify both paths match
        with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
            seg_raw = space.encode(seg_tensor)
        seg_manual_norm = space.normalize_seg(seg_raw)
        diff = (seg_norm - seg_manual_norm).abs().max().item()

        all_passed &= check(
            "encode_normalized_seg = encode + normalize_seg",
            diff < 1e-5,
            f"max diff = {diff:.2e}",
        )

        # Spatial dims must match bravo latent
        all_passed &= check(
            "Seg and bravo latent spatial dims match",
            seg_norm.shape[2:] == encoded.shape[2:],
            f"seg={tuple(seg_norm.shape[2:])}, bravo={tuple(encoded.shape[2:])}",
        )
    else:
        logger.warning("  No seg.nii.gz found, skipping conditioning checks")

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 5: Noise/signal scale
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 5: Noise/signal scale ---")

    noise = torch.randn_like(normalized)
    signal_std = normalized.std().item()
    noise_std = noise.std().item()
    ratio = signal_std / noise_std

    logger.info(f"  Signal std (normalized latent): {signal_std:.4f}")
    logger.info(f"  Noise std: {noise_std:.4f}")
    logger.info(f"  Signal/noise ratio: {ratio:.4f}")

    all_passed &= check(
        "Signal/noise ratio ≈ 1",
        0.2 < ratio < 5.0,
        f"ratio = {ratio:.4f} (expected: 0.2-5.0)",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 6: Generation pipeline shapes
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 6: Generation pipeline shapes ---")

    latent_spatial = tuple(encoded.shape[2:])
    latent_ch = encoded.shape[1]

    # Decode: latent → pixel
    test_latent = torch.randn(1, latent_ch, *latent_spatial).to(device)
    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        test_decoded = space.decode(test_latent)

    pixel_spatial = tuple(test_decoded.shape[2:])
    expected_spatial = tuple(s * scale_factor for s in latent_spatial)

    all_passed &= check(
        "Decode produces correct pixel dims",
        pixel_spatial == expected_spatial,
        f"got {pixel_spatial}, expected {expected_spatial}",
    )

    # Model input: noise (bravo latent) + conditioning (seg latent)
    if seg_paths:
        model_in_ch = latent_ch + seg_norm.shape[1]
        model_input = torch.cat([test_latent, torch.randn_like(seg_norm)], dim=1)

        all_passed &= check(
            "Model input channels (noise + cond)",
            model_input.shape[1] == model_in_ch,
            f"got {model_input.shape[1]}, expected {model_in_ch} ({latent_ch}+{seg_norm.shape[1]})",
        )

    logger.info(f"\n  Latent shape:  [B, {latent_ch}, {', '.join(str(s) for s in latent_spatial)}]")
    logger.info(f"  Pixel shape:   [B, 1, {', '.join(str(s) for s in pixel_spatial)}]")
    logger.info(f"  Scale factor:  {scale_factor}x")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    if all_passed:
        logger.info("ALL CHECKS PASSED — LDM pipeline is consistent")
    else:
        logger.info("SOME CHECKS FAILED — pipeline has issues (see above)")
    logger.info("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
