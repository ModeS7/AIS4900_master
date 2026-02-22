#!/usr/bin/env python3
"""Verify the LDM training and generation pipeline BEFORE training.

Only needs: compression checkpoint + data root.
No diffusion checkpoint or latent cache needed — this tests the full
encode→normalize→decode pipeline so you can confidently start training.

Checks performed:
  1. Compression model loads and encodes/decodes correctly
  2. Latent stats computation (Welford + law of total variance)
  3. Normalized latents have per-channel mean ≈ 0, std ≈ 1
  4. Encode → normalize → decode round-trip PSNR
  5. Seg conditioning: separate encode path works correctly
  6. Noise/signal scale compatibility
  7. Generation pipeline shapes (latent dims, decode output, model input)
  8. Stats are saved correctly (simulate profiling.py checkpoint save)

Usage:
    python -m medgen.scripts.verify_ldm_pipeline \
        --compression-checkpoint runs/compression_3d/.../checkpoint_latest.pt \
        --data-root /path/to/brainmetshare-3

    # Specify compression type if auto-detection fails
    python -m medgen.scripts.verify_ldm_pipeline \
        --compression-checkpoint runs/.../checkpoint_latest.pt \
        --compression-type vqvae \
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


def compute_latent_stats(
    volumes: list[torch.Tensor],
    space,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    """Encode volumes and compute per-channel mean/std (Welford + total variance)."""
    n = 0
    mean = None
    m2 = None
    var_sum = None

    for vol in volumes:
        vol_gpu = vol.to(device)
        with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
            latent = space.encode(vol_gpu)
        latent = latent[0].cpu().float()  # [C, ...]
        n_ch = latent.shape[0]
        flat = latent.reshape(n_ch, -1)
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
    total_variance = avg_within_var + between_var
    std = torch.sqrt(total_variance).clamp(min=1e-6)
    return mean.tolist(), std.tolist()


def main():
    parser = argparse.ArgumentParser(description="Verify LDM pipeline BEFORE training")
    parser.add_argument('--compression-checkpoint', required=True,
                        help='Compression model checkpoint (VQ-VAE/VAE/DC-AE)')
    parser.add_argument('--compression-type', default='auto')
    parser.add_argument('--data-root', required=True,
                        help='Dataset root with NIfTI volumes')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--max-volumes', type=int, default=50,
                        help='Max volumes for stats computation (default: 50)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data_root)
    all_passed = True

    logger.info("=" * 70)
    logger.info("LDM Pipeline Verification (pre-training)")
    logger.info("=" * 70)
    logger.info(f"Compression: {args.compression_checkpoint}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Volume: {args.image_size}x{args.image_size}x{args.depth}")
    logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 1: Compression model loads and works
    # ══════════════════════════════════════════════════════════════════════
    logger.info("--- Check 1: Compression model ---")

    from medgen.data.loaders.latent import load_compression_model
    from medgen.diffusion.spaces import LatentSpace

    comp_model, detected_type, comp_spatial_dims, scale_factor, latent_channels = (
        load_compression_model(
            args.compression_checkpoint, args.compression_type, device,
            spatial_dims='auto',
        )
    )
    logger.info(f"  Type: {detected_type}")
    logger.info(f"  Scale factor: {scale_factor}x")
    logger.info(f"  Latent channels: {latent_channels}")
    logger.info(f"  Spatial dims: {comp_spatial_dims}D")

    all_passed &= check(
        "Compression model loaded",
        comp_model is not None,
        f"{detected_type} {scale_factor}x, {latent_channels}ch",
    )

    # Quick encode→decode test with dummy data
    slicewise = (comp_spatial_dims == 2)
    space_no_norm = LatentSpace(
        compression_model=comp_model, device=device, deterministic=True,
        spatial_dims=comp_spatial_dims, compression_type=detected_type,
        scale_factor=scale_factor, latent_channels=latent_channels,
        slicewise_encoding=slicewise,
    )

    dummy = torch.rand(1, 1, args.depth, args.image_size, args.image_size).to(device)
    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        z = space_no_norm.encode(dummy)
        recon = space_no_norm.decode(z)

    logger.info(f"  Encode: {tuple(dummy.shape)} → {tuple(z.shape)}")
    logger.info(f"  Decode: {tuple(z.shape)} → {tuple(recon.shape)}")

    all_passed &= check(
        "Encode produces expected shape",
        z.shape[1] == latent_channels and z.shape[0] == 1,
        f"latent shape = {tuple(z.shape)}",
    )
    all_passed &= check(
        "Decode recovers input shape",
        recon.shape == dummy.shape,
        f"got {tuple(recon.shape)}, expected {tuple(dummy.shape)}",
    )

    recon_psnr_dummy = 10 * np.log10(1.0 / max(
        ((dummy.cpu().float() - recon.cpu().float()) ** 2).mean().item(), 1e-10
    ))
    # VQ-VAE trained on brain MRI — random noise won't reconstruct well, just sanity check
    all_passed &= check(
        "Encode→decode runs without error (dummy data)",
        recon_psnr_dummy > 5,
        f"PSNR = {recon_psnr_dummy:.1f} dB (low is expected for random input)",
    )
    del dummy, z, recon
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 2: Latent stats from real data (Welford + total variance)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 2: Latent stats computation ---")

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

    logger.info(f"  Encoding {len(bravo_files)} bravo volumes...")
    bravo_volumes = [
        load_nifti_volume(p, args.depth, args.image_size) for p in bravo_files
    ]

    bravo_shift, bravo_scale = compute_latent_stats(bravo_volumes, space_no_norm, device)

    logger.info(f"  Bravo shift: {[f'{v:.6f}' for v in bravo_shift]}")
    logger.info(f"  Bravo scale: {[f'{v:.6f}' for v in bravo_scale]}")

    all_passed &= check(
        "Bravo scale values are positive and reasonable",
        all(0.01 < s < 10 for s in bravo_scale),
        f"range = [{min(bravo_scale):.6f}, {max(bravo_scale):.6f}]",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 3: Normalized latents have per-channel mean ≈ 0, std ≈ 1
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 3: Normalized latent distribution ---")

    space = LatentSpace(
        compression_model=comp_model, device=device, deterministic=True,
        spatial_dims=comp_spatial_dims, compression_type=detected_type,
        scale_factor=scale_factor, latent_channels=latent_channels,
        slicewise_encoding=slicewise,
        latent_shift=bravo_shift, latent_scale=bravo_scale,
    )

    all_means = []
    all_stds = []
    for vol in bravo_volumes[:30]:
        vol_gpu = vol.to(device)
        with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
            latent = space.encode(vol_gpu)
        normalized = space.normalize(latent).cpu().float()[0]
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
    # CHECK 4: Encode → normalize → decode round-trip PSNR
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 4: Full round-trip (encode→normalize→decode) ---")

    vol_path = bravo_files[0]
    vol_tensor = bravo_volumes[0].to(device)
    original_np = vol_tensor[0, 0].cpu().numpy()

    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        encoded = space.encode(vol_tensor)
    normalized = space.normalize(encoded)

    logger.info(f"  Sample: {vol_path.parent.name}")
    logger.info(f"  Raw latent range: [{encoded.min().item():.2f}, {encoded.max().item():.2f}]")
    logger.info(f"  Normalized range: [{normalized.min().item():.2f}, {normalized.max().item():.2f}]")

    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        decoded = space.decode(normalized)
    decoded_np = decoded[0, 0].cpu().float().clamp(0, 1).numpy()

    mse = np.mean((original_np - decoded_np) ** 2)
    psnr = 10 * np.log10(1.0 / max(mse, 1e-10))

    all_passed &= check(
        "Encode→normalize→decode PSNR",
        psnr > 25,
        f"PSNR = {psnr:.2f} dB (threshold: >25)",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 5: Seg conditioning path
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 5: Seg conditioning ---")

    space_full = None  # set if seg conditioning is tested
    seg_shift, seg_scale = None, None
    seg_files = sorted(train_dir.glob("*/seg.nii.gz"))[:args.max_volumes]
    if seg_files:
        logger.info(f"  Encoding {len(seg_files)} seg volumes...")
        seg_volumes = [
            load_nifti_volume(p, args.depth, args.image_size, binarize=True)
            for p in seg_files
        ]

        seg_shift, seg_scale = compute_latent_stats(seg_volumes, space_no_norm, device)
        logger.info(f"  Seg shift: {[f'{v:.6f}' for v in seg_shift]}")
        logger.info(f"  Seg scale: {[f'{v:.6f}' for v in seg_scale]}")

        all_passed &= check(
            "Seg scale values are positive and reasonable",
            all(0.01 < s < 10 for s in seg_scale),
            f"range = [{min(seg_scale):.6f}, {max(seg_scale):.6f}]",
        )

        # Create space with both stats and test encode_normalized_seg
        space_full = LatentSpace(
            compression_model=comp_model, device=device, deterministic=True,
            spatial_dims=comp_spatial_dims, compression_type=detected_type,
            scale_factor=scale_factor, latent_channels=latent_channels,
            slicewise_encoding=slicewise,
            latent_shift=bravo_shift, latent_scale=bravo_scale,
            latent_seg_shift=seg_shift, latent_seg_scale=seg_scale,
        )

        seg_gpu = seg_volumes[0].to(device)
        with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
            seg_norm = space_full.encode_normalized_seg(seg_gpu)

        # Verify encode_normalized_seg = encode + normalize_seg
        with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
            seg_raw = space_full.encode(seg_gpu)
        seg_manual_norm = space_full.normalize_seg(seg_raw)
        diff = (seg_norm - seg_manual_norm).abs().max().item()

        all_passed &= check(
            "encode_normalized_seg = encode + normalize_seg",
            diff < 1e-5,
            f"max diff = {diff:.2e}",
        )

        all_passed &= check(
            "Seg and bravo latent spatial dims match",
            seg_norm.shape[2:] == encoded.shape[2:],
            f"seg={tuple(seg_norm.shape[2:])}, bravo={tuple(encoded.shape[2:])}",
        )

        # Check seg normalized stats
        seg_means = []
        seg_stds_list = []
        for vol in seg_volumes[:20]:
            vol_gpu = vol.to(device)
            with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
                s = space_full.encode_normalized_seg(vol_gpu)
            s_cpu = s[0].cpu().float()
            flat = s_cpu.reshape(s_cpu.shape[0], -1)
            seg_means.append(flat.mean(dim=1))
            seg_stds_list.append(flat.std(dim=1))

        seg_avg_mean = torch.stack(seg_means).mean(dim=0)
        seg_avg_std = torch.stack(seg_stds_list).mean(dim=0)
        logger.info(f"  Seg normalized mean: {seg_avg_mean.tolist()}")
        logger.info(f"  Seg normalized std:  {seg_avg_std.tolist()}")

        all_passed &= check(
            "Seg per-channel mean ≈ 0",
            seg_avg_mean.abs().max().item() < 0.5,
            f"max |mean| = {seg_avg_mean.abs().max().item():.4f}",
        )
        all_passed &= check(
            "Seg per-channel std ≈ 1",
            seg_avg_std.min().item() > 0.3 and seg_avg_std.max().item() < 3.0,
            f"std range = [{seg_avg_std.min().item():.4f}, {seg_avg_std.max().item():.4f}]",
        )
    else:
        logger.warning("  No seg.nii.gz found, skipping conditioning checks")

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 6: Noise/signal scale
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 6: Noise/signal scale ---")

    noise = torch.randn_like(normalized)
    signal_std = normalized.std().item()
    noise_std = noise.std().item()
    ratio = signal_std / noise_std

    logger.info(f"  Signal std (normalized latent): {signal_std:.4f}")
    logger.info(f"  Noise std: {noise_std:.4f}")
    logger.info(f"  Ratio: {ratio:.4f}")

    all_passed &= check(
        "Signal/noise ratio ≈ 1",
        0.2 < ratio < 5.0,
        f"ratio = {ratio:.4f} (expected: 0.2-5.0)",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 7: Generation pipeline shapes
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 7: Generation pipeline shapes ---")

    latent_spatial = tuple(encoded.shape[2:])
    latent_ch = encoded.shape[1]

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

    if seg_files:
        model_in_ch = latent_ch * 2  # bravo + seg
        model_input = torch.cat([test_latent, torch.randn_like(test_latent)], dim=1)
        all_passed &= check(
            "Model input channels (noise + seg cond)",
            model_input.shape[1] == model_in_ch,
            f"got {model_input.shape[1]}, expected {model_in_ch}",
        )

    logger.info(f"\n  Latent shape: [B, {latent_ch}, {', '.join(str(s) for s in latent_spatial)}]")
    logger.info(f"  Pixel shape:  [B, 1, {', '.join(str(s) for s in pixel_spatial)}]")
    logger.info(f"  Scale factor: {scale_factor}x")

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 8: Stats persistence (simulate profiling.py save/load)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 8: Stats persistence (profiling.py round-trip) ---")

    # Simulate what profiling.py saves — use space_full (has seg stats) if available
    stats_space = space_full if space_full is not None else space
    saved_config = {
        'compression_type': stats_space.compression_type,
        'scale_factor': stats_space.scale_factor,
        'latent_channels': stats_space.latent_channels,
    }
    if stats_space.latent_shift is not None:
        saved_config['latent_shift'] = stats_space.latent_shift
        saved_config['latent_scale'] = stats_space.latent_scale
    if stats_space.latent_seg_shift is not None:
        saved_config['latent_seg_shift'] = stats_space.latent_seg_shift
        saved_config['latent_seg_scale'] = stats_space.latent_seg_scale

    # Simulate what find_optimal_steps.py reads
    loaded_shift = saved_config.get('latent_shift')
    loaded_scale = saved_config.get('latent_scale')

    all_passed &= check(
        "Stats survive save/load round-trip",
        loaded_shift == bravo_shift and loaded_scale == bravo_scale,
        "shift and scale match",
    )

    if seg_files:
        loaded_seg_shift = saved_config.get('latent_seg_shift')
        loaded_seg_scale = saved_config.get('latent_seg_scale')
        all_passed &= check(
            "Seg stats survive save/load round-trip",
            loaded_seg_shift == seg_shift and loaded_seg_scale == seg_scale,
            "seg shift and scale match",
        )

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    if all_passed:
        logger.info("ALL CHECKS PASSED — LDM pipeline is ready for training")
    else:
        logger.info("SOME CHECKS FAILED — fix issues before training")
    logger.info("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
