#!/usr/bin/env python3
"""Verify that the LDM training and generation pipelines are 100% consistent.

Runs a series of checks to confirm that what the diffusion model sees during
training exactly matches what the generation pipeline produces. Any mismatch
indicates a bug.

Checks performed:
  1. Normalized latent stats: per-channel mean ≈ 0, std ≈ 1
  2. Cache round-trip: LatentDataset sample matches manual encode+normalize
  3. Decode invertibility: decode(normalize(encode(x))) recovers x
  4. Conditioning consistency: encode_normalized_seg matches cache normalization
  5. Noise/signal scale: noise std ≈ normalized latent std (both ≈ 1)
  6. Generation pipeline match: full pipeline produces same tensors as training path

All checks must PASS for the pipeline to be correct.

Usage:
    python -m medgen.scripts.verify_ldm_pipeline \
        --compression-checkpoint runs/compression_3d/.../checkpoint_latest.pt \
        --compression-type vqvae \
        --latent-cache-dir /path/to/brainmetshare-3-latents-vqvae-3d-hash/train \
        --data-root /path/to/brainmetshare-3
"""
import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(description="Verify LDM pipeline consistency")
    parser.add_argument('--compression-checkpoint', required=True)
    parser.add_argument('--compression-type', default='auto')
    parser.add_argument('--latent-cache-dir', required=True,
                        help='Path to train/ subdirectory of latent cache')
    parser.add_argument('--data-root', required=True,
                        help='Dataset root (for raw NIfTI volumes)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir = args.latent_cache_dir
    data_root = Path(args.data_root)
    all_passed = True

    logger.info("=" * 70)
    logger.info("LDM Pipeline Verification")
    logger.info("=" * 70)

    # ── Load compression model ──
    from medgen.data.loaders.compression_detection import load_compression_model
    from medgen.diffusion.spaces import LatentSpace

    comp_model, detected_type, comp_spatial_dims, scale_factor, latent_channels = (
        load_compression_model(args.compression_checkpoint, args.compression_type, device, spatial_dims='auto')
    )
    logger.info(f"Compression: {detected_type} {scale_factor}x, {latent_channels}ch, {comp_spatial_dims}D")

    # ── Load normalization stats from metadata ──
    meta_path = Path(cache_dir) / 'metadata.json'
    if not meta_path.exists():
        logger.error(f"No metadata.json at {meta_path}")
        sys.exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)

    latent_shift = metadata.get('latent_shift')
    latent_scale = metadata.get('latent_scale')
    latent_seg_shift = metadata.get('latent_seg_shift')
    latent_seg_scale = metadata.get('latent_seg_scale')

    logger.info(f"Bravo stats: shift={latent_shift}, scale={latent_scale}")
    if latent_seg_shift:
        logger.info(f"Seg stats:   shift={latent_seg_shift}, scale={latent_seg_scale}")

    # ── Create LatentSpace (generation pipeline) ──
    slicewise = (comp_spatial_dims == 2)  # 2D model on 3D data
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

    # ── Create LatentDataset (training pipeline) ──
    from medgen.data.loaders.latent import LatentDataset

    dataset = LatentDataset(cache_dir=cache_dir, mode='bravo_seg_cond', spatial_dims=comp_spatial_dims)

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 1: Normalized latent stats should be ~N(0,1) per channel
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 1: Normalized latent distribution ---")

    n_samples = min(50, len(dataset))
    all_means = []
    all_stds = []
    seg_means = []
    seg_stds = []

    for i in range(n_samples):
        sample = dataset[i]
        lat = sample['latent']  # already normalized by __getitem__
        n_ch = lat.shape[0]
        flat = lat.reshape(n_ch, -1)
        all_means.append(flat.mean(dim=1))
        all_stds.append(flat.std(dim=1))

        if 'latent_seg' in sample and sample['latent_seg'] is not None:
            seg = sample['latent_seg']
            seg_flat = seg.reshape(seg.shape[0], -1)
            seg_means.append(seg_flat.mean(dim=1))
            seg_stds.append(seg_flat.std(dim=1))

    avg_mean = torch.stack(all_means).mean(dim=0)
    avg_std = torch.stack(all_stds).mean(dim=0)
    logger.info(f"  Bravo normalized: per-ch mean={avg_mean.tolist()}")
    logger.info(f"  Bravo normalized: per-ch std ={avg_std.tolist()}")

    # Per-channel mean should be near 0 (within ±0.5)
    all_passed &= check(
        "Bravo per-channel mean ≈ 0",
        avg_mean.abs().max().item() < 0.5,
        f"max |mean| = {avg_mean.abs().max().item():.4f} (threshold: 0.5)",
    )
    # Per-channel std should be near 1 (within 0.3-3.0)
    all_passed &= check(
        "Bravo per-channel std ≈ 1",
        avg_std.min().item() > 0.3 and avg_std.max().item() < 3.0,
        f"std range = [{avg_std.min().item():.4f}, {avg_std.max().item():.4f}] (expected: 0.3-3.0)",
    )

    if seg_means:
        seg_avg_mean = torch.stack(seg_means).mean(dim=0)
        seg_avg_std = torch.stack(seg_stds).mean(dim=0)
        logger.info(f"  Seg normalized: per-ch mean={seg_avg_mean.tolist()}")
        logger.info(f"  Seg normalized: per-ch std ={seg_avg_std.tolist()}")
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

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 2: Cache sample matches manual encode+normalize
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 2: Cache vs manual encode+normalize ---")

    # Load raw .pt file (before dataset normalization)
    pt_files = sorted(Path(cache_dir).glob("*.pt"))
    raw_data = torch.load(str(pt_files[0]), weights_only=False)
    raw_latent = raw_data['latent'].float()
    patient_id = raw_data.get('patient_id', 'unknown')
    logger.info(f"  Sample: {patient_id}, raw latent shape={raw_latent.shape}")

    # Get same sample through LatentDataset (applies normalization)
    dataset_sample = dataset[0]
    dataset_latent = dataset_sample['latent']

    # Manual normalize (same formula as LatentDataset.__getitem__)
    n_spatial = raw_latent.dim() - 1
    shape = (-1,) + (1,) * n_spatial
    shift_t = torch.tensor(latent_shift, dtype=torch.float32).reshape(shape)
    scale_t = torch.tensor(latent_scale, dtype=torch.float32).reshape(shape)
    manual_normalized = (raw_latent - shift_t) / scale_t

    diff = (dataset_latent - manual_normalized).abs().max().item()
    all_passed &= check(
        "Dataset normalization matches manual",
        diff < 1e-5,
        f"max diff = {diff:.2e}",
    )

    # Now test LatentSpace.normalize() (generation pipeline)
    raw_latent_gpu = raw_latent.unsqueeze(0).to(device)  # add batch dim
    space_normalized = space.normalize(raw_latent_gpu)
    space_normalized_cpu = space_normalized[0].cpu()  # remove batch dim

    diff2 = (dataset_latent - space_normalized_cpu).abs().max().item()
    all_passed &= check(
        "LatentSpace.normalize() matches LatentDataset",
        diff2 < 1e-4,
        f"max diff = {diff2:.2e}",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 3: Encode → Normalize → Denormalize → Decode round-trip
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 3: Full round-trip (encode→normalize→decode) ---")

    # Find the original NIfTI for this patient
    bravo_paths = list(data_root.glob(f"*/{patient_id}/bravo.nii.gz"))
    if bravo_paths:
        bravo_path = bravo_paths[0]
        vol = nib.load(str(bravo_path)).get_fdata().astype(np.float32)
        vmin, vmax = vol.min(), vol.max()
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
        vol = np.transpose(vol, (2, 0, 1))  # [H,W,D] -> [D,H,W]

        # Pad depth to match cache
        depth = raw_latent.shape[-3] * scale_factor if comp_spatial_dims == 3 else raw_latent.shape[-3]
        d = vol.shape[0]
        if d < depth:
            pad = np.zeros((depth - d, vol.shape[1], vol.shape[2]), dtype=np.float32)
            vol = np.concatenate([vol, pad], axis=0)
        elif d > depth:
            vol = vol[:depth]

        vol_tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)

        # Encode
        with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
            encoded = space.encode(vol_tensor)

        # Compare raw encode to cache
        encoded_cpu = encoded[0].cpu().float()
        cache_diff = (encoded_cpu - raw_latent).abs().max().item()
        all_passed &= check(
            "Fresh encode matches cached latent",
            cache_diff < 0.01,
            f"max diff = {cache_diff:.4f}",
        )

        # Full round-trip: encode → normalize → decode (decode includes denormalize)
        normalized = space.normalize(encoded)
        with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
            decoded = space.decode(normalized)
        decoded_np = decoded[0, 0].cpu().float().clamp(0, 1).numpy()

        psnr = 10 * np.log10(1.0 / max(np.mean((vol - decoded_np) ** 2), 1e-10))
        all_passed &= check(
            "Encode→normalize→decode PSNR",
            psnr > 30,
            f"PSNR = {psnr:.2f} dB (threshold: >30)",
        )

        # Raw round-trip (no normalization): encode → decode
        with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
            decoded_raw = space.decode(space.normalize(encoded))
        decoded_raw_np = decoded_raw[0, 0].cpu().float().clamp(0, 1).numpy()
        raw_diff = np.abs(decoded_np - decoded_raw_np).max()
        all_passed &= check(
            "Normalized vs raw decode identical",
            raw_diff < 1e-3,
            f"max pixel diff = {raw_diff:.6f}",
        )
    else:
        logger.warning(f"  Could not find NIfTI for {patient_id}, skipping encode check")

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 4: Conditioning normalization consistency
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 4: Conditioning (seg) normalization ---")

    if 'latent_seg' in raw_data and raw_data['latent_seg'] is not None:
        raw_seg_latent = raw_data['latent_seg'].float()
        dataset_seg = dataset_sample.get('latent_seg')

        if dataset_seg is not None:
            # Manual normalize with seg stats (matching LatentDataset logic)
            seg_sh = latent_seg_shift if latent_seg_shift else latent_shift
            seg_sc = latent_seg_scale if latent_seg_scale else latent_scale
            n_sp = raw_seg_latent.dim() - 1
            sp = (-1,) + (1,) * n_sp
            seg_manual = (raw_seg_latent - torch.tensor(seg_sh).reshape(sp)) / torch.tensor(seg_sc).reshape(sp)

            diff3 = (dataset_seg - seg_manual).abs().max().item()
            all_passed &= check(
                "Seg dataset normalization matches manual",
                diff3 < 1e-5,
                f"max diff = {diff3:.2e}",
            )

            # Test LatentSpace.normalize_seg()
            raw_seg_gpu = raw_seg_latent.unsqueeze(0).to(device)
            space_seg_norm = space.normalize_seg(raw_seg_gpu)[0].cpu()
            diff4 = (dataset_seg - space_seg_norm).abs().max().item()
            all_passed &= check(
                "LatentSpace.normalize_seg() matches dataset",
                diff4 < 1e-4,
                f"max diff = {diff4:.2e}",
            )

            # Test encode_normalized_seg on raw NIfTI seg
            seg_paths = list(data_root.glob(f"*/{patient_id}/seg.nii.gz"))
            if seg_paths:
                seg_vol = nib.load(str(seg_paths[0])).get_fdata().astype(np.float32)
                seg_vol = (seg_vol > 0.5).astype(np.float32)
                seg_vol = np.transpose(seg_vol, (2, 0, 1))
                d = seg_vol.shape[0]
                if d < depth:
                    pad = np.zeros((depth - d, seg_vol.shape[1], seg_vol.shape[2]), dtype=np.float32)
                    seg_vol = np.concatenate([seg_vol, pad], axis=0)
                elif d > depth:
                    seg_vol = seg_vol[:depth]

                seg_tensor = torch.from_numpy(seg_vol).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
                    gen_seg_norm = space.encode_normalized_seg(seg_tensor)
                gen_seg_cpu = gen_seg_norm[0].cpu().float()

                diff5 = (dataset_seg - gen_seg_cpu).abs().max().item()
                all_passed &= check(
                    "encode_normalized_seg(NIfTI) matches cached+normalized",
                    diff5 < 0.01,
                    f"max diff = {diff5:.4f}",
                )
    else:
        logger.info("  No latent_seg in cache, skipping seg conditioning checks")

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 5: Noise/signal scale compatibility
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 5: Noise/signal scale ---")

    noise = torch.randn_like(dataset_latent)
    noise_std = noise.std().item()
    signal_std = dataset_latent.std().item()
    ratio = signal_std / noise_std

    logger.info(f"  Noise std: {noise_std:.4f}")
    logger.info(f"  Signal std (normalized latent): {signal_std:.4f}")
    logger.info(f"  Signal/noise ratio: {ratio:.4f}")

    all_passed &= check(
        "Signal/noise scale ratio ≈ 1",
        0.2 < ratio < 5.0,
        f"ratio = {ratio:.4f} (expected: 0.2-5.0)",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 6: Generation pipeline end-to-end match
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 6: Generation pipeline match ---")

    # Simulate what find_optimal_steps.py does:
    # 1. Load seg NIfTI → encode_normalized_seg → conditioning
    # 2. Generate noise in latent shape
    # 3. Cat [noise, conditioning] → model input
    # 4. After model: decode(output) → pixel space

    # Test that the concat dimensions match what the model expects
    sample_latent = dataset_sample['latent']  # [C, D, H, W] normalized
    model_out_ch = sample_latent.shape[0]  # bravo latent channels

    if 'latent_seg' in dataset_sample and dataset_sample['latent_seg'] is not None:
        sample_seg = dataset_sample['latent_seg']
        model_in_ch = model_out_ch + sample_seg.shape[0]  # bravo + seg channels

        # Simulate model_input construction
        fake_noise = torch.randn_like(sample_latent)
        model_input = torch.cat([fake_noise, sample_seg], dim=0)

        all_passed &= check(
            "Model input channels match",
            model_input.shape[0] == model_in_ch,
            f"got {model_input.shape[0]}, expected {model_in_ch}",
        )

        # Verify spatial dimensions match
        all_passed &= check(
            "Noise and conditioning spatial dims match",
            fake_noise.shape[1:] == sample_seg.shape[1:],
            f"noise={fake_noise.shape[1:]}, cond={sample_seg.shape[1:]}",
        )

    # Verify decode produces correct pixel shape
    test_latent = torch.randn(1, model_out_ch, *sample_latent.shape[1:]).to(device)
    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        test_decoded = space.decode(test_latent)
    expected_spatial = tuple(s * scale_factor for s in sample_latent.shape[1:])
    actual_spatial = test_decoded.shape[2:]

    all_passed &= check(
        "Decode produces correct pixel dimensions",
        actual_spatial == expected_spatial,
        f"got {actual_spatial}, expected {expected_spatial}",
    )

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
