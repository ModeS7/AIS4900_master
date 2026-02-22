#!/usr/bin/env python3
"""Verify that the WDM training and generation pipelines are 100% consistent.

Runs a series of checks to confirm that:
  - Haar wavelet transform is lossless (forward→inverse = identity)
  - Per-subband normalization produces mean≈0, std≈1
  - Checkpoint-saved wavelet stats match recomputed stats
  - Noise and signal scales are compatible
  - Full encode→decode round-trip preserves data
  - Generation pipeline shapes are correct

All checks must PASS for the pipeline to be correct.

Usage:
    # Verify with checkpoint (reads saved wavelet stats)
    python -m medgen.scripts.verify_wdm_pipeline \
        --checkpoint runs/diffusion_3d/.../checkpoint_latest.pt \
        --data-root /path/to/brainmetshare-3

    # Verify without checkpoint (computes stats from data only)
    python -m medgen.scripts.verify_wdm_pipeline \
        --data-root /path/to/brainmetshare-3
"""
import argparse
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


PASS = "PASS"
FAIL = "FAIL"
SUBBAND_NAMES = ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']


def check(name: str, condition: bool, detail: str = "") -> bool:
    """Log a check result. Returns True if passed."""
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    logger.info(msg)
    return condition


def load_bravo_volumes(
    data_root: Path,
    depth: int,
    image_size: int,
    max_volumes: int = 200,
    rescale: bool = False,
) -> list[torch.Tensor]:
    """Load BRAVO NIfTI volumes with the same preprocessing as training.

    Preprocessing chain (matches volume_3d.py):
      1. Load NIfTI → [H, W, D]
      2. Min-max normalize to [0, 1]
      3. Transpose to [D, H, W]
      4. Pad/crop depth to target
      5. Resize H/W if needed
      6. Optionally rescale [0,1] → [-1,1]

    Returns list of [1, 1, D, H, W] tensors.
    """
    train_dir = data_root / 'train'
    if not train_dir.exists():
        for d in sorted(data_root.iterdir()):
            if d.is_dir() and list(d.glob("*/bravo.nii.gz")):
                train_dir = d
                break

    bravo_files = sorted(train_dir.glob("*/bravo.nii.gz"))[:max_volumes]
    if not bravo_files:
        raise FileNotFoundError(f"No bravo.nii.gz files in {train_dir}")

    volumes = []
    for path in bravo_files:
        vol = nib.load(str(path)).get_fdata().astype(np.float32)
        vmin, vmax = vol.min(), vol.max()
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)

        # [H, W, D] -> [D, H, W]
        vol = np.transpose(vol, (2, 0, 1))

        # Pad/crop depth
        d = vol.shape[0]
        if d < depth:
            pad = np.zeros((depth - d, vol.shape[1], vol.shape[2]), dtype=np.float32)
            vol = np.concatenate([vol, pad], axis=0)
        elif d > depth:
            vol = vol[:depth]

        # Resize H/W if needed
        vol_tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        if vol_tensor.shape[3] != image_size or vol_tensor.shape[4] != image_size:
            vol_tensor = torch.nn.functional.interpolate(
                vol_tensor, size=(depth, image_size, image_size),
                mode='trilinear', align_corners=False,
            )

        if rescale:
            vol_tensor = 2.0 * vol_tensor - 1.0

        volumes.append(vol_tensor)

    return volumes


def compute_wavelet_stats_from_volumes(
    volumes: list[torch.Tensor],
) -> tuple[list[float], list[float]]:
    """Compute per-subband mean/std using law of total variance.

    Same algorithm as WaveletSpace.compute_subband_stats() and _welford_stats().
    """
    from medgen.models.haar_wavelet_3d import haar_forward_3d

    n = 0
    mean = None
    m2 = None
    var_sum = None

    for vol_tensor in volumes:
        coeffs = haar_forward_3d(vol_tensor)  # [1, 8, D/2, H/2, W/2]
        n_ch = coeffs.shape[1]
        sample = coeffs[0]  # [8, D/2, H/2, W/2]
        flat = sample.reshape(n_ch, -1).float()
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
        raise ValueError("Not enough volumes to compute stats")

    avg_within_var = var_sum / n
    between_var = m2 / (n - 1)
    total_variance = avg_within_var + between_var
    std = torch.sqrt(total_variance).clamp(min=1e-6)

    return mean.tolist(), std.tolist()


def main():
    parser = argparse.ArgumentParser(description="Verify WDM pipeline consistency")
    parser.add_argument('--checkpoint', default=None,
                        help='Diffusion model checkpoint (to verify saved wavelet stats)')
    parser.add_argument('--data-root', required=True,
                        help='Dataset root (for raw NIfTI volumes)')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--rescale', action='store_true',
                        help='Rescale [0,1] → [-1,1] before DWT (must match training)')
    parser.add_argument('--max-volumes', type=int, default=100,
                        help='Max volumes to load for stats computation (default: 100)')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    all_passed = True

    logger.info("=" * 70)
    logger.info("WDM Pipeline Verification")
    logger.info("=" * 70)
    logger.info(f"Data root: {data_root}")
    logger.info(f"Volume: {args.image_size}x{args.image_size}x{args.depth}")
    logger.info(f"Rescale: {args.rescale}")
    if args.checkpoint:
        logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info("")

    from medgen.diffusion.spaces import WaveletSpace
    from medgen.models.haar_wavelet_3d import haar_forward_3d, haar_inverse_3d

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 1: Haar wavelet transform is lossless
    # ══════════════════════════════════════════════════════════════════════
    logger.info("--- Check 1: Haar wavelet losslessness ---")

    # Test with random data
    test_vol = torch.randn(2, 1, 8, 16, 16)  # batch=2, small volume
    fwd = haar_forward_3d(test_vol)
    inv = haar_inverse_3d(fwd)
    max_err = (test_vol - inv).abs().max().item()

    all_passed &= check(
        "Haar forward→inverse = identity (random)",
        max_err < 1e-5,
        f"max error = {max_err:.2e}",
    )

    # Verify shape
    all_passed &= check(
        "Haar forward shape",
        fwd.shape == (2, 8, 4, 8, 8),
        f"got {tuple(fwd.shape)}, expected (2, 8, 4, 8, 8)",
    )

    # Energy preservation (orthogonality): ||forward(x)||^2 == ||x||^2
    energy_in = (test_vol ** 2).sum().item()
    energy_out = (fwd ** 2).sum().item()
    energy_ratio = energy_out / energy_in
    all_passed &= check(
        "Haar energy preservation",
        abs(energy_ratio - 1.0) < 1e-5,
        f"ratio = {energy_ratio:.6f} (expected: 1.0)",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 2: WaveletSpace encode/decode round-trip (no normalization)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 2: WaveletSpace encode→decode round-trip ---")

    space_raw = WaveletSpace(rescale=args.rescale)
    test_pixel = torch.rand(1, 1, 8, 16, 16)  # [0, 1] range
    encoded = space_raw.encode(test_pixel)
    decoded = space_raw.decode(encoded)
    rt_err = (test_pixel - decoded).abs().max().item()

    all_passed &= check(
        "Raw WaveletSpace round-trip",
        rt_err < 1e-5,
        f"max error = {rt_err:.2e}",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 3: Compute wavelet stats from data, verify normalization
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 3: Wavelet stats from data ---")

    logger.info(f"Loading up to {args.max_volumes} BRAVO volumes...")
    volumes = load_bravo_volumes(
        data_root, args.depth, args.image_size,
        max_volumes=args.max_volumes, rescale=args.rescale,
    )
    logger.info(f"  Loaded {len(volumes)} volumes")

    shift, scale = compute_wavelet_stats_from_volumes(volumes)

    logger.info("  Per-subband stats:")
    for i, name in enumerate(SUBBAND_NAMES):
        logger.info(f"    {name}: shift={shift[i]:.6f}, scale={scale[i]:.6f}")

    # Verify stats are reasonable
    all_passed &= check(
        "Scale values are positive and finite",
        all(0 < s < 100 for s in scale),
        f"scale range = [{min(scale):.6f}, {max(scale):.6f}]",
    )

    # Create normalized WaveletSpace and verify normalization
    space_norm = WaveletSpace(shift=shift, scale=scale, rescale=args.rescale)

    # Apply to all volumes and check per-subband stats
    all_ch_means = []
    all_ch_stds = []
    for vol in volumes[:50]:  # Use subset for quick check
        coeffs = space_norm.encode(vol)  # normalized wavelet coefficients
        n_ch = coeffs.shape[1]
        flat = coeffs[0].reshape(n_ch, -1)
        all_ch_means.append(flat.mean(dim=1))
        all_ch_stds.append(flat.std(dim=1))

    avg_mean = torch.stack(all_ch_means).mean(dim=0)
    avg_std = torch.stack(all_ch_stds).mean(dim=0)

    logger.info("  After normalization (per-subband):")
    for i, name in enumerate(SUBBAND_NAMES):
        logger.info(f"    {name}: mean={avg_mean[i]:.4f}, std={avg_std[i]:.4f}")

    all_passed &= check(
        "Normalized per-subband mean ≈ 0",
        avg_mean.abs().max().item() < 0.3,
        f"max |mean| = {avg_mean.abs().max().item():.4f} (threshold: 0.3)",
    )
    all_passed &= check(
        "Normalized per-subband std ≈ 1",
        avg_std.min().item() > 0.5 and avg_std.max().item() < 2.0,
        f"std range = [{avg_std.min().item():.4f}, {avg_std.max().item():.4f}] (expected: 0.5-2.0)",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 4: Checkpoint stats match computed stats
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 4: Checkpoint wavelet stats ---")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        ckpt_cfg = ckpt.get('config', {})
        wavelet_cfg = ckpt_cfg.get('wavelet', {})
        del ckpt

        ckpt_shift = wavelet_cfg.get('wavelet_shift')
        ckpt_scale = wavelet_cfg.get('wavelet_scale')
        ckpt_rescale = wavelet_cfg.get('rescale')

        if ckpt_shift is not None:
            logger.info("  Checkpoint wavelet stats found:")
            for i, name in enumerate(SUBBAND_NAMES):
                logger.info(f"    {name}: shift={ckpt_shift[i]:.6f}, scale={ckpt_scale[i]:.6f}")
            logger.info(f"    rescale: {ckpt_rescale}")

            # Compare to freshly computed stats
            shift_diff = max(abs(a - b) for a, b in zip(shift, ckpt_shift))
            scale_diff = max(abs(a - b) for a, b in zip(scale, ckpt_scale))

            all_passed &= check(
                "Checkpoint shift matches computed",
                shift_diff < 0.01,
                f"max diff = {shift_diff:.6f} (threshold: 0.01)",
            )
            all_passed &= check(
                "Checkpoint scale matches computed",
                scale_diff < 0.01,
                f"max diff = {scale_diff:.6f} (threshold: 0.01)",
            )

            if ckpt_rescale is not None:
                all_passed &= check(
                    "Checkpoint rescale matches arg",
                    ckpt_rescale == args.rescale,
                    f"checkpoint={ckpt_rescale}, arg={args.rescale}",
                )
        else:
            logger.warning("  No wavelet stats in checkpoint config!")
            logger.warning("  This checkpoint was saved before the persistence fix.")
            logger.warning("  Generation must use --wavelet-normalize (recompute from data).")
            all_passed &= check(
                "Checkpoint has wavelet stats",
                False,
                "missing — retrain or re-save checkpoint to persist stats",
            )
    else:
        logger.info("  No checkpoint provided, skipping checkpoint comparison")

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 5: Noise/signal scale compatibility
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 5: Noise/signal scale ---")

    # Get a normalized sample
    sample_norm = space_norm.encode(volumes[0])  # [1, 8, D/2, H/2, W/2]
    noise = torch.randn_like(sample_norm)

    signal_std = sample_norm.std().item()
    noise_std = noise.std().item()
    ratio = signal_std / noise_std

    logger.info(f"  Normalized signal std: {signal_std:.4f}")
    logger.info(f"  Noise std: {noise_std:.4f}")
    logger.info(f"  Signal/noise ratio: {ratio:.4f}")

    all_passed &= check(
        "Signal/noise ratio ≈ 1",
        0.2 < ratio < 5.0,
        f"ratio = {ratio:.4f} (expected: 0.2-5.0)",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 6: Normalized encode→decode round-trip
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 6: Normalized encode→decode round-trip ---")

    # Full round-trip: encode (forward DWT + normalize) → decode (denormalize + inverse DWT)
    original = volumes[0].clone()
    encoded_norm = space_norm.encode(original)
    decoded = space_norm.decode(encoded_norm)
    psnr_err = (original - decoded).abs().max().item()

    all_passed &= check(
        "Normalized WaveletSpace round-trip (lossless)",
        psnr_err < 1e-5,
        f"max error = {psnr_err:.2e}",
    )

    # Also test with real data PSNR
    original_np = original[0, 0].numpy()
    decoded_np = decoded[0, 0].numpy()
    mse = np.mean((original_np - decoded_np) ** 2)
    psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
    all_passed &= check(
        "Round-trip PSNR",
        psnr > 60,  # Should be very high since wavelet is lossless
        f"PSNR = {psnr:.1f} dB (expected: >60, wavelet is lossless)",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 7: Generation pipeline shapes
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 7: Generation pipeline shapes ---")

    wavelet_depth = args.depth // 2
    wavelet_hw = args.image_size // 2

    # Verify encode produces correct wavelet shape
    encoded = space_norm.encode(volumes[0])
    expected_shape = (1, 8, wavelet_depth, wavelet_hw, wavelet_hw)
    all_passed &= check(
        "Encode shape",
        tuple(encoded.shape) == expected_shape,
        f"got {tuple(encoded.shape)}, expected {expected_shape}",
    )

    # Simulate generation: noise in wavelet space → decode to pixel space
    gen_noise = torch.randn(1, 8, wavelet_depth, wavelet_hw, wavelet_hw)
    gen_decoded = space_norm.decode(gen_noise)
    expected_pixel_shape = (1, 1, args.depth, args.image_size, args.image_size)
    all_passed &= check(
        "Decode shape (noise → pixel)",
        tuple(gen_decoded.shape) == expected_pixel_shape,
        f"got {tuple(gen_decoded.shape)}, expected {expected_pixel_shape}",
    )

    # Conditioning (seg) through wavelet space
    seg_test = torch.zeros(1, 1, args.depth, args.image_size, args.image_size)
    seg_encoded = space_norm.encode(seg_test)
    all_passed &= check(
        "Conditioning encode shape",
        tuple(seg_encoded.shape) == expected_shape,
        f"got {tuple(seg_encoded.shape)}, expected {expected_shape}",
    )

    # Verify model input construction (noise + conditioning concat)
    model_input = torch.cat([gen_noise, seg_encoded], dim=1)
    expected_in_ch = 16  # 8 (bravo wavelet) + 8 (seg wavelet)
    all_passed &= check(
        "Model input channels (noise + cond)",
        model_input.shape[1] == expected_in_ch,
        f"got {model_input.shape[1]}, expected {expected_in_ch}",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 8: Subband distribution characteristics
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 8: Subband distribution characteristics ---")

    # LLL (lowpass) should have most energy, high subbands should have less
    raw_coeffs = haar_forward_3d(volumes[0])
    subband_energies = []
    for i in range(8):
        energy = (raw_coeffs[0, i] ** 2).mean().item()
        subband_energies.append(energy)
        logger.info(f"  {SUBBAND_NAMES[i]}: energy={energy:.6f}")

    lll_energy = subband_energies[0]
    high_energy_sum = sum(subband_energies[1:])

    all_passed &= check(
        "LLL has dominant energy",
        lll_energy > high_energy_sum,
        f"LLL={lll_energy:.6f} vs sum(high)={high_energy_sum:.6f}",
    )

    # LLL scale should be largest (most variance in lowpass)
    all_passed &= check(
        "LLL has largest scale",
        scale[0] == max(scale),
        f"LLL scale={scale[0]:.6f}, max scale={max(scale):.6f}",
    )

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    if all_passed:
        logger.info("ALL CHECKS PASSED — WDM pipeline is consistent")
    else:
        logger.info("SOME CHECKS FAILED — pipeline has issues (see above)")
    logger.info("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
