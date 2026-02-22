#!/usr/bin/env python3
"""Verify that the WDM training and generation pipelines are consistent.

Only needs: diffusion checkpoint + data root.
No extra files — wavelet stats are read from the checkpoint (saved by profiling.py).

Checks performed:
  1. Haar wavelet losslessness + energy preservation
  2. WaveletSpace encode→decode round-trip (no normalization)
  3. Checkpoint has wavelet stats, normalization produces mean ≈ 0, std ≈ 1
  4. Checkpoint stats match freshly computed stats from data
  5. Noise/signal scale compatibility
  6. Normalized encode→decode round-trip (lossless)
  7. Generation pipeline shapes (encode, decode, model input)
  8. Subband distribution characteristics (LLL dominance)

Usage:
    python -m medgen.scripts.verify_wdm_pipeline \
        --checkpoint runs/diffusion_3d/.../checkpoint_latest.pt \
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
    """Load BRAVO NIfTI volumes with standard preprocessing.

    Returns list of [1, 1, D, H, W] tensors in [0, 1].
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

        if rescale:
            vol_tensor = 2.0 * vol_tensor - 1.0

        volumes.append(vol_tensor)

    logger.info(f"  Loaded {len(volumes)} volumes from {train_dir}")
    return volumes


def compute_wavelet_stats(
    volumes: list[torch.Tensor],
) -> tuple[list[float], list[float]]:
    """Compute per-subband mean/std using law of total variance."""
    from medgen.models.haar_wavelet_3d import haar_forward_3d

    n = 0
    mean = None
    m2 = None
    var_sum = None

    for vol in volumes:
        coeffs = haar_forward_3d(vol)
        n_ch = coeffs.shape[1]
        flat = coeffs[0].reshape(n_ch, -1).float()
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
    parser = argparse.ArgumentParser(description="Verify WDM pipeline consistency")
    parser.add_argument('--checkpoint', required=True,
                        help='Diffusion model checkpoint (has wavelet stats in config)')
    parser.add_argument('--data-root', required=True,
                        help='Dataset root with NIfTI volumes')
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--max-volumes', type=int, default=100,
                        help='Max volumes for stats verification (default: 100)')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    all_passed = True

    logger.info("=" * 70)
    logger.info("WDM Pipeline Verification")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Volume: {args.image_size}x{args.image_size}x{args.depth}")
    logger.info("")

    from medgen.diffusion.spaces import WaveletSpace
    from medgen.models.haar_wavelet_3d import haar_forward_3d, haar_inverse_3d

    # ── Load checkpoint config ──
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ckpt_cfg = ckpt.get('config', {})
    wavelet_cfg = ckpt_cfg.get('wavelet', {})
    del ckpt

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 1: Haar wavelet losslessness + energy preservation
    # ══════════════════════════════════════════════════════════════════════
    logger.info("--- Check 1: Haar wavelet losslessness ---")

    test_vol = torch.randn(2, 1, 8, 16, 16)
    fwd = haar_forward_3d(test_vol)
    inv = haar_inverse_3d(fwd)
    max_err = (test_vol - inv).abs().max().item()

    all_passed &= check(
        "Haar forward→inverse = identity",
        max_err < 1e-5,
        f"max error = {max_err:.2e}",
    )
    all_passed &= check(
        "Haar forward shape",
        fwd.shape == (2, 8, 4, 8, 8),
        f"got {tuple(fwd.shape)}, expected (2, 8, 4, 8, 8)",
    )

    energy_in = (test_vol ** 2).sum().item()
    energy_out = (fwd ** 2).sum().item()
    energy_ratio = energy_out / energy_in
    all_passed &= check(
        "Haar energy preservation",
        abs(energy_ratio - 1.0) < 1e-5,
        f"ratio = {energy_ratio:.6f} (expected: 1.0)",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 2: WaveletSpace encode→decode round-trip (no normalization)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 2: Raw WaveletSpace round-trip ---")

    ckpt_rescale = wavelet_cfg.get('rescale', False)
    space_raw = WaveletSpace(rescale=ckpt_rescale)
    test_pixel = torch.rand(1, 1, 8, 16, 16)
    rt_err = (test_pixel - space_raw.decode(space_raw.encode(test_pixel))).abs().max().item()

    all_passed &= check(
        "Raw WaveletSpace round-trip",
        rt_err < 1e-5,
        f"max error = {rt_err:.2e}",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 3: Checkpoint has wavelet stats, normalization → mean≈0, std≈1
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 3: Checkpoint wavelet stats + normalization ---")

    ckpt_shift = wavelet_cfg.get('wavelet_shift')
    ckpt_scale = wavelet_cfg.get('wavelet_scale')

    all_passed &= check(
        "Checkpoint has wavelet_shift/wavelet_scale",
        ckpt_shift is not None and ckpt_scale is not None,
        "present" if ckpt_shift else "MISSING — retrain with stats persistence fix",
    )

    if ckpt_shift is None:
        logger.error("Cannot continue without wavelet stats. Retrain with the stats persistence fix.")
        sys.exit(1)

    logger.info(f"  rescale: {ckpt_rescale}")
    for i, name in enumerate(SUBBAND_NAMES):
        logger.info(f"  {name}: shift={ckpt_shift[i]:.6f}, scale={ckpt_scale[i]:.6f}")

    all_passed &= check(
        "Scale values are positive",
        all(s > 0 for s in ckpt_scale),
        f"min scale = {min(ckpt_scale):.6f}",
    )

    # Create normalized space and verify on data
    space_norm = WaveletSpace(shift=ckpt_shift, scale=ckpt_scale, rescale=ckpt_rescale)

    logger.info(f"\n  Loading volumes for normalization check...")
    volumes = load_bravo_volumes(
        data_root, args.depth, args.image_size,
        max_volumes=args.max_volumes, rescale=ckpt_rescale,
    )

    all_ch_means = []
    all_ch_stds = []
    for vol in volumes[:50]:
        coeffs = space_norm.encode(vol)
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
    # CHECK 4: Checkpoint stats match freshly computed stats
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 4: Checkpoint stats vs recomputed ---")

    fresh_shift, fresh_scale = compute_wavelet_stats(volumes)

    shift_diff = max(abs(a - b) for a, b in zip(fresh_shift, ckpt_shift))
    scale_diff = max(abs(a - b) for a, b in zip(fresh_scale, ckpt_scale))

    logger.info(f"  Max shift diff: {shift_diff:.6f}")
    logger.info(f"  Max scale diff: {scale_diff:.6f}")

    # Tolerance is loose because training uses DataLoader transforms
    # while we load NIfTI directly — slight preprocessing differences expected
    all_passed &= check(
        "Checkpoint shift ≈ recomputed",
        shift_diff < 0.05,
        f"max diff = {shift_diff:.6f} (threshold: 0.05)",
    )
    all_passed &= check(
        "Checkpoint scale ≈ recomputed",
        scale_diff < 0.05,
        f"max diff = {scale_diff:.6f} (threshold: 0.05)",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 5: Noise/signal scale
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 5: Noise/signal scale ---")

    sample_norm = space_norm.encode(volumes[0])
    noise = torch.randn_like(sample_norm)
    signal_std = sample_norm.std().item()
    noise_std = noise.std().item()
    ratio = signal_std / noise_std

    logger.info(f"  Signal std (normalized): {signal_std:.4f}")
    logger.info(f"  Noise std: {noise_std:.4f}")
    logger.info(f"  Ratio: {ratio:.4f}")

    all_passed &= check(
        "Signal/noise ratio ≈ 1",
        0.2 < ratio < 5.0,
        f"ratio = {ratio:.4f} (expected: 0.2-5.0)",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 6: Normalized encode→decode round-trip (lossless)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 6: Normalized encode→decode round-trip ---")

    original = volumes[0].clone()
    decoded = space_norm.decode(space_norm.encode(original))
    rt_err = (original - decoded).abs().max().item()

    all_passed &= check(
        "Normalized round-trip (lossless)",
        rt_err < 1e-5,
        f"max error = {rt_err:.2e}",
    )

    mse = ((original - decoded) ** 2).mean().item()
    psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
    all_passed &= check(
        "Round-trip PSNR",
        psnr > 60,
        f"PSNR = {psnr:.1f} dB (expected: >60, wavelet is lossless)",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 7: Generation pipeline shapes
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 7: Generation pipeline shapes ---")

    wavelet_depth = args.depth // 2
    wavelet_hw = args.image_size // 2

    encoded = space_norm.encode(volumes[0])
    expected_enc = (1, 8, wavelet_depth, wavelet_hw, wavelet_hw)
    all_passed &= check(
        "Encode shape",
        tuple(encoded.shape) == expected_enc,
        f"got {tuple(encoded.shape)}, expected {expected_enc}",
    )

    gen_noise = torch.randn(1, 8, wavelet_depth, wavelet_hw, wavelet_hw)
    gen_decoded = space_norm.decode(gen_noise)
    expected_pixel = (1, 1, args.depth, args.image_size, args.image_size)
    all_passed &= check(
        "Decode shape (noise → pixel)",
        tuple(gen_decoded.shape) == expected_pixel,
        f"got {tuple(gen_decoded.shape)}, expected {expected_pixel}",
    )

    # Model input: noise + seg conditioning (8 + 8 = 16 channels)
    seg_encoded = space_norm.encode(torch.zeros_like(volumes[0]))
    model_input = torch.cat([gen_noise, seg_encoded], dim=1)
    all_passed &= check(
        "Model input channels (noise + cond)",
        model_input.shape[1] == 16,
        f"got {model_input.shape[1]}, expected 16 (8+8)",
    )

    logger.info(f"\n  Wavelet shape: [B, 8, {wavelet_depth}, {wavelet_hw}, {wavelet_hw}]")
    logger.info(f"  Pixel shape:   [B, 1, {args.depth}, {args.image_size}, {args.image_size}]")

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 8: Subband distribution characteristics
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 8: Subband distribution ---")

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
    all_passed &= check(
        "LLL has largest scale",
        ckpt_scale[0] == max(ckpt_scale),
        f"LLL scale={ckpt_scale[0]:.6f}, max={max(ckpt_scale):.6f}",
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
