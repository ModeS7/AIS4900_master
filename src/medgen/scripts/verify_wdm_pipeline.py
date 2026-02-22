#!/usr/bin/env python3
"""Verify the WDM training and generation pipeline BEFORE training.

Only needs: data root (NIfTI volumes).
No checkpoint needed — this verifies the pipeline mechanics are correct
so you can confidently start a training run.

Checks performed:
  1. Haar wavelet losslessness + energy preservation
  2. WaveletSpace encode→decode round-trip (no normalization)
  3. Per-subband stats computation produces valid normalization (mean≈0, std≈1)
  4. Normalized encode→decode round-trip (lossless)
  5. Noise/signal scale compatibility after normalization
  6. Generation pipeline shapes (encode, decode, model input)
  7. Subband distribution characteristics (LLL dominance)
  8. Conditioning (seg) through wavelet space

Usage:
    python -m medgen.scripts.verify_wdm_pipeline \
        --data-root /path/to/brainmetshare-3

    # With rescale (if training uses wavelet.rescale=true)
    python -m medgen.scripts.verify_wdm_pipeline \
        --data-root /path/to/brainmetshare-3 --rescale
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


def load_volumes(
    data_root: Path,
    modality: str,
    depth: int,
    image_size: int,
    max_volumes: int,
    rescale: bool = False,
    binarize: bool = False,
) -> list[torch.Tensor]:
    """Load NIfTI volumes with standard preprocessing.

    Returns list of [1, 1, D, H, W] tensors.
    """
    train_dir = data_root / 'train'
    if not train_dir.exists():
        for d in sorted(data_root.iterdir()):
            if d.is_dir() and list(d.glob(f"*/{modality}.nii.gz")):
                train_dir = d
                break

    files = sorted(train_dir.glob(f"*/{modality}.nii.gz"))[:max_volumes]
    if not files:
        raise FileNotFoundError(f"No {modality}.nii.gz in {train_dir}")

    volumes = []
    for path in files:
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

        if rescale:
            vol_tensor = 2.0 * vol_tensor - 1.0

        volumes.append(vol_tensor)

    logger.info(f"  Loaded {len(volumes)} {modality} volumes from {train_dir}")
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
    parser = argparse.ArgumentParser(description="Verify WDM pipeline BEFORE training")
    parser.add_argument('--data-root', required=True,
                        help='Dataset root with NIfTI volumes')
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--rescale', action='store_true',
                        help='Rescale [0,1] → [-1,1] before DWT (match wavelet.rescale config)')
    parser.add_argument('--max-volumes', type=int, default=100)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    all_passed = True

    logger.info("=" * 70)
    logger.info("WDM Pipeline Verification (pre-training)")
    logger.info("=" * 70)
    logger.info(f"Data root: {data_root}")
    logger.info(f"Volume: {args.image_size}x{args.image_size}x{args.depth}")
    logger.info(f"Rescale: {args.rescale}")
    logger.info("")

    from medgen.diffusion.spaces import WaveletSpace
    from medgen.models.haar_wavelet_3d import haar_forward_3d, haar_inverse_3d

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

    space_raw = WaveletSpace(rescale=args.rescale)
    test_pixel = torch.rand(1, 1, 8, 16, 16)
    rt_err = (test_pixel - space_raw.decode(space_raw.encode(test_pixel))).abs().max().item()

    all_passed &= check(
        "Raw WaveletSpace round-trip",
        rt_err < 1e-5,
        f"max error = {rt_err:.2e}",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 3: Compute stats from real data, verify normalization
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 3: Stats computation + normalization ---")

    logger.info(f"  Loading up to {args.max_volumes} bravo volumes...")
    volumes = load_volumes(
        data_root, 'bravo', args.depth, args.image_size,
        max_volumes=args.max_volumes, rescale=args.rescale,
    )

    shift, scale = compute_wavelet_stats(volumes)

    logger.info("  Computed per-subband stats:")
    for i, name in enumerate(SUBBAND_NAMES):
        logger.info(f"    {name}: shift={shift[i]:.6f}, scale={scale[i]:.6f}")

    all_passed &= check(
        "Scale values are positive and finite",
        all(0 < s < 100 for s in scale),
        f"scale range = [{min(scale):.6f}, {max(scale):.6f}]",
    )

    # Create normalized space and verify
    space_norm = WaveletSpace(shift=shift, scale=scale, rescale=args.rescale)

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

    logger.info("  After normalization:")
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
    # CHECK 4: Normalized encode→decode round-trip (lossless)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 4: Normalized round-trip on real data ---")

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
    # CHECK 5: Noise/signal scale compatibility
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 5: Noise/signal scale ---")

    sample_norm = space_norm.encode(volumes[0])
    noise = torch.randn_like(sample_norm)
    signal_std = sample_norm.std().item()
    noise_std = noise.std().item()
    ratio = signal_std / noise_std

    logger.info(f"  Signal std (normalized wavelet): {signal_std:.4f}")
    logger.info(f"  Noise std: {noise_std:.4f}")
    logger.info(f"  Ratio: {ratio:.4f}")

    all_passed &= check(
        "Signal/noise ratio ≈ 1",
        0.2 < ratio < 5.0,
        f"ratio = {ratio:.4f} (expected: 0.2-5.0)",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 6: Generation pipeline shapes
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 6: Generation pipeline shapes ---")

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
    # CHECK 7: Subband distribution characteristics
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 7: Subband distribution ---")

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
        scale[0] == max(scale),
        f"LLL scale={scale[0]:.6f}, max={max(scale):.6f}",
    )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 8: Seg conditioning through wavelet space
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n--- Check 8: Seg conditioning ---")

    try:
        seg_volumes = load_volumes(
            data_root, 'seg', args.depth, args.image_size,
            max_volumes=5, rescale=args.rescale, binarize=True,
        )

        seg_encoded = space_norm.encode(seg_volumes[0])
        bravo_encoded = space_norm.encode(volumes[0])

        all_passed &= check(
            "Seg wavelet shape matches bravo",
            seg_encoded.shape == bravo_encoded.shape,
            f"seg={tuple(seg_encoded.shape)}, bravo={tuple(bravo_encoded.shape)}",
        )

        # Seg is mostly zeros → wavelet coefficients should be mostly near shift
        # (after normalization, mostly near 0 but with different distribution)
        seg_nonzero = (seg_volumes[0] > 0).float().mean().item()
        logger.info(f"  Seg volume sparsity: {1 - seg_nonzero:.1%} zero")

        # Verify concat works
        model_input = torch.cat([bravo_encoded, seg_encoded], dim=1)
        all_passed &= check(
            "Bravo + seg concat = 16 channels",
            model_input.shape[1] == 16,
            f"got {model_input.shape[1]}",
        )

        # Round-trip seg through wavelet
        seg_decoded = space_norm.decode(seg_encoded)
        seg_rt_err = (seg_volumes[0] - seg_decoded).abs().max().item()
        all_passed &= check(
            "Seg wavelet round-trip (lossless)",
            seg_rt_err < 1e-5,
            f"max error = {seg_rt_err:.2e}",
        )
    except FileNotFoundError:
        logger.warning("  No seg.nii.gz found, skipping conditioning checks")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    if all_passed:
        logger.info("ALL CHECKS PASSED — WDM pipeline is ready for training")
    else:
        logger.info("SOME CHECKS FAILED — fix issues before training")
    logger.info("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
