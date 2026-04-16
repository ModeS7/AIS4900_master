#!/usr/bin/env python3
"""Compare frequency-domain degradation methods for restoration training.

Computes and compares two approaches:
1. Empirical Transfer Function: H(f) = sqrt(PSD_generated / PSD_real)
2. Analytical Wiener Filter: H(f) = α̅ / (α̅ + (1-α̅)·|f|²)

Applies both to real volumes, then verifies the degraded volumes match
the frequency profile of generated (MSE-smoothed) volumes.

Outputs:
- Comparison plots (power spectra, visual slices)
- Transfer function saved as .npz for use in pair generation
- Per-method statistics

Usage:
    python -m medgen.scripts.compare_degradation_methods \
        --real-dir /path/to/brainmetshare-3/train \
        --generated-dir /path/to/generated/exp1_1_bravo_imagenet_525 \
        --output-dir /path/to/degradation_comparison \
        --num-volumes 20 --depth 160
"""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.optimize import minimize_scalar

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


# ── Power spectrum computation ──────────────────────────────────────

def compute_radial_power_spectrum_3d(
    volume: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute radially averaged 3D power spectrum.

    Returns:
        (freq_bins, power): 1D arrays of radial frequency and mean PSD.
    """
    fft = np.fft.fftn(volume)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2

    d, h, w = volume.shape
    cd, ch, cw = d // 2, h // 2, w // 2
    dz, dy, dx = np.ogrid[-cd:d - cd, -ch:h - ch, -cw:w - cw]
    radius = np.sqrt((dz / d) ** 2 + (dy / h) ** 2 + (dx / w) ** 2)

    max_radius = 0.5
    num_bins = min(d, h, w) // 2
    bin_edges = np.linspace(0, max_radius, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    radial_power = np.zeros(num_bins)
    for i in range(num_bins):
        mask = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
        if mask.sum() > 0:
            radial_power[i] = power[mask].mean()

    return bin_centers, radial_power


def compute_average_spectrum(
    volumes: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute average radial power spectrum across volumes."""
    all_spectra = []
    for vol in volumes:
        freqs, power = compute_radial_power_spectrum_3d(vol)
        all_spectra.append(power)
    return freqs, np.mean(all_spectra, axis=0)


# ── Degradation methods ─────────────────────────────────────────────

def compute_empirical_transfer_function(
    psd_real: np.ndarray,
    psd_generated: np.ndarray,
) -> np.ndarray:
    """Compute H(f) = sqrt(PSD_gen / PSD_real), clamped to [0, 1].

    The transfer function attenuates each frequency bin to match
    the generated (MSE-smoothed) spectrum.
    """
    ratio = np.where(
        psd_real > 1e-20,
        psd_generated / psd_real,
        1.0,
    )
    # Clamp: can't amplify, only attenuate
    ratio = np.clip(ratio, 0.0, 1.0)
    return np.sqrt(ratio)


def wiener_filter(freqs: np.ndarray, alpha_bar: float) -> np.ndarray:
    """Analytical Wiener filter: H(f) = α̅ / (α̅ + (1-α̅)·|f|²).

    From STIG (AAAI 2024): models the frequency response of MSE-optimal
    diffusion denoising.
    """
    f_sq = freqs ** 2
    return alpha_bar / (alpha_bar + (1.0 - alpha_bar) * f_sq)


def fit_wiener_alpha(
    freqs: np.ndarray,
    empirical_h: np.ndarray,
) -> float:
    """Find α̅ that best matches empirical transfer function."""
    def loss(log_alpha: float) -> float:
        alpha = np.exp(log_alpha)
        h_wiener = wiener_filter(freqs, alpha)
        return float(np.mean((h_wiener - empirical_h) ** 2))

    result = minimize_scalar(loss, bounds=(-10, 10), method='bounded')
    return float(np.exp(result.x))


def apply_frequency_degradation(
    volume: np.ndarray,
    freqs: np.ndarray,
    transfer_func: np.ndarray,
    randomize_std: float = 0.0,
) -> np.ndarray:
    """Apply frequency-domain degradation preserving phase.

    Args:
        volume: 3D array [D, H, W] in [0, 1].
        freqs: 1D radial frequency bins (from PSD computation).
        transfer_func: 1D transfer function H(f) per frequency bin.
        randomize_std: If > 0, add Gaussian noise to log(H) for variation.

    Returns:
        Degraded volume [D, H, W] in [0, 1].
    """
    d, h, w = volume.shape
    cd, ch, cw = d // 2, h // 2, w // 2

    # Build 3D radius grid (matching PSD computation)
    dz, dy, dx = np.ogrid[-cd:d - cd, -ch:h - ch, -cw:w - cw]
    radius = np.sqrt((dz / d) ** 2 + (dy / h) ** 2 + (dx / w) ** 2)

    # Interpolate 1D transfer function to 3D
    max_radius = 0.5
    num_bins = len(freqs)
    bin_edges = np.linspace(0, max_radius, num_bins + 1)

    # Build 3D filter by mapping each voxel's radius to the corresponding H(f)
    filter_3d = np.ones_like(radius)
    for i in range(num_bins):
        mask = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
        h_val = transfer_func[i]
        if randomize_std > 0:
            # Randomize in log space to stay positive
            h_val = h_val * np.exp(np.random.normal(0, randomize_std))
            h_val = np.clip(h_val, 0.0, 1.0)
        filter_3d[mask] = h_val

    # Apply in frequency domain
    fft = np.fft.fftn(volume)
    fft_shift = np.fft.fftshift(fft)
    fft_filtered = fft_shift * filter_3d
    result = np.fft.ifftn(np.fft.ifftshift(fft_filtered)).real

    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ── Volume loading ──────────────────────────────────────────────────

def load_volume(path: Path, depth: int) -> np.ndarray:
    """Load NIfTI, normalize to [0,1], transpose to [D,H,W], pad depth."""
    img = nib.load(str(path))
    vol = img.get_fdata().astype(np.float32)
    vmax = vol.max()
    if vmax > 0:
        vol /= vmax
    vol = np.transpose(vol, (2, 0, 1))  # [H,W,D] -> [D,H,W]
    if vol.shape[0] < depth:
        vol = np.pad(vol, ((0, depth - vol.shape[0]), (0, 0), (0, 0)))
    elif vol.shape[0] > depth:
        vol = vol[:depth]
    return vol


def load_real_volumes(
    data_dir: Path, depth: int, max_vols: int,
) -> list[np.ndarray]:
    """Load bravo volumes from patient directories."""
    volumes = []
    for patient_dir in sorted(data_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        bravo = patient_dir / "bravo.nii.gz"
        if bravo.exists():
            volumes.append(load_volume(bravo, depth))
            if len(volumes) >= max_vols:
                break
    return volumes


def load_generated_volumes(
    gen_dir: Path, depth: int, max_vols: int,
) -> list[np.ndarray]:
    """Load generated bravo volumes (nested or flat structure)."""
    volumes = []

    # Try nested structure first: vol_*/bravo.nii.gz
    for sub in sorted(gen_dir.iterdir()):
        if sub.is_dir():
            bravo = sub / "bravo.nii.gz"
            if bravo.exists():
                volumes.append(load_volume(bravo, depth))
                if len(volumes) >= max_vols:
                    return volumes

    # Flat structure: *.nii.gz
    if not volumes:
        for f in sorted(gen_dir.glob("*.nii.gz")):
            volumes.append(load_volume(f, depth))
            if len(volumes) >= max_vols:
                break

    return volumes


# ── Visualization ───────────────────────────────────────────────────

def plot_spectra_comparison(
    freqs: np.ndarray,
    psd_real: np.ndarray,
    psd_generated: np.ndarray,
    psd_degraded_emp: np.ndarray,
    psd_degraded_wiener: np.ndarray,
    output_path: Path,
) -> None:
    """Plot power spectra: real, generated, and both degradation methods."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Log-scale spectra
    ax = axes[0]
    ax.semilogy(freqs, psd_real, 'b-', label='Real', linewidth=2)
    ax.semilogy(freqs, psd_generated, 'r-', label='Generated', linewidth=2)
    ax.semilogy(freqs, psd_degraded_emp, 'g--', label='Degraded (empirical)', linewidth=1.5)
    ax.semilogy(freqs, psd_degraded_wiener, 'm--', label='Degraded (Wiener)', linewidth=1.5)
    ax.set_xlabel('Radial Frequency')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Power Spectra Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ratio to real (should match generated/real ratio)
    ax = axes[1]
    ratio_gen = np.where(psd_real > 1e-20, psd_generated / psd_real, 1.0)
    ratio_emp = np.where(psd_real > 1e-20, psd_degraded_emp / psd_real, 1.0)
    ratio_wie = np.where(psd_real > 1e-20, psd_degraded_wiener / psd_real, 1.0)
    ax.plot(freqs, ratio_gen, 'r-', label='Generated/Real (target)', linewidth=2)
    ax.plot(freqs, ratio_emp, 'g--', label='Empirical degraded/Real', linewidth=1.5)
    ax.plot(freqs, ratio_wie, 'm--', label='Wiener degraded/Real', linewidth=1.5)
    ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.3)
    ax.set_xlabel('Radial Frequency')
    ax.set_ylabel('PSD Ratio')
    ax.set_title('Spectral Ratio (should match red line)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.5)

    # Transfer functions
    ax = axes[2]
    h_emp = compute_empirical_transfer_function(psd_real, psd_generated)
    alpha_fit = fit_wiener_alpha(freqs, h_emp)
    h_wiener = wiener_filter(freqs, alpha_fit)
    ax.plot(freqs, h_emp, 'g-', label='Empirical H(f)', linewidth=2)
    ax.plot(freqs, h_wiener, 'm-', label=f'Wiener H(f) (α̅={alpha_fit:.4f})', linewidth=2)
    ax.set_xlabel('Radial Frequency')
    ax.set_ylabel('H(f)')
    ax.set_title('Transfer Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved spectra comparison: {output_path}")


def plot_visual_comparison(
    real_vol: np.ndarray,
    generated_vol: np.ndarray,
    degraded_emp: np.ndarray,
    degraded_wiener: np.ndarray,
    output_path: Path,
    slice_idx: int | None = None,
) -> None:
    """Show axial slices: Real, Generated, Degraded (empirical), Degraded (Wiener)."""
    if slice_idx is None:
        slice_idx = real_vol.shape[0] // 2

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    images = [real_vol, generated_vol, degraded_emp, degraded_wiener]
    titles = ['Real (clean)', 'Generated (MSE-smoothed)', 'Degraded (empirical)', 'Degraded (Wiener)']

    # Top row: full slices
    for ax, img, title in zip(axes[0], images, titles):
        ax.imshow(img[slice_idx], cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    # Bottom row: zoomed (center crop)
    h, w = real_vol.shape[1], real_vol.shape[2]
    crop_h, crop_w = h // 3, w // 3
    ch, cw = h // 2, w // 2
    for ax, img, title in zip(axes[1], images, titles):
        crop = img[slice_idx, ch - crop_h:ch + crop_h, cw - crop_w:cw + crop_w]
        ax.imshow(crop, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'{title} (zoomed)', fontsize=11)
        ax.axis('off')

    plt.suptitle(f'Axial Slice {slice_idx} — Degradation Method Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved visual comparison: {output_path}")


def plot_difference_maps(
    real_vol: np.ndarray,
    degraded_emp: np.ndarray,
    degraded_wiener: np.ndarray,
    output_path: Path,
    slice_idx: int | None = None,
) -> None:
    """Show what each degradation removes (residual = real - degraded)."""
    if slice_idx is None:
        slice_idx = real_vol.shape[0] // 2

    diff_emp = real_vol - degraded_emp
    diff_wie = real_vol - degraded_wiener

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    vmax = max(np.abs(diff_emp[slice_idx]).max(), np.abs(diff_wie[slice_idx]).max())

    axes[0].imshow(diff_emp[slice_idx], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0].set_title(f'Empirical residual (L1={np.abs(diff_emp).mean():.4f})')
    axes[0].axis('off')

    axes[1].imshow(diff_wie[slice_idx], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title(f'Wiener residual (L1={np.abs(diff_wie).mean():.4f})')
    axes[1].axis('off')

    # Difference between the two residuals
    diff_diff = diff_emp - diff_wie
    axes[2].imshow(diff_diff[slice_idx], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[2].set_title(f'Empirical - Wiener (L1={np.abs(diff_diff).mean():.4f})')
    axes[2].axis('off')

    plt.suptitle(f'Residuals (what degradation removes) — Slice {slice_idx}', fontsize=13)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved difference maps: {output_path}")


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare frequency-domain degradation methods",
    )
    parser.add_argument("--real-dir", type=str, required=True,
                        help="Directory with real patient volumes (train split)")
    parser.add_argument("--generated-dir", type=str, required=True,
                        help="Directory with generated volumes (exp1_1)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for comparison results")
    parser.add_argument("--num-volumes", type=int, default=20,
                        help="Number of volumes for spectrum computation")
    parser.add_argument("--depth", type=int, default=160)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load volumes ──
    logger.info(f"Loading real volumes from {args.real_dir}...")
    real_vols = load_real_volumes(Path(args.real_dir), args.depth, args.num_volumes)
    logger.info(f"Loaded {len(real_vols)} real volumes")

    logger.info(f"Loading generated volumes from {args.generated_dir}...")
    gen_vols = load_generated_volumes(Path(args.generated_dir), args.depth, args.num_volumes)
    logger.info(f"Loaded {len(gen_vols)} generated volumes")

    if not real_vols or not gen_vols:
        logger.error("Need both real and generated volumes")
        return

    # ── Compute average spectra ──
    logger.info("Computing average power spectra...")
    freqs, psd_real = compute_average_spectrum(real_vols)
    _, psd_gen = compute_average_spectrum(gen_vols)

    # ── Compute transfer functions ──
    logger.info("Computing empirical transfer function...")
    h_empirical = compute_empirical_transfer_function(psd_real, psd_gen)

    logger.info("Fitting Wiener filter parameter...")
    alpha_bar = fit_wiener_alpha(freqs, h_empirical)
    h_wiener = wiener_filter(freqs, alpha_bar)
    logger.info(f"Best-fit Wiener α̅ = {alpha_bar:.6f}")

    # Compute match quality
    mse_wiener_vs_emp = float(np.mean((h_wiener - h_empirical) ** 2))
    logger.info(f"Wiener vs Empirical MSE: {mse_wiener_vs_emp:.6f}")

    # ── Save transfer functions ──
    np.savez(
        str(output_dir / 'transfer_functions.npz'),
        freqs=freqs,
        h_empirical=h_empirical,
        h_wiener=h_wiener,
        psd_real=psd_real,
        psd_generated=psd_gen,
        alpha_bar_fit=alpha_bar,
    )
    logger.info(f"Saved transfer functions to {output_dir / 'transfer_functions.npz'}")

    # ── Apply degradations to test volumes ──
    n_test = min(5, len(real_vols))
    logger.info(f"Applying degradations to {n_test} test volumes...")

    all_psd_emp = []
    all_psd_wie = []

    for i in range(n_test):
        real = real_vols[i]

        degraded_emp = apply_frequency_degradation(real, freqs, h_empirical)
        degraded_wie = apply_frequency_degradation(real, freqs, h_wiener)

        # Compute spectra of degraded volumes
        _, psd_emp = compute_radial_power_spectrum_3d(degraded_emp)
        _, psd_wie = compute_radial_power_spectrum_3d(degraded_wie)
        all_psd_emp.append(psd_emp)
        all_psd_wie.append(psd_wie)

        # Pick a generated volume for visual comparison
        gen = gen_vols[i % len(gen_vols)]

        # Visual comparison
        plot_visual_comparison(
            real, gen, degraded_emp, degraded_wie,
            output_dir / f'visual_comparison_{i:02d}.png',
        )

        # Difference maps
        plot_difference_maps(
            real, degraded_emp, degraded_wie,
            output_dir / f'difference_maps_{i:02d}.png',
        )

    # Average spectra of degraded volumes
    avg_psd_emp = np.mean(all_psd_emp, axis=0)
    avg_psd_wie = np.mean(all_psd_wie, axis=0)

    # ── Plot spectra comparison ──
    plot_spectra_comparison(
        freqs, psd_real, psd_gen, avg_psd_emp, avg_psd_wie,
        output_dir / 'spectra_comparison.png',
    )

    # ── Summary statistics ──
    # How well does each degradation match the generated spectrum?
    ratio_gen = np.where(psd_real > 1e-20, psd_gen / psd_real, 1.0)
    ratio_emp = np.where(psd_real > 1e-20, avg_psd_emp / psd_real, 1.0)
    ratio_wie = np.where(psd_real > 1e-20, avg_psd_wie / psd_real, 1.0)

    mse_emp_vs_target = float(np.mean((ratio_emp - ratio_gen) ** 2))
    mse_wie_vs_target = float(np.mean((ratio_wie - ratio_gen) ** 2))

    logger.info("=== Summary ===")
    logger.info(f"Wiener best-fit α̅: {alpha_bar:.6f}")
    logger.info(f"Spectral match MSE (empirical vs target): {mse_emp_vs_target:.6f}")
    logger.info(f"Spectral match MSE (Wiener vs target):    {mse_wie_vs_target:.6f}")
    if mse_emp_vs_target < mse_wie_vs_target:
        logger.info(">>> Empirical transfer function matches generated spectrum better")
    else:
        logger.info(">>> Wiener filter matches generated spectrum better")

    # Save summary
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write(f"Wiener best-fit alpha_bar: {alpha_bar:.6f}\n")
        f.write(f"Wiener vs Empirical H(f) MSE: {mse_wiener_vs_emp:.6f}\n")
        f.write(f"Spectral match MSE (empirical vs target): {mse_emp_vs_target:.6f}\n")
        f.write(f"Spectral match MSE (Wiener vs target): {mse_wie_vs_target:.6f}\n")
        f.write(f"Better method: {'empirical' if mse_emp_vs_target < mse_wie_vs_target else 'wiener'}\n")
        f.write(f"\nNum real volumes: {len(real_vols)}\n")
        f.write(f"Num generated volumes: {len(gen_vols)}\n")
        f.write(f"Num test comparisons: {n_test}\n")

    logger.info(f"All results saved to {output_dir}")


if __name__ == '__main__':
    main()
