#!/usr/bin/env python3
"""Visualize FFT amplitude comparison: real vs generated volumes.

Creates detailed frequency-domain visualizations:
1. 2D FFT amplitude maps for central axial slices (real vs generated)
2. Amplitude ratio maps showing where generated is weaker
3. Brain-masked 3D radial power spectrum (excludes background zeros)
4. Per-frequency-band amplitude comparison
5. Recomputed empirical transfer function with brain masking

Usage:
    python -m medgen.scripts.visualize_fft_comparison \
        --real-dir /path/to/brainmetshare-3/train \
        --generated-dir /path/to/generated/exp1_1_bravo_imagenet_525 \
        --output-dir /path/to/fft_comparison \
        --num-volumes 10 --depth 160
"""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


# ── Volume loading ──────────────────────────────────────────────────

def load_volume(path: Path, depth: int) -> np.ndarray:
    """Load NIfTI, normalize to [0,1], transpose to [D,H,W], pad depth."""
    img = nib.load(str(path))
    vol = img.get_fdata().astype(np.float32)
    vmax = vol.max()
    if vmax > 0:
        vol /= vmax
    vol = np.transpose(vol, (2, 0, 1))
    if vol.shape[0] < depth:
        vol = np.pad(vol, ((0, depth - vol.shape[0]), (0, 0), (0, 0)))
    elif vol.shape[0] > depth:
        vol = vol[:depth]
    return vol


def load_real_volumes(data_dir: Path, depth: int, max_vols: int) -> list[np.ndarray]:
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


def load_generated_volumes(gen_dir: Path, depth: int, max_vols: int) -> list[np.ndarray]:
    volumes = []
    for sub in sorted(gen_dir.iterdir()):
        if sub.is_dir():
            bravo = sub / "bravo.nii.gz"
            if bravo.exists():
                volumes.append(load_volume(bravo, depth))
                if len(volumes) >= max_vols:
                    return volumes
    if not volumes:
        for f in sorted(gen_dir.glob("*.nii.gz")):
            volumes.append(load_volume(f, depth))
            if len(volumes) >= max_vols:
                break
    return volumes


# ── Brain mask ──────────────────────────────────────────────────────

def compute_brain_mask(volume: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """Simple intensity-based brain mask."""
    return (volume > threshold).astype(np.float32)


# ── Spectrum computation ────────────────────────────────────────────

def compute_masked_radial_spectrum_3d(
    volume: np.ndarray, mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute radially averaged 3D power spectrum, optionally within mask.

    If mask is provided, zero out background before FFT and normalize
    by the number of masked voxels.
    """
    if mask is not None:
        vol = volume * mask
    else:
        vol = volume

    fft = np.fft.fftn(vol)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2

    d, h, w = vol.shape
    cd, ch, cw = d // 2, h // 2, w // 2
    dz, dy, dx = np.ogrid[-cd:d - cd, -ch:h - ch, -cw:w - cw]
    radius = np.sqrt((dz / d) ** 2 + (dy / h) ** 2 + (dx / w) ** 2)

    max_radius = 0.5
    num_bins = min(d, h, w) // 2
    bin_edges = np.linspace(0, max_radius, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    radial_power = np.zeros(num_bins)
    for i in range(num_bins):
        ring = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
        if ring.sum() > 0:
            radial_power[i] = power[ring].mean()

    return bin_centers, radial_power


def compute_2d_fft_amplitude(slice_2d: np.ndarray) -> np.ndarray:
    """Compute log-amplitude spectrum of a 2D slice."""
    fft = np.fft.fft2(slice_2d)
    fft_shift = np.fft.fftshift(fft)
    amplitude = np.abs(fft_shift)
    # Log scale for visualization (avoid log(0))
    return np.log1p(amplitude)


# ── Visualizations ──────────────────────────────────────────────────

def plot_2d_fft_maps(
    real_vols: list[np.ndarray],
    gen_vols: list[np.ndarray],
    output_dir: Path,
    n_examples: int = 3,
) -> None:
    """Show 2D FFT amplitude maps for central axial slices."""
    for idx in range(min(n_examples, len(real_vols), len(gen_vols))):
        real = real_vols[idx]
        gen = gen_vols[idx]
        s = real.shape[0] // 2  # central axial slice

        real_slice = real[s]
        gen_slice = gen[s]

        amp_real = compute_2d_fft_amplitude(real_slice)
        amp_gen = compute_2d_fft_amplitude(gen_slice)

        # Ratio (where is generated weaker?)
        # Use raw amplitudes for ratio, not log
        raw_real = np.abs(np.fft.fftshift(np.fft.fft2(real_slice)))
        raw_gen = np.abs(np.fft.fftshift(np.fft.fft2(gen_slice)))
        ratio = np.where(raw_real > 1e-10, raw_gen / raw_real, 1.0)
        ratio = np.clip(ratio, 0, 2)

        fig, axes = plt.subplots(2, 4, figsize=(22, 11))

        # Top row: spatial domain
        axes[0, 0].imshow(real_slice, cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title('Real (spatial)', fontsize=12)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(gen_slice, cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title('Generated (spatial)', fontsize=12)
        axes[0, 1].axis('off')

        # Difference in spatial domain
        diff = real_slice - gen_slice
        vmax_diff = np.abs(diff).max()
        axes[0, 2].imshow(diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
        axes[0, 2].set_title(f'Real - Generated (L1={np.abs(diff).mean():.4f})', fontsize=12)
        axes[0, 2].axis('off')

        # Zoomed center crop
        h, w = real_slice.shape
        ch, cw = h // 2, w // 2
        crop = h // 4
        axes[0, 3].imshow(real_slice[ch-crop:ch+crop, cw-crop:cw+crop], cmap='gray', vmin=0, vmax=1)
        axes[0, 3].set_title('Real (zoomed center)', fontsize=12)
        axes[0, 3].axis('off')

        # Bottom row: frequency domain
        vmax_amp = max(amp_real.max(), amp_gen.max())
        axes[1, 0].imshow(amp_real, cmap='hot', vmin=0, vmax=vmax_amp)
        axes[1, 0].set_title('Real FFT (log amplitude)', fontsize=12)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(amp_gen, cmap='hot', vmin=0, vmax=vmax_amp)
        axes[1, 1].set_title('Generated FFT (log amplitude)', fontsize=12)
        axes[1, 1].axis('off')

        # Amplitude difference
        amp_diff = amp_real - amp_gen
        vmax_ad = np.abs(amp_diff).max()
        axes[1, 2].imshow(amp_diff, cmap='RdBu_r', vmin=-vmax_ad, vmax=vmax_ad)
        axes[1, 2].set_title('FFT amplitude difference (real - gen)', fontsize=12)
        axes[1, 2].axis('off')

        # Ratio map
        im = axes[1, 3].imshow(ratio, cmap='RdYlGn', vmin=0, vmax=2)
        axes[1, 3].set_title('Amplitude ratio (gen/real, 1.0=equal)', fontsize=12)
        axes[1, 3].axis('off')
        plt.colorbar(im, ax=axes[1, 3], shrink=0.8)

        plt.suptitle(f'FFT Comparison — Volume {idx} — Axial Slice {s}', fontsize=14)
        plt.tight_layout()
        plt.savefig(str(output_dir / f'fft_2d_comparison_{idx:02d}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved 2D FFT comparison {idx}")


def plot_radial_spectra(
    real_vols: list[np.ndarray],
    gen_vols: list[np.ndarray],
    output_dir: Path,
) -> None:
    """Plot radial power spectra: unmasked vs brain-masked."""
    logger.info("Computing unmasked spectra...")
    all_real_unmasked = []
    all_gen_unmasked = []
    for vol in real_vols:
        _, psd = compute_masked_radial_spectrum_3d(vol, mask=None)
        all_real_unmasked.append(psd)
    for vol in gen_vols:
        _, psd = compute_masked_radial_spectrum_3d(vol, mask=None)
        all_gen_unmasked.append(psd)

    logger.info("Computing brain-masked spectra...")
    all_real_masked = []
    all_gen_masked = []
    for vol in real_vols:
        mask = compute_brain_mask(vol)
        _, psd = compute_masked_radial_spectrum_3d(vol, mask=mask)
        all_real_masked.append(psd)
    for vol in gen_vols:
        mask = compute_brain_mask(vol)
        _, psd = compute_masked_radial_spectrum_3d(vol, mask=mask)
        all_gen_masked.append(psd)

    freqs, _ = compute_masked_radial_spectrum_3d(real_vols[0])

    avg_real_u = np.mean(all_real_unmasked, axis=0)
    avg_gen_u = np.mean(all_gen_unmasked, axis=0)
    avg_real_m = np.mean(all_real_masked, axis=0)
    avg_gen_m = np.mean(all_gen_masked, axis=0)

    # Also compute std for confidence bands
    std_real_m = np.std(all_real_masked, axis=0)
    std_gen_m = np.std(all_gen_masked, axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # ── Row 1: Unmasked (full volume) ──
    ax = axes[0, 0]
    ax.semilogy(freqs, avg_real_u, 'b-', label='Real', linewidth=2)
    ax.semilogy(freqs, avg_gen_u, 'r-', label='Generated', linewidth=2)
    ax.set_title('Unmasked — Power Spectra', fontsize=12)
    ax.set_xlabel('Radial Frequency')
    ax.set_ylabel('PSD (log)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ratio_u = np.where(avg_real_u > 1e-20, avg_gen_u / avg_real_u, 1.0)
    ax.plot(freqs, ratio_u, 'k-', linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('Unmasked — PSD Ratio (gen/real)', fontsize=12)
    ax.set_xlabel('Radial Frequency')
    ax.set_ylabel('Ratio')
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    h_u = np.sqrt(np.clip(ratio_u, 0, 1))
    ax.plot(freqs, h_u, 'g-', linewidth=2, label='H(f) unmasked')
    ax.set_title('Unmasked — Transfer Function', fontsize=12)
    ax.set_xlabel('Radial Frequency')
    ax.set_ylabel('H(f)')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Row 2: Brain-masked ──
    ax = axes[1, 0]
    ax.semilogy(freqs, avg_real_m, 'b-', label='Real (masked)', linewidth=2)
    ax.fill_between(freqs,
                     np.maximum(avg_real_m - std_real_m, 1e-20),
                     avg_real_m + std_real_m, alpha=0.2, color='blue')
    ax.semilogy(freqs, avg_gen_m, 'r-', label='Generated (masked)', linewidth=2)
    ax.fill_between(freqs,
                     np.maximum(avg_gen_m - std_gen_m, 1e-20),
                     avg_gen_m + std_gen_m, alpha=0.2, color='red')
    ax.set_title('Brain-Masked — Power Spectra', fontsize=12)
    ax.set_xlabel('Radial Frequency')
    ax.set_ylabel('PSD (log)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ratio_m = np.where(avg_real_m > 1e-20, avg_gen_m / avg_real_m, 1.0)
    ax.plot(freqs, ratio_m, 'k-', linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('Brain-Masked — PSD Ratio (gen/real)', fontsize=12)
    ax.set_xlabel('Radial Frequency')
    ax.set_ylabel('Ratio')
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    h_m = np.sqrt(np.clip(ratio_m, 0, 1))
    ax.plot(freqs, h_u, 'g--', linewidth=1.5, label='H(f) unmasked', alpha=0.6)
    ax.plot(freqs, h_m, 'g-', linewidth=2, label='H(f) brain-masked')
    ax.set_title('Transfer Functions — Unmasked vs Masked', fontsize=12)
    ax.set_xlabel('Radial Frequency')
    ax.set_ylabel('H(f)')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Radial Power Spectrum: Unmasked (top) vs Brain-Masked (bottom)', fontsize=14)
    plt.tight_layout()
    plt.savefig(str(output_dir / 'radial_spectra_masked_vs_unmasked.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved radial spectra comparison")

    # Save the transfer functions
    np.savez(
        str(output_dir / 'transfer_functions_masked.npz'),
        freqs=freqs,
        h_unmasked=h_u,
        h_brain_masked=h_m,
        psd_real_unmasked=avg_real_u,
        psd_gen_unmasked=avg_gen_u,
        psd_real_masked=avg_real_m,
        psd_gen_masked=avg_gen_m,
        ratio_unmasked=ratio_u,
        ratio_masked=ratio_m,
    )
    logger.info("Saved transfer functions to transfer_functions_masked.npz")


def plot_per_band_comparison(
    real_vols: list[np.ndarray],
    gen_vols: list[np.ndarray],
    output_dir: Path,
    n_bands: int = 8,
) -> None:
    """Show amplitude in frequency bands as bar chart."""
    freqs, _ = compute_masked_radial_spectrum_3d(real_vols[0])
    n_freq = len(freqs)
    band_size = n_freq // n_bands

    real_bands = np.zeros(n_bands)
    gen_bands = np.zeros(n_bands)

    for vol in real_vols:
        mask = compute_brain_mask(vol)
        _, psd = compute_masked_radial_spectrum_3d(vol, mask=mask)
        for b in range(n_bands):
            start = b * band_size
            end = min((b + 1) * band_size, n_freq)
            real_bands[b] += psd[start:end].mean()

    for vol in gen_vols:
        mask = compute_brain_mask(vol)
        _, psd = compute_masked_radial_spectrum_3d(vol, mask=mask)
        for b in range(n_bands):
            start = b * band_size
            end = min((b + 1) * band_size, n_freq)
            gen_bands[b] += psd[start:end].mean()

    real_bands /= len(real_vols)
    gen_bands /= len(gen_vols)

    band_labels = [f'{freqs[b*band_size]:.3f}-{freqs[min((b+1)*band_size-1, n_freq-1)]:.3f}'
                   for b in range(n_bands)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(n_bands)
    width = 0.35

    ax = axes[0]
    ax.bar(x - width/2, np.log10(real_bands + 1e-20), width, label='Real', color='blue', alpha=0.7)
    ax.bar(x + width/2, np.log10(gen_bands + 1e-20), width, label='Generated', color='red', alpha=0.7)
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('log₁₀(Mean PSD)')
    ax.set_title('Per-Band Power (brain-masked)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, rotation=45, ha='right', fontsize=9)
    ax.legend()

    ax = axes[1]
    ratio = gen_bands / np.maximum(real_bands, 1e-20)
    colors = ['green' if r > 0.9 else 'orange' if r > 0.7 else 'red' for r in ratio]
    ax.bar(x, ratio, color=colors, alpha=0.8)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Generated / Real')
    ax.set_title('Per-Band Power Ratio (1.0 = perfect match)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.5)

    for i, r in enumerate(ratio):
        ax.text(i, r + 0.02, f'{r:.2f}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(str(output_dir / 'per_band_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved per-band comparison")


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FFT amplitude comparison: real vs generated")
    parser.add_argument("--real-dir", type=str, required=True)
    parser.add_argument("--generated-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-volumes", type=int, default=10)
    parser.add_argument("--depth", type=int, default=160)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading real volumes from {args.real_dir}...")
    real_vols = load_real_volumes(Path(args.real_dir), args.depth, args.num_volumes)
    logger.info(f"Loaded {len(real_vols)} real volumes")

    logger.info(f"Loading generated volumes from {args.generated_dir}...")
    gen_vols = load_generated_volumes(Path(args.generated_dir), args.depth, args.num_volumes)
    logger.info(f"Loaded {len(gen_vols)} generated volumes")

    # 1. 2D FFT amplitude maps
    logger.info("=== 2D FFT Amplitude Maps ===")
    plot_2d_fft_maps(real_vols, gen_vols, output_dir, n_examples=5)

    # 2. Radial spectra: unmasked vs brain-masked
    logger.info("=== Radial Power Spectra (unmasked vs brain-masked) ===")
    plot_radial_spectra(real_vols, gen_vols, output_dir)

    # 3. Per-band bar chart
    logger.info("=== Per-Band Comparison ===")
    plot_per_band_comparison(real_vols, gen_vols, output_dir, n_bands=8)

    logger.info(f"All results saved to {output_dir}")


if __name__ == '__main__':
    main()
