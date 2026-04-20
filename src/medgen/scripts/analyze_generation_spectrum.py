#!/usr/bin/env python3
"""Radial 3D power-spectrum analysis across fine-tuned generators (Phase 1 #1).

Compares generated brain MRI volumes from multiple fine-tuned models against
real (held-out) reference volumes in the frequency domain. Produces:

  Figure 1: Overlay of radially averaged power spectra (log-log), one curve
            per model + one for real. Direct visual answer to "which model
            recovers which frequency band?".

  Figure 2: Ratio to real per model (model_spectrum / real_spectrum). Shows
            over/under-shoot per band in a single plot.

  Figure 3: Band-energy bars. Energy integrated over six named bands
            (very_low / low / low_mid / mid / high / very_high) per model.

  Figure 4: Absolute deficit vs real — |real - model| per band. Highlights
            which bands each method improved the most.

  JSON:     All numeric values — per-model mean spectrum, band energies,
            deficits, ratios. For thesis tables.

Usage:
    python -m medgen.scripts.analyze_generation_spectrum \\
        --compare-dir /path/to/generated/compare_exp37_20260420-165132 \\
        --real-dir    /path/to/brainmetshare-3/test1 \\
        --output-dir  runs/eval/spectrum_analysis \\
        --max-volumes 10
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# Frequency band definitions. Radial-frequency ranges in units of Nyquist (0.5).
# Chosen to separate "coarse structure" from "vessel-scale detail":
BANDS: dict[str, tuple[float, float]] = {
    'very_low':  (0.00, 0.05),   # Overall brain shape, coarse structure
    'low':       (0.05, 0.10),   # Ventricle edges, cortex envelope
    'low_mid':   (0.10, 0.20),   # Large-scale parenchyma variation
    'mid':       (0.20, 0.30),   # Parenchyma texture grain
    'high':      (0.30, 0.40),   # Fine detail, small-scale features
    'very_high': (0.40, 0.50),   # Vessels, noise-scale structure (Nyquist/4 → Nyquist)
}

# Default model colors — consistent palette for 7 models
MODEL_COLORS: dict[str, str] = {
    'real':         'black',
    'exp1_1_1000':  'gray',
    'exp32_1_1000': 'tab:blue',
    'exp32_2_1000': 'tab:cyan',
    'exp32_3_1000': 'tab:purple',
    'exp37_1':      'tab:orange',
    'exp37_2':      'tab:red',
    'exp37_3':      'tab:brown',
}


def load_volume(path: Path, depth: int = 160) -> np.ndarray:
    """Load NIfTI -> [D, H, W] numpy float32 normalized to [0, 1]."""
    vol = nib.load(str(path)).get_fdata().astype(np.float32)
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    vol = np.transpose(vol, (2, 0, 1))  # [H, W, D] -> [D, H, W]
    d = vol.shape[0]
    if d < depth:
        vol = np.pad(vol, ((0, depth - d), (0, 0), (0, 0)))
    elif d > depth:
        vol = vol[:depth]
    return vol


def find_bravo_files(root: Path, max_volumes: int) -> list[Path]:
    """Collect up to `max_volumes` bravo.nii.gz files from root's subdirs."""
    files = sorted(root.glob("*/bravo.nii.gz"))
    if not files:
        files = sorted(root.glob("*.nii.gz"))
    return files[:max_volumes]


def compute_radial_power_spectrum_3d(
    volume: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute radially averaged 3D power spectrum.

    Returns (bin_centers, radial_power) both shape (n_bins,).
    Radial frequency is normalized such that Nyquist = 0.5.
    """
    fft = np.fft.fftn(volume)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2

    d, h, w = volume.shape
    cd, ch, cw = d // 2, h // 2, w // 2
    dz, dy, dx = np.ogrid[-cd:d - cd, -ch:h - ch, -cw:w - cw]
    radius = np.sqrt((dz / d) ** 2 + (dy / h) ** 2 + (dx / w) ** 2)

    max_radius = 0.5  # Nyquist
    num_bins = min(d, h, w) // 2
    bin_edges = np.linspace(0, max_radius, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    radial_power = np.zeros(num_bins)
    for i in range(num_bins):
        mask = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
        if mask.sum() > 0:
            radial_power[i] = power[mask].mean()

    return bin_centers, radial_power


def compute_mean_spectrum(files: list[Path], depth: int) -> tuple[np.ndarray, np.ndarray]:
    """Average the radial power spectrum across a list of volume files."""
    spectra = []
    freqs = None
    for i, fp in enumerate(files, start=1):
        vol = load_volume(fp, depth=depth)
        freqs, power = compute_radial_power_spectrum_3d(vol)
        spectra.append(power)
        if i % 5 == 0:
            logger.info(f"    loaded {i}/{len(files)}")
    mean = np.mean(spectra, axis=0)
    return freqs, mean


def band_energy(freqs: np.ndarray, power: np.ndarray,
                band: tuple[float, float]) -> float:
    """Integrate power (with frequency weighting 4π r² dr) over a band."""
    lo, hi = band
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return 0.0
    # 3D shell volume weighting: integrate r² × power(r) dr
    r = freqs[mask]
    p = power[mask]
    dr = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    return float(np.sum(4.0 * np.pi * r * r * p) * dr)


def plot_spectra_overlay(data: dict, output_base: Path) -> None:
    """Figure 1: radial spectrum overlay for all models + real."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, d in data.items():
        color = MODEL_COLORS.get(name)
        lw = 2.5 if name == 'real' else 1.4
        ax.loglog(d['freqs'], d['power'], label=name, color=color,
                  linewidth=lw, linestyle='-', alpha=0.85)

    # Shade named bands lightly
    ymin, ymax = ax.get_ylim()
    for _bname, (lo, hi) in BANDS.items():
        ax.axvspan(lo, hi, alpha=0.04, color='gray')
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel('radial frequency (Nyquist = 0.5)')
    ax.set_ylabel('mean radial power |FFT|²')
    ax.set_title('Radial 3D power spectrum: real vs fine-tuned generators')
    ax.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', fontsize=9, ncol=2)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_ratio_to_real(data: dict, output_base: Path) -> None:
    """Figure 2: per-model spectrum ratio vs real (deviation from 1.0 = band
    where the model is under/over-shooting real)."""
    if 'real' not in data:
        logger.warning("  real not in data — skipping ratio plot")
        return
    real_power = data['real']['power']
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, d in data.items():
        if name == 'real':
            continue
        color = MODEL_COLORS.get(name)
        ratio = np.where(real_power > 0, d['power'] / real_power, 1.0)
        ax.semilogx(d['freqs'], ratio, label=name, color=color,
                    linewidth=1.5, alpha=0.85)

    ax.axhline(1.0, color='black', linewidth=1.0, linestyle='--', alpha=0.6,
               label='real (ratio=1)')
    # Shade bands
    for _bname, (lo, hi) in BANDS.items():
        ax.axvspan(lo, hi, alpha=0.04, color='gray')

    ax.set_xlabel('radial frequency')
    ax.set_ylabel('model power / real power')
    ax.set_title('Spectrum ratio to real — <1 = deficit, >1 = overshoot')
    ax.set_xlim(data['real']['freqs'][0], 0.5)
    ax.set_ylim(0, 2.0)
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', fontsize=9, ncol=2)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_band_energy(data: dict, output_base: Path) -> None:
    """Figure 3: bar chart of band-integrated energy per model."""
    band_names = list(BANDS.keys())
    model_names = list(data.keys())
    energies = np.zeros((len(model_names), len(band_names)))
    for i, m in enumerate(model_names):
        for j, bn in enumerate(band_names):
            energies[i, j] = band_energy(data[m]['freqs'], data[m]['power'], BANDS[bn])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(band_names))
    width = 0.9 / max(len(model_names), 1)
    for i, m in enumerate(model_names):
        color = MODEL_COLORS.get(m)
        ax.bar(x + (i - len(model_names) / 2 + 0.5) * width, energies[i],
               width=width, label=m, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(band_names)
    ax.set_ylabel('band-integrated energy (4π∫r²|F|² dr)')
    ax.set_yscale('log')
    ax.set_title('Band-integrated power per model')
    ax.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=9, ncol=2)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_band_ratio_bars(data: dict, output_base: Path) -> None:
    """Figure 4: ratio of each model's band energy to real's band energy.

    Direct visual answer to 'which model fixed which band?'. A bar at 1.0 means
    the model matches real in that band; <1 is deficit, >1 is overshoot.
    """
    if 'real' not in data:
        return
    band_names = list(BANDS.keys())
    real_energies = np.array([
        band_energy(data['real']['freqs'], data['real']['power'], BANDS[bn])
        for bn in band_names
    ])

    model_names = [m for m in data.keys() if m != 'real']
    ratios = np.zeros((len(model_names), len(band_names)))
    for i, m in enumerate(model_names):
        for j, bn in enumerate(band_names):
            e = band_energy(data[m]['freqs'], data[m]['power'], BANDS[bn])
            ratios[i, j] = e / real_energies[j] if real_energies[j] > 0 else 0.0

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(band_names))
    width = 0.9 / max(len(model_names), 1)
    for i, m in enumerate(model_names):
        color = MODEL_COLORS.get(m)
        ax.bar(x + (i - len(model_names) / 2 + 0.5) * width, ratios[i],
               width=width, label=m, color=color, alpha=0.85)

    ax.axhline(1.0, color='black', linewidth=1.0, linestyle='--', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(band_names)
    ax.set_ylabel('model_energy / real_energy (per band)')
    ax.set_title('Per-band energy ratio to real — 1.0 is target')
    ax.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=9, ncol=2)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def print_band_table(data: dict) -> None:
    """Print band-energy + ratio-to-real table to stdout."""
    if 'real' not in data:
        return
    band_names = list(BANDS.keys())
    real_e = [band_energy(data['real']['freqs'], data['real']['power'], BANDS[bn])
              for bn in band_names]

    print()
    print("=" * 92)
    print("Band-integrated energy (× 1e6) and ratio-to-real per model")
    print("=" * 92)
    header = f"{'model':<16}" + " ".join(f"{bn:>11}" for bn in band_names)
    print(header)
    print("-" * len(header))
    # Real row
    row = f"{'real':<16}" + " ".join(f"{e * 1e-6:>11.3f}" for e in real_e)
    print(row + "    (reference)")
    print("-" * len(header))
    # Other models (ratio)
    for m in data.keys():
        if m == 'real':
            continue
        row = f"{m:<16}"
        for j, bn in enumerate(band_names):
            e = band_energy(data[m]['freqs'], data[m]['power'], BANDS[bn])
            ratio = e / real_e[j] if real_e[j] > 0 else 0.0
            row += f"  {ratio:>9.2%}"
        print(row)
    print("=" * 92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.1 — Radial spectrum analysis")
    parser.add_argument('--compare-dir', required=True,
                        help='Directory containing one subdir per model (e.g. compare_exp37_*)')
    parser.add_argument('--real-dir', default='/home/mode/NTNU/MedicalDataSets/brainmetshare-3/test1',
                        help='Directory of reference real volumes')
    parser.add_argument('--output-dir', default='runs/eval/spectrum_analysis')
    parser.add_argument('--max-volumes', type=int, default=10,
                        help='Max volumes per model (default 10 — matches compare_dir size)')
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--thesis-dir',
                        default='/home/mode/NTNU/AIS4900_doc/AIS4900-master-thesis/Images/spectrum_analysis',
                        help='Optional thesis Images directory to copy PNGs/PDFs into (empty to disable)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover model subdirs
    compare_root = Path(args.compare_dir)
    if not compare_root.is_dir():
        raise SystemExit(f"compare-dir not found: {compare_root}")
    model_dirs = sorted([p for p in compare_root.iterdir() if p.is_dir()])
    logger.info(f"Found {len(model_dirs)} models in {compare_root.name}: "
                + ", ".join(p.name for p in model_dirs))

    data: dict[str, dict] = {}

    # Real reference
    real_root = Path(args.real_dir)
    logger.info(f"Loading real reference from {real_root}")
    real_files = find_bravo_files(real_root, args.max_volumes)
    if not real_files:
        raise SystemExit(f"No bravo.nii.gz in {real_root}")
    logger.info(f"  real: {len(real_files)} volumes")
    freqs, power = compute_mean_spectrum(real_files, args.depth)
    data['real'] = {'freqs': freqs, 'power': power, 'n_volumes': len(real_files)}

    # Each generated model
    for md in model_dirs:
        logger.info(f"Loading {md.name}")
        files = find_bravo_files(md, args.max_volumes)
        if not files:
            logger.warning(f"  no volumes in {md} — skipping")
            continue
        logger.info(f"  {md.name}: {len(files)} volumes")
        freqs_m, power_m = compute_mean_spectrum(files, args.depth)
        data[md.name] = {'freqs': freqs_m, 'power': power_m, 'n_volumes': len(files)}

    # Plot
    logger.info("Plotting")
    plot_spectra_overlay(data, output_dir / 'spectrum_overlay')
    plot_ratio_to_real(data, output_dir / 'spectrum_ratio_to_real')
    plot_band_energy(data, output_dir / 'band_energy')
    plot_band_ratio_bars(data, output_dir / 'band_ratio_to_real')

    # Text table
    print_band_table(data)

    # JSON with numeric results
    summary = {
        'compare_dir': str(compare_root),
        'real_dir': str(real_root),
        'max_volumes': args.max_volumes,
        'bands': {k: list(v) for k, v in BANDS.items()},
        'models': {},
    }
    for name, d in data.items():
        bands_e = {bn: band_energy(d['freqs'], d['power'], BANDS[bn]) for bn in BANDS}
        ratios = {}
        if name != 'real' and 'real' in data:
            real_e = {bn: band_energy(data['real']['freqs'], data['real']['power'], BANDS[bn])
                      for bn in BANDS}
            ratios = {bn: (bands_e[bn] / real_e[bn] if real_e[bn] > 0 else 0.0) for bn in BANDS}
        summary['models'][name] = {
            'n_volumes': d['n_volumes'],
            'freqs': d['freqs'].tolist(),
            'power': d['power'].tolist(),
            'band_energies': bands_e,
            'band_ratios_to_real': ratios,
        }
    with open(output_dir / 'spectrum_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {output_dir / 'spectrum_results.json'}")

    # Optional thesis copy
    if args.thesis_dir:
        thesis_dir = Path(args.thesis_dir)
        thesis_dir.mkdir(parents=True, exist_ok=True)
        for name in ('spectrum_overlay', 'spectrum_ratio_to_real',
                     'band_energy', 'band_ratio_to_real'):
            for ext in ('png', 'pdf'):
                src = output_dir / f'{name}.{ext}'
                if src.exists():
                    (thesis_dir / f'{name}.{ext}').write_bytes(src.read_bytes())
        logger.info(f"Copied figures to thesis dir: {thesis_dir}")


if __name__ == '__main__':
    main()
