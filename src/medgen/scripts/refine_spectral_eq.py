#!/usr/bin/env python3
"""Spectral equalization refinement (Wiener-style) — T1B of the blur-attack plan.

Computes the radial gain `gain(f) = sqrt(P_real(f) / P_synth(f))` from a set
of real volumes and a set of synthetic volumes, then applies it as a per-volume
Fourier filter to each synthetic volume:

    FFT(synth) → multiply by gain at each frequency's radial bin → IFFT

This is amplitude-only correction — phases (= spatial structure) of synth are
preserved. No real-data transplant at inference, no anatomical mismatch.

Sweeps `--gain-clip-max` to control noise amplification at frequencies where
synth has near-zero energy but real has lots.

Usage:
    python -m medgen.scripts.refine_spectral_eq \\
        --synth-dirs <dir1> <dir2> \\
        --real-dir   /path/to/test_new \\
        --output-dir runs/eval/spectral_eq_$(date +%Y%m%d-%H%M%S) \\
        --gain-clip-maxes 2.0 3.0 5.0
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from medgen.data.utils import save_nifti
from medgen.metrics.quality import compute_lpips_3d
from medgen.scripts.analyze_generation_spectrum import (
    BANDS,
    compute_radial_power_spectrum_3d,
    load_volume,
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def find_volumes(root: Path, n: int | None = None) -> list[Path]:
    files = sorted(root.glob("*/bravo.nii.gz"))
    if not files:
        files = sorted(root.glob("*.nii.gz"))
    if not files:
        raise SystemExit(f"No NIfTI files under {root}")
    return files[:n] if n else files


def average_spectrum(volumes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    bins_ref, accum = None, None
    for v in volumes:
        bins, power = compute_radial_power_spectrum_3d(v)
        if accum is None:
            bins_ref, accum = bins, np.zeros_like(power)
        accum = accum + power
    return bins_ref, accum / max(1, len(volumes))


def make_radial_index_map(d: int, h: int, w: int, n_bins: int) -> np.ndarray:
    """Per-voxel radial-bin index in the FFT-shifted spectrum (Nyquist=0.5 normalization)."""
    cd, ch, cw = d // 2, h // 2, w // 2
    dz, dy, dx = np.ogrid[-cd:d - cd, -ch:h - ch, -cw:w - cw]
    radius = np.sqrt((dz / d) ** 2 + (dy / h) ** 2 + (dx / w) ** 2)
    # bin edges 0..0.5
    bin_idx = np.clip((radius / 0.5 * n_bins).astype(np.int32), 0, n_bins - 1)
    return bin_idx


def smooth(arr: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return arr
    pad = window // 2
    padded = np.pad(arr, pad, mode='edge')
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode='valid')


def apply_spectral_eq(volume: np.ndarray, gain_per_bin: np.ndarray, bin_idx_map: np.ndarray) -> np.ndarray:
    """volume: [D,H,W] in [0,1]. gain_per_bin: [n_bins]. bin_idx_map: [D,H,W] int."""
    fft = np.fft.fftshift(np.fft.fftn(volume))
    gain_3d = gain_per_bin[bin_idx_map]
    fft_eq = fft * gain_3d
    out = np.real(np.fft.ifftn(np.fft.ifftshift(fft_eq)))
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def band_energy(bins: np.ndarray, power: np.ndarray) -> dict[str, float]:
    out = {}
    for name, (lo, hi) in BANDS.items():
        mask = (bins >= lo) & (bins < hi)
        out[name] = float(power[mask].sum()) if mask.any() else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--synth-dirs', nargs='+', required=True,
                        help='One or more synth dirs to refine')
    parser.add_argument('--real-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--gain-clip-maxes', type=float, nargs='+',
                        default=[2.0, 3.0, 5.0])
    parser.add_argument('--num-synth', type=int, default=10,
                        help='How many synth volumes to refine per dir')
    parser.add_argument('--num-real', type=int, default=20,
                        help='Real volumes used to compute reference spectrum')
    parser.add_argument('--num-real-lpips', type=int, default=5,
                        help='Real volumes used in cross-pair LPIPS evaluation')
    parser.add_argument('--gain-smooth-window', type=int, default=3,
                        help='Moving-average smoothing of gain curve (bins)')
    parser.add_argument('--depth', type=int, default=160)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Real reference spectrum ───────────────────────────────────
    real_files = find_volumes(Path(args.real_dir), args.num_real)
    log.info(f"Real reference: {len(real_files)} volumes")
    real_np = [load_volume(f, args.depth) for f in real_files]
    bins, real_spec = average_spectrum(real_np)
    n_bins = len(bins)

    # Volumes for LPIPS evaluation (cross pairs)
    real_lpips_np = real_np[:args.num_real_lpips]
    real_lpips_t = torch.cat(
        [torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(device).float() for a in real_lpips_np],
        dim=0,
    )

    # ── Per synth dir ─────────────────────────────────────────────
    results = {
        'real_dir': str(Path(args.real_dir).resolve()),
        'gain_clip_maxes': args.gain_clip_maxes,
        'gain_smooth_window': args.gain_smooth_window,
        'n_real_for_spectrum': len(real_np),
        'n_real_for_lpips': len(real_lpips_np),
        'per_dir': {},
    }

    for synth_dir_str in args.synth_dirs:
        synth_dir = Path(synth_dir_str)
        tag = synth_dir.name
        log.info(f"\n========== {tag} ==========")
        synth_files = find_volumes(synth_dir, args.num_synth)
        log.info(f"Synth: {len(synth_files)} volumes")
        synth_np = [load_volume(f, args.depth) for f in synth_files]
        _, synth_spec = average_spectrum(synth_np)
        synth_t = torch.cat(
            [torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(device).float() for a in synth_np],
            dim=0,
        )

        # Baseline LPIPS(real, synth) cross-pairs
        baseline_vals = []
        for i in range(real_lpips_t.shape[0]):
            for j in range(synth_t.shape[0]):
                baseline_vals.append(float(compute_lpips_3d(
                    real_lpips_t[i:i + 1], synth_t[j:j + 1], device=device, chunk_size=32,
                )))
        baseline_lpips = float(np.mean(baseline_vals))
        log.info(f"Baseline LPIPS(real, synth) = {baseline_lpips:.4f}")

        # Gain curve (unclipped)
        eps = 1e-8
        raw_gain = np.sqrt(np.maximum(real_spec, eps) / np.maximum(synth_spec, eps))
        smoothed_gain = smooth(raw_gain, window=args.gain_smooth_window)

        # Pre-compute the radial-bin index map (same for all volumes of fixed shape)
        d, h, w = synth_np[0].shape
        bin_idx_map = make_radial_index_map(d, h, w, n_bins)

        per_clip = {}
        for clip_max in args.gain_clip_maxes:
            log.info(f"\n  --- gain clip-max = {clip_max} ---")
            gain_clipped = np.clip(smoothed_gain, 0.0, clip_max)

            sub_dir = out_dir / tag / f"clip_{clip_max:.1f}"
            mixed_dir = sub_dir / 'volumes'
            mixed_dir.mkdir(parents=True, exist_ok=True)

            mixed_np: list[np.ndarray] = []
            for arr, fp in zip(synth_np, synth_files):
                mixed = apply_spectral_eq(arr, gain_clipped, bin_idx_map)
                mixed_np.append(mixed)
                subj = fp.parent.name
                subj_dir = mixed_dir / subj
                subj_dir.mkdir(parents=True, exist_ok=True)
                save_nifti(np.transpose(mixed, (1, 2, 0)), str(subj_dir / 'bravo.nii.gz'))

            _, mixed_spec = average_spectrum(mixed_np)

            mixed_t = torch.cat(
                [torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(device).float() for a in mixed_np],
                dim=0,
            )
            lpips_vals = []
            for i in range(real_lpips_t.shape[0]):
                for j in range(mixed_t.shape[0]):
                    lpips_vals.append(float(compute_lpips_3d(
                        real_lpips_t[i:i + 1], mixed_t[j:j + 1], device=device, chunk_size=32,
                    )))
            mixed_lpips = float(np.mean(lpips_vals))
            delta_pct = 100 * (mixed_lpips - baseline_lpips) / baseline_lpips

            # Band ratios
            bands_real = band_energy(bins, real_spec)
            bands_synth = band_energy(bins, synth_spec)
            bands_mixed = band_energy(bins, mixed_spec)
            verdict_rows = []
            for b in BANDS:
                s = bands_synth[b] / bands_real[b] if bands_real[b] else float('nan')
                m = bands_mixed[b] / bands_real[b] if bands_real[b] else float('nan')
                verdict_rows.append(f"  {b:10s}  synth/real={s:6.3f}  eq/real={m:6.3f}  Δ={m - s:+.3f}")
            log.info(f"  LPIPS  baseline={baseline_lpips:.4f}  eq={mixed_lpips:.4f}  Δ={delta_pct:+.2f}%")
            log.info("  " + "\n  ".join(verdict_rows))

            # Spectrum overlay plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            for name, spec, color in [
                ('real', real_spec, 'black'),
                ('synth', synth_spec, 'gray'),
                (f'eq (clip={clip_max})', mixed_spec, 'tab:red'),
            ]:
                ax1.loglog(bins[1:], spec[1:], label=name, color=color, linewidth=2)
            ax1.set_xlabel('Radial frequency (Nyquist=0.5)')
            ax1.set_ylabel('Power')
            ax1.set_title(f'Spectrum (clip={clip_max})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.semilogx(bins[1:], synth_spec[1:] / real_spec[1:],
                         label='synth / real', color='gray', linewidth=2)
            ax2.semilogx(bins[1:], mixed_spec[1:] / real_spec[1:],
                         label='eq / real', color='tab:red', linewidth=2)
            ax2.semilogx(bins[1:], gain_clipped[1:],
                         label=f'gain (clip={clip_max})', color='tab:blue',
                         linewidth=1, linestyle='--')
            ax2.axhline(1.0, color='black', linestyle=':', alpha=0.5)
            ax2.set_xlabel('Radial frequency (Nyquist=0.5)')
            ax2.set_ylabel('Ratio / Gain')
            ax2.set_title('Ratio to real (1.0=matched) + gain curve')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(sub_dir / 'spectrum.png', dpi=120)
            plt.close(fig)

            per_clip[f'{clip_max}'] = {
                'gain_clip_max': clip_max,
                'lpips_mean': mixed_lpips,
                'lpips_baseline': baseline_lpips,
                'lpips_delta_pct': delta_pct,
                'band_ratio_synth_to_real': {b: bands_synth[b] / bands_real[b] if bands_real[b] else None for b in BANDS},
                'band_ratio_eq_to_real': {b: bands_mixed[b] / bands_real[b] if bands_real[b] else None for b in BANDS},
                'band_verdict': verdict_rows,
            }

        results['per_dir'][tag] = {
            'synth_dir': str(synth_dir.resolve()),
            'n_synth': len(synth_files),
            'baseline_lpips': baseline_lpips,
            'gain_curve_raw': raw_gain.tolist(),
            'gain_curve_smoothed': smoothed_gain.tolist(),
            'bins': bins.tolist(),
            'per_clip': per_clip,
        }

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved: {out_dir / 'results.json'}")
    log.info(f"Refined volumes: {out_dir}/<dir>/clip_<X.X>/volumes/<subj>/bravo.nii.gz")


if __name__ == '__main__':
    main()
