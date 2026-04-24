#!/usr/bin/env python3
"""Final comparison across blur-attack experiments.

Takes one or more method names → directory pairs, compares each method's
refined volumes against a fixed real reference set on:

  - LPIPS (3D, RadImageNet ResNet50, slice-averaged)
  - L1 mean absolute error vs nearest real (cross-pair mean)
  - Radial PSD slope in f ∈ [0.15, 0.45] (HF region)
  - Per-band energy ratios to real

Outputs:
  <output>/results.csv
  <output>/bar_chart.png      — LPIPS + L1 per method
  <output>/visual_grid.png    — N subjects × M methods, mid-axial slice
  <output>/spectrum_overlay.png — radial PSD per method
  <output>/results.json

Usage:
    python -m medgen.scripts.eval_blur_attack_compare \\
        --methods baseline:<dir> exp42b:<dir> sdedit:<dir> spectral:<dir> \\
                  exp43_irsde:<dir> exp44_fm:<dir> \\
        --real-dir /path/to/test_new \\
        --output-dir runs/eval/blur_attack_$(date +%Y%m%d-%H%M%S) \\
        --num-volumes 10 --num-real 10
"""
import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

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


def parse_methods(items: list[str]) -> list[tuple[str, Path]]:
    out = []
    for it in items:
        if ':' not in it:
            raise SystemExit(f"Bad --methods entry (expect 'name:dir'): {it}")
        name, p = it.split(':', 1)
        d = Path(p)
        if not d.is_dir():
            raise SystemExit(f"Method '{name}' dir does not exist: {d}")
        out.append((name, d))
    return out


def average_spectrum(volumes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    bins_ref, accum = None, None
    for v in volumes:
        bins, power = compute_radial_power_spectrum_3d(v)
        if accum is None:
            bins_ref, accum = bins, np.zeros_like(power)
        accum = accum + power
    return bins_ref, accum / max(1, len(volumes))


def band_energy(bins: np.ndarray, power: np.ndarray) -> dict[str, float]:
    out = {}
    for name, (lo, hi) in BANDS.items():
        mask = (bins >= lo) & (bins < hi)
        out[name] = float(power[mask].sum()) if mask.any() else 0.0
    return out


def hf_psd_slope(bins: np.ndarray, power: np.ndarray, fmin: float, fmax: float) -> float:
    """Linear regression slope of log(power) vs log(f) in the HF window."""
    mask = (bins >= fmin) & (bins <= fmax) & (power > 0)
    if mask.sum() < 3:
        return float('nan')
    x = np.log(bins[mask])
    y = np.log(power[mask])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def cross_lpips(real_t: torch.Tensor, method_t: torch.Tensor, device: torch.device) -> float:
    """Mean LPIPS_3d over all real_i × method_j cross pairs."""
    vals = []
    for i in range(real_t.shape[0]):
        for j in range(method_t.shape[0]):
            vals.append(float(compute_lpips_3d(
                real_t[i:i + 1], method_t[j:j + 1], device=device, chunk_size=32,
            )))
    return float(np.mean(vals)) if vals else float('nan')


def cross_l1(real_arrs: list[np.ndarray], method_arrs: list[np.ndarray]) -> float:
    """Mean L1 over all real × method cross pairs."""
    vals = []
    for r in real_arrs:
        for m in method_arrs:
            vals.append(float(np.mean(np.abs(r - m))))
    return float(np.mean(vals)) if vals else float('nan')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', required=True,
                        help='One or more name:dir pairs')
    parser.add_argument('--real-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-volumes', type=int, default=10,
                        help='Volumes per method to evaluate')
    parser.add_argument('--num-real', type=int, default=10)
    parser.add_argument('--num-real-lpips', type=int, default=5,
                        help='Real volumes used in LPIPS cross-pair (full N×M is expensive)')
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--hf-min', type=float, default=0.15)
    parser.add_argument('--hf-max', type=float, default=0.45)
    parser.add_argument('--grid-subjects', type=int, default=5,
                        help='Number of subjects in the visual grid')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    methods = parse_methods(args.methods)
    log.info(f"Methods: {[n for n, _ in methods]}")

    # ── Real reference ─────────────────────────────────────────────
    real_files = find_volumes(Path(args.real_dir), args.num_real)
    log.info(f"Real reference: {len(real_files)} volumes")
    real_np = [load_volume(f, args.depth) for f in real_files]
    bins, real_spec = average_spectrum(real_np)
    real_bands = band_energy(bins, real_spec)
    real_slope = hf_psd_slope(bins, real_spec, args.hf_min, args.hf_max)
    log.info(f"Real HF PSD slope (f∈[{args.hf_min},{args.hf_max}]): {real_slope:.3f}")

    real_lpips_np = real_np[:args.num_real_lpips]
    real_lpips_t = torch.cat(
        [torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(device).float() for a in real_lpips_np],
        dim=0,
    )

    # ── Per-method metrics ─────────────────────────────────────────
    method_results: dict[str, dict] = {}
    method_arrays: dict[str, list[np.ndarray]] = {}
    method_specs: dict[str, np.ndarray] = {}

    for name, mdir in methods:
        log.info(f"\n========== {name} ({mdir}) ==========")
        files = find_volumes(mdir, args.num_volumes)
        log.info(f"  {len(files)} volumes")
        arrs = [load_volume(f, args.depth) for f in files]
        method_arrays[name] = arrs

        _, spec = average_spectrum(arrs)
        method_specs[name] = spec
        bands = band_energy(bins, spec)
        slope = hf_psd_slope(bins, spec, args.hf_min, args.hf_max)

        m_t = torch.cat(
            [torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(device).float() for a in arrs],
            dim=0,
        )
        lpips = cross_lpips(real_lpips_t, m_t, device)
        l1 = cross_l1(real_np[:args.num_real_lpips], arrs)

        method_results[name] = {
            'dir': str(mdir.resolve()),
            'n_volumes': len(arrs),
            'lpips_vs_real': lpips,
            'l1_vs_real': l1,
            'hf_psd_slope': slope,
            'hf_psd_slope_real': real_slope,
            'slope_diff_to_real': slope - real_slope,
            'band_ratio_to_real': {b: bands[b] / real_bands[b] if real_bands[b] else None for b in BANDS},
        }
        log.info(
            f"  LPIPS={lpips:.4f}  L1={l1:.4f}  slope={slope:.3f} "
            f"(Δreal={slope - real_slope:+.3f})"
        )

    # ── CSV + JSON ─────────────────────────────────────────────────
    csv_path = out_dir / 'results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'lpips', 'l1', 'hf_psd_slope', 'slope_diff_to_real']
                   + [f'band_{b}' for b in BANDS])
        for name in method_results:
            r = method_results[name]
            row = [name, r['lpips_vs_real'], r['l1_vs_real'], r['hf_psd_slope'],
                   r['slope_diff_to_real']]
            row += [r['band_ratio_to_real'][b] if r['band_ratio_to_real'][b] is not None else ''
                    for b in BANDS]
            w.writerow(row)
    log.info(f"\nSaved CSV: {csv_path}")

    summary = {
        'real_dir': str(Path(args.real_dir).resolve()),
        'num_real': len(real_np),
        'num_real_lpips': len(real_lpips_np),
        'hf_window': [args.hf_min, args.hf_max],
        'real_hf_psd_slope': real_slope,
        'real_band_energy': real_bands,
        'methods': method_results,
    }
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Bar chart: LPIPS + L1 ─────────────────────────────────────
    names = list(method_results.keys())
    lpips_vals = [method_results[n]['lpips_vs_real'] for n in names]
    l1_vals = [method_results[n]['l1_vs_real'] for n in names]
    slope_diffs = [abs(method_results[n]['slope_diff_to_real']) for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(names, lpips_vals, color='tab:red')
    axes[0].set_title('LPIPS vs real (lower = better)')
    axes[0].set_ylabel('LPIPS')
    axes[0].tick_params(axis='x', rotation=30)
    axes[1].bar(names, l1_vals, color='tab:blue')
    axes[1].set_title('L1 vs real (lower = better)')
    axes[1].set_ylabel('L1')
    axes[1].tick_params(axis='x', rotation=30)
    axes[2].bar(names, slope_diffs, color='tab:green')
    axes[2].set_title('|Δ HF PSD slope vs real| (lower = better)')
    axes[2].set_ylabel('|Δslope|')
    axes[2].tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / 'bar_chart.png', dpi=120)
    plt.close(fig)

    # ── Spectrum overlay ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(bins[1:], real_spec[1:], label='real', color='black', linewidth=2)
    cmap = plt.get_cmap('tab10')
    for i, name in enumerate(names):
        ax.loglog(bins[1:], method_specs[name][1:], label=name, color=cmap(i % 10))
    ax.axvspan(args.hf_min, args.hf_max, alpha=0.1, color='red', label='HF window')
    ax.set_xlabel('Radial frequency (Nyquist=0.5)')
    ax.set_ylabel('Power')
    ax.set_title('Radial power spectrum across methods')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'spectrum_overlay.png', dpi=120)
    plt.close(fig)

    # ── Visual grid: subjects × methods, mid-axial slice ───────────
    n_subj = min(args.grid_subjects, min(len(arrs) for arrs in method_arrays.values()))
    n_methods = len(methods)
    n_cols = n_methods + 1  # +1 for real reference
    fig, axes = plt.subplots(n_subj, n_cols, figsize=(2.5 * n_cols, 2.5 * n_subj))
    if n_subj == 1:
        axes = axes.reshape(1, -1)
    for r in range(n_subj):
        ax = axes[r, 0]
        ax.imshow(real_np[r % len(real_np)][real_np[0].shape[0] // 2], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title('real', fontsize=10)
        for c, name in enumerate(names):
            arr = method_arrays[name][r]
            mid = arr.shape[0] // 2
            axes[r, c + 1].imshow(arr[mid], cmap='gray')
            axes[r, c + 1].set_xticks([])
            axes[r, c + 1].set_yticks([])
            if r == 0:
                axes[r, c + 1].set_title(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / 'visual_grid.png', dpi=120)
    plt.close(fig)

    log.info(f"\nSaved: {out_dir}/{{results.csv,results.json,bar_chart.png,visual_grid.png,spectrum_overlay.png}}")
    log.info("\nSummary table:")
    log.info(f"{'method':<20} {'LPIPS':>8} {'L1':>8} {'slope':>7} {'Δslope':>7}")
    for name in names:
        r = method_results[name]
        log.info(f"{name:<20} {r['lpips_vs_real']:>8.4f} {r['l1_vs_real']:>8.4f} "
                 f"{r['hf_psd_slope']:>7.3f} {r['slope_diff_to_real']:>+7.3f}")


if __name__ == '__main__':
    main()
