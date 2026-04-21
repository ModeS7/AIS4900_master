#!/usr/bin/env python3
"""Frangi vesselness analysis across fine-tuned generators (Phase 1 #2).

Applies the Frangi vesselness filter to each brain volume and measures how
prominent tubular (vessel-like) structures are. Independent of the spectral
analysis — directly tests the visual claim "model X has more visible vessels
than model Y" in a quantitatively defensible way.

Frangi responds to locally tubular structure based on the Hessian eigenvalue
ratios. High response = tube-like; low = blob-like or flat. Inside the brain
mask, the distribution of Frangi responses summarizes vessel prominence.

Outputs:
  Figure 1: Mean vesselness score per model (bar chart) vs real reference
  Figure 2: Distribution of above-threshold voxels per volume (violin plot)
  Figure 3: Example axial slices with Frangi response overlaid (real + best 2 + worst 2 models)
  JSON:    Numeric values — per-volume mean/sum/threshold-passing vesselness

Usage:
    python -m medgen.scripts.analyze_vessel_prominence \\
        --compare-dir /path/to/generated/compare_exp37_20260420-165132 \\
        --real-dir    /path/to/brainmetshare-3/test1 \\
        --output-dir  runs/eval/vessel_analysis \\
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
from scipy.ndimage import binary_fill_holes
from skimage.filters import frangi

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# Frangi scales in voxel units — brain vessel scales at 128³ resolution.
# Voxel size ≈ 0.94mm (x,y), 1mm (z). Major vessels are 2-6mm diameter →
# 1-3 voxels radius. Covering [0.5, 1, 1.5, 2, 3] captures small-to-medium vessels.
FRANGI_SIGMAS = (0.5, 1.0, 1.5, 2.0, 3.0)

# Threshold above which Frangi response counts as "vessel-like voxel".
# Chosen empirically — 0.05 is on the conservative end but excludes most non-vessel texture.
VESSEL_THRESHOLD = 0.05

# Consistent colors with spectrum script
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
    """Load NIfTI -> [D, H, W] float32 in [0, 1]."""
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


def brain_mask_simple(volume: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Simple brain mask for restricting vesselness to inside-brain voxels."""
    m = (volume > threshold).astype(np.uint8)
    # Fill holes slice-wise (cheaper than 3D for large volumes)
    out = np.zeros_like(m)
    for i in range(m.shape[0]):
        out[i] = binary_fill_holes(m[i])
    return out.astype(bool)


def compute_vesselness(
    volume: np.ndarray,
    sigmas: tuple = FRANGI_SIGMAS,
) -> np.ndarray:
    """3D Frangi response.

    Returns float array same shape as `volume`, values in [0, 1]. Higher
    response = more tube-like structure at any of the scales in `sigmas`.
    """
    # scikit-image frangi handles ND natively. black_ridges=False targets
    # bright tubes on darker background (matches contrast-enhanced MRI).
    return frangi(
        volume,
        sigmas=sigmas,
        alpha=0.5,
        beta=0.5,
        gamma=None,   # auto-compute from input
        black_ridges=False,
    ).astype(np.float32)


def find_bravo_files(root: Path, max_volumes: int) -> list[Path]:
    files = sorted(root.glob("*/bravo.nii.gz"))
    if not files:
        files = sorted(root.glob("*.nii.gz"))
    return files[:max_volumes]


def summarize_one(volume: np.ndarray, brain_mask: np.ndarray) -> dict[str, float]:
    """Compute vesselness summary for a single volume."""
    v = compute_vesselness(volume)
    inside = v[brain_mask]
    if inside.size == 0:
        return {'mean': 0.0, 'p95': 0.0, 'p99': 0.0, 'frac_above_thresh': 0.0, 'sum': 0.0}
    return {
        'mean': float(inside.mean()),
        'p95': float(np.percentile(inside, 95)),
        'p99': float(np.percentile(inside, 99)),
        'frac_above_thresh': float((inside > VESSEL_THRESHOLD).mean()),
        'sum': float(inside.sum()),
    }


def process_model(files: list[Path], depth: int, label: str) -> list[dict[str, float]]:
    """Run Frangi summary on each volume; return list of per-volume stats."""
    stats: list[dict[str, float]] = []
    for i, fp in enumerate(files, start=1):
        vol = load_volume(fp, depth=depth)
        mask = brain_mask_simple(vol)
        s = summarize_one(vol, mask)
        s['file'] = fp.parent.name
        stats.append(s)
        if i % 2 == 0:
            logger.info(f"    [{label}] {i}/{len(files)}  mean={s['mean']:.4f}  "
                        f"frac>{VESSEL_THRESHOLD}={s['frac_above_thresh']:.4f}")
    return stats


def aggregate(stats: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-volume stats to mean + std."""
    if not stats:
        return {}
    keys = [k for k in stats[0] if k != 'file']
    agg: dict[str, float] = {}
    for k in keys:
        vals = np.array([s[k] for s in stats], dtype=np.float64)
        agg[f'{k}_mean'] = float(vals.mean())
        agg[f'{k}_std'] = float(vals.std())
    return agg


def plot_bars(results: dict, metric: str, ylabel: str, output_base: Path) -> None:
    """Bar chart of per-model metric mean with error bars."""
    names = list(results.keys())
    means = [results[n]['agg'][f'{metric}_mean'] for n in names]
    stds = [results[n]['agg'][f'{metric}_std'] for n in names]
    colors = [MODEL_COLORS.get(n, 'lightgray') for n in names]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=4,
                  edgecolor='black', linewidth=0.5)
    # Highlight real bar
    if 'real' in names:
        ri = names.index('real')
        bars[ri].set_edgecolor('black')
        bars[ri].set_linewidth(2.0)
        # Horizontal guide at real's value
        ax.axhline(means[ri], color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(f"Frangi vesselness: {metric} (inside brain mask)")
    ax.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_violin(results: dict, metric: str, ylabel: str, output_base: Path) -> None:
    """Violin plot of per-volume metric distribution per model."""
    names = list(results.keys())
    data = [[s[metric] for s in results[n]['per_volume']] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    parts = ax.violinplot(data, positions=np.arange(len(names)), showmedians=True,
                          showextrema=True, widths=0.7)
    # Color each violin
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(MODEL_COLORS.get(names[i], 'lightgray'))
        pc.set_alpha(0.65)
    # Reference line at real median
    if 'real' in names:
        real_median = float(np.median(data[names.index('real')]))
        ax.axhline(real_median, color='black', linewidth=0.8, linestyle='--',
                   alpha=0.5, label=f'real median = {real_median:.4f}')
        ax.legend(loc='best', fontsize=9)

    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=25, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(f"Frangi {metric} — distribution across volumes")
    ax.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_axial_examples(
    volumes: dict[str, np.ndarray],
    vesselness: dict[str, np.ndarray],
    output_base: Path,
) -> None:
    """One axial slice per model with vesselness overlay — visual sanity check."""
    names = list(volumes.keys())
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 6.2), squeeze=False)
    slice_idx = next(iter(volumes.values())).shape[0] // 2

    for i, name in enumerate(names):
        img = volumes[name][slice_idx]
        v = vesselness[name][slice_idx]

        axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(name, fontsize=9)
        axes[0, i].axis('off')

        axes[1, i].imshow(img, cmap='gray', vmin=0, vmax=1)
        # Overlay vesselness in red where response is above threshold
        axes[1, i].imshow(np.ma.masked_where(v < VESSEL_THRESHOLD, v),
                          cmap='Reds', alpha=0.7, vmin=VESSEL_THRESHOLD, vmax=max(v.max(), 0.2))
        axes[1, i].set_title(f'{name} (Frangi overlay)', fontsize=9)
        axes[1, i].axis('off')

    plt.suptitle(f'Axial slice {slice_idx} — original (top) / with Frangi overlay (bottom)',
                 fontsize=10)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=180, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def print_summary_table(results: dict) -> None:
    """Print the results table for thesis."""
    if 'real' not in results:
        return
    real_agg = results['real']['agg']
    keys = ['mean', 'frac_above_thresh', 'p95', 'p99']

    print()
    print("=" * 95)
    print(f"Frangi vesselness summary (threshold > {VESSEL_THRESHOLD})")
    print("=" * 95)
    header = f"{'model':<16}" + "".join(f"{k:>16}" for k in keys) + f"  {'% of real (mean)':>16}"
    print(header)
    print("-" * len(header))
    for name, d in results.items():
        agg = d['agg']
        row = f"{name:<16}"
        for k in keys:
            row += f"{agg[k + '_mean']:>12.5f}±{agg[k + '_std']:.4f}"
        if name != 'real' and real_agg['mean_mean'] > 0:
            pct = 100.0 * agg['mean_mean'] / real_agg['mean_mean']
            row += f"  {pct:>14.1f}%"
        else:
            row += f"  {'(reference)':>16}"
        print(row)
    print("=" * 95)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.2 — Frangi vesselness analysis")
    parser.add_argument('--compare-dir', required=True)
    parser.add_argument('--real-dir', default='/home/mode/NTNU/MedicalDataSets/brainmetshare-3/test1')
    parser.add_argument('--output-dir', default='runs/eval/vessel_analysis')
    parser.add_argument('--max-volumes', type=int, default=10)
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--thesis-dir',
                        default='/home/mode/NTNU/AIS4900_doc/AIS4900-master-thesis/Images/vessel_analysis')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compare_root = Path(args.compare_dir)
    if not compare_root.is_dir():
        raise SystemExit(f"compare-dir not found: {compare_root}")
    model_dirs = sorted([p for p in compare_root.iterdir() if p.is_dir()])
    logger.info(f"Found {len(model_dirs)} models")

    results: dict[str, dict] = {}

    # Real reference
    real_root = Path(args.real_dir)
    logger.info("Processing real reference")
    real_files = find_bravo_files(real_root, args.max_volumes)
    if not real_files:
        raise SystemExit(f"No bravo volumes in {real_root}")
    per_vol = process_model(real_files, args.depth, 'real')
    results['real'] = {'per_volume': per_vol, 'agg': aggregate(per_vol),
                       'n_volumes': len(per_vol)}

    # Each generator
    for md in model_dirs:
        logger.info(f"Processing {md.name}")
        files = find_bravo_files(md, args.max_volumes)
        if not files:
            logger.warning(f"  no volumes in {md}, skipping")
            continue
        per_vol = process_model(files, args.depth, md.name)
        results[md.name] = {'per_volume': per_vol, 'agg': aggregate(per_vol),
                            'n_volumes': len(per_vol)}

    # Plots — two metrics (mean and frac_above_thresh) visualized
    logger.info("Plotting")
    plot_bars(results, 'mean', 'Mean Frangi response (inside brain)',
              output_dir / 'vessel_mean_bars')
    plot_bars(results, 'frac_above_thresh',
              f'Fraction of voxels with vesselness > {VESSEL_THRESHOLD}',
              output_dir / 'vessel_frac_bars')
    plot_violin(results, 'mean', 'Mean Frangi (per volume)',
                output_dir / 'vessel_mean_violin')
    plot_violin(results, 'frac_above_thresh',
                'Fraction above threshold (per volume)',
                output_dir / 'vessel_frac_violin')

    # Example axial-slice overlays — use the first volume of each model
    logger.info("Rendering axial-slice overlays")
    volumes_by_name: dict[str, np.ndarray] = {}
    vesselness_by_name: dict[str, np.ndarray] = {}
    for name in results:
        if name == 'real':
            fp = real_files[0]
        else:
            md = next(m for m in model_dirs if m.name == name)
            files = find_bravo_files(md, 1)
            if not files:
                continue
            fp = files[0]
        vol = load_volume(fp, args.depth)
        volumes_by_name[name] = vol
        vesselness_by_name[name] = compute_vesselness(vol)
    plot_axial_examples(volumes_by_name, vesselness_by_name,
                        output_dir / 'vessel_axial_overlay')

    # Text table
    print_summary_table(results)

    # JSON
    with open(output_dir / 'vessel_results.json', 'w') as f:
        json.dump({
            'compare_dir': str(compare_root),
            'real_dir': str(real_root),
            'max_volumes': args.max_volumes,
            'frangi_sigmas': list(FRANGI_SIGMAS),
            'vessel_threshold': VESSEL_THRESHOLD,
            'models': {
                name: {
                    'n_volumes': d['n_volumes'],
                    'agg': d['agg'],
                    'per_volume': d['per_volume'],
                } for name, d in results.items()
            },
        }, f, indent=2)
    logger.info(f"Saved summary: {output_dir / 'vessel_results.json'}")

    # Thesis copy
    if args.thesis_dir:
        try:
            thesis_dir = Path(args.thesis_dir)
            thesis_dir.mkdir(parents=True, exist_ok=True)
            for name in ('vessel_mean_bars', 'vessel_frac_bars',
                         'vessel_mean_violin', 'vessel_frac_violin',
                         'vessel_axial_overlay'):
                for ext in ('png', 'pdf'):
                    src = output_dir / f'{name}.{ext}'
                    if src.exists():
                        (thesis_dir / f'{name}.{ext}').write_bytes(src.read_bytes())
            logger.info(f"Copied figures to thesis dir: {thesis_dir}")
        except OSError as e:
            logger.warning(f"Could not copy to thesis dir {args.thesis_dir}: {e}")


if __name__ == '__main__':
    main()
