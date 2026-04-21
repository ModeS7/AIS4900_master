#!/usr/bin/env python3
"""Cortical-shell vesselness analysis (Phase 1 #2b).

Targeted follow-up to `analyze_vessel_prominence.py`. The aggregate Frangi
analysis computed vesselness *inside* the brain mask, which excludes the very
region where surface vessels live: the cortical shell just inside the brain
boundary. This script restricts the vesselness statistics to a thin inner
shell — where some exp32 volumes visually appeared to have surface vessel
patterns even when the aggregate-mean test showed no effect.

Two key methodology changes vs the original:

  1. Restrict statistics to a **cortical shell** = (brain_mask) XOR (eroded
     brain_mask). Width controlled by --shell-width (default 3 voxels).
     Vessels near/at the surface contribute; deep-tissue Frangi response is
     excluded.
  2. **Per-volume scatter** (not aggregate bar): each volume is one dot so
     outlier volumes (the two or three exp32 samples that visually showed
     vessels) stand out instead of being averaged away.

Outputs:
  Figure 1: Per-volume shell-Frangi scatter, one dot per volume
  Figure 2: Top-3 volumes per model — axial/sagittal MIPs of Frangi in shell
  JSON:     Per-volume shell-Frangi stats + sorted top-K file list per model

Usage:
    python -m medgen.scripts.analyze_cortical_vessels \\
        --compare-dir /path/to/generated/compare_exp37_20260420-165132 \\
        --real-dir    /path/to/brainmetshare-3/test1 \\
        --output-dir  runs/eval/cortical_vessels \\
        --shell-width 3 \\
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
from scipy.ndimage import binary_erosion, binary_fill_holes
from skimage.filters import frangi

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# Same Frangi scales as the aggregate script — we want direct comparability.
FRANGI_SIGMAS = (0.5, 1.0, 1.5, 2.0, 3.0)
VESSEL_THRESHOLD = 0.05

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
    vol = np.transpose(vol, (2, 0, 1))
    d = vol.shape[0]
    if d < depth:
        vol = np.pad(vol, ((0, depth - d), (0, 0), (0, 0)))
    elif d > depth:
        vol = vol[:depth]
    return vol


def brain_mask_simple(volume: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    m = (volume > threshold).astype(np.uint8)
    out = np.zeros_like(m)
    for i in range(m.shape[0]):
        out[i] = binary_fill_holes(m[i])
    return out.astype(bool)


def cortical_shell(brain_mask: np.ndarray, width: int = 3) -> np.ndarray:
    """Inner cortical shell = brain_mask AND NOT erosion(brain_mask, width).

    Voxels that are inside the brain but within `width` voxels of the surface.
    3D erosion with the default 3D cross structuring element, so width is
    measured in L1-voxel units (close to Euclidean for small widths).
    """
    eroded = binary_erosion(brain_mask, iterations=width)
    return brain_mask & ~eroded


def compute_vesselness(volume: np.ndarray, sigmas: tuple = FRANGI_SIGMAS) -> np.ndarray:
    return frangi(
        volume,
        sigmas=sigmas,
        alpha=0.5,
        beta=0.5,
        gamma=None,
        black_ridges=False,
    ).astype(np.float32)


def find_bravo_files(root: Path, max_volumes: int) -> list[Path]:
    files = sorted(root.glob("*/bravo.nii.gz"))
    if not files:
        files = sorted(root.glob("*.nii.gz"))
    return files[:max_volumes]


def summarize_shell(
    volume: np.ndarray,
    brain_mask: np.ndarray,
    shell_mask: np.ndarray,
) -> dict[str, float]:
    """Compute Frangi once and split stats by region (shell vs full brain)."""
    v = compute_vesselness(volume)
    shell_vals = v[shell_mask]
    brain_vals = v[brain_mask]
    stats = {
        'shell_voxels':       int(shell_mask.sum()),
        'brain_voxels':       int(brain_mask.sum()),
        'shell_mean':         float(shell_vals.mean()) if shell_vals.size else 0.0,
        'shell_p95':          float(np.percentile(shell_vals, 95)) if shell_vals.size else 0.0,
        'shell_p99':          float(np.percentile(shell_vals, 99)) if shell_vals.size else 0.0,
        'shell_frac_thresh':  float((shell_vals > VESSEL_THRESHOLD).mean()) if shell_vals.size else 0.0,
        'brain_mean':         float(brain_vals.mean()) if brain_vals.size else 0.0,
    }
    # Shell-over-brain ratio — does Frangi *concentrate* in the shell relative
    # to the full brain interior? Ratio > 1 means yes, surface has more tube
    # response than deep tissue.
    stats['shell_over_brain'] = (stats['shell_mean'] / stats['brain_mean']
                                 if stats['brain_mean'] > 1e-8 else 0.0)
    return stats


def process_model(
    files: list[Path],
    depth: int,
    shell_width: int,
    label: str,
) -> list[dict]:
    stats: list[dict] = []
    for i, fp in enumerate(files, start=1):
        vol = load_volume(fp, depth=depth)
        mask = brain_mask_simple(vol)
        shell = cortical_shell(mask, width=shell_width)
        s = summarize_shell(vol, mask, shell)
        s['file'] = str(fp)
        s['name'] = fp.parent.name
        stats.append(s)
        if i % 2 == 0:
            logger.info(f"    [{label}] {i}/{len(files)}  "
                        f"shell_mean={s['shell_mean']:.4f}  "
                        f"shell/brain={s['shell_over_brain']:.2f}")
    return stats


def plot_scatter(
    results: dict,
    metric: str,
    ylabel: str,
    output_base: Path,
) -> None:
    """Per-volume scatter — each dot is one volume, x is model (jittered)."""
    names = list(results.keys())
    fig, ax = plt.subplots(figsize=(11, 5.5))

    rng = np.random.default_rng(0)
    for i, name in enumerate(names):
        pv = results[name]['per_volume']
        ys = [s[metric] for s in pv]
        xs = i + rng.uniform(-0.15, 0.15, size=len(ys))
        color = MODEL_COLORS.get(name, 'lightgray')
        ax.scatter(xs, ys, color=color, alpha=0.8, s=55,
                   edgecolor='black', linewidth=0.5,
                   label=f'{name} (n={len(ys)})')

    if 'real' in names:
        real_vals = [s[metric] for s in results['real']['per_volume']]
        ax.axhline(float(np.median(real_vals)), color='black', linewidth=0.8,
                   linestyle='--', alpha=0.6, label='real median')

    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=25, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Cortical-shell Frangi — {metric} (one dot per volume)')
    ax.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def render_top_mips(
    results: dict,
    shell_width: int,
    depth: int,
    metric: str,
    top_k: int,
    output_base: Path,
) -> None:
    """For each model's top-K volumes by shell metric: render axial + sagittal MIPs
    of the Frangi response masked to the shell only. Shows WHERE surface vessels
    (if any) are distributed.
    """
    names = list(results.keys())
    fig, axes = plt.subplots(len(names), top_k * 2, figsize=(3.0 * top_k * 2, 2.5 * len(names)),
                             squeeze=False)

    for row, name in enumerate(names):
        pv = results[name]['per_volume']
        ordered = sorted(pv, key=lambda s: s[metric], reverse=True)[:top_k]
        for col, stats in enumerate(ordered):
            fp = Path(stats['file'])
            vol = load_volume(fp, depth=depth)
            mask = brain_mask_simple(vol)
            shell = cortical_shell(mask, width=shell_width)
            v = compute_vesselness(vol)
            v_shell = np.where(shell, v, 0.0)

            # Axial MIP along z
            axial_mip = v_shell.max(axis=0)
            axes[row, col * 2].imshow(axial_mip, cmap='hot', vmin=0, vmax=0.2)
            axes[row, col * 2].set_title(
                f'{name} rank={col + 1}\n{stats["name"]}  m={stats[metric]:.4f}',
                fontsize=7,
            )
            axes[row, col * 2].axis('off')

            # Sagittal MIP along W
            sag_mip = v_shell.max(axis=2)
            axes[row, col * 2 + 1].imshow(sag_mip, cmap='hot', vmin=0, vmax=0.2, aspect='auto')
            axes[row, col * 2 + 1].set_title('sagittal', fontsize=7)
            axes[row, col * 2 + 1].axis('off')

    plt.suptitle(
        f'Top-{top_k} volumes per model by {metric} — shell Frangi MIPs (axial/sagittal)',
        fontsize=10,
    )
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=180, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def render_slice_overlay(
    results: dict,
    shell_width: int,
    depth: int,
    metric: str,
    output_base: Path,
) -> None:
    """One axial slice per model (best-shell-Frangi volume). Shows T1 with
    shell overlaid + Frangi-in-shell overlay.
    """
    names = list(results.keys())
    fig, axes = plt.subplots(3, len(names), figsize=(2.6 * len(names), 7.4),
                             squeeze=False)

    for col, name in enumerate(names):
        pv = results[name]['per_volume']
        best = max(pv, key=lambda s: s[metric])
        fp = Path(best['file'])
        vol = load_volume(fp, depth=depth)
        mask = brain_mask_simple(vol)
        shell = cortical_shell(mask, width=shell_width)
        v = compute_vesselness(vol)
        zmid = vol.shape[0] // 2

        axes[0, col].imshow(vol[zmid], cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f'{name}\n{best["name"]}', fontsize=8)
        axes[0, col].axis('off')

        axes[1, col].imshow(vol[zmid], cmap='gray', vmin=0, vmax=1)
        axes[1, col].imshow(np.ma.masked_where(~shell[zmid], shell[zmid]),
                            cmap='autumn', alpha=0.4)
        axes[1, col].set_title('shell region', fontsize=8)
        axes[1, col].axis('off')

        v_slice = np.where(shell[zmid], v[zmid], 0.0)
        axes[2, col].imshow(vol[zmid], cmap='gray', vmin=0, vmax=1)
        axes[2, col].imshow(np.ma.masked_where(v_slice < VESSEL_THRESHOLD, v_slice),
                            cmap='Reds', alpha=0.85,
                            vmin=VESSEL_THRESHOLD, vmax=max(v_slice.max(), 0.2))
        axes[2, col].set_title(f'shell Frangi  mean={best[metric]:.4f}', fontsize=8)
        axes[2, col].axis('off')

    plt.suptitle('Cortical-shell Frangi — best-shell volume per model (axial mid-slice)',
                 fontsize=10)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=180, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def print_summary_table(results: dict) -> None:
    if 'real' not in results:
        return
    real_pv = results['real']['per_volume']
    real_shell = np.array([s['shell_mean'] for s in real_pv])
    real_median = float(np.median(real_shell))

    print()
    print("=" * 110)
    print("Cortical-shell Frangi summary")
    print("=" * 110)
    header = (f"{'model':<16}{'n':>4}"
              f"{'shell_mean (med)':>20}{'shell_p95 (med)':>18}"
              f"{'shell/brain (med)':>20}{'top3 shell_mean':>22}"
              f"{'% real':>10}")
    print(header)
    print("-" * len(header))
    for name, d in results.items():
        pv = d['per_volume']
        shell_means = np.array([s['shell_mean'] for s in pv])
        shell_p95 = np.array([s['shell_p95'] for s in pv])
        shell_ratio = np.array([s['shell_over_brain'] for s in pv])
        top3 = np.sort(shell_means)[::-1][:3]
        pct = (100.0 * float(np.median(shell_means)) / real_median) if real_median > 0 else 0.0
        top3_str = ",".join(f"{v:.4f}" for v in top3)
        print(
            f"{name:<16}{len(pv):>4}"
            f"{float(np.median(shell_means)):>20.5f}"
            f"{float(np.median(shell_p95)):>18.5f}"
            f"{float(np.median(shell_ratio)):>20.3f}"
            f"{top3_str:>22s}"
            f"{pct:>9.1f}%"
        )
    print("=" * 110)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.2b — cortical-shell Frangi")
    parser.add_argument('--compare-dir', required=True)
    parser.add_argument('--real-dir',
                        default='/home/mode/NTNU/MedicalDataSets/brainmetshare-3/test1')
    parser.add_argument('--output-dir', default='runs/eval/cortical_vessels')
    parser.add_argument('--shell-width', type=int, default=3,
                        help='Shell thickness in voxels (inner erosion iterations)')
    parser.add_argument('--max-volumes', type=int, default=10)
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--top-k', type=int, default=3,
                        help='Render MIPs for top-K volumes per model')
    parser.add_argument('--thesis-dir',
                        default='/home/mode/NTNU/AIS4900_doc/AIS4900-master-thesis/Images/cortical_vessels')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compare_root = Path(args.compare_dir)
    if not compare_root.is_dir():
        raise SystemExit(f"compare-dir not found: {compare_root}")
    model_dirs = sorted([p for p in compare_root.iterdir() if p.is_dir()])
    logger.info(f"Found {len(model_dirs)} models, shell width = {args.shell_width}")

    results: dict[str, dict] = {}

    real_root = Path(args.real_dir)
    logger.info("Processing real reference")
    real_files = find_bravo_files(real_root, args.max_volumes)
    if not real_files:
        raise SystemExit(f"No bravo volumes in {real_root}")
    pv = process_model(real_files, args.depth, args.shell_width, 'real')
    results['real'] = {'per_volume': pv, 'n_volumes': len(pv)}

    for md in model_dirs:
        logger.info(f"Processing {md.name}")
        files = find_bravo_files(md, args.max_volumes)
        if not files:
            logger.warning(f"  no volumes in {md}, skipping")
            continue
        pv = process_model(files, args.depth, args.shell_width, md.name)
        results[md.name] = {'per_volume': pv, 'n_volumes': len(pv)}

    logger.info("Plotting per-volume scatter")
    plot_scatter(results, 'shell_mean', 'Mean Frangi in cortical shell',
                 output_dir / 'shell_mean_scatter')
    plot_scatter(results, 'shell_over_brain',
                 'Shell mean / full-brain mean (concentration ratio)',
                 output_dir / 'shell_ratio_scatter')
    plot_scatter(results, 'shell_frac_thresh',
                 f'Fraction of shell voxels with Frangi > {VESSEL_THRESHOLD}',
                 output_dir / 'shell_frac_scatter')

    logger.info("Rendering top-K MIPs")
    render_top_mips(results, args.shell_width, args.depth,
                    metric='shell_mean', top_k=args.top_k,
                    output_base=output_dir / 'top_shell_mips')

    logger.info("Rendering best-slice overlays")
    render_slice_overlay(results, args.shell_width, args.depth,
                         metric='shell_mean',
                         output_base=output_dir / 'shell_slice_overlay')

    print_summary_table(results)

    with open(output_dir / 'cortical_results.json', 'w') as f:
        json.dump({
            'compare_dir': str(compare_root),
            'real_dir': str(real_root),
            'max_volumes': args.max_volumes,
            'shell_width': args.shell_width,
            'frangi_sigmas': list(FRANGI_SIGMAS),
            'vessel_threshold': VESSEL_THRESHOLD,
            'models': {
                name: {
                    'n_volumes': d['n_volumes'],
                    'per_volume': [
                        {k: v for k, v in s.items() if k != 'file'}
                        for s in d['per_volume']
                    ],
                    'top_shell_mean_files': [
                        s['file'] for s in sorted(
                            d['per_volume'], key=lambda s: s['shell_mean'], reverse=True
                        )[:args.top_k]
                    ],
                } for name, d in results.items()
            },
        }, f, indent=2)
    logger.info(f"Saved: {output_dir / 'cortical_results.json'}")

    if args.thesis_dir:
        try:
            thesis_dir = Path(args.thesis_dir)
            thesis_dir.mkdir(parents=True, exist_ok=True)
            for name in ('shell_mean_scatter', 'shell_ratio_scatter', 'shell_frac_scatter',
                         'top_shell_mips', 'shell_slice_overlay'):
                for ext in ('png', 'pdf'):
                    src = output_dir / f'{name}.{ext}'
                    if src.exists():
                        (thesis_dir / f'{name}.{ext}').write_bytes(src.read_bytes())
            logger.info(f"Copied figures to thesis dir: {thesis_dir}")
        except OSError as e:
            logger.warning(f"Could not copy to thesis dir {args.thesis_dir}: {e}")


if __name__ == '__main__':
    main()
