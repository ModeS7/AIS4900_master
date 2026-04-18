#!/usr/bin/env python3
"""Ablation study: PCA shape filter across resolution and component count.

Evaluates whether the deployed (128x128x80, k=60) PCA configuration is well-
calibrated by comparing against a coarser (64x64x40, k=30) configuration on:

  1. Reconstruction fidelity on held-out real brains (lower = better fit)
  2. Discriminative AUC for real-vs-generated brains (higher = better filter)
  3. Sensitivity of AUC to component count k (detects over/under-specification)

Both PCA models are fit on train+val only; test1 serves as held-out real.

Usage:
    python -m medgen.scripts.ablate_pca_filter \\
        --pca-coarse data/ablation/brain_pca_64x64x40.npz \\
        --pca-fine   data/ablation/brain_pca_128x128x80.npz \\
        --real-dir   /path/to/brainmetshare-3/test1 \\
        --gen-dir    /path/to/generated/exp1_1_1000_bravo_525 \\
        --output-dir runs/eval/pca_ablation
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
import torch

from medgen.metrics.brain_mask import create_brain_mask

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


K_SWEEP = [10, 20, 30, 40, 60, 80, 100]
FIG_SIZE = (7.5, 5.0)


def load_volume(path: Path, depth: int = 160, image_size: int = 256) -> np.ndarray:
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
    if vol.shape[1] != image_size or vol.shape[2] != image_size:
        t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
        t = torch.nn.functional.interpolate(
            t, size=(depth, image_size, image_size), mode='trilinear', align_corners=False,
        )
        vol = t.squeeze().numpy()
    return vol


def downsample_mask(mask: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    down = torch.nn.functional.interpolate(
        t, size=target_shape, mode='trilinear', align_corners=False,
    )
    return (down.squeeze().numpy() > 0.5).astype(np.float32)


def collect_masks(root: Path, target_shapes: list[tuple[int, int, int]]) -> dict:
    """Load all bravo.nii.gz in root, produce binary masks at every target shape."""
    paths = sorted(root.glob("*/bravo.nii.gz"))
    logger.info(f"  {root.name}: {len(paths)} volumes")
    by_shape: dict[tuple[int, int, int], list[np.ndarray]] = {s: [] for s in target_shapes}
    for i, p in enumerate(paths):
        vol = load_volume(p)
        mask = create_brain_mask(vol, threshold=0.05, fill_holes=True, dilate_pixels=0)
        for shape in target_shapes:
            by_shape[shape].append(downsample_mask(mask, shape).flatten())
        if (i + 1) % 20 == 0:
            logger.info(f"    {i + 1}/{len(paths)}")
    return {s: np.array(v, dtype=np.float32) for s, v in by_shape.items()}


def reconstruction_error(masks_flat: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Per-sample MSE reconstruction error. components shape: [k, n_voxels]."""
    centered = masks_flat - mean
    proj = centered @ components.T
    recon = proj @ components + mean
    return np.mean((masks_flat - recon) ** 2, axis=1)


def compute_auc(real_errors: np.ndarray, gen_errors: np.ndarray) -> float:
    """AUC for classifier 'larger error = generated' (higher AUC = better filter)."""
    n_r, n_g = len(real_errors), len(gen_errors)
    if n_r == 0 or n_g == 0:
        return float('nan')
    # Rank-sum AUC = P(gen > real) + 0.5 * P(tied)
    all_errors = np.concatenate([real_errors, gen_errors])
    labels = np.concatenate([np.zeros(n_r), np.ones(n_g)]).astype(bool)
    order = np.argsort(all_errors, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_errors) + 1)
    # Fix ties: average rank within each tied group
    sorted_vals = all_errors[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    rank_sum_gen = ranks[labels].sum()
    return float((rank_sum_gen - n_g * (n_g + 1) / 2) / (n_r * n_g))


def load_pca(path: Path) -> dict:
    data = np.load(path)
    out = {
        'mean': data['mean'],
        'components': data['components'],
        'pca_shape': tuple(int(x) for x in data['pca_shape']),
        'n_samples': int(data['n_samples'][0]),
    }
    if 'full_singular_values' in data.files:
        s = data['full_singular_values'].astype(np.float64)
        ev = (s ** 2) / (out['n_samples'] - 1)
        out['explained_ratio'] = ev / ev.sum()
    else:
        ev = data['explained_variance'].astype(np.float64)
        out['explained_ratio'] = ev / ev.sum()
    return out


def plot_cumulative_comparison(
    coarse: dict, fine: dict, output_base: Path,
    k_coarse_mark: int = 30, k_fine_mark: int = 60,
) -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for label, pca, color, mark in [
        ('64×64×40', coarse, 'steelblue', k_coarse_mark),
        ('128×128×80', fine, 'darkorange', k_fine_mark),
    ]:
        cum = np.cumsum(pca['explained_ratio'])
        components = np.arange(1, len(cum) + 1)
        ax.plot(components, cum, linewidth=2, color=color, label=label)
        if mark <= len(cum):
            y = cum[mark - 1]
            ax.scatter([mark], [y], color=color, s=40, zorder=5, edgecolor='k', linewidth=0.8)
            ax.annotate(
                f'k={mark}\n{y:.1%}',
                xy=(mark, y), xytext=(mark + 5, y - 0.08),
                fontsize=9, color=color,
                arrowprops=dict(arrowstyle='->', color=color, alpha=0.7, lw=0.8),
            )

    ax.set_xlabel('number of components')
    ax.set_ylabel('cumulative explained variance ratio')
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle='-', linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right')

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_auc_vs_k(results: dict, output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for label, color in [('coarse', 'steelblue'), ('fine', 'darkorange')]:
        ks = results[label]['k_values']
        aucs = results[label]['auc']
        ax.plot(ks, aucs, 'o-', linewidth=2, color=color, label={
            'coarse': '64×64×40', 'fine': '128×128×80',
        }[label])
    ax.set_xlabel('number of components (k)')
    ax.set_ylabel('AUC (P(err_gen > err_real))')
    ax.set_ylim(0.0, 1.02)
    ax.axhline(0.5, color='k', linewidth=0.8, linestyle='--', alpha=0.6,
               label='random (0.5)')
    ax.axhline(1.0, color='k', linewidth=0.6, linestyle=':', alpha=0.4)
    ax.grid(True, linestyle='-', linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='best')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_recon_fidelity(results: dict, output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for label, color in [('coarse', 'steelblue'), ('fine', 'darkorange')]:
        ks = results[label]['k_values']
        mean_real = results[label]['real_error_mean']
        ax.plot(ks, mean_real, 'o-', linewidth=2, color=color, label={
            'coarse': '64×64×40', 'fine': '128×128×80',
        }[label])
    ax.set_xlabel('number of components (k)')
    ax.set_ylabel('mean reconstruction MSE (held-out real)')
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='-', linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='best')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_threshold_vs_k(results: dict, output_base: Path) -> None:
    """How the filter threshold shrinks with k (more components = tighter fit)."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for label, color in [('coarse', 'steelblue'), ('fine', 'darkorange')]:
        ks = results[label]['k_values']
        thr = results[label]['threshold']
        ax.plot(ks, thr, 'o-', linewidth=2, color=color, label={
            'coarse': '64×64×40', 'fine': '128×128×80',
        }[label])
    ax.set_xlabel('number of components (k)')
    ax.set_ylabel('filter threshold (3 × max training MSE)')
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='-', linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='best')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_pass_rates(results: dict, output_base: Path) -> None:
    """What fraction of real / generated brains pass the filter at each k."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, label, pretty in [
        (axes[0], 'coarse', '64×64×40'),
        (axes[1], 'fine', '128×128×80'),
    ]:
        ks = results[label]['k_values']
        real_pass = results[label]['real_pass_rate']
        gen_pass = results[label]['gen_pass_rate']
        ax.plot(ks, [100 * x for x in real_pass], 'o-', color='steelblue',
                linewidth=2, label='held-out real (want high)')
        ax.plot(ks, [100 * x for x in gen_pass], 's-', color='firebrick',
                linewidth=2, label='generated (lower = more filtered out)')
        ax.set_xlabel('number of components (k)')
        ax.set_ylabel('pass rate (%)')
        ax.set_ylim(0, 105)
        ax.set_title(pretty, fontsize=11)
        ax.grid(True, linestyle='-', linewidth=0.4, alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc='lower left')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_histograms_with_threshold(
    results: dict, output_base: Path,
    coarse_k: int = 30, fine_k: int = 60,
) -> None:
    """Error histograms with filter threshold marked — shows who gets rejected."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, label, k, pretty in [
        (axes[0], 'coarse', coarse_k, '64×64×40'),
        (axes[1], 'fine', fine_k, '128×128×80'),
    ]:
        idx = results[label]['k_values'].index(k)
        real_errs = np.array(results[label]['real_errors_by_k'][idx])
        gen_errs = np.array(results[label]['gen_errors_by_k'][idx])
        thr = results[label]['threshold'][idx]
        real_pass = results[label]['real_pass_rate'][idx]
        gen_pass = results[label]['gen_pass_rate'][idx]

        xmax = max(real_errs.max(), gen_errs.max(), thr) * 1.05
        bins = np.linspace(0, xmax, 40)
        ax.hist(real_errs, bins=bins, alpha=0.6, label=f'real (n={len(real_errs)})', color='steelblue')
        ax.hist(gen_errs, bins=bins, alpha=0.6, label=f'generated (n={len(gen_errs)})', color='firebrick')
        ax.axvline(thr, color='black', linewidth=2, linestyle='--',
                   label=f'threshold {thr:.4f}')
        ax.set_title(
            f'{pretty}, k={k}   '
            f'(real pass: {real_pass:.0%}, generated pass: {gen_pass:.0%})',
            fontsize=10,
        )
        ax.set_xlabel('reconstruction MSE')
        ax.set_ylabel('count')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle='-', linewidth=0.4, alpha=0.4)
        ax.set_axisbelow(True)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_error_histograms(
    results: dict, output_base: Path,
    coarse_k: int = 30, fine_k: int = 60,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, label, k in [(axes[0], 'coarse', coarse_k), (axes[1], 'fine', fine_k)]:
        idx = results[label]['k_values'].index(k)
        real_errs = np.array(results[label]['real_errors_by_k'][idx])
        gen_errs = np.array(results[label]['gen_errors_by_k'][idx])
        auc = results[label]['auc'][idx]

        bins = np.linspace(0, max(real_errs.max(), gen_errs.max()) * 1.05, 40)
        ax.hist(real_errs, bins=bins, alpha=0.6, label=f'real (n={len(real_errs)})', color='steelblue')
        ax.hist(gen_errs, bins=bins, alpha=0.6, label=f'generated (n={len(gen_errs)})', color='firebrick')
        pretty = {'coarse': '64×64×40', 'fine': '128×128×80'}[label]
        ax.set_title(f'{pretty}, k={k}   (AUC={auc:.3f})', fontsize=11)
        ax.set_xlabel('reconstruction MSE')
        ax.set_ylabel('count')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='-', linewidth=0.4, alpha=0.4)
        ax.set_axisbelow(True)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA filter ablation: resolution x k")
    parser.add_argument('--pca-coarse', default='data/ablation/brain_pca_64x64x40.npz')
    parser.add_argument('--pca-fine',   default='data/ablation/brain_pca_128x128x80.npz')
    parser.add_argument('--real-dir',   default='/home/mode/NTNU/MedicalDataSets/brainmetshare-3/test1',
                        help='Held-out real brains (not used in PCA fit)')
    parser.add_argument('--gen-dir',    default='/home/mode/NTNU/MedicalDataSets/generated/exp1_1_1000_bravo_525',
                        help='Generated brain volumes from the diffusion model')
    parser.add_argument('--train-root', default='/home/mode/NTNU/MedicalDataSets/brainmetshare-3',
                        help='Dataset root; train+val used for threshold calibration')
    parser.add_argument('--threshold-multiplier', type=float, default=3.0,
                        help='Threshold = max training error × this multiplier (default 3)')
    parser.add_argument('--output-dir', default='runs/eval/pca_ablation')
    parser.add_argument('--thesis-dir',
                        default='/home/mode/NTNU/AIS4900_doc/AIS4900-master-thesis/Images/pca_components')
    parser.add_argument('--coarse-k', type=int, default=30,
                        help='k value to highlight for coarse config (first-attempt default)')
    parser.add_argument('--fine-k',   type=int, default=60,
                        help='k value to highlight for fine config (final default)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load PCAs
    logger.info("Loading PCA models")
    coarse = load_pca(Path(args.pca_coarse))
    fine = load_pca(Path(args.pca_fine))
    logger.info(f"  coarse: shape={coarse['pca_shape']}, components={coarse['components'].shape[0]}")
    logger.info(f"  fine:   shape={fine['pca_shape']}, components={fine['components'].shape[0]}")

    # -- Collect masks at both PCA resolutions
    logger.info("Loading held-out real brains")
    real = collect_masks(Path(args.real_dir), [coarse['pca_shape'], fine['pca_shape']])
    logger.info(f"  real: {len(next(iter(real.values())))} volumes")

    logger.info("Loading generated brains")
    gen = collect_masks(Path(args.gen_dir), [coarse['pca_shape'], fine['pca_shape']])
    logger.info(f"  generated: {len(next(iter(gen.values())))} volumes")

    logger.info("Loading training (train+val) brains — for threshold calibration")
    train_root = Path(args.train_root)
    train_masks: dict = {s: [] for s in (coarse['pca_shape'], fine['pca_shape'])}
    for split in ('train', 'val'):
        part = collect_masks(train_root / split, list(train_masks.keys()))
        for shape, arr in part.items():
            train_masks[shape].append(arr)
    train = {s: np.concatenate(v, axis=0) for s, v in train_masks.items()}
    logger.info(f"  training: {len(next(iter(train.values())))} volumes")

    # -- Sweep k for each resolution
    results: dict = {}
    for label, pca in [('coarse', coarse), ('fine', fine)]:
        shape = pca['pca_shape']
        max_k = min(max(K_SWEEP), pca['components'].shape[0])
        ks = [k for k in K_SWEEP if k <= max_k]

        per_k: dict = {
            'k_values': ks, 'auc': [], 'real_error_mean': [], 'gen_error_mean': [],
            'real_errors_by_k': [], 'gen_errors_by_k': [],
            'threshold': [], 'real_pass_rate': [], 'gen_pass_rate': [],
            'train_max_error': [],
        }

        real_flat = real[shape]
        gen_flat = gen[shape]
        train_flat = train[shape]

        for k in ks:
            comps_k = pca['components'][:k]
            re = reconstruction_error(real_flat, pca['mean'], comps_k)
            ge = reconstruction_error(gen_flat, pca['mean'], comps_k)
            te = reconstruction_error(train_flat, pca['mean'], comps_k)

            # Threshold is the deployed-filter logic: max(train_error) × multiplier
            train_max = float(te.max())
            threshold = train_max * args.threshold_multiplier
            real_pass = float(np.mean(re < threshold))
            gen_pass = float(np.mean(ge < threshold))

            per_k['auc'].append(compute_auc(re, ge))
            per_k['real_error_mean'].append(float(re.mean()))
            per_k['gen_error_mean'].append(float(ge.mean()))
            per_k['real_errors_by_k'].append(re.tolist())
            per_k['gen_errors_by_k'].append(ge.tolist())
            per_k['threshold'].append(threshold)
            per_k['real_pass_rate'].append(real_pass)
            per_k['gen_pass_rate'].append(gen_pass)
            per_k['train_max_error'].append(train_max)

        results[label] = per_k
        logger.info(f"  {label}: AUC by k = " +
                    ", ".join(f"k={k}:{a:.3f}" for k, a in zip(ks, per_k['auc'])))

    # -- Plots
    logger.info("Plotting")
    plot_cumulative_comparison(
        coarse, fine, output_dir / 'pca_cumulative_comparison',
        k_coarse_mark=args.coarse_k, k_fine_mark=args.fine_k,
    )
    plot_auc_vs_k(results, output_dir / 'pca_auc_vs_k')
    plot_recon_fidelity(results, output_dir / 'pca_recon_fidelity')
    plot_error_histograms(
        results, output_dir / 'pca_error_histograms',
        coarse_k=args.coarse_k, fine_k=args.fine_k,
    )
    plot_threshold_vs_k(results, output_dir / 'pca_threshold_vs_k')
    plot_pass_rates(results, output_dir / 'pca_pass_rates')
    plot_histograms_with_threshold(
        results, output_dir / 'pca_histograms_with_threshold',
        coarse_k=args.coarse_k, fine_k=args.fine_k,
    )

    # -- Also copy the canonical figures to the thesis dir
    if args.thesis_dir:
        thesis_dir = Path(args.thesis_dir)
        thesis_dir.mkdir(parents=True, exist_ok=True)
        for name in ('pca_cumulative_comparison', 'pca_auc_vs_k',
                     'pca_recon_fidelity', 'pca_error_histograms',
                     'pca_threshold_vs_k', 'pca_pass_rates',
                     'pca_histograms_with_threshold'):
            for ext in ('png', 'pdf'):
                src = output_dir / f'{name}.{ext}'
                dst = thesis_dir / f'{name}.{ext}'
                dst.write_bytes(src.read_bytes())
        logger.info(f"Copied to thesis: {thesis_dir}")

    # -- JSON for the thesis caption
    json_path = output_dir / 'ablation_results.json'
    summary = {
        'pca_coarse_path': args.pca_coarse,
        'pca_fine_path': args.pca_fine,
        'real_dir': args.real_dir,
        'gen_dir': args.gen_dir,
        'coarse_pca_shape': list(coarse['pca_shape']),
        'fine_pca_shape': list(fine['pca_shape']),
        'coarse': {k: v for k, v in results['coarse'].items()
                   if k not in ('real_errors_by_k', 'gen_errors_by_k')},
        'fine':   {k: v for k, v in results['fine'].items()
                   if k not in ('real_errors_by_k', 'gen_errors_by_k')},
    }
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {json_path}")

    # -- Print caption numbers
    print()
    print("=" * 72)
    print("Ablation summary (numbers for thesis caption)")
    print("=" * 72)
    for label, pretty in [('coarse', '64×64×40'), ('fine', '128×128×80')]:
        r = results[label]
        print(f"\n{pretty}:")
        print(f"  {'k':>3} {'threshold':>10} {'real_pass':>10} {'gen_pass':>10} {'AUC':>7}")
        for k, thr, rp, gp, auc in zip(
            r['k_values'], r['threshold'],
            r['real_pass_rate'], r['gen_pass_rate'], r['auc'],
        ):
            print(f"  {k:3d} {thr:10.6f} {rp:9.1%} {gp:9.1%} {auc:7.4f}")
    print()
    print("Key comparison (first-attempt vs final choice):")
    c_idx = results['coarse']['k_values'].index(args.coarse_k)
    f_idx = results['fine']['k_values'].index(args.fine_k)
    rc = results['coarse']
    rf = results['fine']
    print(f"  64×64×40,   k={args.coarse_k:3d}: threshold={rc['threshold'][c_idx]:.4f}  "
          f"real_pass={rc['real_pass_rate'][c_idx]:.1%}  gen_pass={rc['gen_pass_rate'][c_idx]:.1%}")
    print(f"  128×128×80, k={args.fine_k:3d}: threshold={rf['threshold'][f_idx]:.4f}  "
          f"real_pass={rf['real_pass_rate'][f_idx]:.1%}  gen_pass={rf['gen_pass_rate'][f_idx]:.1%}")
    d_gen = rf['gen_pass_rate'][f_idx] - rc['gen_pass_rate'][c_idx]
    d_real = rf['real_pass_rate'][f_idx] - rc['real_pass_rate'][c_idx]
    print(f"  Δ gen_pass:  {d_gen:+.1%}  (negative = more generated rejected by final config)")
    print(f"  Δ real_pass: {d_real:+.1%}  (closer to 0 = real brains unaffected)")
    print("=" * 72)


if __name__ == '__main__':
    main()
