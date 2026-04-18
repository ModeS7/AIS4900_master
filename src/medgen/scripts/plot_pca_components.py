#!/usr/bin/env python3
"""Generate PCA explained-variance plots for thesis.

Produces:
  Plot 1 (cumulative): cumulative explained variance ratio vs component index,
          spanning the full PCA rank (n_samples - 1), with reference lines at
          the truncation choices (30 and 60 components).
  Plot 2 (scree):      per-component variance ratio on log scale.

Both plots are saved as PNG and PDF to the thesis Images directory.

Requires the .npz to contain `full_singular_values` (the entire S array from
the SVD, pre-truncation). Re-run compute_brain_pca.py after its update to
populate this key — the legacy 60-component `explained_variance` array is
insufficient because it hides the tail of the curve.

Usage:
    python -m medgen.scripts.plot_pca_components \\
        --npz data/brain_pca_256x256x160.npz \\
        --output-dir /path/to/thesis/Images/pca_components
"""
import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


REF_COMPONENTS = (30, 60)  # First-attempt and final choices


def _load_variance(npz_path: Path) -> tuple[np.ndarray, int]:
    """Return (explained_variance_ratio_full, n_samples).

    Prefer `full_singular_values` (entire S); else fall back to the truncated
    `explained_variance` with a warning (plot will only span saved components).
    """
    data = np.load(npz_path)
    n_samples = int(data['n_samples'][0])

    if 'full_singular_values' in data.files:
        s = data['full_singular_values'].astype(np.float64)
        ev = (s ** 2) / (n_samples - 1)
        total = ev.sum()
        logger.info(f"Using full_singular_values: rank={len(s)}, total_var={total:.4g}")
        return ev / total, n_samples

    logger.warning(
        "full_singular_values missing. Falling back to truncated explained_variance "
        "— plot will NOT span the full rank. Re-run compute_brain_pca.py to populate."
    )
    ev = data['explained_variance'].astype(np.float64)
    # We cannot recover the tail; dividing by truncated sum overstates ratios.
    return ev / ev.sum(), n_samples


def plot_cumulative(ratios: np.ndarray, output_base: Path) -> None:
    """Plot 1: cumulative explained variance ratio."""
    n = len(ratios)
    cumulative = np.cumsum(ratios)
    components = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(components, cumulative, color='steelblue', linewidth=2)

    for k in REF_COMPONENTS:
        if k > n:
            continue
        y = cumulative[k - 1]
        ax.axvline(k, color='firebrick', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y, color='firebrick', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.annotate(
            f'k={k}\n{y:.1%}',
            xy=(k, y), xytext=(k + 5, y - 0.08),
            fontsize=9,
            arrowprops=dict(arrowstyle='->', color='firebrick', alpha=0.7, lw=0.8),
        )

    ax.set_xlabel('number of components')
    ax.set_ylabel('cumulative explained variance ratio')
    ax.set_xlim(1, n)
    ax.set_ylim(0, 1.02)

    step = 10 if n <= 50 else 20
    ax.set_xticks(np.arange(0, n + 1, step))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.grid(True, which='both', linestyle='-', linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_scree(ratios: np.ndarray, output_base: Path) -> None:
    """Plot 2: per-component variance ratio (log scale).

    Clipped to the effective numerical rank — components whose explained
    variance ratio falls below 1e-8 are numerical SVD noise, not signal,
    and plotting them to 10^-30 would mislead the reader.
    """
    n = int(np.sum(ratios >= 1e-8))
    components = np.arange(1, n + 1)
    ratios_plot = ratios[:n]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.semilogy(components, ratios_plot, color='darkorange', linewidth=1.5, marker='.', markersize=3)

    for k in REF_COMPONENTS:
        if k > n:
            continue
        ax.axvline(k, color='firebrick', linestyle='--', linewidth=1, alpha=0.6)

    ax.set_xlabel('component index')
    ax.set_ylabel('explained variance ratio (log scale)')
    ax.set_xlim(1, n)

    step = 10 if n <= 50 else 20
    ax.set_xticks(np.arange(0, n + 1, step))
    ax.grid(True, which='both', linestyle='-', linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf (clipped to effective rank {n})")


def print_caption_numbers(ratios: np.ndarray, n_samples: int) -> None:
    """Print values that should appear in the figure caption / thesis text."""
    cumulative = np.cumsum(ratios)
    n = len(ratios)

    print()
    print("=" * 60)
    print("Numbers for figure caption / thesis text")
    print("=" * 60)
    print(f"Total samples fit: {n_samples}")
    print(f"PCA rank (saved):  {n}")
    print()
    print("Cumulative variance ratio at specific components:")
    for k in (30, 60, 100):
        if k > n:
            print(f"  k={k}: N/A (only {n} components available)")
        else:
            print(f"  k={k:3d}: {cumulative[k - 1]:.4f}  ({cumulative[k - 1]:.2%})")
    print()
    print("Components required to reach a cumulative variance threshold:")
    for target in (0.90, 0.95, 0.99):
        hits = np.where(cumulative >= target)[0]
        if len(hits) == 0:
            print(f"  >={target:.2%}: not reached within {n} components")
        else:
            k = int(hits[0]) + 1
            print(f"  >={target:.2%}: k={k:3d}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PCA explained variance for thesis")
    parser.add_argument(
        '--npz', default='data/brain_pca_256x256x160.npz',
        help='Path to brain_pca .npz file (must contain full_singular_values)',
    )
    parser.add_argument(
        '--output-dir',
        default='/home/mode/NTNU/AIS4900_doc/AIS4900-master-thesis/Images/pca_components',
        help='Where to save PNG+PDF outputs',
    )
    args = parser.parse_args()

    npz_path = Path(args.npz)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ratios, n_samples = _load_variance(npz_path)

    logger.info(f"Plotting to {output_dir}")
    plot_cumulative(ratios, output_dir / 'pca_cumulative')
    plot_scree(ratios, output_dir / 'pca_scree')
    print_caption_numbers(ratios, n_samples)


if __name__ == '__main__':
    main()
