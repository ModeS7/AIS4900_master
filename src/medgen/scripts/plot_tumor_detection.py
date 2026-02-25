"""Plot per-tumor detection analysis from saved JSON records.

Produces two plots:
1. Scatter: Feret diameter (mm) vs Dice score, colored by size category
2. Bar: Detection rate by tumor size category

Usage:
    python -m medgen.scripts.plot_tumor_detection path/to/per_tumor_detection.json [--output-dir .]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# RANO-BM boundaries (mm)
_RANO_BOUNDARIES = [10, 20, 30]

_SIZE_COLORS = {
    'tiny': '#1f77b4',    # blue
    'small': '#2ca02c',   # green
    'medium': '#ff7f0e',  # orange
    'large': '#d62728',   # red
}

_SIZE_ORDER = ['tiny', 'small', 'medium', 'large']


def plot_scatter(tumors: list[dict], detection_threshold: float, output_path: Path) -> None:
    """Scatter plot: Feret diameter vs Dice, colored by size category."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for size in _SIZE_ORDER:
        records = [t for t in tumors if t['size_cat'] == size]
        if not records:
            continue
        x = [t['feret_mm'] for t in records]
        y = [t['dice'] for t in records]
        ax.scatter(x, y, c=_SIZE_COLORS[size], label=size, alpha=0.6, s=20, edgecolors='none')

    # Detection threshold line
    ax.axhline(y=detection_threshold, color='gray', linestyle='--', linewidth=1,
               label=f'Detection threshold ({detection_threshold})')

    # RANO-BM vertical boundary lines
    for boundary in _RANO_BOUNDARIES:
        ax.axvline(x=boundary, color='lightgray', linestyle=':', linewidth=0.8)
        ax.text(boundary, 1.02, f'{boundary}mm', transform=ax.get_xaxis_transform(),
                ha='center', fontsize=8, color='gray')

    ax.set_xlabel('Feret Diameter (mm)')
    ax.set_ylabel('Dice Score')
    ax.set_title('Per-Tumor Detection: Feret Diameter vs Dice Score')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved scatter plot: {output_path}")


def plot_detection_bars(summary: dict, tumors: list[dict], output_path: Path) -> None:
    """Bar chart: Detection rate by tumor size category."""
    fig, ax = plt.subplots(figsize=(8, 5))

    rates = []
    counts = []
    colors = []
    labels = []

    for size in _SIZE_ORDER:
        key = f'detection_rate_{size}'
        rate = summary.get(key, 0.0)
        n = sum(1 for t in tumors if t['size_cat'] == size)
        rates.append(rate)
        counts.append(n)
        colors.append(_SIZE_COLORS[size])
        labels.append(size)

    x = np.arange(len(labels))
    bars = ax.bar(x, rates, color=colors, edgecolor='black', linewidth=0.5)

    # Add percentage + count labels on bars
    for bar, rate, n in zip(bars, rates, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{rate:.0%}\nn={n}', ha='center', va='bottom', fontsize=10)

    # Overall detection rate
    overall = summary.get('detection_rate', 0.0)
    ax.axhline(y=overall, color='black', linestyle='--', linewidth=1,
               label=f'Overall: {overall:.0%}')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Detection Rate')
    ax.set_title('Detection Rate by Tumor Size')
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # FP annotation
    fp = summary.get('false_positives', 0)
    ax.text(0.98, 0.95, f'False Positives: {fp:.0f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved bar plot: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot per-tumor detection analysis')
    parser.add_argument('json_path', type=str, help='Path to per_tumor_detection.json')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: same as JSON)')
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: {json_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    tumors = data['tumors']
    summary = data['summary']
    threshold = data.get('detection_threshold', 0.1)

    if not tumors:
        print("No tumor records found in JSON", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else json_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(tumors)} tumor records")
    print(f"Overall detection rate: {summary.get('detection_rate', 0):.1%}")
    print(f"False positives: {summary.get('false_positives', 0):.0f}")

    plot_scatter(tumors, threshold, output_dir / 'tumor_feret_vs_dice.png')
    plot_detection_bars(summary, tumors, output_dir / 'detection_rate_by_size.png')


if __name__ == '__main__':
    main()
