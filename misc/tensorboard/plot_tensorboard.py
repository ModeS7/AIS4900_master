#!/usr/bin/env python3
"""
Plot TensorBoard training logs.
Reads TensorBoard event files and creates visualizations of training metrics.
"""

import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict

# Configuration
TENSORBOARD_ROOT = Path(__file__).parent.parent / 'tensorboard_logs' / 'successful'
OUTPUT_DIR = Path(__file__).parent / 'plots'


def load_tensorboard_data(log_dir):
    """
    Load all scalar data from a TensorBoard log directory.

    Returns:
        dict: {metric_name: [(step, value), ...]}
    """
    event_acc = EventAccumulator(str(log_dir))
    event_acc.Reload()

    data = {}
    for tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]

    return data


def plot_single_run(log_dir, output_dir=None, metrics=None, log_scale=False, ylim=None):
    """Plot metrics for a single training run."""
    log_path = Path(log_dir)
    run_name = log_path.name

    print(f"\nLoading data from: {run_name}")
    data = load_tensorboard_data(log_path)

    if not data:
        print(f"No scalar data found in {log_path}")
        return

    available_metrics = list(data.keys())
    print(f"Available metrics: {', '.join(available_metrics)}")

    # Filter metrics if specified
    if metrics:
        metrics_to_plot = [m for m in metrics if m in available_metrics]
        if not metrics_to_plot:
            print(f"None of the specified metrics found. Available: {available_metrics}")
            return
    else:
        metrics_to_plot = available_metrics

    # Create plots
    n_metrics = len(metrics_to_plot)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        steps, values = zip(*data[metric])

        ax.plot(steps, values, linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title(metric)
        if log_scale:
            ax.set_yscale('log')
        if ylim is not None:
            ax.set_ylim(top=ylim)
        ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f"Training Metrics: {run_name}", fontsize=14, y=1.00)
    plt.tight_layout()

    # Save plot
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / f"{run_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_comparison(log_dirs, metric, output_dir=None, smooth=0, log_scale=False, ylim=None):
    """
    Compare a single metric across multiple runs.

    Args:
        log_dirs: List of log directories
        metric: Metric name to compare
        output_dir: Where to save the plot
        smooth: Smoothing window size (0 = no smoothing)
        log_scale: Use logarithmic y-axis
        ylim: Maximum y-axis value
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for log_dir in log_dirs:
        log_path = Path(log_dir)
        run_name = log_path.name

        data = load_tensorboard_data(log_path)

        if metric not in data:
            print(f"Metric '{metric}' not found in {run_name}")
            continue

        steps, values = zip(*data[metric])

        # Apply smoothing if requested
        if smooth > 0:
            values = np.convolve(values, np.ones(smooth)/smooth, mode='valid')
            steps = steps[:len(values)]

        ax.plot(steps, values, label=run_name, linewidth=2, alpha=0.8)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Value (log scale)' if log_scale else 'Value', fontsize=12)
    ax.set_title(f'Comparison: {metric}', fontsize=14)
    if log_scale:
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(top=ylim)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / f"comparison_{metric.replace('/', '_')}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")

    plt.show()


def list_runs(root_dir=TENSORBOARD_ROOT):
    """List all available training runs."""
    root = Path(root_dir)
    runs = sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith('.')])

    print(f"\n{'='*80}")
    print(f"Available Training Runs ({len(runs)})")
    print(f"{'='*80}")

    for idx, run in enumerate(runs, 1):
        # Check if it has event files
        event_files = list(run.glob("events.out.tfevents.*"))
        if event_files:
            print(f"{idx:2d}. {run.name}")

    print(f"{'='*80}\n")


def list_metrics(log_dir):
    """List all metrics in a log directory."""
    data = load_tensorboard_data(log_dir)

    print(f"\n{'='*80}")
    print(f"Available Metrics in: {Path(log_dir).name}")
    print(f"{'='*80}")

    for idx, metric in enumerate(sorted(data.keys()), 1):
        num_points = len(data[metric])
        print(f"{idx:2d}. {metric} ({num_points} data points)")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Plot TensorBoard training logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available runs
  python plot_tensorboard.py --list

  # Plot all metrics from a single run
  python plot_tensorboard.py --run Diffusion_bravo_128_20250919-101308

  # Plot specific metrics
  python plot_tensorboard.py --run Diffusion_bravo_128_20250919-101308 --metrics loss

  # List metrics in a run
  python plot_tensorboard.py --run Diffusion_bravo_128_20250919-101308 --list-metrics

  # Compare loss across multiple runs
  python plot_tensorboard.py --compare loss --runs Diffusion_bravo_128_20250919-101308 RFlow_seg_128_recovered_20250919-132722

  # Compare with smoothing
  python plot_tensorboard.py --compare loss --runs run1 run2 --smooth 10

  # Compare with logarithmic scale
  python plot_tensorboard.py --compare "Loss/Total" --runs run1 run2 --log

  # Compare with custom y-axis limit
  python plot_tensorboard.py --compare "Loss/Total" --runs run1 run2 --ylim 0.02

  # Specify custom tensorboard root
  python plot_tensorboard.py --root /path/to/logs --list
        """
    )

    parser.add_argument('--root', type=str, default=str(TENSORBOARD_ROOT),
                        help='Root directory containing TensorBoard logs')
    parser.add_argument('--output', type=str, default=str(OUTPUT_DIR),
                        help='Output directory for plots')

    # List options
    parser.add_argument('--list', action='store_true',
                        help='List all available runs')
    parser.add_argument('--list-metrics', action='store_true',
                        help='List metrics in a specific run (requires --run)')

    # Single run plotting
    parser.add_argument('--run', type=str,
                        help='Name of training run to plot')
    parser.add_argument('--metrics', type=str, nargs='+',
                        help='Specific metrics to plot (default: all)')

    # Comparison plotting
    parser.add_argument('--compare', type=str,
                        help='Metric to compare across runs')
    parser.add_argument('--runs', type=str, nargs='+',
                        help='List of run names to compare')
    parser.add_argument('--smooth', type=int, default=0,
                        help='Smoothing window size (default: 0)')
    parser.add_argument('--log', action='store_true',
                        help='Use logarithmic scale for y-axis')
    parser.add_argument('--ylim', type=float, default=0.01,
                        help='Maximum y-axis value (default: 0.01)')

    args = parser.parse_args()

    # List all runs
    if args.list:
        list_runs(args.root)
        return

    # List metrics in a run
    if args.list_metrics:
        if not args.run:
            print("Error: --list-metrics requires --run")
            return
        log_dir = Path(args.root) / args.run
        if not log_dir.exists():
            print(f"Error: Run directory not found: {log_dir}")
            return
        list_metrics(log_dir)
        return

    # Compare metric across runs
    if args.compare:
        if not args.runs or len(args.runs) < 2:
            print("Error: --compare requires at least 2 runs specified with --runs")
            return

        log_dirs = [Path(args.root) / run for run in args.runs]
        for log_dir in log_dirs:
            if not log_dir.exists():
                print(f"Error: Run directory not found: {log_dir}")
                return

        plot_comparison(log_dirs, args.compare, args.output, args.smooth, args.log, args.ylim)
        return

    # Plot single run
    if args.run:
        log_dir = Path(args.root) / args.run
        if not log_dir.exists():
            print(f"Error: Run directory not found: {log_dir}")
            return

        plot_single_run(log_dir, args.output, args.metrics, args.log, args.ylim)
        return

    # No action specified
    parser.print_help()


if __name__ == '__main__':
    main()
