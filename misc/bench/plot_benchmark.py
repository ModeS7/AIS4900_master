#!/usr/bin/env python3
"""
Visualize benchmark results.
Creates comparison plots for training and generation performance.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path(__file__).parent / 'benchmark_results'


def plot_training_results(csv_path, output_dir=None):
    """Plot training benchmark results."""
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Iterations per second
    ax = axes[0]
    configs = df['config']
    its = df['mean_it_per_sec']
    its_err = df['std_it_per_sec']

    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    bars = ax.bar(range(len(configs)), its, yerr=its_err, capsize=5, color=colors, alpha=0.8)

    ax.set_ylabel('Iterations per Second', fontsize=12)
    ax.set_title('Training Throughput', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=15, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val, err) in enumerate(zip(bars, its, its_err)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + err + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Calculate and show speedup
    baseline_its = its.iloc[0]
    speedups = its / baseline_its
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        if i > 0:  # Skip baseline
            ax.text(bar.get_x() + bar.get_width()/2., 1.0,
                    f'{speedup:.2f}x', ha='center', va='bottom',
                    fontsize=9, color='blue', fontweight='bold')

    # Plot 2: VRAM usage
    ax = axes[1]
    vram_peak = df['vram_peak_gb']
    vram_current = df['vram_current_gb']

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, vram_peak, width, label='Peak', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, vram_current, width, label='Current', color='#3498db', alpha=0.8)

    ax.set_ylabel('VRAM Usage (GB)', fontsize=12)
    ax.set_title('Memory Usage', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    # Plot 3: Combined efficiency (throughput / VRAM)
    ax = axes[2]
    efficiency = its / vram_peak
    bars = ax.bar(range(len(configs)), efficiency, color=colors, alpha=0.8)

    ax.set_ylabel('Iterations per Second per GB VRAM', fontsize=12)
    ax.set_title('Memory Efficiency', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=15, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, efficiency):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Training Performance Benchmark', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Save plot
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / f"training_benchmark_{Path(csv_path).stem}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training plot saved to: {save_path}")

    plt.show()


def plot_generation_results(csv_path, output_dir=None):
    """Plot generation benchmark results."""
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Samples per second
    ax = axes[0]
    configs = df['config']
    sps = df['mean_samples_per_sec']
    sps_err = df['std_samples_per_sec']

    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    bars = ax.bar(range(len(configs)), sps, yerr=sps_err, capsize=5, color=colors, alpha=0.8)

    ax.set_ylabel('Samples per Second', fontsize=12)
    ax.set_title('Generation Throughput', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=15, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val, err) in enumerate(zip(bars, sps, sps_err)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + err + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Calculate and show speedup
    baseline_sps = sps.iloc[0]
    speedups = sps / baseline_sps
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        if i > 0:  # Skip baseline
            y_pos = height * 0.1 if height > 0.1 else 0.01
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{speedup:.2f}x', ha='center', va='bottom',
                    fontsize=9, color='blue', fontweight='bold')

    # Plot 2: VRAM usage
    ax = axes[1]
    vram_peak = df['vram_peak_gb']
    vram_current = df['vram_current_gb']

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, vram_peak, width, label='Peak', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, vram_current, width, label='Current', color='#3498db', alpha=0.8)

    ax.set_ylabel('VRAM Usage (GB)', fontsize=12)
    ax.set_title('Memory Usage', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    # Plot 3: Time per batch
    ax = axes[2]
    time_per_batch = df['mean_sec_per_batch']
    time_err = df['std_sec_per_batch']
    bars = ax.bar(range(len(configs)), time_per_batch, yerr=time_err,
                  capsize=5, color=colors, alpha=0.8)

    batch_size = df['batch_size'].iloc[0]
    gen_steps = df['gen_steps'].iloc[0]
    ax.set_ylabel('Seconds per Batch', fontsize=12)
    ax.set_title(f'Generation Time\n({batch_size} images, {gen_steps} steps)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=15, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val, err in zip(bars, time_per_batch, time_err):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + err + 0.5,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Generation Performance Benchmark', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Save plot
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / f"generation_benchmark_{Path(csv_path).stem}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Generation plot saved to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--training', type=str,
                        help='Path to training benchmark CSV')
    parser.add_argument('--generation', type=str,
                        help='Path to generation benchmark CSV')
    parser.add_argument('--latest', action='store_true',
                        help='Plot latest benchmark results')
    parser.add_argument('--output', type=str, default=str(RESULTS_DIR),
                        help='Output directory for plots')

    args = parser.parse_args()

    results_dir = Path(RESULTS_DIR)

    # Plot latest results
    if args.latest:
        # Find latest training results
        training_files = sorted(results_dir.glob('training_benchmark_*.csv'))
        if training_files:
            print(f"Plotting latest training results: {training_files[-1].name}")
            plot_training_results(training_files[-1], args.output)
        else:
            print("No training benchmark results found")

        # Find latest generation results
        gen_files = sorted(results_dir.glob('generation_benchmark_*.csv'))
        if gen_files:
            print(f"Plotting latest generation results: {gen_files[-1].name}")
            plot_generation_results(gen_files[-1], args.output)
        else:
            print("No generation benchmark results found")

        return

    # Plot specific files
    if args.training:
        plot_training_results(args.training, args.output)

    if args.generation:
        plot_generation_results(args.generation, args.output)

    if not args.training and not args.generation and not args.latest:
        parser.print_help()


if __name__ == '__main__':
    main()
