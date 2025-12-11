#!/usr/bin/env python3
"""
Plot learning rate scheduler over training epochs.

Usage:
    python plot_lr_sch.py --epochs 500 --scheduler cosine --lr 1e-4
    python plot_lr_sch.py --epochs 500 --scheduler linear --lr 1e-4 --warmup 10
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def get_cosine_schedule(lr: float, epochs: int, warmup_epochs: int = 0, min_lr: float = 0.0):
    """Cosine annealing learning rate schedule with optional warmup."""
    lrs = []

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            # Linear warmup
            lr_value = lr * (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            lr_value = min_lr + (lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        lrs.append(lr_value)

    return lrs


def get_linear_schedule(lr: float, epochs: int, warmup_epochs: int = 0, min_lr: float = 0.0):
    """Linear decay learning rate schedule with optional warmup."""
    lrs = []

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            # Linear warmup
            lr_value = lr * (epoch + 1) / warmup_epochs
        else:
            # Linear decay
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            lr_value = lr - (lr - min_lr) * progress

        lrs.append(lr_value)

    return lrs


def get_step_schedule(lr: float, epochs: int, step_size: int = 100, gamma: float = 0.1):
    """Step decay learning rate schedule."""
    lrs = []

    for epoch in range(epochs):
        lr_value = lr * (gamma ** (epoch // step_size))
        lrs.append(lr_value)

    return lrs


def get_exponential_schedule(lr: float, epochs: int, gamma: float = 0.95):
    """Exponential decay learning rate schedule."""
    lrs = []

    for epoch in range(epochs):
        lr_value = lr * (gamma ** epoch)
        lrs.append(lr_value)

    return lrs


def get_constant_schedule(lr: float, epochs: int):
    """Constant learning rate (no decay)."""
    return [lr] * epochs


def plot_scheduler(scheduler_type: str, lr: float, epochs: int, output_path: Path = None, **kwargs):
    """Plot learning rate schedule."""

    # Get learning rate values
    if scheduler_type == 'cosine':
        lrs = get_cosine_schedule(lr, epochs, **kwargs)
        title = f'Cosine Annealing LR Schedule (warmup={kwargs.get("warmup_epochs", 0)})'
    elif scheduler_type == 'linear':
        lrs = get_linear_schedule(lr, epochs, **kwargs)
        title = f'Linear Decay LR Schedule (warmup={kwargs.get("warmup_epochs", 0)})'
    elif scheduler_type == 'step':
        lrs = get_step_schedule(lr, epochs, **kwargs)
        title = f'Step Decay LR Schedule (step_size={kwargs.get("step_size", 100)}, gamma={kwargs.get("gamma", 0.1)})'
    elif scheduler_type == 'exponential':
        lrs = get_exponential_schedule(lr, epochs, **kwargs)
        title = f'Exponential Decay LR Schedule (gamma={kwargs.get("gamma", 0.95)})'
    elif scheduler_type == 'constant':
        lrs = get_constant_schedule(lr, epochs)
        title = 'Constant LR Schedule'
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    epochs_range = range(1, epochs + 1)
    ax.plot(epochs_range, lrs, linewidth=2, color='#2E86AB')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add statistics
    stats_text = f"Initial LR: {lr:.2e}\n"
    stats_text += f"Final LR: {lrs[-1]:.2e}\n"
    stats_text += f"Min LR: {min(lrs):.2e}\n"
    stats_text += f"Max LR: {max(lrs):.2e}"

    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot learning rate scheduler over training epochs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cosine schedule with warmup
  python plot_lr_sch.py --epochs 500 --scheduler cosine --lr 1e-4 --warmup 10

  # Linear decay
  python plot_lr_sch.py --epochs 500 --scheduler linear --lr 1e-4 --warmup 5

  # Step decay
  python plot_lr_sch.py --epochs 500 --scheduler step --lr 1e-4 --step-size 100 --gamma 0.5

  # Exponential decay
  python plot_lr_sch.py --epochs 500 --scheduler exponential --lr 1e-4 --gamma 0.99

  # Constant (no decay)
  python plot_lr_sch.py --epochs 500 --scheduler constant --lr 1e-4

  # Save to file
  python plot_lr_sch.py --epochs 500 --scheduler cosine --lr 1e-4 --warmup 10 -o scheduler.png
        """
    )

    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs (default: 500, matches production)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'linear', 'step', 'exponential', 'constant'],
                        help='Scheduler type (default: cosine, matches production)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate (default: 1e-4, matches production)')
    parser.add_argument('--warmup', type=int, default=0,
                        help='Number of warmup epochs (default: 0, matches production)')
    parser.add_argument('--min-lr', type=float, default=1e-5,
                        help='Minimum learning rate for cosine/linear (default: 1e-5, matches production eta_min)')
    parser.add_argument('--step-size', type=int, default=100,
                        help='Step size for step scheduler (default: 100)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Decay factor for step/exponential scheduler (default: 0.1)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file path (default: show plot)')

    args = parser.parse_args()

    # Prepare kwargs based on scheduler type
    kwargs = {}
    if args.scheduler in ['cosine', 'linear']:
        kwargs['warmup_epochs'] = args.warmup
        kwargs['min_lr'] = args.min_lr
    elif args.scheduler == 'step':
        kwargs['step_size'] = args.step_size
        kwargs['gamma'] = args.gamma
    elif args.scheduler == 'exponential':
        kwargs['gamma'] = args.gamma

    # Plot
    output_path = Path(args.output) if args.output else None
    plot_scheduler(args.scheduler, args.lr, args.epochs, output_path, **kwargs)


if __name__ == '__main__':
    main()
