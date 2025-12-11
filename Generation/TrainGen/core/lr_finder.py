"""
Learning Rate Finder utility for diffusion models.

This module provides a learning rate range test to find optimal learning rates
by sweeping through a range and plotting loss vs learning rate.

Usage:
    python -m core.lr_finder --mode seg --strategy ddpm --min_lr 1e-7 --max_lr 1e-1
"""
import argparse
import os
import sys
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast
from torch.optim import AdamW
from tqdm import tqdm

from monai.losses import PerceptualLoss
from monai.networks.nets import DiffusionModelUNet

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config import PathConfig
from core.data import create_dataloader, create_dual_image_dataloader
from core.modes import ConditionalDualMode, ConditionalSingleMode, SegmentationMode
from core.strategies import DDPMStrategy, RFlowStrategy


def find_lr(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    strategy,
    mode,
    perceptual_loss_fn: torch.nn.Module,
    device: torch.device,
    min_lr: float = 1e-7,
    max_lr: float = 1e-1,
    num_steps: int = 100,
    perceptual_weight: float = 0.001,
    smoothing: float = 0.05
) -> Tuple[List[float], List[float]]:
    """Run learning rate range test.

    Args:
        model: The model to test.
        dataloader: DataLoader providing training batches.
        strategy: Diffusion strategy instance.
        mode: Training mode instance.
        perceptual_loss_fn: Perceptual loss function.
        device: Device to run on.
        min_lr: Minimum learning rate to test.
        max_lr: Maximum learning rate to test.
        num_steps: Number of LR steps to test.
        perceptual_weight: Weight for perceptual loss.
        smoothing: Exponential smoothing factor for loss.

    Returns:
        Tuple of (learning_rates, losses).
    """
    # Setup optimizer with minimum LR
    optimizer = AdamW(model.parameters(), lr=min_lr)

    # Compute multiplicative factor for LR increase
    lr_mult = (max_lr / min_lr) ** (1.0 / num_steps)

    learning_rates: List[float] = []
    losses: List[float] = []
    smoothed_loss: Optional[float] = None
    best_loss = float('inf')

    model.train()
    data_iter = iter(dataloader)
    step = 0

    pbar = tqdm(total=num_steps, desc="LR Finder")

    while step < num_steps:
        # Get next batch (cycle through dataloader)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Forward pass
        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=torch.bfloat16):
            prepared = mode.prepare_batch(batch, device)
            images = prepared['images']
            labels_dict = {'labels': prepared.get('labels')}

            if isinstance(images, dict):
                noise = {key: torch.randn_like(img).to(device) for key, img in images.items()}
            else:
                noise = torch.randn_like(images).to(device)

            timesteps = strategy.sample_timesteps(images)
            noisy_images = strategy.add_noise(images, noise, timesteps)
            model_input = mode.format_model_input(noisy_images, labels_dict)
            prediction = strategy.predict_noise_or_velocity(model, model_input, timesteps)
            mse_loss, predicted_clean = strategy.compute_loss(prediction, images, noise, noisy_images, timesteps)

            if isinstance(predicted_clean, dict):
                p_loss = sum(
                    perceptual_loss_fn(pred.float(), images[key].float())
                    for key, pred in predicted_clean.items()
                )
            else:
                p_loss = perceptual_loss_fn(predicted_clean.float(), images.float())

            total_loss = mse_loss + perceptual_weight * p_loss

        # Backward and step
        total_loss.backward()
        optimizer.step()

        loss_val = total_loss.item()

        # Smooth the loss
        if smoothed_loss is None:
            smoothed_loss = loss_val
        else:
            smoothed_loss = smoothing * loss_val + (1 - smoothing) * smoothed_loss

        # Track best loss
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss

        # Stop if loss explodes (4x the best loss)
        if step > 10 and smoothed_loss > 4 * best_loss:
            print(f"\nStopping early: loss exploded at LR={optimizer.param_groups[0]['lr']:.2e}")
            break

        # Record
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        losses.append(smoothed_loss)

        # Update LR
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult

        step += 1
        pbar.update(1)
        pbar.set_postfix(lr=f"{current_lr:.2e}", loss=f"{smoothed_loss:.4f}")

    pbar.close()
    return learning_rates, losses


def plot_lr_finder(
    learning_rates: List[float],
    losses: List[float],
    save_path: str,
    suggested_lr: Optional[float] = None
) -> None:
    """Plot learning rate finder results.

    Args:
        learning_rates: List of learning rates tested.
        losses: List of corresponding losses.
        save_path: Path to save the plot.
        suggested_lr: Optional suggested LR to mark on plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, losses, 'b-', linewidth=2)
    plt.xscale('log')

    # Find and mark the suggested LR (steepest descent point)
    if suggested_lr is None and len(losses) > 10:
        # Find point of steepest descent
        gradients = np.gradient(losses)
        min_grad_idx = np.argmin(gradients)
        suggested_lr = learning_rates[min_grad_idx]

    if suggested_lr is not None:
        plt.axvline(x=suggested_lr, color='r', linestyle='--', linewidth=2,
                    label=f'Suggested LR: {suggested_lr:.2e}')
        plt.legend()

    plt.xlabel('Learning Rate', fontsize=12)
    plt.ylabel('Loss (smoothed)', fontsize=12)
    plt.title('Learning Rate Finder', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Mark the minimum loss region
    min_loss_idx = np.argmin(losses)
    plt.scatter([learning_rates[min_loss_idx]], [losses[min_loss_idx]],
                color='green', s=100, zorder=5, label=f'Min loss at LR={learning_rates[min_loss_idx]:.2e}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {save_path}")


def main():
    """Run the LR finder."""
    parser = argparse.ArgumentParser(description='Learning Rate Finder for Diffusion Models')
    parser.add_argument('--mode', type=str, choices=['seg', 'bravo', 'dual'], default='seg',
                        help='Training mode')
    parser.add_argument('--strategy', type=str, choices=['ddpm', 'rflow'], default='ddpm',
                        help='Diffusion strategy')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-1, help='Maximum learning rate')
    parser.add_argument('--num_steps', type=int, default=1000, help='Number of LR steps to test')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--image_size', type=int, default=128, help='Image size')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--compute', type=str, choices=['local', 'cluster'], default='local',
                        help='Compute environment')
    parser.add_argument('--output', type=str, default='lr_finder_plot.png',
                        help='Output plot filename')
    args = parser.parse_args()

    device = torch.device('cuda')
    path_config = PathConfig(compute=args.compute)

    # Create strategy
    if args.strategy == 'ddpm':
        strategy = DDPMStrategy()
    else:
        strategy = RFlowStrategy()
    scheduler = strategy.setup_scheduler(args.num_timesteps, args.image_size)

    # Create mode
    modes = {
        'seg': SegmentationMode,
        'bravo': ConditionalSingleMode,
        'dual': ConditionalDualMode
    }
    mode = modes[args.mode]()
    model_config = mode.get_model_config()

    # Create model
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256
    ).to(device)

    # Create perceptual loss
    perceptual_loss_fn = PerceptualLoss(
        spatial_dims=2,
        network_type="radimagenet_resnet50",
        cache_dir=str(path_config.cache_dir),
        pretrained=True,
    ).to(device)

    # Create dataloader
    if args.mode == 'dual':
        dataloader, _ = create_dual_image_dataloader(
            path_config=path_config,
            image_keys=['t1_pre', 't1_gd'],
            conditioning='seg',
            image_size=args.image_size,
            batch_size=args.batch_size
        )
    else:
        image_type = 'seg' if args.mode == 'seg' else 'bravo'
        dataloader, _ = create_dataloader(
            path_config=path_config,
            image_type=image_type,
            image_size=args.image_size,
            batch_size=args.batch_size
        )

    print(f"\nRunning LR Finder:")
    print(f"  Mode: {args.mode}")
    print(f"  Strategy: {args.strategy}")
    print(f"  LR Range: {args.min_lr:.2e} - {args.max_lr:.2e}")
    print(f"  Steps: {args.num_steps}")
    print()

    # Run LR finder
    learning_rates, losses = find_lr(
        model=model,
        dataloader=dataloader,
        strategy=strategy,
        mode=mode,
        perceptual_loss_fn=perceptual_loss_fn,
        device=device,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_steps=args.num_steps
    )

    # Plot results
    output_path = str(path_config.base_prefix / 'AIS4005_IP' / 'misc' / args.output)
    plot_lr_finder(learning_rates, losses, output_path)

    # Print suggestions
    if len(losses) > 10:
        gradients = np.gradient(losses)
        min_grad_idx = np.argmin(gradients)
        suggested_lr = learning_rates[min_grad_idx]
        min_loss_idx = np.argmin(losses)
        min_loss_lr = learning_rates[min_loss_idx]

        print(f"\nResults:")
        print(f"  Suggested LR (steepest descent): {suggested_lr:.2e}")
        print(f"  LR at minimum loss: {min_loss_lr:.2e}")
        print(f"  Recommended: Use LR between {suggested_lr:.2e} and {min_loss_lr:.2e}")


if __name__ == '__main__':
    main()
