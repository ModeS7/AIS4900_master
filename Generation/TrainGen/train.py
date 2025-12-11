"""
Training entry point for diffusion models on medical images.

This module provides the main training script for training diffusion models
(DDPM or Rectified Flow) on brain MRI data with various training modes.

Usage:
    python train.py --epochs 500 --mode seg --strategy ddpm --compute local
    python train.py --epochs 500 --mode bravo --strategy rflow --compute cluster
    python train.py --epochs 500 --mode dual --strategy ddpm --multi_gpu
"""
import argparse
import os
import sys
from typing import Literal

import torch

# Add TrainGen directory for core module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add project root for config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import PathConfig
from core.data import create_dataloader, create_dual_image_dataloader
from core.trainer import DiffusionTrainer

# Enable CUDA optimizations
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch._dynamo.config.cache_size_limit = 32

TrainingMode = Literal['seg', 'bravo', 'dual']
DiffusionStrategy = Literal['ddpm', 'rflow']


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for training configuration.

    Returns:
        Parsed argument namespace with training parameters.
    """
    parser = argparse.ArgumentParser(
        description='Train diffusion model for medical image generation'
    )
    parser.add_argument(
        '--epochs', type=int, default=500,
        help='Number of training epochs (default: 500)'
    )
    parser.add_argument(
        '--val_interval', type=int, default=30,
        help='Validation interval in epochs (default: 30)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Batch size for training (default: 16)'
    )
    parser.add_argument(
        '--image_size', type=int, default=128,
        help='Image size for training (default: 128)'
    )
    parser.add_argument(
        '--compute', type=str, choices=['local', 'cluster'], default='local',
        help='Compute environment (default: local)'
    )
    parser.add_argument(
        '--num_timesteps', type=int, default=1000,
        help='Number of diffusion timesteps (default: 1000)'
    )
    parser.add_argument(
        '--warmup_epochs', type=int, default=5,
        help='Number of epochs for learning rate warmup (default: 5)'
    )
    parser.add_argument(
        '--mode', type=str, choices=['seg', 'bravo', 'dual'], default='seg',
        help='Training mode: seg (masks), bravo (bravo+mask), dual (T1 pre+post+mask)'
    )
    parser.add_argument(
        '--strategy', type=str, choices=['ddpm', 'rflow'], default='ddpm',
        help='Diffusion strategy: ddpm or rflow (default: ddpm)'
    )
    parser.add_argument(
        '--multi_gpu', action='store_true',
        help='Enable multi-GPU distributed training'
    )
    parser.add_argument(
        '--find_lr', action='store_true',
        help='Run LR finder before training to automatically select optimal learning rate'
    )
    parser.add_argument(
        '--no_ema', action='store_true',
        help='Disable EMA (Exponential Moving Average) weights tracking'
    )
    parser.add_argument(
        '--no_min_snr', action='store_true',
        help='Disable Min-SNR loss weighting'
    )

    return parser.parse_args()


def train_segmentation_mode(
    args: argparse.Namespace,
    strategy: DiffusionStrategy,
    use_multi_gpu: bool
) -> None:
    """Train segmentation mask generation model (unconditional).

    Args:
        args: Parsed command line arguments.
        strategy: Diffusion strategy ('ddpm' or 'rflow').
        use_multi_gpu: Whether to use multi-GPU training.
    """
    print(f"\n{'=' * 50}")
    print(f"Training segmentation masks with {strategy}")
    print("Unconditional generation (no conditioning)")
    print(f"{'=' * 50}\n")

    trainer = DiffusionTrainer(
        strategy=strategy,
        mode='seg',
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=1e-4,
        perceptual_weight=0.001,
        num_timesteps=args.num_timesteps,
        warmup_epochs=args.warmup_epochs,
        val_interval=args.val_interval,
        compute=args.compute,
        use_multi_gpu=use_multi_gpu,
        use_ema=not args.no_ema,
        use_min_snr=not args.no_min_snr
    )

    dataloader, train_dataset = create_dataloader(
        path_config=trainer.path_config,
        image_type='seg',
        image_size=args.image_size,
        batch_size=args.batch_size,
        use_distributed=use_multi_gpu,
        rank=trainer.rank if use_multi_gpu else 0,
        world_size=trainer.world_size if use_multi_gpu else 1
    )

    trainer.setup_model(train_dataset)

    # Run LR finder if requested
    if args.find_lr:
        optimal_lr = trainer.find_optimal_lr(dataloader)
        trainer.update_learning_rate(optimal_lr)

    trainer.train(dataloader, train_dataset)


def train_bravo_mode(
    args: argparse.Namespace,
    strategy: DiffusionStrategy,
    use_multi_gpu: bool
) -> None:
    """Train BRAVO image generation model conditioned on segmentation.

    Args:
        args: Parsed command line arguments.
        strategy: Diffusion strategy ('ddpm' or 'rflow').
        use_multi_gpu: Whether to use multi-GPU training.
    """
    print(f"\n{'=' * 50}")
    print(f"Training BRAVO images with {strategy}")
    print("Conditioned on segmentation masks")
    print(f"{'=' * 50}\n")

    trainer = DiffusionTrainer(
        strategy=strategy,
        mode='bravo',
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=1e-4,
        perceptual_weight=0.001,
        num_timesteps=args.num_timesteps,
        warmup_epochs=args.warmup_epochs,
        val_interval=args.val_interval,
        compute=args.compute,
        use_multi_gpu=use_multi_gpu,
        use_ema=not args.no_ema,
        use_min_snr=not args.no_min_snr
    )

    dataloader, train_dataset = create_dataloader(
        path_config=trainer.path_config,
        image_type='bravo',
        image_size=args.image_size,
        batch_size=args.batch_size,
        use_distributed=use_multi_gpu,
        rank=trainer.rank if use_multi_gpu else 0,
        world_size=trainer.world_size if use_multi_gpu else 1
    )

    trainer.setup_model(train_dataset)

    # Run LR finder if requested
    if args.find_lr:
        optimal_lr = trainer.find_optimal_lr(dataloader)
        trainer.update_learning_rate(optimal_lr)

    trainer.train(dataloader, train_dataset)


def train_dual_mode(
    args: argparse.Namespace,
    strategy: DiffusionStrategy,
    use_multi_gpu: bool
) -> None:
    """Train dual-image model (T1 pre + T1 gd) conditioned on segmentation.

    Model learns anatomical consistency between pre- and post-contrast images.

    Args:
        args: Parsed command line arguments.
        strategy: Diffusion strategy ('ddpm' or 'rflow').
        use_multi_gpu: Whether to use multi-GPU training.
    """
    print(f"\n{'=' * 50}")
    print(f"Training dual-image (T1 pre + T1 gd) with {strategy}")
    print("Conditioned on segmentation masks")
    print("Model learns anatomical consistency between images")
    print(f"{'=' * 50}\n")

    trainer = DiffusionTrainer(
        strategy=strategy,
        mode='dual',
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=1e-4,
        perceptual_weight=0.001,
        num_timesteps=args.num_timesteps,
        warmup_epochs=args.warmup_epochs,
        val_interval=args.val_interval,
        compute=args.compute,
        use_multi_gpu=use_multi_gpu,
        use_ema=not args.no_ema,
        use_min_snr=not args.no_min_snr
    )

    image_keys = ['t1_pre', 't1_gd']

    dataloader, train_dataset = create_dual_image_dataloader(
        path_config=trainer.path_config,
        image_keys=image_keys,
        conditioning='seg',
        image_size=args.image_size,
        batch_size=args.batch_size,
        use_distributed=use_multi_gpu,
        rank=trainer.rank if use_multi_gpu else 0,
        world_size=trainer.world_size if use_multi_gpu else 1
    )

    trainer.setup_model(train_dataset)

    # Run LR finder if requested
    if args.find_lr:
        optimal_lr = trainer.find_optimal_lr(dataloader)
        trainer.update_learning_rate(optimal_lr)

    trainer.train(dataloader, train_dataset)


def main() -> None:
    """Main entry point for training."""
    args = parse_arguments()

    strategy: DiffusionStrategy = getattr(args, 'strategy', 'ddpm')
    mode: TrainingMode = getattr(args, 'mode', 'seg')
    use_multi_gpu: bool = getattr(args, 'multi_gpu', False)
    find_lr: bool = getattr(args, 'find_lr', False)
    use_ema: bool = not getattr(args, 'no_ema', False)
    use_min_snr: bool = not getattr(args, 'no_min_snr', False)

    print(f"Strategy: {strategy} | Mode: {mode} | Multi-GPU: {use_multi_gpu} | Find LR: {find_lr}")
    print(f"EMA: {use_ema} | Min-SNR: {use_min_snr}")

    if mode == 'seg':
        train_segmentation_mode(args, strategy, use_multi_gpu)
    elif mode == 'bravo':
        train_bravo_mode(args, strategy, use_multi_gpu)
    elif mode == 'dual':
        train_dual_mode(args, strategy, use_multi_gpu)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from ['seg', 'bravo', 'dual']")


if __name__ == "__main__":
    main()
