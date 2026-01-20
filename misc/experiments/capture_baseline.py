#!/usr/bin/env python3
"""
Capture baseline metrics for regression testing.

This script captures deterministic metrics BEFORE any refactoring changes.
The captured baseline is used to verify that the refactored code produces
identical results.

Usage:
    # Capture baselines:
    python misc/experiments/capture_baseline.py --spatial_dims 2 --mode bravo --output tests/integration/baselines/2d_bravo_baseline.json
    python misc/experiments/capture_baseline.py --spatial_dims 2 --mode seg --output tests/integration/baselines/2d_seg_baseline.json
    python misc/experiments/capture_baseline.py --spatial_dims 3 --mode bravo --output tests/integration/baselines/3d_bravo_baseline.json
    python misc/experiments/capture_baseline.py --spatial_dims 3 --mode seg --output tests/integration/baselines/3d_seg_baseline.json

    # Compare against baseline:
    python misc/experiments/capture_baseline.py --spatial_dims 2 --mode bravo --compare tests/integration/baselines/2d_bravo_baseline.json
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

# Add project root to path (misc/experiments -> misc -> project_root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def set_deterministic(seed: int = 42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_synthetic_batch_2d(batch_size: int = 4, image_size: int = 64, channels: int = 2, seed: int = 0) -> torch.Tensor:
    """Create deterministic 2D batch (bravo mode format: [B, 2, H, W])."""
    set_deterministic(seed)
    # Channel 0: BRAVO image, Channel 1: segmentation mask
    batch = torch.randn(batch_size, channels, image_size, image_size)
    # Make seg channel binary-ish
    batch[:, -1] = (batch[:, -1] > 0.5).float()
    return batch


def create_synthetic_batch_3d(batch_size: int = 1, depth: int = 32, height: int = 64, width: int = 64, mode: str = 'bravo', seed: int = 0):
    """Create deterministic 3D batch.

    For bravo/dual: dict with 'image' [B, 1, D, H, W] and 'seg' [B, 1, D, H, W]
    For seg: tuple (seg_tensor, size_bins) - SegmentationConditionedMode expects this

    Note: 3D seg mode uses SegmentationConditionedMode which expects tuple format,
    while 2D seg mode uses SegmentationMode which handles both dict and tensor.
    This is one of the divergences being fixed in the refactor.
    """
    set_deterministic(seed)

    if mode == 'seg':
        # 3D seg mode (SegmentationConditionedMode): expects tuple (seg, size_bins)
        seg = (torch.randn(batch_size, 1, depth, height, width) > 0.5).float()
        size_bins = torch.zeros(batch_size, 9)  # Placeholder size bins
        size_bins[:, 0] = 1  # At least one tumor in first bin
        return (seg, size_bins)
    else:
        # Conditional modes: image to generate + seg conditioning
        batch = {
            'image': torch.randn(batch_size, 1, depth, height, width),
            'seg': (torch.randn(batch_size, 1, depth, height, width) > 0.5).float(),
        }
        return batch


def hash_tensor(tensor: torch.Tensor) -> str:
    """Create deterministic hash of tensor for comparison."""
    # Convert to numpy for consistent hashing
    arr = tensor.detach().cpu().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def hash_model_state(model: nn.Module) -> str:
    """Hash model state for verification."""
    state_bytes = b""
    for name, param in sorted(model.named_parameters()):
        state_bytes += param.detach().cpu().numpy().tobytes()
    return hashlib.sha256(state_bytes).hexdigest()[:32]


def create_minimal_config(spatial_dims: int, mode: str = 'bravo') -> Dict[str, Any]:
    """Create minimal config dict for trainer initialization using Hydra compose."""
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    # Use Hydra to compose the full config from actual config files
    config_dir = str(project_root / "configs")

    # Initialize Hydra (suppress output)
    try:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    except Exception:
        pass

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # Compose with appropriate mode
        if spatial_dims == 3:
            cfg = compose(
                config_name="diffusion_3d",
                overrides=[
                    f"mode={mode}",
                    "strategy=rflow",
                    "paths=local",
                    # Small model for fast testing
                    "model.channels=[64,128]",  # Divisible by num_head_channels (32)
                    "model.attention_levels=[false,true]",
                    "model.num_res_blocks=1",
                    "model.num_head_channels=32",
                    # Minimal training settings
                    "training.epochs=10",  # Need at least warmup_epochs + 1
                    "training.warmup_epochs=0",
                    "training.batch_size=1",
                    "training.use_ema=false",
                    "training.use_compile=false",
                    "training.compile_fused_forward=false",
                    "training.generation_metrics.enabled=false",
                    "training.logging.flops=false",
                    "training.perceptual_weight=0.0",
                    # Volume settings
                    "volume.height=64",
                    "volume.width=64",
                    "volume.depth=32",
                ]
            )
        else:
            cfg = compose(
                config_name="diffusion",
                overrides=[
                    f"mode={mode}",
                    "strategy=rflow",
                    "paths=local",
                    # Small model for fast testing
                    "model.image_size=64",
                    "model.channels=[64,128]",  # Divisible by num_head_channels (32)
                    "model.attention_levels=[false,true]",
                    "model.num_res_blocks=1",
                    "model.num_head_channels=32",
                    # Minimal training settings
                    "training.epochs=10",  # Need at least warmup_epochs + 1
                    "training.warmup_epochs=0",
                    "training.batch_size=4",
                    "training.use_ema=false",
                    "training.use_compile=false",
                    "training.compile_fused_forward=false",
                    "training.generation_metrics.enabled=false",
                    "training.logging.flops=false",
                    "training.perceptual_weight=0.0",
                ]
            )

    # Make config writable and override output dirs
    OmegaConf.set_struct(cfg, False)
    cfg.paths.output_dir = '/tmp/baseline_capture'
    cfg.paths.model_dir = '/tmp/baseline_capture'

    return cfg


def capture_train_step_metrics(
    trainer,
    batch: Any,
    spatial_dims: int,
    step: int,
) -> Dict[str, Any]:
    """Capture metrics from a single training step."""
    # Run train step
    result = trainer.train_step(batch)

    # Capture gradients
    grad_norms = {}
    for name, param in trainer.model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    # Capture key weights (first and last layer)
    weight_samples = {}
    params = list(trainer.model.named_parameters())
    if params:
        # First layer
        name, param = params[0]
        weight_samples[f"first_{name}"] = param.detach()[:5].flatten()[:10].tolist() if param.numel() >= 10 else param.detach().flatten().tolist()
        # Last layer
        name, param = params[-1]
        weight_samples[f"last_{name}"] = param.detach()[:5].flatten()[:10].tolist() if param.numel() >= 10 else param.detach().flatten().tolist()

    return {
        'step': step,
        'total_loss': result.total_loss,
        'mse_loss': result.mse_loss,
        'perceptual_loss': getattr(result, 'perceptual_loss', 0.0),
        'grad_norm_sum': sum(grad_norms.values()),
        'weight_samples': weight_samples,
    }


def capture_baseline(
    spatial_dims: int,
    num_steps: int = 10,
    seed: int = 42,
    mode: str = 'bravo',
) -> Dict[str, Any]:
    """
    Capture baseline metrics for regression testing.

    Args:
        spatial_dims: 2 for 2D, 3 for 3D
        num_steps: Number of training steps to capture
        seed: Random seed for reproducibility
        mode: Training mode ('seg', 'bravo', 'dual')

    Returns:
        Dict with captured metrics
    """
    logger.info(f"Capturing baseline for {spatial_dims}D {mode} mode...")

    # Set deterministic
    set_deterministic(seed)

    # Create config
    cfg = create_minimal_config(spatial_dims, mode)

    # Create trainer (unified 2D/3D via spatial_dims parameter)
    from medgen.pipeline.trainer import DiffusionTrainer
    trainer = DiffusionTrainer(cfg, spatial_dims=spatial_dims)

    # Setup model - create minimal mock dataset just for the interface
    class MockDataset:
        def __len__(self):
            return 10

    trainer.setup_model(MockDataset())

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer.model.to(device)

    # Capture step metrics
    step_metrics = []
    for step in range(num_steps):
        set_deterministic(seed + step)  # Different seed per step for diversity

        # Create batch
        if spatial_dims == 2:
            batch = create_synthetic_batch_2d(
                batch_size=cfg.training.batch_size,
                image_size=cfg.model.image_size,
                channels=2 if mode in ('bravo', 'dual') else 1,
                seed=seed + step,
            ).to(device)
        else:
            batch = create_synthetic_batch_3d(
                batch_size=cfg.training.batch_size,
                depth=cfg.volume.depth,
                height=cfg.volume.height,
                width=cfg.volume.width,
                mode=mode,
                seed=seed + step,
            )
            # Move batch to device (handle dict or tuple)
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
            elif isinstance(batch, tuple):
                batch = tuple(v.to(device) for v in batch)

        metrics = capture_train_step_metrics(trainer, batch, spatial_dims, step)
        step_metrics.append(metrics)

        logger.info(f"  Step {step}: loss={metrics['total_loss']:.6f}")

    # Final model state hash
    final_state_hash = hash_model_state(trainer.model)

    return {
        'metadata': {
            'spatial_dims': spatial_dims,
            'mode': mode,
            'seed': seed,
            'num_steps': num_steps,
            'device': str(device),
            'torch_version': torch.__version__,
        },
        'step_metrics': step_metrics,
        'final_state_hash': final_state_hash,
    }


def compare_baselines(baseline: Dict[str, Any], current: Dict[str, Any], tolerance: float = 1e-5) -> bool:
    """
    Compare captured baseline with current run.

    Returns True if they match within tolerance.
    """
    all_match = True

    # Compare metadata
    for key in ['spatial_dims', 'mode', 'seed', 'num_steps']:
        if baseline['metadata'][key] != current['metadata'][key]:
            logger.error(f"Metadata mismatch: {key} = {baseline['metadata'][key]} vs {current['metadata'][key]}")
            all_match = False

    # Compare step metrics
    for i, (base_step, curr_step) in enumerate(zip(baseline['step_metrics'], current['step_metrics'])):
        loss_diff = abs(base_step['total_loss'] - curr_step['total_loss'])
        if loss_diff > tolerance:
            logger.error(f"Step {i} loss mismatch: {base_step['total_loss']:.8f} vs {curr_step['total_loss']:.8f} (diff={loss_diff:.8f})")
            all_match = False
        else:
            logger.info(f"Step {i} loss matches: {base_step['total_loss']:.8f} (diff={loss_diff:.2e})")

    # Compare final state hash
    if baseline['final_state_hash'] != current['final_state_hash']:
        logger.error(f"Final state hash mismatch!")
        logger.error(f"  Baseline: {baseline['final_state_hash']}")
        logger.error(f"  Current:  {current['final_state_hash']}")
        all_match = False
    else:
        logger.info(f"Final state hash matches: {baseline['final_state_hash']}")

    return all_match


def main():
    parser = argparse.ArgumentParser(description='Capture baseline metrics for regression testing')
    parser.add_argument('--spatial_dims', type=int, choices=[2, 3], default=2,
                        help='Spatial dimensions (2 or 3)')
    parser.add_argument('--mode', type=str, default='bravo',
                        choices=['seg', 'bravo'],
                        help='Training mode')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for baseline JSON')
    parser.add_argument('--compare', type=str, default=None,
                        help='Compare against existing baseline file')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='Number of steps to capture')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Capture current metrics
    current = capture_baseline(
        spatial_dims=args.spatial_dims,
        num_steps=args.num_steps,
        seed=args.seed,
        mode=args.mode,
    )

    if args.compare:
        # Compare mode
        logger.info(f"\nComparing against baseline: {args.compare}")
        with open(args.compare, 'r') as f:
            baseline = json.load(f)

        if compare_baselines(baseline, current):
            logger.info("\n✓ All metrics match baseline!")
            sys.exit(0)
        else:
            logger.error("\n✗ Metrics differ from baseline!")
            sys.exit(1)

    elif args.output:
        # Save mode
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(current, f, indent=2)

        logger.info(f"\nBaseline saved to: {output_path}")

    else:
        # Just print
        logger.info("\nCaptured metrics (use --output to save):")
        for step in current['step_metrics']:
            logger.info(f"  Step {step['step']}: loss={step['total_loss']:.6f}")
        logger.info(f"Final state hash: {current['final_state_hash']}")


if __name__ == '__main__':
    main()
