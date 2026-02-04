"""Profiling and metadata utilities for diffusion training.

This module provides functions for:
- Model FLOPs measurement
- Training metadata collection
- Model configuration serialization
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from medgen.pipeline.trainer import DiffusionTrainer

logger = logging.getLogger(__name__)


def get_trainer_type() -> str:
    """Return trainer type identifier for metadata.

    Returns:
        String identifier for this trainer type.
    """
    return 'diffusion'


def get_metadata_extra(trainer: 'DiffusionTrainer') -> Dict[str, Any]:
    """Collect diffusion-specific metadata for checkpoints.

    Args:
        trainer: The DiffusionTrainer instance.

    Returns:
        Dict with diffusion-specific metadata fields.
    """
    return {
        'strategy': trainer.strategy_name,
        'mode': trainer.mode_name,
        'image_size': trainer.image_size,
        'num_timesteps': trainer.num_timesteps,
        'batch_size': trainer.batch_size,
        'learning_rate': trainer.learning_rate,
        'use_sam': trainer.use_sam,
        'use_ema': trainer.use_ema,
        'created_at': datetime.now().isoformat(),
    }


def get_model_config(trainer: 'DiffusionTrainer') -> Dict[str, Any]:
    """Get model configuration for checkpoint.

    Includes architecture params so checkpoints are self-describing
    and can be loaded without hardcoding defaults.

    Args:
        trainer: The DiffusionTrainer instance.

    Returns:
        Dict with model configuration.
    """
    model_cfg = trainer.mode.get_model_config()
    config = {
        'model_type': trainer.model_type,
        'in_channels': model_cfg['in_channels'],
        'out_channels': model_cfg['out_channels'],
        'strategy': trainer.strategy_name,
        'mode': trainer.mode_name,
        'spatial_dims': trainer.cfg.model.get('spatial_dims', 2),
    }

    # Architecture params differ between UNet and transformer
    if trainer.is_transformer:
        config.update({
            'image_size': trainer.cfg.model.image_size,
            'patch_size': trainer.cfg.model.patch_size,
            'variant': trainer.cfg.model.variant,
            'mlp_ratio': trainer.cfg.model.get('mlp_ratio', 4.0),
            'conditioning': trainer.cfg.model.get('conditioning', 'concat'),
            'qk_norm': trainer.cfg.model.get('qk_norm', True),
        })
    else:
        config.update({
            'channels': list(trainer.cfg.model.channels),
            'attention_levels': list(trainer.cfg.model.attention_levels),
            'num_res_blocks': trainer.cfg.model.num_res_blocks,
            'num_head_channels': trainer.cfg.model.num_head_channels,
        })

    return config


def measure_model_flops(
    trainer: 'DiffusionTrainer',
    train_loader: DataLoader,
) -> None:
    """Measure model FLOPs using batch_size=1 to avoid OOM during torch.compile.

    Args:
        trainer: The DiffusionTrainer instance.
        train_loader: Training data loader.
    """
    if not trainer.log_flops:
        return

    try:
        batch = next(iter(train_loader))
        prepared = trainer.mode.prepare_batch(batch, trainer.device)
        images = prepared['images']
        labels = prepared.get('labels')

        # Slice to batch_size=1 to avoid OOM during torch.compile tracing
        # torch.compile compiles for specific shapes; using full batch can cause
        # excessive memory during the compilation graph creation
        if isinstance(images, dict):
            images = {key: img[:1] for key, img in images.items()}
            noise = {key: torch.randn_like(img).to(trainer.device) for key, img in images.items()}
        else:
            images = images[:1]
            noise = torch.randn_like(images).to(trainer.device)

        if labels is not None:
            labels = labels[:1]
        labels_dict = {'labels': labels}

        timesteps = trainer.strategy.sample_timesteps(images)
        noisy_images = trainer.strategy.add_noise(images, noise, timesteps)

        # For ControlNet (Stage 1 or 2): use only noisy images
        if trainer.use_controlnet or trainer.controlnet_stage1:
            model_input = noisy_images
        else:
            model_input = trainer.mode.format_model_input(noisy_images, labels_dict)

        trainer._flops_tracker.measure(
            trainer.model_raw,
            model_input[:1] if isinstance(model_input, torch.Tensor) else model_input,
            steps_per_epoch=len(train_loader),
            batch_size=trainer.cfg.training.batch_size,
            timesteps=timesteps[:1] if isinstance(timesteps, torch.Tensor) else timesteps,
            is_main_process=trainer.is_main_process,
        )
    except Exception as e:
        if trainer.is_main_process:
            logger.warning(f"Could not measure FLOPs: {e}")


def update_metadata_final(
    trainer: 'DiffusionTrainer',
    final_loss: float,
    final_mse: float,
    total_time: float,
) -> None:
    """Update metadata with final training stats.

    Args:
        trainer: The DiffusionTrainer instance.
        final_loss: Final training loss.
        final_mse: Final MSE loss.
        total_time: Total training time in seconds.
    """
    metadata_path = os.path.join(trainer.save_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        metadata['final_loss'] = final_loss
        metadata['final_mse'] = final_mse
        metadata['total_time_seconds'] = total_time
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    # Save JSON history files (regional_losses.json, timestep_losses.json, etc.)
    trainer._unified_metrics.save_json_histories(trainer.save_dir)
