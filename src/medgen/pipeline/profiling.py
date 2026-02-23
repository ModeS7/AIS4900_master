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
from typing import TYPE_CHECKING, Any

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


def get_metadata_extra(trainer: 'DiffusionTrainer') -> dict[str, Any]:
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
        'use_ema': trainer.use_ema,
        'created_at': datetime.now().isoformat(),
    }


def get_model_config(trainer: 'DiffusionTrainer') -> dict[str, Any]:
    """Get model configuration for checkpoint.

    Includes architecture params so checkpoints are self-describing
    and can be loaded without hardcoding defaults.

    Args:
        trainer: The DiffusionTrainer instance.

    Returns:
        Dict with model configuration.
    """
    model_cfg = trainer.mode.get_model_config()
    from .base_config import ModelConfig
    mc = ModelConfig.from_hydra(trainer.cfg)

    config = {
        'model_type': trainer.model_type,
        'in_channels': model_cfg['in_channels'],
        'out_channels': model_cfg['out_channels'],
        'strategy': trainer.strategy_name,
        'mode': trainer.mode_name,
        'spatial_dims': mc.spatial_dims,
    }

    # Architecture params differ between UNet and transformer
    if trainer.is_transformer:
        config.update({
            'image_size': mc.image_size,
            'patch_size': mc.patch_size,
            'variant': mc.variant,
            'mlp_ratio': mc.mlp_ratio,
            'conditioning': mc.conditioning,
            'qk_norm': getattr(trainer.cfg.model, 'qk_norm', True),
        })
        # HDiT-specific
        if mc.level_depths is not None:
            config['level_depths'] = mc.level_depths
    else:
        config.update({
            'channels': list(mc.channels),
            'attention_levels': list(mc.attention_levels),
            'num_res_blocks': mc.num_res_blocks,
            'num_head_channels': mc.num_head_channels,
        })

    # Pixel config (for brain-only normalization and [-1,1] rescaling)
    from medgen.diffusion.spaces import LatentSpace, PixelSpace, WaveletSpace
    if isinstance(trainer.space, PixelSpace):
        pixel_cfg: dict[str, Any] = {}
        if trainer.space.shift is not None:
            pixel_cfg['pixel_shift'] = trainer.space.shift
            pixel_cfg['pixel_scale'] = trainer.space.scale
        if trainer.space._rescale:
            pixel_cfg['rescale'] = True
        if pixel_cfg:
            config['pixel'] = pixel_cfg

    # Wavelet config (for WDM — save stats so generation doesn't recompute)
    if isinstance(trainer.space, WaveletSpace):
        wavelet_config: dict[str, Any] = {
            'rescale': trainer.space.rescale,
        }
        if trainer.space.shift is not None:
            wavelet_config['wavelet_shift'] = trainer.space.shift
            wavelet_config['wavelet_scale'] = trainer.space.scale
        config['wavelet'] = wavelet_config

    # Latent config (for LDM — save normalization stats so generation
    # doesn't need the latent cache metadata.json)
    if isinstance(trainer.space, LatentSpace):
        latent_config: dict[str, Any] = {
            'compression_type': trainer.space.compression_type,
            'scale_factor': trainer.space.scale_factor,
            'latent_channels': trainer.space.latent_channels,
        }
        if trainer.space.latent_shift is not None:
            latent_config['latent_shift'] = trainer.space.latent_shift
            latent_config['latent_scale'] = trainer.space.latent_scale
        if trainer.space.latent_seg_shift is not None:
            latent_config['latent_seg_shift'] = trainer.space.latent_seg_shift
            latent_config['latent_seg_scale'] = trainer.space.latent_seg_scale
        config['latent'] = latent_config

    # Size bin config (for seg_conditioned mode)
    if getattr(trainer, 'use_size_bin_embedding', False):
        config['size_bin'] = {
            'num_bins': trainer.size_bin_num_bins,
            'max_count': trainer.size_bin_max_count,
            'embed_dim': trainer.size_bin_embed_dim,
            'projection_hidden_dim': trainer.size_bin_projection_hidden_dim,
            'projection_num_layers': trainer.size_bin_projection_num_layers,
            'aux_loss_weight': getattr(trainer, 'size_bin_aux_loss_weight', 0.0),
        }

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
        # Use a temporary single-process loader to avoid poisoning the
        # train_loader's persistent workers.  Creating then abandoning an
        # iterator on a DataLoader with persistent_workers=True can deadlock
        # when the training loop later tries to create its own iterator
        # (workers are stuck with un-consumed prefetched items).
        temp_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=train_loader.collate_fn,
        )
        batch = next(iter(temp_loader))
        prepared = trainer.mode.prepare_batch(batch, trainer.device)
        images = prepared['images']
        labels = prepared.get('labels')
        is_latent = prepared.get('is_latent', False)
        labels_is_latent = prepared.get('labels_is_latent', False)

        # Encode to diffusion space (S2D, wavelet, latent, or identity for pixel)
        # Must match what train_epoch does so dummy input has correct channels
        # Skip encoding if data is already in latent space (from latent dataloader)
        if not is_latent:
            images = trainer.space.encode_batch(images)
        if labels is not None and not labels_is_latent:
            labels = trainer.space.encode(labels)

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
            batch_size=trainer.batch_size,
            timesteps=timesteps[:1] if isinstance(timesteps, torch.Tensor) else timesteps,
            is_main_process=trainer.is_main_process,
        )
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if trainer.is_main_process:
            logger.warning(f"FLOPs measurement failed (OOM or CUDA error): {e}")
    except StopIteration:
        if trainer.is_main_process:
            logger.warning("FLOPs measurement failed: empty dataloader")
    except (ImportError, AssertionError) as e:
        if trainer.is_main_process:
            logger.warning(f"FLOPs measurement skipped ({type(e).__name__}: {e})")
    except (TypeError, ValueError, AttributeError) as e:
        # Prevent training crash from unexpected FLOPs errors
        if trainer.is_main_process:
            logger.warning(f"FLOPs measurement failed unexpectedly: {type(e).__name__}: {e}")


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
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata['final_loss'] = final_loss
        metadata['final_mse'] = final_mse
        metadata['total_time_seconds'] = total_time
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    # Save JSON history files (regional_losses.json, timestep_losses.json, etc.)
    trainer._unified_metrics.save_json_histories(trainer.save_dir)
