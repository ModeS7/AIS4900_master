"""Configuration validation.

Provides validation functions to catch configuration errors at startup
rather than during training.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Config


def validate_config(config: 'Config') -> list[str]:
    """Validate configuration at startup.

    Catches common configuration errors early to prevent training failures
    that would otherwise occur mid-run.

    Args:
        config: Configuration object to validate.

    Returns:
        List of error messages (empty if valid).

    Example:
        >>> errors = validate_config(config)
        >>> if errors:
        ...     for error in errors:
        ...         print(f"Config error: {error}")
        ...     sys.exit(1)
    """
    errors: list[str] = []

    # Validate training config
    if config.training.batch_size <= 0:
        errors.append("training.batch_size must be positive")
    if config.training.learning_rate <= 0:
        errors.append("training.learning_rate must be positive")
    if config.training.epochs <= 0:
        errors.append("training.epochs must be positive")
    if config.training.warmup_epochs < 0:
        errors.append("training.warmup_epochs must be non-negative")
    if config.training.warmup_epochs >= config.training.epochs:
        errors.append("training.warmup_epochs must be less than training.epochs")
    if config.training.gradient_clip_norm <= 0:
        errors.append("training.gradient_clip_norm must be positive")

    # Validate EMA config
    if config.training.ema.enabled:
        if not (0.9 <= config.training.ema.decay < 1.0):
            errors.append("training.ema.decay must be in [0.9, 1.0)")
        if config.training.ema.update_after_step < 0:
            errors.append("training.ema.update_after_step must be non-negative")
        if config.training.ema.update_every < 1:
            errors.append("training.ema.update_every must be at least 1")

    # Validate model config
    if config.model.image_size <= 0:
        errors.append("model.image_size must be positive")
    if config.model.spatial_dims not in (2, 3):
        errors.append("model.spatial_dims must be 2 or 3")
    if config.model.in_channels <= 0:
        errors.append("model.in_channels must be positive")
    if config.model.out_channels <= 0:
        errors.append("model.out_channels must be positive")
    if config.model.num_res_blocks < 1:
        errors.append("model.num_res_blocks must be at least 1")

    # Validate channel/attention level consistency
    if len(config.model.channels) != len(config.model.attention_levels):
        errors.append(
            f"model.channels length ({len(config.model.channels)}) must match "
            f"model.attention_levels length ({len(config.model.attention_levels)})"
        )

    # Validate 3D requires volume config
    if config.model.spatial_dims == 3 and config.volume is None:
        errors.append("volume config required when model.spatial_dims=3")

    # Validate volume config if present
    if config.volume is not None:
        if config.volume.height <= 0:
            errors.append("volume.height must be positive")
        if config.volume.width <= 0:
            errors.append("volume.width must be positive")
        if config.volume.depth <= 0:
            errors.append("volume.depth must be positive")
        if config.volume.pad_depth_to < config.volume.depth:
            errors.append("volume.pad_depth_to must be >= volume.depth")

    # Validate strategy config
    if config.strategy.name not in ('ddpm', 'rflow'):
        errors.append(f"strategy.name must be 'ddpm' or 'rflow', got '{config.strategy.name}'")
    if config.strategy.num_train_timesteps <= 0:
        errors.append("strategy.num_train_timesteps must be positive")

    # Validate mode config
    if config.mode.name not in (
        'seg', 'bravo', 'dual', 'multi', 'multi_modality',
        'seg_conditioned', 'seg_conditioned_input', 'bravo_seg_cond'
    ):
        errors.append(
            f"mode.name must be one of: seg, bravo, dual, multi, multi_modality, "
            f"seg_conditioned, seg_conditioned_input, bravo_seg_cond; got '{config.mode.name}'"
        )
    if config.mode.in_channels <= 0:
        errors.append("mode.in_channels must be positive")

    # Validate latent config if enabled
    if config.latent.enabled:
        if config.latent.compression_checkpoint is None:
            errors.append("latent.compression_checkpoint required when latent.enabled=True")
        if config.latent.scale_factor is not None and config.latent.scale_factor <= 0:
            errors.append("latent.scale_factor must be positive")

    # Validate paths
    if not config.paths.data_dir:
        errors.append("paths.data_dir is required")
    if not config.paths.model_dir:
        errors.append("paths.model_dir is required")

    return errors


def validate_and_raise(config: 'Config') -> None:
    """Validate configuration and raise if invalid.

    Convenience function that validates and raises a ValueError with
    all error messages if validation fails.

    Args:
        config: Configuration object to validate.

    Raises:
        ValueError: If configuration is invalid, with all error messages.
    """
    errors = validate_config(config)
    if errors:
        error_list = '\n  - '.join(errors)
        raise ValueError(f"Configuration validation failed:\n  - {error_list}")
