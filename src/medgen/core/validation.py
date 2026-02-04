"""Configuration validation utilities.

Provides reusable validation functions for training configurations.
Each function returns a list of error strings (empty if validation passes).
"""
import logging
import os
from collections.abc import Callable

import torch
from omegaconf import DictConfig

from .constants import ModeType

logger = logging.getLogger(__name__)


def validate_common_config(cfg: DictConfig) -> list[str]:
    """Validate common training parameters.

    Checks:
    - training.epochs > 0
    - training.batch_size > 0
    - training.learning_rate > 0
    - paths.data_dir exists
    - CUDA availability

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error strings (empty if validation passes).
    """
    errors = []

    # Training params
    if cfg.training.epochs <= 0:
        errors.append(f"epochs must be > 0, got {cfg.training.epochs}")
    if cfg.training.batch_size <= 0:
        errors.append(f"batch_size must be > 0, got {cfg.training.batch_size}")
    if cfg.training.learning_rate <= 0:
        errors.append(f"learning_rate must be > 0, got {cfg.training.learning_rate}")

    # Paths - check if data directory exists
    if not os.path.exists(cfg.paths.data_dir):
        errors.append(f"Data directory does not exist: {cfg.paths.data_dir}")

    # Check CUDA availability
    if not torch.cuda.is_available():
        errors.append("CUDA is not available. Training requires GPU.")

    return errors


def validate_model_config(cfg: DictConfig) -> list[str]:
    """Validate model configuration.

    Checks:
    - model.image_size > 0
    - Warns if image_size is not power of 2

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error strings (empty if validation passes).
    """
    errors = []

    if cfg.model.image_size <= 0:
        errors.append(f"image_size must be > 0, got {cfg.model.image_size}")

    if cfg.model.image_size & (cfg.model.image_size - 1) != 0:
        logger.warning(f"image_size {cfg.model.image_size} is not a power of 2 (may cause issues)")

    return errors


def validate_diffusion_config(cfg: DictConfig) -> list[str]:
    """Validate diffusion-specific configuration.

    Checks:
    - strategy.name in ['ddpm', 'rflow']
    - mode.name in ModeType enum

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error strings (empty if validation passes).
    """
    errors = []

    # Strategy
    if cfg.strategy.name not in ['ddpm', 'rflow']:
        errors.append(f"Unknown strategy: {cfg.strategy.name}")

    # Mode
    valid_modes = [m.value for m in ModeType]
    if cfg.mode.name not in valid_modes:
        errors.append(f"Unknown mode: {cfg.mode.name}. Valid modes: {valid_modes}")

    return errors


def validate_vae_config(cfg: DictConfig) -> list[str]:
    """Validate VAE-specific configuration.

    Checks:
    - vae section exists
    - vae.latent_channels > 0
    - vae.channels not empty

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error strings (empty if validation passes).
    """
    errors = []

    if not hasattr(cfg, 'vae'):
        errors.append("VAE configuration missing. Add 'vae' section to config.")
    else:
        if cfg.vae.latent_channels <= 0:
            errors.append(f"vae.latent_channels must be > 0, got {cfg.vae.latent_channels}")
        if len(cfg.vae.channels) == 0:
            errors.append("vae.channels must not be empty")

    return errors


def validate_vqvae_config(cfg: DictConfig) -> list[str]:
    """Validate VQ-VAE specific configuration.

    Checks:
    - vqvae section exists
    - vqvae.num_embeddings > 0
    - vqvae.embedding_dim > 0
    - vqvae.channels not empty

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error strings (empty if validation passes).
    """
    errors = []

    if not hasattr(cfg, 'vqvae'):
        errors.append("VQ-VAE configuration missing. Add 'vqvae' section to config.")
    else:
        if cfg.vqvae.num_embeddings <= 0:
            errors.append(f"vqvae.num_embeddings must be > 0, got {cfg.vqvae.num_embeddings}")
        if cfg.vqvae.embedding_dim <= 0:
            errors.append(f"vqvae.embedding_dim must be > 0, got {cfg.vqvae.embedding_dim}")
        if len(cfg.vqvae.channels) == 0:
            errors.append("vqvae.channels must not be empty")

    return errors


def validate_training_config(cfg: DictConfig) -> list[str]:
    """Validate training configuration for common issues.

    Checks:
    - use_compile + gradient_checkpointing conflict

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error strings (empty if validation passes).
    """
    errors = []
    training = cfg.get('training', {})

    # Check compile + gradient_checkpointing conflict
    # torch.compile with reduce-overhead mode uses CUDA graphs which conflict
    # with gradient checkpointing's dynamic recomputation
    use_compile = training.get('use_compile', False)
    use_checkpointing = training.get('gradient_checkpointing', False)

    if use_compile and use_checkpointing:
        errors.append(
            "use_compile=True and gradient_checkpointing=True cannot be used together. "
            "torch.compile with reduce-overhead mode uses CUDA graphs which conflict "
            "with gradient checkpointing's dynamic recomputation. Set one to False."
        )

    return errors


def validate_strategy_mode_compatibility(cfg: DictConfig) -> list[str]:
    """Check strategy and mode are compatible.

    Validates:
    - RFlow should use continuous timesteps
    - DDPM should use discrete timesteps
    - Multi-modality mode requires mode embedding

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error strings (empty if validation passes).
    """
    errors = []
    strategy = cfg.strategy.get('name', 'rflow')
    mode = cfg.mode.get('name', 'seg')

    # RFlow should use continuous timesteps
    if strategy == 'rflow' and cfg.strategy.get('use_discrete_timesteps', False):
        errors.append(
            "strategy=rflow typically uses continuous timesteps. "
            "Set use_discrete_timesteps=false or use strategy=ddpm."
        )

    # DDPM should use discrete timesteps
    if strategy == 'ddpm' and not cfg.strategy.get('use_discrete_timesteps', True):
        errors.append(
            "strategy=ddpm requires discrete timesteps. "
            "Set use_discrete_timesteps=true or use strategy=rflow."
        )

    # Multi-modality mode requires mode embedding
    if mode == 'multi_modality' and not cfg.training.get('use_mode_embedding', False):
        errors.append(
            "mode=multi_modality requires use_mode_embedding=true. "
            "Add training.use_mode_embedding=true to your config."
        )

    return errors


def validate_3d_config(cfg: DictConfig) -> list[str]:
    """Validate 3D-specific configuration.

    Checks:
    - spatial_dims=3 requires volume configuration
    - Volume dimensions must be positive
    - 3D model config is compatible

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error strings (empty if validation passes).
    """
    errors = []
    spatial_dims = cfg.model.get('spatial_dims', 2)

    if spatial_dims != 3:
        return errors  # Not 3D, skip

    # Must have volume config
    if 'volume' not in cfg or cfg.volume is None:
        errors.append(
            "spatial_dims=3 requires volume configuration. "
            "Add volume section with depth, height, width."
        )
        return errors

    # Check dimensions are positive
    for dim in ['depth', 'height', 'width']:
        val = cfg.volume.get(dim, 0)
        if val <= 0:
            errors.append(f"volume.{dim} must be positive, got {val}")

    return errors


def validate_latent_config(cfg: DictConfig) -> list[str]:
    """Validate latent diffusion configuration.

    Checks:
    - latent.enabled=true requires compression checkpoint
    - Checkpoint file exists (if path provided)

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error strings (empty if validation passes).
    """
    errors = []

    latent_cfg = cfg.get('latent', {})
    if not latent_cfg.get('enabled', False):
        return errors  # Not latent, skip

    # Must have checkpoint
    checkpoint = latent_cfg.get('compression_checkpoint')
    if not checkpoint:
        errors.append(
            "latent.enabled=true requires latent.compression_checkpoint. "
            "Provide path to trained VAE/VQ-VAE/DC-AE checkpoint."
        )

    # Check checkpoint exists (if path provided)
    if checkpoint and not os.path.exists(checkpoint):
        errors.append(f"Compression checkpoint not found: {checkpoint}")

    return errors


def validate_regional_logging(cfg: DictConfig) -> list[str]:
    """Validate regional logging configuration.

    Checks that regional losses logging is compatible with the mode.
    Regional losses need segmentation masks as conditioning (e.g., bravo, dual).

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error strings (empty if validation passes).
    """
    errors = []

    logging_cfg = cfg.training.get('logging', {})
    if not logging_cfg.get('regional_losses', False):
        return errors

    mode = cfg.mode.get('name', 'seg')
    # Regional losses need segmentation masks as separate conditioning
    # seg mode generates masks (has no separate mask input)
    # seg_conditioned generates masks from size_bin input (has no pixel mask)
    if mode in ('seg', 'seg_conditioned'):
        errors.append(
            "logging.regional_losses=true requires a mode with segmentation masks "
            "as conditioning (e.g., bravo, dual). seg mode has no separate mask."
        )

    return errors


def run_validation(
    cfg: DictConfig,
    validators: list[Callable[[DictConfig], list[str]]],
) -> None:
    """Run multiple validators and raise if any errors.

    Args:
        cfg: Hydra configuration object.
        validators: List of validation functions to run.

    Raises:
        ValueError: If any validation fails.
    """
    all_errors = []
    for validator in validators:
        errors = validator(cfg)
        all_errors.extend(errors)

    if all_errors:
        raise ValueError(
            "Configuration validation failed:\n  - " + "\n  - ".join(all_errors)
        )
