"""Configuration validation utilities.

Provides reusable validation functions for training configurations.
Each function returns a list of error strings (empty if validation passes).
"""
import logging
import os
from typing import Callable, List

import torch
from omegaconf import DictConfig

from .constants import ModeType

logger = logging.getLogger(__name__)


def validate_common_config(cfg: DictConfig) -> List[str]:
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


def validate_model_config(cfg: DictConfig) -> List[str]:
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


def validate_diffusion_config(cfg: DictConfig) -> List[str]:
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


def validate_vae_config(cfg: DictConfig) -> List[str]:
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


def validate_vqvae_config(cfg: DictConfig) -> List[str]:
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


def validate_training_config(cfg: DictConfig) -> List[str]:
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


def run_validation(
    cfg: DictConfig,
    validators: List[Callable[[DictConfig], List[str]]],
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
