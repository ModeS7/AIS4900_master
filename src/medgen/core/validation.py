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
    errors: list[str] = []

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
    errors: list[str] = []

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
    errors: list[str] = []

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
    errors: list[str] = []

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
    errors: list[str] = []

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
    - gradient_clip_norm > 0
    - warmup_epochs >= 0

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error strings (empty if validation passes).
    """
    errors: list[str] = []
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

    # Validate gradient_clip_norm
    gradient_clip_norm = training.get('gradient_clip_norm', 1.0)
    if gradient_clip_norm <= 0:
        errors.append(f"training.gradient_clip_norm must be > 0, got {gradient_clip_norm}")

    # Validate warmup_epochs
    warmup_epochs = training.get('warmup_epochs', 0)
    if warmup_epochs < 0:
        errors.append(f"training.warmup_epochs must be >= 0, got {warmup_epochs}")

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
    errors: list[str] = []
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
    if mode in ('multi', 'multi_modality') and not cfg.training.get('use_mode_embedding', False):
        errors.append(
            "mode=multi requires use_mode_embedding=true. "
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
    errors: list[str] = []
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
    errors: list[str] = []

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

    Regional losses/weighting need segmentation masks as conditioning.
    For seg modes (no separate mask), these features are silently skipped
    at runtime — no need to block the entire run.

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error strings (empty — regional config is always valid).
    """
    return []


def validate_strategy_config(cfg: DictConfig) -> list[str]:
    """Validate strategy configuration.

    Checks:
    - strategy.num_train_timesteps > 0

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error strings (empty if validation passes).
    """
    errors: list[str] = []
    strategy = cfg.get('strategy', {})

    num_timesteps = strategy.get('num_train_timesteps', 1000)
    if num_timesteps <= 0:
        errors.append(f"strategy.num_train_timesteps must be > 0, got {num_timesteps}")

    return errors


def validate_ema_config(cfg: DictConfig) -> list[str]:
    """Validate EMA configuration.

    Checks:
    - training.ema.decay in [0.9, 1.0)
    - training.ema.update_after_step >= 0
    - training.ema.update_every > 0

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error strings (empty if validation passes).
    """
    errors: list[str] = []
    training = cfg.get('training', {})

    if not training.get('use_ema', False):
        return errors  # EMA disabled, skip validation

    ema = training.get('ema', {})

    decay = ema.get('decay', 0.9999)
    if not (0.9 <= decay < 1.0):
        errors.append(f"training.ema.decay must be in [0.9, 1.0), got {decay}")

    update_after_step = ema.get('update_after_step', 100)
    if update_after_step < 0:
        errors.append(f"training.ema.update_after_step must be >= 0, got {update_after_step}")

    update_every = ema.get('update_every', 10)
    if update_every <= 0:
        errors.append(f"training.ema.update_every must be > 0, got {update_every}")

    return errors


def validate_optimizer_config(cfg: DictConfig) -> list[str]:
    """Validate optimizer configuration.

    Checks:
    - training.optimizer.betas is list of 2 floats in [0, 1)
    - training.optimizer.weight_decay >= 0

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error strings (empty if validation passes).
    """
    from omegaconf import ListConfig
    errors: list[str] = []
    training = cfg.get('training', {})
    optimizer = training.get('optimizer', {})

    # Validate betas
    betas = optimizer.get('betas', [0.9, 0.999])
    if not isinstance(betas, (list, tuple, ListConfig)):
        errors.append(f"training.optimizer.betas must be a list, got {type(betas).__name__}")
    elif len(betas) != 2:
        errors.append(f"training.optimizer.betas must have exactly 2 values, got {len(betas)}")
    else:
        for i, beta in enumerate(betas):
            if not isinstance(beta, (int, float)):
                errors.append(f"training.optimizer.betas[{i}] must be a number, got {type(beta).__name__}")
            elif not (0 <= beta < 1):
                errors.append(f"training.optimizer.betas[{i}] must be in [0, 1), got {beta}")

    # Validate weight_decay
    weight_decay = optimizer.get('weight_decay', 0.0)
    if weight_decay < 0:
        errors.append(f"training.optimizer.weight_decay must be >= 0, got {weight_decay}")

    return errors


def validate_space_to_depth_config(cfg: DictConfig) -> list[str]:
    """Validate space-to-depth configuration.

    Checks:
    - Skip if not enabled
    - Requires spatial_dims=3
    - Incompatible with latent diffusion
    - Volume dimensions must be divisible by factors

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error strings (empty if validation passes).
    """
    errors: list[str] = []

    s2d_cfg = cfg.get('space_to_depth', {})
    if not s2d_cfg.get('enabled', False):
        return errors  # Not enabled, skip

    # Requires 3D
    spatial_dims = cfg.model.get('spatial_dims', 2)
    if spatial_dims != 3:
        errors.append(
            "space_to_depth requires spatial_dims=3. "
            "Space-to-depth rearrangement is only meaningful for 3D volumes."
        )

    # Incompatible with latent diffusion
    latent_cfg = cfg.get('latent', {})
    if latent_cfg.get('enabled', False):
        errors.append(
            "space_to_depth and latent diffusion cannot be used together. "
            "Disable one of latent.enabled or space_to_depth.enabled."
        )

    # Check volume divisibility
    if spatial_dims == 3 and 'volume' in cfg and cfg.volume is not None:
        sf = s2d_cfg.get('spatial_factor', 2)
        df = s2d_cfg.get('depth_factor', 2)

        height = cfg.volume.get('height', 0)
        width = cfg.volume.get('width', 0)
        depth = cfg.volume.get('pad_depth_to', cfg.volume.get('depth', 0))

        if height > 0 and height % sf != 0:
            errors.append(
                f"volume.height ({height}) must be divisible by "
                f"space_to_depth.spatial_factor ({sf})"
            )
        if width > 0 and width % sf != 0:
            errors.append(
                f"volume.width ({width}) must be divisible by "
                f"space_to_depth.spatial_factor ({sf})"
            )
        if df > 1 and depth > 0 and depth % df != 0:
            errors.append(
                f"volume depth ({depth}) must be divisible by "
                f"space_to_depth.depth_factor ({df})"
            )

    return errors


def validate_wavelet_config(cfg: DictConfig) -> list[str]:
    """Validate wavelet (Haar) configuration.

    Checks:
    - Skip if not enabled
    - Requires spatial_dims=3
    - Incompatible with latent diffusion
    - Incompatible with space_to_depth (can't use both)
    - Volume dimensions must be divisible by 2

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error strings (empty if validation passes).
    """
    errors: list[str] = []

    wavelet_cfg = cfg.get('wavelet', {})
    if not wavelet_cfg.get('enabled', False):
        return errors  # Not enabled, skip

    # Requires 3D
    spatial_dims = cfg.model.get('spatial_dims', 2)
    if spatial_dims != 3:
        errors.append(
            "wavelet requires spatial_dims=3. "
            "Haar 3D wavelet decomposition is only meaningful for 3D volumes."
        )

    # Incompatible with latent diffusion
    latent_cfg = cfg.get('latent', {})
    if latent_cfg.get('enabled', False):
        errors.append(
            "wavelet and latent diffusion cannot be used together. "
            "Disable one of latent.enabled or wavelet.enabled."
        )

    # Incompatible with space_to_depth
    s2d_cfg = cfg.get('space_to_depth', {})
    if s2d_cfg.get('enabled', False):
        errors.append(
            "wavelet and space_to_depth cannot be used together. "
            "Disable one of space_to_depth.enabled or wavelet.enabled."
        )

    # Check volume divisibility (Haar always uses factor 2)
    if spatial_dims == 3 and 'volume' in cfg and cfg.volume is not None:
        height = cfg.volume.get('height', 0)
        width = cfg.volume.get('width', 0)
        depth = cfg.volume.get('pad_depth_to', cfg.volume.get('depth', 0))

        if height > 0 and height % 2 != 0:
            errors.append(
                f"volume.height ({height}) must be divisible by 2 for wavelet decomposition"
            )
        if width > 0 and width % 2 != 0:
            errors.append(
                f"volume.width ({width}) must be divisible by 2 for wavelet decomposition"
            )
        if depth > 0 and depth % 2 != 0:
            errors.append(
                f"volume depth ({depth}) must be divisible by 2 for wavelet decomposition"
            )

    return errors


def validate_augmentation_config(cfg: DictConfig) -> list[str]:
    """Validate score augmentation configuration.

    Checks:
    - score_aug probability values are in [0, 1]

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error strings (empty if validation passes).
    """
    errors: list[str] = []
    training = cfg.get('training', {})
    score_aug = training.get('score_aug', {})

    if not score_aug:
        return errors  # No score_aug config

    prob_fields = ['compose_prob', 'nondestructive_prob', 'destructive_prob', 'dropout_prob']
    for field in prob_fields:
        if field in score_aug:
            prob = score_aug[field]
            if not isinstance(prob, (int, float)):
                errors.append(f"score_aug.{field} must be a number, got {type(prob).__name__}")
            elif not (0 <= prob <= 1):
                errors.append(f"score_aug.{field} must be in [0, 1], got {prob}")

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
    all_errors: list[str] = []
    for validator in validators:
        errors = validator(cfg)
        all_errors.extend(errors)

    if all_errors:
        raise ValueError(
            "Configuration validation failed:\n  - " + "\n  - ".join(all_errors)
        )
