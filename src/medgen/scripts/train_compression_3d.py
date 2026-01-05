"""Unified training entry point for 3D compression models.

This module provides shared training logic for 3D VAE and 3D VQ-VAE models.
Individual entry point scripts (train_vae_3d.py, train_vqvae_3d.py)
delegate to train_compression_3d() with their specific trainer type.

This consolidation eliminates ~70 lines of duplicate code while maintaining
backward-compatible entry points for each model type.
"""
import logging
from typing import Callable, Dict, Type

from omegaconf import DictConfig, OmegaConf

from medgen.core import (
    setup_cuda_optimizations,
    validate_common_config,
    validate_model_config,
    run_validation,
)
from medgen.data import (
    create_vae_3d_dataloader,
    create_vae_3d_validation_dataloader,
    create_vae_3d_test_dataloader,
    create_vae_3d_multi_modality_dataloader,
    create_vae_3d_multi_modality_validation_dataloader,
    create_vae_3d_multi_modality_test_dataloader,
    create_vae_3d_single_modality_validation_loader,
)
from medgen.pipeline import VAE3DTrainer, VQVAE3DTrainer
from .common import override_vae_channels, run_test_evaluation, get_image_keys, create_per_modality_val_loaders_3d

# Enable CUDA optimizations at module import
setup_cuda_optimizations()

log = logging.getLogger(__name__)


# =============================================================================
# Config Validators
# =============================================================================

def validate_vae_3d_config(cfg: DictConfig) -> list:
    """Validate 3D VAE configuration.

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    # Check volume config exists
    if 'volume' not in cfg:
        errors.append("Missing 'volume' config section for 3D VAE")
        return errors

    volume = cfg.volume
    if volume.get('depth', 0) <= 0:
        errors.append("volume.depth must be positive")
    if volume.get('height', 0) <= 0:
        errors.append("volume.height must be positive")
    if volume.get('width', 0) <= 0:
        errors.append("volume.width must be positive")

    # Check depth is divisible by 8 for clean compression
    if volume.get('depth', 0) % 8 != 0:
        errors.append(f"volume.depth ({volume.get('depth')}) must be divisible by 8 for 8x compression")

    # Check vae_3d config exists
    if 'vae_3d' not in cfg:
        errors.append("Missing 'vae_3d' config section")

    return errors


def validate_vqvae_3d_config(cfg: DictConfig) -> list:
    """Validate 3D VQ-VAE configuration.

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    # Check volume config exists
    if 'volume' not in cfg:
        errors.append("Missing 'volume' config section for 3D VQ-VAE")
        return errors

    volume = cfg.volume
    if volume.get('depth', 0) <= 0:
        errors.append("volume.depth must be positive")
    if volume.get('height', 0) <= 0:
        errors.append("volume.height must be positive")
    if volume.get('width', 0) <= 0:
        errors.append("volume.width must be positive")

    # Check depth is divisible by compression factor
    n_downsamples = len(cfg.vqvae_3d.get('channels', [64, 128]))
    divisor = 2 ** n_downsamples
    if volume.get('depth', 0) % divisor != 0:
        errors.append(f"volume.depth ({volume.get('depth')}) must be divisible by {divisor} for {divisor}x compression")

    # Check vqvae_3d config exists
    if 'vqvae_3d' not in cfg:
        errors.append("Missing 'vqvae_3d' config section")

    return errors


# =============================================================================
# Trainer Registry
# =============================================================================

class Trainer3DConfig:
    """Configuration for a 3D compression trainer type."""

    def __init__(
        self,
        trainer_class: Type,
        validator: Callable[[DictConfig], list],
        config_section: str,
        display_name: str,
    ):
        """Initialize trainer configuration.

        Args:
            trainer_class: Trainer class to instantiate.
            validator: Config validation function (returns list of errors).
            config_section: Name of config section (e.g., 'vae_3d', 'vqvae_3d').
            display_name: Human-readable name for logging.
        """
        self.trainer_class = trainer_class
        self.validator = validator
        self.config_section = config_section
        self.display_name = display_name


# Registry mapping trainer_type to configuration
TRAINER_3D_REGISTRY: Dict[str, Trainer3DConfig] = {
    'vae_3d': Trainer3DConfig(
        trainer_class=VAE3DTrainer,
        validator=validate_vae_3d_config,
        config_section='vae_3d',
        display_name='3D VAE',
    ),
    'vqvae_3d': Trainer3DConfig(
        trainer_class=VQVAE3DTrainer,
        validator=validate_vqvae_3d_config,
        config_section='vqvae_3d',
        display_name='3D VQ-VAE',
    ),
}


# =============================================================================
# Core Training Function
# =============================================================================

def train_compression_3d(cfg: DictConfig, trainer_type: str) -> None:
    """Unified training function for 3D compression models.

    Args:
        cfg: Hydra configuration object.
        trainer_type: Type of trainer ('vae_3d', 'vqvae_3d').

    Raises:
        ValueError: If trainer_type is not in TRAINER_3D_REGISTRY.

    Example:
        >>> @hydra.main(config_path="...", config_name="vae_3d")
        >>> def main(cfg):
        ...     train_compression_3d(cfg, trainer_type='vae_3d')
    """
    if trainer_type not in TRAINER_3D_REGISTRY:
        raise ValueError(f"Unknown trainer_type: {trainer_type}. Valid options: {list(TRAINER_3D_REGISTRY.keys())}")

    config = TRAINER_3D_REGISTRY[trainer_type]

    # Validate configuration before proceeding
    run_validation(cfg, [
        validate_common_config,
        validate_model_config,
        config.validator,
    ])

    mode = cfg.mode.name
    use_multi_gpu = cfg.training.get('use_multi_gpu', False)
    is_multi_modality = mode == 'multi_modality'

    # Override in_channels (mode configs are shared with diffusion)
    in_channels = override_vae_channels(cfg, mode)

    # Log resolved configuration
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Build trainer-specific extra info
    extra_info = _build_extra_info_3d(trainer_type, cfg)

    # Log training header
    log.info("")
    log.info("=" * 60)
    log.info(f"Training {config.display_name} for {mode} mode")
    log.info(f"Channels: {in_channels}")
    log.info(f"Volume: {cfg.volume.width}x{cfg.volume.height}x{cfg.volume.depth}")
    log.info(extra_info)
    log.info(f"Batch size: {cfg.training.batch_size} | Epochs: {cfg.training.epochs}")
    log.info(f"Multi-GPU: {use_multi_gpu}")
    if is_multi_modality:
        image_keys = get_image_keys(cfg, is_3d=True)
        log.info(f"Modalities: {image_keys}")
    log.info("=" * 60)
    log.info("")

    # Create trainer
    trainer = config.trainer_class(cfg)
    log.info(f"Validation: every epoch, figures at interval {trainer.figure_interval}")

    # Create 3D dataloader
    if is_multi_modality:
        dataloader, train_dataset = create_vae_3d_multi_modality_dataloader(
            cfg=cfg,
            use_distributed=use_multi_gpu,
            rank=trainer.rank if use_multi_gpu else 0,
            world_size=trainer.world_size if use_multi_gpu else 1,
        )
    else:
        dataloader, train_dataset = create_vae_3d_dataloader(
            cfg=cfg,
            modality=mode,
            use_distributed=use_multi_gpu,
            rank=trainer.rank if use_multi_gpu else 0,
            world_size=trainer.world_size if use_multi_gpu else 1,
        )
    log.info(f"Training dataset: {len(train_dataset)} volumes")

    # Create validation dataloader
    val_loader = None
    if is_multi_modality:
        val_result = create_vae_3d_multi_modality_validation_dataloader(cfg=cfg)
    else:
        val_result = create_vae_3d_validation_dataloader(cfg=cfg, modality=mode)
    if val_result is not None:
        val_loader, val_dataset = val_result
        log.info(f"Validation dataset: {len(val_dataset)} volumes")
    else:
        log.info("No val/ directory found - using train samples for validation")

    # Create per-modality validation loaders
    per_modality_val_loaders = {}
    if is_multi_modality:
        image_keys = get_image_keys(cfg, is_3d=True)
        per_modality_val_loaders = create_per_modality_val_loaders_3d(
            cfg, image_keys, create_vae_3d_single_modality_validation_loader, log
        )

    # Setup model
    pretrained_checkpoint = cfg.get('pretrained_checkpoint', None)
    if pretrained_checkpoint:
        log.info(f"Loading pretrained weights from: {pretrained_checkpoint}")
    trainer.setup_model(pretrained_checkpoint=pretrained_checkpoint)

    # Train
    trainer.train(
        dataloader,
        train_dataset,
        val_loader=val_loader,
        per_modality_val_loaders=per_modality_val_loaders if per_modality_val_loaders else None,
    )

    # Test evaluation
    if is_multi_modality:
        test_result = create_vae_3d_multi_modality_test_dataloader(cfg=cfg)
    else:
        test_result = create_vae_3d_test_dataloader(cfg=cfg, modality=mode)

    run_test_evaluation(trainer, test_result, log, eval_method="evaluate_test_set")

    # Close TensorBoard writer
    trainer.close_writer()


def _build_extra_info_3d(trainer_type: str, cfg: DictConfig) -> str:
    """Build trainer-specific extra info string for logging header.

    Args:
        trainer_type: Type of trainer.
        cfg: Configuration object.

    Returns:
        Extra info string for logging.
    """
    if trainer_type == 'vae_3d':
        return f"Latent channels: {cfg.vae_3d.latent_channels}"
    elif trainer_type == 'vqvae_3d':
        n_downsamples = len(cfg.vqvae_3d.channels)
        latent_h = cfg.volume.height // (2 ** n_downsamples)
        latent_w = cfg.volume.width // (2 ** n_downsamples)
        latent_d = cfg.volume.depth // (2 ** n_downsamples)
        return (
            f"Latent: {latent_w}x{latent_h}x{latent_d} ({n_downsamples}x compression) | "
            f"Codebook: {cfg.vqvae_3d.num_embeddings} x {cfg.vqvae_3d.embedding_dim}"
        )
    return ""
