"""Unified training entry point for 2D compression models.

This module provides shared training logic for VAE, VQ-VAE, and DC-AE models.
Individual entry point scripts (train_vae.py, train_vqvae.py, train_dcae.py)
delegate to train_compression() with their specific trainer type.

This consolidation eliminates ~80 lines of duplicate code while maintaining
backward-compatible entry points for each model type.
"""
import logging
from typing import Callable, Dict, Optional, Tuple, Type

from omegaconf import DictConfig, OmegaConf

from medgen.core import (
    setup_cuda_optimizations,
    validate_common_config,
    validate_model_config,
    validate_vae_config,
    validate_vqvae_config,
    run_validation,
)
from medgen.data import (
    create_vae_dataloader,
    create_vae_validation_dataloader,
    create_vae_test_dataloader,
    create_multi_modality_dataloader,
    create_multi_modality_validation_dataloader,
    create_multi_modality_test_dataloader,
    create_single_modality_validation_loader,
)
from medgen.pipeline import VAETrainer, VQVAETrainer
from .common import override_vae_channels, run_test_evaluation, create_per_modality_val_loaders, get_image_keys

# Enable CUDA optimizations at module import
setup_cuda_optimizations()

log = logging.getLogger(__name__)


# =============================================================================
# Trainer Registry
# =============================================================================

class TrainerConfig:
    """Configuration for a compression trainer type."""

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
            config_section: Name of config section (e.g., 'vae', 'vqvae').
            display_name: Human-readable name for logging.
        """
        self.trainer_class = trainer_class
        self.validator = validator
        self.config_section = config_section
        self.display_name = display_name


# Registry mapping trainer_type to configuration
TRAINER_REGISTRY: Dict[str, TrainerConfig] = {
    'vae': TrainerConfig(
        trainer_class=VAETrainer,
        validator=validate_vae_config,
        config_section='vae',
        display_name='VAE',
    ),
    'vqvae': TrainerConfig(
        trainer_class=VQVAETrainer,
        validator=validate_vqvae_config,
        config_section='vqvae',
        display_name='VQ-VAE',
    ),
}


# =============================================================================
# Core Training Function
# =============================================================================

def train_compression(cfg: DictConfig, trainer_type: str) -> None:
    """Unified training function for 2D compression models.

    Args:
        cfg: Hydra configuration object.
        trainer_type: Type of trainer ('vae', 'vqvae').

    Raises:
        ValueError: If trainer_type is not in TRAINER_REGISTRY.

    Example:
        >>> @hydra.main(config_path="...", config_name="vae")
        >>> def main(cfg):
        ...     train_compression(cfg, trainer_type='vae')
    """
    if trainer_type not in TRAINER_REGISTRY:
        raise ValueError(f"Unknown trainer_type: {trainer_type}. Valid options: {list(TRAINER_REGISTRY.keys())}")

    config = TRAINER_REGISTRY[trainer_type]

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

    # Build trainer-specific info for header
    config_section = getattr(cfg, config.config_section)
    extra_info = _build_extra_info(trainer_type, config_section)

    # Log training header
    log.info("")
    log.info("=" * 60)
    log.info(f"Training {config.display_name} for {mode} mode{extra_info}")
    log.info(f"Channels: {in_channels} | Image size: {cfg.model.image_size}")
    log.info(f"Batch size: {cfg.training.batch_size} | Epochs: {cfg.training.epochs}")
    log.info(f"Multi-GPU: {use_multi_gpu} | EMA: {cfg.training.use_ema}")
    log.info("=" * 60)
    log.info("")

    # Create trainer
    trainer = config.trainer_class(cfg)
    log.info(f"Validation: every epoch, figures at interval {trainer.figure_interval}")

    # Create dataloaders
    augment = cfg.training.get('augment', True)
    per_modality_val_loaders = {}

    if is_multi_modality:
        image_keys = get_image_keys(cfg)
        dataloader, train_dataset = create_multi_modality_dataloader(
            cfg=cfg,
            image_keys=image_keys,
            image_size=cfg.model.image_size,
            batch_size=cfg.training.batch_size,
            use_distributed=use_multi_gpu,
            rank=trainer.rank if use_multi_gpu else 0,
            world_size=trainer.world_size if use_multi_gpu else 1,
            augment=augment
        )
        log.info(f"Training {config.display_name} on multi_modality mode (modalities: {image_keys})")
        log.info(f"Training dataset: {len(train_dataset)} slices")

        # Create combined validation dataloader
        val_loader = None
        val_result = create_multi_modality_validation_dataloader(
            cfg=cfg,
            image_keys=image_keys,
            image_size=cfg.model.image_size,
            batch_size=cfg.training.batch_size
        )
        if val_result is not None:
            val_loader, val_dataset = val_result
            log.info(f"Validation dataset: {len(val_dataset)} slices (combined)")

        # Create per-modality validation loaders
        per_modality_val_loaders = create_per_modality_val_loaders(
            cfg=cfg,
            image_keys=image_keys,
            create_loader_fn=create_single_modality_validation_loader,
            image_size=cfg.model.image_size,
            batch_size=cfg.training.batch_size,
            log=log
        )
    else:
        # Standard single/dual modality mode
        dataloader, train_dataset = create_vae_dataloader(
            cfg=cfg,
            modality=mode,
            use_distributed=use_multi_gpu,
            rank=trainer.rank if use_multi_gpu else 0,
            world_size=trainer.world_size if use_multi_gpu else 1,
            augment=augment
        )
        log.info(f"Training {config.display_name} on {mode} mode ({in_channels} channel{'s' if in_channels > 1 else ''})")
        log.info(f"Training dataset: {len(train_dataset)} slices")

        # Create validation dataloader
        val_loader = None
        val_result = create_vae_validation_dataloader(cfg=cfg, modality=mode)
        if val_result is not None:
            val_loader, val_dataset = val_result
            log.info(f"Validation dataset: {len(val_dataset)} slices")

    if val_loader is None:
        log.info("No val/ directory found - using train samples for validation")

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
        per_modality_val_loaders=per_modality_val_loaders if per_modality_val_loaders else None
    )

    # Test evaluation
    if is_multi_modality:
        test_result = create_multi_modality_test_dataloader(
            cfg=cfg,
            image_keys=image_keys,
            image_size=cfg.model.image_size,
            batch_size=cfg.training.batch_size
        )
    else:
        test_result = create_vae_test_dataloader(cfg=cfg, modality=mode)

    run_test_evaluation(trainer, test_result, log)

    # Close TensorBoard writer
    trainer.close_writer()


def _build_extra_info(trainer_type: str, config_section: DictConfig) -> str:
    """Build trainer-specific extra info string for logging header.

    Args:
        trainer_type: Type of trainer.
        config_section: Config section (cfg.vae or cfg.vqvae).

    Returns:
        Extra info string (e.g., " (GAN: disabled)").
    """
    if trainer_type == 'vqvae':
        disable_gan = config_section.get('disable_gan', False)
        gan_status = "disabled" if disable_gan else "enabled"
        return f" (GAN: {gan_status})"
    return ""
