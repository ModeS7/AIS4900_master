"""
Training entry point for VAE models.

This module provides the main training script for AutoencoderKL models
that will be used for latent diffusion training.

Usage:
    # Default config
    python -m medgen.scripts.train_vae

    # Override via CLI
    python -m medgen.scripts.train_vae vae.latent_channels=8 model.image_size=256

    # Cluster training
    python -m medgen.scripts.train_vae paths=cluster
"""
import logging

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from medgen.core import (
    setup_cuda_optimizations,
    validate_common_config,
    validate_model_config,
    validate_vae_config,
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
from medgen.pipeline import VAETrainer
from .common import override_vae_channels, run_test_evaluation, create_per_modality_val_loaders

# Enable CUDA optimizations
setup_cuda_optimizations()

log = logging.getLogger(__name__)


def validate_config(cfg: DictConfig) -> None:
    """Validate configuration values before training.

    Args:
        cfg: Hydra configuration object.

    Raises:
        ValueError: If any configuration value is invalid.
    """
    run_validation(cfg, [
        validate_common_config,
        validate_model_config,
        validate_vae_config,
    ])


@hydra.main(version_base=None, config_path="../../../configs", config_name="vae")
def main(cfg: DictConfig) -> None:
    """Main VAE training entry point.

    Args:
        cfg: Hydra configuration object composed from YAML files.
    """
    # Validate configuration before proceeding
    validate_config(cfg)

    mode = cfg.mode.name
    use_multi_gpu = cfg.training.get('use_multi_gpu', False)

    # Check if multi_modality mode
    is_multi_modality = mode == 'multi_modality'

    # Override in_channels for VAE training (mode configs are shared with diffusion)
    vae_in_channels = override_vae_channels(cfg, mode)

    # Log resolved configuration
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    log.info("")
    log.info("=" * 60)
    log.info(f"Training VAE for {mode} mode")
    log.info(f"Channels: {vae_in_channels} | Image size: {cfg.model.image_size}")
    log.info(f"Batch size: {cfg.training.batch_size} | Latent channels: {cfg.vae.latent_channels}")
    log.info(f"Epochs: {cfg.training.epochs} | Multi-GPU: {use_multi_gpu}")
    log.info(f"EMA: {cfg.training.use_ema}")
    log.info("=" * 60)
    log.info("")

    # Create trainer
    trainer = VAETrainer(cfg)
    log.info(f"Validation interval: every {trainer.val_interval} epochs ({cfg.training.epochs // trainer.val_interval} validations)")

    # Create dataloader using correct VAE data loading (no seg concatenation)
    augment = cfg.training.get('augment', True)

    # Per-modality validation loaders for multi_modality mode
    per_modality_val_loaders = {}

    if is_multi_modality:
        # Multi-modality mode: pool all modalities
        image_keys = cfg.mode.get('image_keys', ['bravo', 't1_pre', 't1_gd'])
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
        log.info(f"Training VAE on multi_modality mode (modalities: {image_keys})")
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

        # Create per-modality validation loaders for individual metrics
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
        log.info(f"Training VAE on {mode} mode ({vae_in_channels} channel{'s' if vae_in_channels > 1 else ''})")
        log.info(f"Training dataset: {len(train_dataset)} slices")

        # Create validation dataloader (if val/ directory exists)
        val_loader = None
        val_result = create_vae_validation_dataloader(cfg=cfg, modality=mode)
        if val_result is not None:
            val_loader, val_dataset = val_result
            log.info(f"Validation dataset: {len(val_dataset)} slices")

    if val_loader is None:
        log.info("No val/ directory found - using train samples for validation")

    # Setup model (with optional pretrained weights)
    pretrained_checkpoint = cfg.get('pretrained_checkpoint', None)
    if pretrained_checkpoint:
        log.info(f"Loading pretrained weights from: {pretrained_checkpoint}")
    trainer.setup_model(pretrained_checkpoint=pretrained_checkpoint)

    # Train with optional validation loader and per-modality loaders
    trainer.train(
        dataloader,
        train_dataset,
        val_loader=val_loader,
        per_modality_val_loaders=per_modality_val_loaders if per_modality_val_loaders else None
    )

    # Run test evaluation if test_new/ directory exists
    if is_multi_modality:
        image_keys = cfg.mode.get('image_keys', ['bravo', 't1_pre', 't1_gd'])
        test_result = create_multi_modality_test_dataloader(
            cfg=cfg,
            image_keys=image_keys,
            image_size=cfg.model.image_size,
            batch_size=cfg.training.batch_size
        )
    else:
        test_result = create_vae_test_dataloader(cfg=cfg, modality=mode)

    run_test_evaluation(trainer, test_result, log)

    # Close TensorBoard writer after all logging
    trainer.close_writer()


if __name__ == "__main__":
    main()
