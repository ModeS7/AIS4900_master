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
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from medgen.core import DEFAULT_DUAL_IMAGE_KEYS, setup_cuda_optimizations
from medgen.data import create_dataloader, create_dual_image_dataloader
from medgen.diffusion.vae_trainer import VAETrainer

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
    errors = []

    # Training params
    if cfg.training.epochs <= 0:
        errors.append(f"epochs must be > 0, got {cfg.training.epochs}")
    if cfg.training.batch_size <= 0:
        errors.append(f"batch_size must be > 0, got {cfg.training.batch_size}")
    if cfg.training.learning_rate <= 0:
        errors.append(f"learning_rate must be > 0, got {cfg.training.learning_rate}")

    # Model params
    if cfg.model.image_size <= 0:
        errors.append(f"image_size must be > 0, got {cfg.model.image_size}")
    if cfg.model.image_size & (cfg.model.image_size - 1) != 0:
        log.warning(f"image_size {cfg.model.image_size} is not a power of 2 (may cause issues)")

    # VAE params
    if not hasattr(cfg, 'vae'):
        errors.append("VAE configuration missing. Add 'vae' section to config.")
    else:
        if cfg.vae.latent_channels <= 0:
            errors.append(f"vae.latent_channels must be > 0, got {cfg.vae.latent_channels}")
        if len(cfg.vae.channels) == 0:
            errors.append("vae.channels must not be empty")

    # Paths - check if data directory exists
    if not os.path.exists(cfg.paths.data_dir):
        errors.append(f"Data directory does not exist: {cfg.paths.data_dir}")

    # Check CUDA availability
    if not torch.cuda.is_available():
        errors.append("CUDA is not available. Training requires GPU.")

    if errors:
        raise ValueError("Configuration validation failed:\n  - " + "\n  - ".join(errors))


@hydra.main(version_base=None, config_path="../../../configs", config_name="train_vae")
def main(cfg: DictConfig) -> None:
    """Main VAE training entry point.

    Args:
        cfg: Hydra configuration object composed from YAML files.
    """
    # Validate configuration before proceeding
    validate_config(cfg)

    # Print resolved configuration
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    mode = cfg.mode.name
    use_multi_gpu = cfg.training.get('use_multi_gpu', False)

    print(f"\n{'=' * 60}")
    print(f"Training VAE for {mode} mode")
    print(f"Image size: {cfg.model.image_size} | Batch size: {cfg.training.batch_size}")
    print(f"Latent channels: {cfg.vae.latent_channels} | KL weight: {cfg.vae.kl_weight}")
    print(f"Epochs: {cfg.training.epochs} | Multi-GPU: {use_multi_gpu}")
    print(f"EMA: {cfg.training.use_ema}")
    print(f"{'=' * 60}\n")

    # Create trainer
    trainer = VAETrainer(cfg)

    # Create dataloader based on mode
    # For VAE, we typically train on single images regardless of diffusion mode
    # But we use the same data loading infrastructure
    if mode == 'dual':
        # For dual mode VAE, we need to decide how to handle multiple images
        # Option 1: Train VAE on each image type separately (recommended)
        # Option 2: Train VAE on concatenated images
        # Here we train on individual images from the dual dataset
        image_keys = list(cfg.mode.image_keys) if 'image_keys' in cfg.mode else DEFAULT_DUAL_IMAGE_KEYS
        dataloader, train_dataset = create_dual_image_dataloader(
            cfg=cfg,
            image_keys=image_keys,
            conditioning='seg',
            use_distributed=use_multi_gpu,
            rank=trainer.rank if use_multi_gpu else 0,
            world_size=trainer.world_size if use_multi_gpu else 1
        )
        log.info(f"Training VAE on dual images: {image_keys}")
    else:
        # seg or bravo
        image_type = 'seg' if mode == 'seg' else 'bravo'
        dataloader, train_dataset = create_dataloader(
            cfg=cfg,
            image_type=image_type,
            use_distributed=use_multi_gpu,
            rank=trainer.rank if use_multi_gpu else 0,
            world_size=trainer.world_size if use_multi_gpu else 1
        )
        log.info(f"Training VAE on {image_type} images")

    # Setup model
    trainer.setup_model()

    # Train
    trainer.train(dataloader, train_dataset)


if __name__ == "__main__":
    main()
