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
from omegaconf import DictConfig, OmegaConf, open_dict

from medgen.core import ModeType, setup_cuda_optimizations
from medgen.data import create_vae_dataloader
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

    mode = cfg.mode.name
    use_multi_gpu = cfg.training.get('use_multi_gpu', False)

    # Override in_channels for VAE training (mode configs are shared with diffusion)
    # VAE: single modality = 1 channel, dual = 2 channels (t1_pre + t1_gd, NO seg)
    if mode == ModeType.DUAL:
        vae_in_channels = 2  # t1_pre + t1_gd
    else:
        vae_in_channels = 1  # bravo, seg, t1_pre, t1_gd individually

    # Override mode.in_channels for VAE
    with open_dict(cfg):
        cfg.mode.in_channels = vae_in_channels
        cfg.mode.out_channels = vae_in_channels

    # Print resolved configuration
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    print(f"\n{'=' * 60}")
    print(f"Training VAE for {mode} mode")
    print(f"Channels: {vae_in_channels} | Image size: {cfg.model.image_size}")
    print(f"Batch size: {cfg.training.batch_size} | Latent channels: {cfg.vae.latent_channels}")
    print(f"Epochs: {cfg.training.epochs} | Multi-GPU: {use_multi_gpu}")
    print(f"EMA: {cfg.training.use_ema}")
    print(f"{'=' * 60}\n")

    # Create trainer
    trainer = VAETrainer(cfg)

    # Create dataloader using correct VAE data loading (no seg concatenation)
    dataloader, train_dataset = create_vae_dataloader(
        cfg=cfg,
        modality=mode,
        use_distributed=use_multi_gpu,
        rank=trainer.rank if use_multi_gpu else 0,
        world_size=trainer.world_size if use_multi_gpu else 1
    )
    log.info(f"Training VAE on {mode} mode ({vae_in_channels} channel{'s' if vae_in_channels > 1 else ''})")

    # Setup model
    trainer.setup_model()

    # Train
    trainer.train(dataloader, train_dataset)


if __name__ == "__main__":
    main()
