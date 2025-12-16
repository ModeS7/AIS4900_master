"""
Training entry point for diffusion models on medical images.

This module provides the main training script using Hydra for configuration
management. Trains diffusion models (DDPM or Rectified Flow) on brain MRI data.

Usage:
    # Default config (ddpm, 128, bravo, local)
    python -m medgen.scripts.train

    # Override via CLI
    python -m medgen.scripts.train strategy=rflow model=unet_256

    # Cluster training
    python -m medgen.scripts.train paths=cluster

    # Quick debug
    python -m medgen.scripts.train training=fast_debug
"""
import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from medgen.core import DEFAULT_DUAL_IMAGE_KEYS, ModeType, setup_cuda_optimizations
from medgen.data import (
    create_dataloader,
    create_dual_image_dataloader,
    create_validation_dataloader,
    create_dual_image_validation_dataloader,
)
from medgen.diffusion import DiffusionTrainer
from medgen.diffusion.spaces import PixelSpace, load_vae_for_latent_space

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

    # Strategy
    if cfg.strategy.name not in ['ddpm', 'rflow']:
        errors.append(f"Unknown strategy: {cfg.strategy.name}")

    # Mode
    valid_modes = [m.value for m in ModeType]
    if cfg.mode.name not in valid_modes:
        errors.append(f"Unknown mode: {cfg.mode.name}. Valid modes: {valid_modes}")

    # Paths - check if data directory exists
    if not os.path.exists(cfg.paths.data_dir):
        errors.append(f"Data directory does not exist: {cfg.paths.data_dir}")

    # Check CUDA availability
    if not torch.cuda.is_available():
        errors.append("CUDA is not available. Training requires GPU.")

    if errors:
        raise ValueError("Configuration validation failed:\n  - " + "\n  - ".join(errors))


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entry point.

    Args:
        cfg: Hydra configuration object composed from YAML files.
    """
    # Validate configuration before proceeding
    validate_config(cfg)

    # Print resolved configuration
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    mode = cfg.mode.name
    strategy = cfg.strategy.name
    use_multi_gpu = cfg.training.use_multi_gpu

    # Create space based on config (latent or pixel)
    latent_cfg = cfg.get('latent', {})
    use_latent = latent_cfg.get('enabled', False)

    if use_latent:
        vae_checkpoint = latent_cfg.get('vae_checkpoint')
        if not vae_checkpoint:
            raise ValueError("latent.vae_checkpoint must be specified when latent.enabled=true")
        if not os.path.exists(vae_checkpoint):
            raise ValueError(f"VAE checkpoint not found: {vae_checkpoint}")

        device = torch.device("cuda")
        space = load_vae_for_latent_space(vae_checkpoint, device)
        space_name = "latent"
        log.info(f"Loaded VAE from {vae_checkpoint}")
    else:
        space = PixelSpace()
        space_name = "pixel"

    print(f"\n{'=' * 60}")
    print(f"Training {mode} mode with {strategy} strategy ({space_name} space)")
    print(f"Image size: {cfg.model.image_size} | Batch size: {cfg.training.batch_size}")
    print(f"Epochs: {cfg.training.epochs} | Multi-GPU: {use_multi_gpu}")
    print(f"EMA: {cfg.training.use_ema} | Min-SNR: {cfg.training.use_min_snr}")
    print(f"{'=' * 60}\n")

    # Create trainer
    trainer = DiffusionTrainer(cfg, space=space)

    # Create dataloader based on mode
    if mode == ModeType.DUAL:
        image_keys = list(cfg.mode.image_keys) if 'image_keys' in cfg.mode else DEFAULT_DUAL_IMAGE_KEYS
        dataloader, train_dataset = create_dual_image_dataloader(
            cfg=cfg,
            image_keys=image_keys,
            conditioning='seg',
            use_distributed=use_multi_gpu,
            rank=trainer.rank if use_multi_gpu else 0,
            world_size=trainer.world_size if use_multi_gpu else 1
        )
    else:
        # seg or bravo
        image_type = 'seg' if mode == ModeType.SEG else 'bravo'
        dataloader, train_dataset = create_dataloader(
            cfg=cfg,
            image_type=image_type,
            use_distributed=use_multi_gpu,
            rank=trainer.rank if use_multi_gpu else 0,
            world_size=trainer.world_size if use_multi_gpu else 1
        )

    log.info(f"Training dataset: {len(train_dataset)} slices")

    # Create validation dataloader (if val/ directory exists)
    val_loader = None
    if mode == ModeType.DUAL:
        val_result = create_dual_image_validation_dataloader(
            cfg=cfg,
            image_keys=image_keys,
            conditioning='seg',
        )
    else:
        val_result = create_validation_dataloader(
            cfg=cfg,
            image_type=image_type,
        )

    if val_result is not None:
        val_loader, val_dataset = val_result
        log.info(f"Validation dataset: {len(val_dataset)} slices")
    else:
        log.info("No val/ directory found - using train loss for best model selection")

    # Setup model
    trainer.setup_model(train_dataset)

    # Train with optional validation loader
    trainer.train(dataloader, train_dataset, val_loader=val_loader)

    # Close TensorBoard writer after training
    trainer.close_writer()


if __name__ == "__main__":
    main()
