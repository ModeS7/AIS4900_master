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
    ModeType,
    setup_cuda_optimizations,
    validate_common_config,
    validate_model_config,
    validate_vae_config,
    run_validation,
)
from medgen.data import create_vae_dataloader, create_vae_validation_dataloader, create_vae_test_dataloader
from medgen.pipeline import VAETrainer

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
    augment = cfg.training.get('augment', True)
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
    else:
        log.info("No val/ directory found - using train samples for validation")

    # Setup model
    trainer.setup_model()

    # Train with optional validation loader
    trainer.train(dataloader, train_dataset, val_loader=val_loader)

    # Run test evaluation if test_new/ directory exists
    test_result = create_vae_test_dataloader(cfg=cfg, modality=mode)
    if test_result is not None:
        test_loader, test_dataset = test_result
        log.info(f"Test dataset: {len(test_dataset)} slices")
        # Evaluate on both best and latest checkpoints
        trainer.evaluate_test_set(test_loader, checkpoint_name="best")
        trainer.evaluate_test_set(test_loader, checkpoint_name="latest")
    else:
        log.info("No test_new/ directory found - skipping test evaluation")

    # Close TensorBoard writer after all logging
    trainer.close_writer()


if __name__ == "__main__":
    main()
