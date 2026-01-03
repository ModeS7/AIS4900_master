"""
Training entry point for 3D VQ-VAE models.

This module provides the main training script for 3D VQ-VAE models
for volumetric medical image compression with discrete latent space.

Advantages over 3D KL-VAE:
- Lower memory (no mu/logvar branches)
- Discrete codebook (cleaner latent space)

Usage:
    # Default config (dual mode, 4x compression)
    python -m medgen.scripts.train_vqvae_3d

    # Multi-modality (all sequences)
    python -m medgen.scripts.train_vqvae_3d mode=multi_modality

    # Cluster training
    python -m medgen.scripts.train_vqvae_3d paths=cluster
"""
import logging

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from medgen.core import (
    ModeType,
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
from medgen.pipeline import VQVAE3DTrainer
from medgen.scripts.common import run_test_evaluation

# Enable CUDA optimizations
setup_cuda_optimizations()

log = logging.getLogger(__name__)


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

    # Check depth is divisible by 4 for 4x compression (2 downsample levels)
    n_downsamples = len(cfg.vqvae_3d.get('channels', [64, 128]))
    divisor = 2 ** n_downsamples
    if volume.get('depth', 0) % divisor != 0:
        errors.append(f"volume.depth ({volume.get('depth')}) must be divisible by {divisor} for {divisor}x compression")

    # Check vqvae_3d config exists
    if 'vqvae_3d' not in cfg:
        errors.append("Missing 'vqvae_3d' config section")

    return errors


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
        validate_vqvae_3d_config,
    ])


@hydra.main(version_base=None, config_path="../../../configs", config_name="vqvae_3d")
def main(cfg: DictConfig) -> None:
    """Main 3D VQ-VAE training entry point.

    Args:
        cfg: Hydra configuration object composed from YAML files.
    """
    # Validate configuration before proceeding
    validate_config(cfg)

    mode = cfg.mode.name
    use_multi_gpu = cfg.training.get('use_multi_gpu', False)
    is_multi_modality = (mode == 'multi_modality')

    # Set channels based on mode
    if mode == ModeType.DUAL:
        vae_in_channels = 2  # t1_pre + t1_gd
    else:
        vae_in_channels = 1  # Single modality or multi_modality (each volume is 1 channel)

    # Override mode.in_channels
    with open_dict(cfg):
        cfg.mode.in_channels = vae_in_channels
        cfg.mode.out_channels = vae_in_channels

    # Log resolved configuration
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Compute latent shape
    n_downsamples = len(cfg.vqvae_3d.channels)
    latent_h = cfg.volume.height // (2 ** n_downsamples)
    latent_w = cfg.volume.width // (2 ** n_downsamples)
    latent_d = cfg.volume.depth // (2 ** n_downsamples)

    log.info("")
    log.info("=" * 60)
    log.info(f"Training 3D VQ-VAE for {mode} mode")
    log.info(f"Channels: {vae_in_channels}")
    log.info(f"Volume: {cfg.volume.width}x{cfg.volume.height}x{cfg.volume.depth}")
    log.info(f"Latent: {latent_w}x{latent_h}x{latent_d} ({n_downsamples}x compression)")
    log.info(f"Codebook: {cfg.vqvae_3d.num_embeddings} x {cfg.vqvae_3d.embedding_dim}")
    log.info(f"Batch size: {cfg.training.batch_size}")
    log.info(f"Epochs: {cfg.training.epochs} | Multi-GPU: {use_multi_gpu}")
    if is_multi_modality:
        image_keys = cfg.mode.get('image_keys', ['bravo', 'flair', 't1_pre', 't1_gd'])
        log.info(f"Modalities: {image_keys}")
    log.info("=" * 60)
    log.info("")

    # Create trainer
    trainer = VQVAE3DTrainer(cfg)
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

    # Create per-modality validation loaders for multi_modality mode
    per_modality_val_loaders = {}
    if is_multi_modality:
        image_keys = cfg.mode.get('image_keys', ['bravo', 'flair', 't1_pre', 't1_gd'])
        for modality_name in image_keys:
            loader = create_vae_3d_single_modality_validation_loader(cfg, modality_name)
            if loader is not None:
                per_modality_val_loaders[modality_name] = loader
                log.info(f"  Per-modality 3D validation for {modality_name}: {len(loader.dataset)} volumes")
        if per_modality_val_loaders:
            log.info(f"Created {len(per_modality_val_loaders)} per-modality validation loaders")

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


if __name__ == "__main__":
    main()
