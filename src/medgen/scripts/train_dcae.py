"""
DC-AE training script.

Trains DC-AE (Deep Compression Autoencoder) for high-compression 2D MRI encoding.

Usage:
    # Train from scratch with f32 (default)
    python -m medgen.scripts.train_dcae

    # Fine-tune from pretrained
    python -m medgen.scripts.train_dcae dcae.pretrained="mit-han-lab/dc-ae-f32c32-in-1.0-diffusers"

    # Use f64 compression
    python -m medgen.scripts.train_dcae dcae=f64

    # Cluster training
    python -m medgen.scripts.train_dcae paths=cluster
"""
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from medgen.core import (
    setup_cuda_optimizations,
    validate_common_config,
    validate_model_config,
    run_validation,
)
from medgen.data import create_vae_dataloader, create_vae_validation_dataloader, create_vae_test_dataloader
from medgen.data.loaders.multi_modality import (
    create_multi_modality_dataloader,
    create_multi_modality_validation_dataloader,
    create_multi_modality_test_dataloader,
    create_single_modality_validation_loader,
)
from medgen.data.loaders.seg_compression import (
    create_seg_compression_dataloader,
    create_seg_compression_validation_dataloader,
    create_seg_compression_test_dataloader,
)
from medgen.pipeline import DCAETrainer
from .common import override_vae_channels, run_test_evaluation, create_per_modality_val_loaders, get_image_keys

# Enable CUDA optimizations
setup_cuda_optimizations()

log = logging.getLogger(__name__)


def validate_dcae_config(cfg: DictConfig) -> list:
    """Validate DC-AE configuration.

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    # Check dcae config exists
    if 'dcae' not in cfg:
        errors.append("Missing 'dcae' config section")
        return errors

    dcae = cfg.dcae
    if dcae.get('image_size', 0) <= 0:
        errors.append("dcae.image_size must be positive")
    if dcae.get('latent_channels', 0) <= 0:
        errors.append("dcae.latent_channels must be positive")
    if dcae.get('compression_ratio', 0) <= 0:
        errors.append("dcae.compression_ratio must be positive")

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
        validate_dcae_config,
    ])


@hydra.main(version_base=None, config_path="../../../configs", config_name="dcae")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
    # Validate configuration before proceeding
    validate_config(cfg)

    mode = cfg.mode.name

    # Override in_channels for DC-AE training (mode configs are shared with diffusion)
    dcae_in_channels = override_vae_channels(cfg, mode)

    # Log resolved configuration
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Log training header
    log.info("")
    log.info("=" * 60)
    log.info(f"Training DC-AE for {mode} mode")
    log.info(f"Compression: {cfg.dcae.compression_ratio}× | Latent channels: {cfg.dcae.latent_channels}")
    log.info(f"Image size: {cfg.dcae.image_size} | Channels: {dcae_in_channels}")
    log.info(f"Batch size: {cfg.training.batch_size} | Epochs: {cfg.training.epochs}")
    log.info(f"Pretrained: {cfg.dcae.get('pretrained', 'None (from scratch)')}")
    log.info("=" * 60)
    log.info("")

    # Create trainer
    trainer = DCAETrainer(cfg)
    trainer.setup_model()

    # Create data loaders
    mode_name = cfg.mode.get('name', 'multi_modality')
    is_seg_mode = cfg.dcae.get('seg_mode', False)

    image_keys = get_image_keys(cfg, is_3d=False)  # DC-AE is 2D, uses all 4 modalities by default

    if is_seg_mode or mode_name == 'seg_compression':
        # Segmentation mask compression mode
        train_loader, train_dataset = create_seg_compression_dataloader(
            cfg=cfg,
            image_size=cfg.dcae.image_size,
            batch_size=cfg.training.batch_size,
            augment=True,
        )
        val_result = create_seg_compression_validation_dataloader(
            cfg=cfg,
            image_size=cfg.dcae.image_size,
            batch_size=cfg.training.batch_size,
        )
        val_loader = val_result[0] if val_result else None
    elif mode_name == 'multi_modality':
        # Multi-modality: mixed slices from all modalities
        train_loader, train_dataset = create_multi_modality_dataloader(
            cfg=cfg,
            image_keys=image_keys,
            image_size=cfg.dcae.image_size,
            batch_size=cfg.training.batch_size,
            augment=True,
        )
        val_result = create_multi_modality_validation_dataloader(
            cfg=cfg,
            image_keys=image_keys,
            image_size=cfg.dcae.image_size,
            batch_size=cfg.training.batch_size,
        )
        val_loader = val_result[0] if val_result else None
    else:
        # Single modality
        train_loader, train_dataset = create_vae_dataloader(
            cfg=cfg,
            modality=mode_name,
            augment=True,
        )
        val_result = create_vae_validation_dataloader(cfg=cfg, modality=mode_name)
        val_loader = val_result[0] if val_result else None

    log.info(f"Train samples: {len(train_dataset)}")
    if val_loader:
        log.info(f"Validation batches: {len(val_loader)}")
    else:
        log.warning("No validation data found")

    # Create per-modality validation loaders for multi_modality mode
    # (not applicable for seg_compression which only has seg masks)
    per_modality_val_loaders = {}
    if mode_name == 'multi_modality' and not is_seg_mode:
        per_modality_val_loaders = create_per_modality_val_loaders(
            cfg=cfg,
            image_keys=image_keys,
            create_loader_fn=create_single_modality_validation_loader,
            image_size=cfg.dcae.image_size,
            batch_size=cfg.training.batch_size,
            log=log
        )

    # Load checkpoint if resuming
    start_epoch = 0
    if cfg.pretrained_checkpoint:
        start_epoch = trainer.load_checkpoint(cfg.pretrained_checkpoint)

    # Train
    trainer.train(
        train_loader=train_loader,
        train_dataset=train_dataset,
        val_loader=val_loader,
        start_epoch=start_epoch,
        per_modality_val_loaders=per_modality_val_loaders if per_modality_val_loaders else None,
    )

    # Run test evaluation if test_new/ directory exists
    if is_seg_mode or mode_name == 'seg_compression':
        test_result = create_seg_compression_test_dataloader(
            cfg=cfg,
            image_size=cfg.dcae.image_size,
            batch_size=cfg.training.batch_size,
        )
    elif mode_name == 'multi_modality':
        test_result = create_multi_modality_test_dataloader(
            cfg=cfg,
            image_keys=image_keys,
            image_size=cfg.dcae.image_size,
            batch_size=cfg.training.batch_size,
        )
    else:
        test_result = create_vae_test_dataloader(cfg=cfg, modality=mode_name)

    run_test_evaluation(trainer, test_result, log, eval_method="evaluate_test_set")

    # Cleanup
    trainer.close_writer()


if __name__ == "__main__":
    main()
