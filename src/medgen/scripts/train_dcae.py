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

from medgen.data import create_vae_dataloader, create_vae_validation_dataloader, create_vae_test_dataloader
from medgen.data.loaders.multi_modality import (
    create_multi_modality_dataloader,
    create_multi_modality_validation_dataloader,
    create_multi_modality_test_dataloader,
    create_single_modality_validation_loader,
)
from medgen.pipeline import DCAETrainer
from .common import override_vae_channels, run_test_evaluation, create_per_modality_val_loaders

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="dcae")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
    mode = cfg.mode.name

    # Override in_channels for DC-AE training (mode configs are shared with diffusion)
    dcae_in_channels = override_vae_channels(cfg, mode)

    # Log config
    logger.info("DC-AE Training Configuration:")
    logger.info(f"  Compression: {cfg.dcae.compression_ratio}×")
    logger.info(f"  Latent channels: {cfg.dcae.latent_channels}")
    logger.info(f"  Pretrained: {cfg.dcae.get('pretrained', 'None (from scratch)')}")
    logger.info(f"  Epochs: {cfg.training.epochs}")
    logger.info(f"  Batch size: {cfg.training.batch_size}")
    logger.info(f"  Mode: {mode} ({dcae_in_channels} channel{'s' if dcae_in_channels > 1 else ''})")

    # Create trainer
    trainer = DCAETrainer(cfg)
    trainer.setup_model()

    # Create data loaders
    mode_name = cfg.mode.get('name', 'multi_modality')

    image_keys = cfg.mode.get('image_keys', ['bravo', 'flair', 't1_pre', 't1_gd'])

    if mode_name == 'multi_modality':
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

    logger.info(f"Train samples: {len(train_dataset)}")
    if val_loader:
        logger.info(f"Validation batches: {len(val_loader)}")
    else:
        logger.warning("No validation data found")

    # Create per-modality validation loaders for multi_modality mode
    per_modality_val_loaders = {}
    if mode_name == 'multi_modality':
        per_modality_val_loaders = create_per_modality_val_loaders(
            cfg=cfg,
            image_keys=image_keys,
            create_loader_fn=create_single_modality_validation_loader,
            image_size=cfg.dcae.image_size,
            batch_size=cfg.training.batch_size,
            log=logger
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
    if mode_name == 'multi_modality':
        test_result = create_multi_modality_test_dataloader(
            cfg=cfg,
            image_keys=image_keys,
            image_size=cfg.dcae.image_size,
            batch_size=cfg.training.batch_size,
        )
    else:
        test_result = create_vae_test_dataloader(cfg=cfg, modality=mode_name)

    run_test_evaluation(trainer, test_result, logger, eval_method="evaluate_test")

    # Cleanup
    trainer.close_writer()


if __name__ == "__main__":
    main()
