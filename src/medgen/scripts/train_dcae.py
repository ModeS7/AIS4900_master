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
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from medgen.core import ModeType

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from medgen.data import create_vae_dataloader, create_vae_validation_dataloader, create_vae_test_dataloader
from medgen.data.loaders.multi_modality import (
    create_multi_modality_dataloader,
    create_multi_modality_validation_dataloader,
    create_multi_modality_test_dataloader,
)
from medgen.pipeline import DCAETrainer

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="dcae")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
    mode = cfg.mode.name

    # Override in_channels for DC-AE training (mode configs are shared with diffusion)
    # DC-AE: single modality = 1 channel, dual = 2 channels (t1_pre + t1_gd, NO seg)
    if mode == ModeType.DUAL:
        dcae_in_channels = 2  # t1_pre + t1_gd
    else:
        dcae_in_channels = 1  # bravo, seg, t1_pre, t1_gd, multi_modality individually

    # Override mode.in_channels for DC-AE
    with open_dict(cfg):
        cfg.mode.in_channels = dcae_in_channels
        cfg.mode.out_channels = dcae_in_channels

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

    if test_result is not None:
        test_loader, test_dataset = test_result
        logger.info(f"Test dataset: {len(test_dataset)} slices")
        # Evaluate on both best and latest checkpoints
        trainer.evaluate_test(test_loader, checkpoint_name="best")
        trainer.evaluate_test(test_loader, checkpoint_name="latest")
    else:
        logger.info("No test_new/ directory found - skipping test evaluation")

    # Cleanup
    trainer.close_writer()


if __name__ == "__main__":
    main()
