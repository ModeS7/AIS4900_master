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

    # Fast debug mode
    python -m medgen.scripts.train training=fast_debug
"""
import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from medgen.core import (
    DEFAULT_DUAL_IMAGE_KEYS,
    ModeType,
    setup_cuda_optimizations,
    validate_common_config,
    validate_model_config,
    validate_diffusion_config,
    run_validation,
)
from medgen.data import (
    create_dataloader,
    create_dual_image_dataloader,
    create_validation_dataloader,
    create_dual_image_validation_dataloader,
    create_test_dataloader,
    create_dual_image_test_dataloader,
    create_multi_diffusion_dataloader,
    create_multi_diffusion_validation_dataloader,
    create_multi_diffusion_test_dataloader,
    create_single_modality_diffusion_val_loader,
)
from medgen.pipeline import DiffusionTrainer
from medgen.pipeline.spaces import PixelSpace, load_vae_for_latent_space

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
        validate_diffusion_config,
    ])


@hydra.main(version_base=None, config_path="../../../configs", config_name="diffusion")
def main(cfg: DictConfig) -> None:
    """Main training entry point.

    Args:
        cfg: Hydra configuration object composed from YAML files.
    """
    # Validate configuration before proceeding
    validate_config(cfg)

    # Log resolved configuration
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

    log.info("")
    log.info("=" * 60)
    log.info(f"Training {mode} mode with {strategy} strategy ({space_name} space)")
    log.info(f"Image size: {cfg.model.image_size} | Batch size: {cfg.training.batch_size}")
    log.info(f"Epochs: {cfg.training.epochs} | Multi-GPU: {use_multi_gpu}")
    log.info(f"EMA: {cfg.training.use_ema} | Min-SNR: {cfg.training.use_min_snr}")
    log.info("=" * 60)
    log.info("")

    # Create trainer
    trainer = DiffusionTrainer(cfg, space=space)
    log.info(f"Validation: every epoch, figures at interval {trainer.figure_interval}")

    # Create dataloader based on mode
    augment = cfg.training.get('augment', True)
    image_keys = None  # Will be set for multi and dual modes

    if mode == 'multi':
        # Multi-modality diffusion with mode embedding
        image_keys = list(cfg.mode.image_keys) if 'image_keys' in cfg.mode else ['bravo', 'flair', 't1_pre', 't1_gd']
        dataloader, train_dataset = create_multi_diffusion_dataloader(
            cfg=cfg,
            image_keys=image_keys,
            use_distributed=use_multi_gpu,
            rank=trainer.rank if use_multi_gpu else 0,
            world_size=trainer.world_size if use_multi_gpu else 1,
            augment=augment
        )
    elif mode == ModeType.DUAL:
        image_keys = list(cfg.mode.image_keys) if 'image_keys' in cfg.mode else DEFAULT_DUAL_IMAGE_KEYS
        dataloader, train_dataset = create_dual_image_dataloader(
            cfg=cfg,
            image_keys=image_keys,
            conditioning='seg',
            use_distributed=use_multi_gpu,
            rank=trainer.rank if use_multi_gpu else 0,
            world_size=trainer.world_size if use_multi_gpu else 1,
            augment=augment
        )
    else:
        # seg or bravo
        image_type = 'seg' if mode == ModeType.SEG else 'bravo'
        dataloader, train_dataset = create_dataloader(
            cfg=cfg,
            image_type=image_type,
            use_distributed=use_multi_gpu,
            rank=trainer.rank if use_multi_gpu else 0,
            world_size=trainer.world_size if use_multi_gpu else 1,
            augment=augment
        )

    log.info(f"Training dataset: {len(train_dataset)} slices")

    # Create validation dataloader (if val/ directory exists)
    # Pass world_size to reduce batch size for DDP (avoids OOM on single GPU)
    val_loader = None
    world_size = trainer.world_size if use_multi_gpu else 1

    if mode == 'multi':
        val_result = create_multi_diffusion_validation_dataloader(
            cfg=cfg,
            image_keys=image_keys,
            world_size=world_size,
        )
        # Create per-modality validation loaders for multi-modality metrics
        per_modality_val_loaders = {}
        for modality in image_keys:
            loader = create_single_modality_diffusion_val_loader(cfg, modality)
            if loader:
                per_modality_val_loaders[modality] = loader
        trainer.per_modality_val_loaders = per_modality_val_loaders
        if per_modality_val_loaders:
            log.info(f"Per-modality validation loaders: {list(per_modality_val_loaders.keys())}")
    elif mode == ModeType.DUAL:
        val_result = create_dual_image_validation_dataloader(
            cfg=cfg,
            image_keys=image_keys,
            conditioning='seg',
            world_size=world_size,
        )
    else:
        val_result = create_validation_dataloader(
            cfg=cfg,
            image_type=image_type,
            world_size=world_size,
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

    # Create test dataloader and evaluate (if test_new/ directory exists)
    if mode == 'multi':
        test_result = create_multi_diffusion_test_dataloader(
            cfg=cfg,
            image_keys=image_keys,
        )
    elif mode == ModeType.DUAL:
        test_result = create_dual_image_test_dataloader(
            cfg=cfg,
            image_keys=image_keys,
            conditioning='seg',
        )
    else:
        test_result = create_test_dataloader(
            cfg=cfg,
            image_type=image_type,
        )

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
