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
    create_seg_conditioned_dataloader,
    create_seg_conditioned_validation_dataloader,
    create_seg_conditioned_test_dataloader,
)
from medgen.data.loaders.latent import (
    LatentCacheBuilder,
    create_latent_dataloader,
    create_latent_validation_dataloader,
    create_latent_test_dataloader,
    load_compression_model,
)
from medgen.pipeline import DiffusionTrainer
from medgen.pipeline.spaces import PixelSpace, LatentSpace

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
    compression_model = None
    compression_type = None
    cache_dir = None

    if use_latent:
        compression_checkpoint = latent_cfg.get('compression_checkpoint')
        if not compression_checkpoint:
            raise ValueError(
                "latent.compression_checkpoint must be specified when latent.enabled=true"
            )
        if not os.path.exists(compression_checkpoint):
            raise ValueError(f"Compression checkpoint not found: {compression_checkpoint}")

        device = torch.device("cuda")

        # Load compression model
        compression_type_config = latent_cfg.get('compression_type', 'auto')
        compression_model, compression_type = load_compression_model(
            compression_checkpoint,
            compression_type_config,
            device,
        )

        # Create LatentSpace wrapper
        space = LatentSpace(compression_model, device, deterministic=True)
        space_name = f"latent ({compression_type})"

        # Determine cache directory
        cache_dir = latent_cfg.get('cache_dir')
        if cache_dir is None:
            cache_dir = f"{cfg.paths.data_dir}-latents-{compression_type}"

        log.info(f"Loaded {compression_type} compression model from {compression_checkpoint}")
        log.info(f"Latent cache directory: {cache_dir}")
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

    # Latent diffusion: build cache and use latent dataloaders
    if use_latent:
        # Build cache if needed
        cache_builder = LatentCacheBuilder(
            compression_model=compression_model,
            device=torch.device("cuda"),
            mode=mode,
            image_size=cfg.model.image_size,
            compression_type=compression_type,
        )

        compression_checkpoint = latent_cfg.get('compression_checkpoint')
        auto_encode = latent_cfg.get('auto_encode', True)

        # Check and build cache for train split
        train_cache_dir = os.path.join(cache_dir, 'train')
        if not cache_builder.validate_cache(train_cache_dir, compression_checkpoint):
            if auto_encode:
                log.info("Train cache invalid/missing, encoding dataset...")
                # Create pixel dataloader temporarily for encoding
                if mode == 'multi':
                    image_keys = list(cfg.mode.image_keys) if 'image_keys' in cfg.mode else ['bravo', 'flair', 't1_pre', 't1_gd']
                    _, pixel_dataset = create_multi_diffusion_dataloader(
                        cfg=cfg, image_keys=image_keys, augment=False
                    )
                elif mode == ModeType.DUAL:
                    image_keys = list(cfg.mode.image_keys) if 'image_keys' in cfg.mode else DEFAULT_DUAL_IMAGE_KEYS
                    _, pixel_dataset = create_dual_image_dataloader(
                        cfg=cfg, image_keys=image_keys, conditioning='seg', augment=False
                    )
                else:
                    image_type = 'seg' if mode == ModeType.SEG else 'bravo'
                    _, pixel_dataset = create_dataloader(
                        cfg=cfg, image_type=image_type, augment=False
                    )
                cache_builder.build_cache(
                    pixel_dataset, train_cache_dir, compression_checkpoint,
                    batch_size=latent_cfg.get('batch_size', 32),
                    num_workers=latent_cfg.get('num_workers', 4),
                )
            else:
                raise ValueError(f"Train cache invalid and auto_encode=false: {train_cache_dir}")

        # Check and build cache for val split
        val_cache_dir = os.path.join(cache_dir, 'val')
        val_pixel_dir = os.path.join(cfg.paths.data_dir, 'val')
        if os.path.exists(val_pixel_dir):
            if not cache_builder.validate_cache(val_cache_dir, compression_checkpoint):
                if auto_encode:
                    log.info("Val cache invalid/missing, encoding dataset...")
                    if mode == 'multi':
                        val_result = create_multi_diffusion_validation_dataloader(cfg=cfg, image_keys=image_keys)
                    elif mode == ModeType.DUAL:
                        val_result = create_dual_image_validation_dataloader(
                            cfg=cfg, image_keys=image_keys, conditioning='seg'
                        )
                    else:
                        val_result = create_validation_dataloader(cfg=cfg, image_type=image_type)
                    if val_result:
                        _, val_pixel_dataset = val_result
                        cache_builder.build_cache(
                            val_pixel_dataset, val_cache_dir, compression_checkpoint,
                            batch_size=latent_cfg.get('batch_size', 32),
                            num_workers=latent_cfg.get('num_workers', 4),
                        )
                else:
                    log.warning(f"Val cache invalid and auto_encode=false: {val_cache_dir}")

        # Create latent dataloaders
        dataloader, train_dataset = create_latent_dataloader(
            cfg=cfg,
            cache_dir=cache_dir,
            split='train',
            mode=mode,
            shuffle=True,
            use_distributed=use_multi_gpu,
            rank=trainer.rank if use_multi_gpu else 0,
            world_size=trainer.world_size if use_multi_gpu else 1,
        )
        log.info(f"Training dataset (latent): {len(train_dataset)} samples")

    else:
        # Standard pixel-space dataloaders
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
        elif mode == ModeType.SEG_CONDITIONED:
            # Seg conditioned mode - returns (seg, size_bins) tuples
            dataloader, train_dataset = create_seg_conditioned_dataloader(
                cfg=cfg,
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

    if use_latent:
        # Latent validation dataloader
        val_result = create_latent_validation_dataloader(
            cfg=cfg,
            cache_dir=cache_dir,
            mode=mode,
            world_size=world_size,
        )
    elif mode == 'multi':
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
    elif mode == ModeType.SEG_CONDITIONED:
        val_result = create_seg_conditioned_validation_dataloader(
            cfg=cfg,
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
        log.info(f"Validation dataset: {len(val_dataset)} samples")
    else:
        log.info("No val/ directory found - using train loss for best model selection")

    # Setup model
    trainer.setup_model(train_dataset)

    # Train with optional validation loader
    trainer.train(dataloader, train_dataset, val_loader=val_loader)

    # Create test dataloader and evaluate (if test_new/ directory exists)
    if use_latent:
        test_result = create_latent_test_dataloader(
            cfg=cfg,
            cache_dir=cache_dir,
            mode=mode,
        )
    elif mode == 'multi':
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
    elif mode == ModeType.SEG_CONDITIONED:
        test_result = create_seg_conditioned_test_dataloader(cfg=cfg)
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
