"""
Unified training entry point for diffusion models on medical images.

This module provides the main training script using Hydra for configuration
management. Supports both 2D and 3D training via model.spatial_dims config.

Usage:
    # 2D training (default)
    python -m medgen.scripts.train mode=bravo strategy=rflow

    # 3D training via spatial_dims
    python -m medgen.scripts.train mode=bravo strategy=rflow model.spatial_dims=3

    # 3D training with volume config
    python -m medgen.scripts.train mode=bravo model.spatial_dims=3 \\
        volume.depth=32 volume.height=128 volume.width=128

    # Cluster training
    python -m medgen.scripts.train paths=cluster

    # Fast debug mode
    python -m medgen.scripts.train training=fast_debug
"""
import logging
import os
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from medgen.core import (
    ModeFactory,
    get_modality_for_mode,
    run_validation,
    setup_cuda_optimizations,
    validate_3d_config,
    validate_augmentation_config,
    validate_common_config,
    validate_diffusion_config,
    validate_ema_config,
    validate_latent_config,
    validate_model_config,
    validate_optimizer_config,
    validate_regional_logging,
    validate_space_to_depth_config,
    validate_strategy_config,
    validate_strategy_mode_compatibility,
    validate_training_config,
    validate_wavelet_config,
)
from medgen.data.loaders.latent import (
    LatentCacheBuilder,
    create_latent_dataloader,
    create_latent_test_dataloader,
    create_latent_validation_dataloader,
    load_compression_model,
)
from medgen.diffusion import LatentSpace, PixelSpace, SpaceToDepthSpace, WaveletSpace
from medgen.pipeline import DiffusionTrainer

# Enable CUDA optimizations
setup_cuda_optimizations()

# Flush logs immediately (SLURM buffers output when not a TTY)
logging.basicConfig(stream=sys.stderr, force=False)
for handler in logging.root.handlers:
    handler.flush = lambda: sys.stderr.flush()

logger = logging.getLogger(__name__)


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
        validate_training_config,
        validate_strategy_mode_compatibility,
        validate_3d_config,
        validate_latent_config,
        validate_space_to_depth_config,
        validate_wavelet_config,
        validate_regional_logging,
        validate_strategy_config,
        validate_ema_config,
        validate_optimizer_config,
        validate_augmentation_config,
    ])


@hydra.main(version_base=None, config_path="../../../configs", config_name="diffusion")
def main(cfg: DictConfig) -> None:
    """Main training entry point.

    Args:
        cfg: Hydra configuration object composed from YAML files.
    """
    # Validate configuration before proceeding
    validate_config(cfg)

    # Log resolved configuration (saved to config.yaml in run dir, debug-only to stdout)
    logger.debug(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Detect spatial dimensions (2D or 3D)
    spatial_dims = cfg.model.get('spatial_dims', 2)

    # Route to 3D training if spatial_dims=3
    if spatial_dims == 3:
        _train_3d(cfg)
        return

    # Continue with 2D training
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

        # Load compression model (returns model, type, spatial_dims, scale_factor, latent_channels)
        compression_type_config = latent_cfg.get('compression_type', 'auto')
        compression_model, compression_type, _, detected_scale, detected_latent_ch = load_compression_model(
            compression_checkpoint,
            compression_type_config,
            device,
        )

        # Allow config overrides for scale_factor and latent_channels
        scale_factor = latent_cfg.get('scale_factor') or detected_scale
        latent_channels = latent_cfg.get('latent_channels') or detected_latent_ch

        # Write detected values back to config so model factory can use them
        with open_dict(cfg):
            cfg.latent.scale_factor = scale_factor
            cfg.latent.latent_channels = latent_channels

        space_name = f"latent ({compression_type}, {scale_factor}x)"

        # Determine cache directory - include checkpoint hash to avoid conflicts
        # between different compression models with same type
        cache_dir = latent_cfg.get('cache_dir')
        if cache_dir is None:
            checkpoint_hash = LatentCacheBuilder.compute_checkpoint_hash(compression_checkpoint)
            cache_dir = f"{cfg.paths.data_dir}-latents-{compression_type}-{checkpoint_hash}"

        logger.info(f"Loaded {compression_type} compression model from {compression_checkpoint}")
        logger.info(f"Scale factor: {scale_factor}x, Latent channels: {latent_channels}")
        logger.info(f"Latent cache directory: {cache_dir}")
    else:
        space_name = "pixel"

    logger.info(
        f"Training: {mode} mode, {strategy} strategy, {space_name} space | "
        f"Image: {cfg.model.image_size} | Batch: {cfg.training.batch_size} | "
        f"Epochs: {cfg.training.epochs} | EMA: {cfg.training.use_ema}"
    )

    # Get mode configuration using ModeFactory
    mode_config = ModeFactory.get_mode_config(cfg)
    augment = cfg.training.get('augment', True)

    # Latent diffusion: build cache, create space with normalization stats, then dataloaders
    if use_latent:
        # Build cache if needed
        cache_builder = LatentCacheBuilder(
            compression_model=compression_model,
            device=torch.device("cuda"),
            mode=mode,
            image_size=cfg.model.image_size,
            compression_type=compression_type,
            verbose=cfg.training.get('verbose', True),
        )

        compression_checkpoint = latent_cfg.get('compression_checkpoint')
        auto_encode = latent_cfg.get('auto_encode', True)

        # Check if seg_mask is required in cache (for regional losses or seg-conditioned modes)
        logging_cfg = cfg.training.get('logging', {})
        require_seg_mask = (
            logging_cfg.get('regional_losses', False) or
            mode in ('bravo_seg_cond', 'seg', 'seg_conditioned', 'seg_conditioned_input')
        )

        # Check and build cache for train split
        train_cache_dir = os.path.join(cache_dir, 'train')
        if not cache_builder.validate_cache(
            train_cache_dir, compression_checkpoint, require_seg_mask=require_seg_mask
        ):
            if auto_encode:
                logger.info("Train cache invalid/missing, encoding dataset...")
                # Create pixel dataloader temporarily for encoding (no augmentation)
                _, pixel_dataset = ModeFactory.create_pixel_loader_for_latent_cache(
                    cfg, mode_config, split='train'
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
            if not cache_builder.validate_cache(
                val_cache_dir, compression_checkpoint, require_seg_mask=require_seg_mask
            ):
                if auto_encode:
                    logger.info("Val cache invalid/missing, encoding dataset...")
                    try:
                        _, val_pixel_dataset = ModeFactory.create_pixel_loader_for_latent_cache(
                            cfg, mode_config, split='val'
                        )
                        cache_builder.build_cache(
                            val_pixel_dataset, val_cache_dir, compression_checkpoint,
                            batch_size=latent_cfg.get('batch_size', 32),
                            num_workers=latent_cfg.get('num_workers', 4),
                        )
                    except ValueError:
                        logger.warning("No validation data found for cache building")
                else:
                    logger.warning(f"Val cache invalid and auto_encode=false: {val_cache_dir}")

        # Read normalization stats from train cache metadata (backfill if missing)
        import json
        latent_shift = None
        latent_scale = None
        train_meta_path = os.path.join(train_cache_dir, 'metadata.json')
        if os.path.exists(train_meta_path):
            with open(train_meta_path) as f:
                train_metadata = json.load(f)
            latent_shift = train_metadata.get('latent_shift')
            latent_scale = train_metadata.get('latent_scale')

            # Backfill: old caches may lack stats
            if latent_shift is None:
                logger.info("Computing normalization stats for existing cache...")
                stats = LatentCacheBuilder.compute_channel_stats(train_cache_dir)
                if stats:
                    latent_shift = stats['latent_shift']
                    latent_scale = stats['latent_scale']
                    train_metadata.update(stats)
                    with open(train_meta_path, 'w') as f:
                        json.dump(train_metadata, f, indent=2)
                    logger.info(f"Latent stats: shift={latent_shift}, scale={latent_scale}")

        # Create LatentSpace with normalization stats (after cache ensures stats exist)
        space = LatentSpace(
            compression_model, device,
            deterministic=True,
            compression_type=compression_type,
            scale_factor=scale_factor,
            latent_channels=latent_channels,
            latent_shift=latent_shift,
            latent_scale=latent_scale,
        )

        # Create trainer (needs space)
        trainer = DiffusionTrainer(cfg, space=space)

        # Create latent dataloaders (for training)
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
        logger.info(f"Training dataset (latent): {len(train_dataset)} samples")

        # Create pixel-space loaders for reference feature caching
        # FID/KID metrics are computed in pixel space, not latent space
        image_type = ModeFactory.get_image_type_for_mode(mode_config.mode)
        logger.info(f"Creating pixel-space loaders for reference features (type: {image_type})")
        pixel_train_loader, _ = ModeFactory.create_pixel_loader_for_latent_cache(
            cfg, mode_config, split='train'
        )
        pixel_val_result = ModeFactory.create_val_dataloader(cfg, mode_config)
        pixel_val_loader = pixel_val_result[0] if pixel_val_result else None

    else:
        rescale = cfg.training.get('rescale_data', False)
        space = PixelSpace(rescale=rescale)

        # Create trainer
        trainer = DiffusionTrainer(cfg, space=space)

        # Standard pixel-space dataloaders using ModeFactory
        dataloader, train_dataset = ModeFactory.create_train_dataloader(
            cfg=cfg,
            mode_config=mode_config,
            use_distributed=use_multi_gpu,
            rank=trainer.rank if use_multi_gpu else 0,
            world_size=trainer.world_size if use_multi_gpu else 1,
            augment=augment,
        )

        logger.info(f"Training dataset: {len(train_dataset)} slices")
        # Pixel-space training: no separate loaders needed for reference features
        pixel_train_loader = None
        pixel_val_loader = None

    logger.info(f"Validation: every epoch, figures at interval {trainer.figure_interval}")

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
    else:
        # Pixel-space validation dataloader using ModeFactory
        val_result = ModeFactory.create_val_dataloader(cfg, mode_config, world_size=world_size)

        # Create per-modality validation loaders for multi-modality metrics (MULTI mode only)
        per_modality_val_loaders = ModeFactory.create_per_modality_val_loaders(cfg, mode_config)
        if per_modality_val_loaders:
            trainer.per_modality_val_loaders = per_modality_val_loaders
            logger.info(f"Per-modality validation loaders: {list(per_modality_val_loaders.keys())}")

    if val_result is not None:
        val_loader, val_dataset = val_result
        logger.info(f"Validation dataset: {len(val_dataset)} samples")
    else:
        logger.info("No val/ directory found - using train loss for best model selection")

    # Setup model
    trainer.setup_model(train_dataset)

    # Resume from checkpoint if specified
    start_epoch = 0
    resume_from = cfg.training.get('resume_from', None)
    if resume_from:
        start_epoch = trainer.load_checkpoint(resume_from)

    # Set pixel-space loaders for latent diffusion reference features
    trainer.pixel_train_loader = pixel_train_loader
    trainer.pixel_val_loader = pixel_val_loader

    # Train with optional validation loader
    trainer.train(
        dataloader, train_dataset,
        val_loader=val_loader,
        start_epoch=start_epoch,
    )

    # On wall-time signal: skip test eval, let SLURM script handle resubmission
    if getattr(trainer, '_sigterm_received', False):
        logger.info("Wall time signal received — skipping test evaluation")
        trainer.close_writer()
        return

    # Create test dataloader and evaluate (if test_new/ directory exists)
    if use_latent:
        test_result = create_latent_test_dataloader(
            cfg=cfg,
            cache_dir=cache_dir,
            mode=mode,
        )
    else:
        # Pixel-space test dataloader using ModeFactory
        test_result = ModeFactory.create_test_dataloader(cfg, mode_config)

    if test_result is not None:
        test_loader, test_dataset = test_result
        logger.info(f"Test dataset: {len(test_dataset)} slices")
        # Evaluate on both best and latest checkpoints
        trainer.evaluate_test_set(test_loader, checkpoint_name="best")
        trainer.evaluate_test_set(test_loader, checkpoint_name="latest")
    else:
        logger.info("No test_new/ directory found - skipping test evaluation")

    # Close TensorBoard writer after all logging
    trainer.close_writer()

    # Mark training complete for SLURM job chaining
    from pathlib import Path
    (Path(trainer.save_dir) / '.training_complete').touch()


def _train_3d(cfg: DictConfig) -> None:
    """3D volumetric training entry point.

    Called when model.spatial_dims=3 is set in config.

    Args:
        cfg: Hydra configuration object.
    """

    from medgen.data.loaders.latent import (
        LatentCacheBuilder3D,
        create_latent_3d_dataloader,
        create_latent_3d_validation_dataloader,
        load_compression_model,
    )
    from medgen.data.loaders.volume_3d import (
        SingleModality3DDatasetWithSeg,
        create_vae_3d_dataloader,
        create_vae_3d_validation_dataloader,
    )

    mode = cfg.mode.name
    strategy = cfg.strategy.name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get mode configuration using ModeFactory (available for both latent and pixel paths)
    mode_config = ModeFactory.get_mode_config(cfg)

    logger.info(
        f"Training 3D: {mode} mode, {strategy} strategy | "
        f"Volume: {cfg.volume.depth}x{cfg.volume.height}x{cfg.volume.width} | "
        f"Batch: {cfg.training.batch_size} | Epochs: {cfg.training.epochs}"
    )

    # Check latent diffusion config
    latent_cfg = cfg.get('latent', {})
    use_latent = latent_cfg.get('enabled', False)

    if use_latent:
        logger.info("=== Latent Diffusion Mode ===")

        # Get checkpoint path
        checkpoint_path = latent_cfg.get('compression_checkpoint')
        if checkpoint_path is None:
            raise ValueError("latent.compression_checkpoint required when latent.enabled=true")

        compression_type = latent_cfg.get('compression_type', 'auto')
        spatial_dims_cfg = latent_cfg.get('spatial_dims', 'auto')

        # Load 3D compression model (returns model, type, dims, scale_factor, latent_channels)
        logger.info(f"Loading compression model from: {checkpoint_path}")
        compression_model, comp_type, detected_dims, detected_scale, detected_latent_ch = load_compression_model(
            checkpoint_path,
            compression_type,
            device,
            spatial_dims=spatial_dims_cfg,
        )

        # Check for slicewise encoding (2D encoder applied slice-by-slice for 3D)
        slicewise_encoding = latent_cfg.get('slicewise_encoding', False)

        # Validate dimensions
        if slicewise_encoding:
            # Slicewise mode: expect 2D encoder for 3D volume training
            if detected_dims != 2:
                raise ValueError(
                    f"slicewise_encoding=true requires 2D compression model, got {detected_dims}D."
                )
            logger.info("Slicewise encoding: 2D encoder will be applied slice-by-slice to 3D volumes")
        else:
            # Standard mode: expect 3D encoder for 3D volume training
            if detected_dims != 3:
                raise ValueError(
                    f"Expected 3D compression model for spatial_dims=3, got {detected_dims}D. "
                    f"Use slicewise_encoding=true for 2D encoders."
                )

        # Allow config overrides
        scale_factor = latent_cfg.get('scale_factor') or detected_scale
        depth_scale_factor = latent_cfg.get('depth_scale_factor') or scale_factor
        latent_channels = latent_cfg.get('latent_channels') or detected_latent_ch

        # Write detected values back to config so model factory can use them
        with open_dict(cfg):
            cfg.latent.scale_factor = scale_factor
            cfg.latent.depth_scale_factor = depth_scale_factor
            cfg.latent.latent_channels = latent_channels

        logger.info(f"Loaded {comp_type} compression model ({detected_dims}D)")
        logger.info(f"Scale factor: {scale_factor}x (depth: {depth_scale_factor}x), Latent channels: {latent_channels}")

        # Load separate seg compression model if specified (dual-encoder setup)
        seg_compression_model = None
        seg_checkpoint_path = latent_cfg.get('seg_compression_checkpoint')
        if seg_checkpoint_path is not None:
            logger.info(f"Loading SEG compression model from: {seg_checkpoint_path}")
            seg_compression_model, seg_comp_type, _, _, _ = load_compression_model(
                seg_checkpoint_path,
                compression_type,  # Use same type as bravo encoder
                device,
                spatial_dims=spatial_dims_cfg,
            )
            logger.info(f"Loaded SEG encoder: {seg_comp_type}")

        # Check for slicewise encoding (2D encoder applied slice-by-slice for 3D)
        slicewise_encoding = latent_cfg.get('slicewise_encoding', False)
        if slicewise_encoding:
            logger.info("Using slicewise encoding: 2D encoder applied slice-by-slice")

        # Setup cache directory - include checkpoint hash(es) to avoid conflicts
        # between different compression models with same type (e.g., 4x vs 8x VQ-VAE)
        cache_dir = latent_cfg.get('cache_dir')
        if cache_dir is None:
            checkpoint_hash = LatentCacheBuilder3D.compute_checkpoint_hash(checkpoint_path)
            if seg_checkpoint_path is not None:
                # Dual encoder: include both hashes (first 8 chars each)
                seg_hash = LatentCacheBuilder3D.compute_checkpoint_hash(seg_checkpoint_path)
                combined_hash = f"{checkpoint_hash[:8]}_{seg_hash[:8]}"
                cache_dir = f"{cfg.paths.data_dir}-latents-{comp_type}-3d-{combined_hash}"
            else:
                cache_dir = f"{cfg.paths.data_dir}-latents-{comp_type}-3d-{checkpoint_hash}"
        logger.info(f"Cache directory: {cache_dir}")

        # Create cache builder
        cache_builder = LatentCacheBuilder3D(
            compression_model=compression_model,
            device=device,
            mode=mode,
            volume_shape=(cfg.volume.depth, cfg.volume.height, cfg.volume.width),
            compression_type=comp_type,
            verbose=cfg.training.get('verbose', True),
            slicewise_encoding=slicewise_encoding,
            seg_compression_model=seg_compression_model,
        )

        # Build cache for train/val if needed
        auto_encode = latent_cfg.get('auto_encode', True)
        validate_cache = latent_cfg.get('validate_cache', True)

        # Check if seg_mask is required in cache (for regional losses or seg-conditioned modes)
        logging_cfg = cfg.training.get('logging', {})
        require_seg_mask = (
            logging_cfg.get('regional_losses', False) or
            mode in ('bravo_seg_cond', 'seg', 'seg_conditioned', 'seg_conditioned_input')
        )

        for split in ['train', 'val', 'test_new']:
            split_cache = os.path.join(cache_dir, split)

            needs_encoding = not os.path.exists(split_cache)
            if not needs_encoding and validate_cache:
                needs_encoding = not cache_builder.validate_cache(
                    split_cache, checkpoint_path,
                    require_seg_mask=require_seg_mask,
                    seg_checkpoint_path=seg_checkpoint_path,
                )

            if needs_encoding:
                if not auto_encode:
                    # Only raise error for train split - val/test are optional
                    if split == 'train':
                        raise ValueError(
                            f"Cache missing/invalid for {split} and auto_encode=false. "
                            f"Set latent.auto_encode=true to build cache automatically."
                        )
                    else:
                        logger.info(f"No {split} cache found (optional)")
                        continue

                logger.info(f"Building {split} cache (this may take a while)...")

                # Map mode to modality for pixel dataset loading
                # bravo_seg_cond uses bravo.nii.gz, seg_conditioned uses seg.nii.gz, etc.
                pixel_modality = get_modality_for_mode(mode)

                # Create pixel-space dataset for encoding
                if split == 'train':
                    pixel_loader, pixel_dataset = create_vae_3d_dataloader(cfg, pixel_modality)
                elif split == 'val':
                    result = create_vae_3d_validation_dataloader(cfg, pixel_modality)
                    if result is None:
                        logger.warning(f"No {split} data found, skipping")
                        continue
                    pixel_loader, pixel_dataset = result
                else:  # test_new
                    # Check if test_new directory exists
                    test_dir = os.path.join(cfg.paths.data_dir, 'test_new')
                    if not os.path.exists(test_dir):
                        logger.info("No test_new data found, skipping test cache")
                        continue
                    # Use SingleModality3DDatasetWithSeg for test data
                    pixel_dataset = SingleModality3DDatasetWithSeg(
                        data_dir=test_dir,
                        modality=pixel_modality,
                        height=cfg.volume.height,
                        width=cfg.volume.width,
                        pad_depth_to=cfg.volume.pad_depth_to,
                        pad_mode=cfg.volume.get('pad_mode', 'replicate'),
                        slice_step=cfg.volume.get('slice_step', 1),
                    )

                # Build cache
                cache_builder.build_cache(
                    pixel_dataset, split_cache, checkpoint_path,
                    seg_checkpoint_path=seg_checkpoint_path,
                )

        # Create latent dataloaders (for training)
        train_loader, train_dataset = create_latent_3d_dataloader(
            cfg, cache_dir, 'train', mode
        )
        logger.info(f"Train dataset (latent): {len(train_dataset)} volumes")

        val_result = create_latent_3d_validation_dataloader(cfg, cache_dir, mode)
        if val_result is not None:
            val_loader, val_dataset = val_result
            logger.info(f"Val dataset (latent): {len(val_dataset)} volumes")
        else:
            val_loader = None
            logger.warning("No validation dataset found")

        # Create pixel-space loaders for reference feature caching
        # FID/KID metrics are computed in pixel space, not latent space
        pixel_modality = get_modality_for_mode(mode)
        logger.info(f"Creating pixel-space loaders for reference features (modality: {pixel_modality})")
        pixel_train_loader, _ = create_vae_3d_dataloader(cfg, pixel_modality)
        pixel_val_result = create_vae_3d_validation_dataloader(cfg, pixel_modality)
        pixel_val_loader = pixel_val_result[0] if pixel_val_result else None

        # Read normalization stats from train cache metadata (backfill if missing)
        import json
        latent_shift = None
        latent_scale = None
        train_cache_dir = os.path.join(cache_dir, 'train')
        train_meta_path = os.path.join(train_cache_dir, 'metadata.json')
        if os.path.exists(train_meta_path):
            with open(train_meta_path) as f:
                train_metadata = json.load(f)
            latent_shift = train_metadata.get('latent_shift')
            latent_scale = train_metadata.get('latent_scale')

            # Backfill: old caches may lack stats
            if latent_shift is None:
                logger.info("Computing normalization stats for existing cache...")
                stats = LatentCacheBuilder.compute_channel_stats(train_cache_dir)
                if stats:
                    latent_shift = stats['latent_shift']
                    latent_scale = stats['latent_scale']
                    train_metadata.update(stats)
                    with open(train_meta_path, 'w') as f:
                        json.dump(train_metadata, f, indent=2)
                    logger.info(f"Latent stats: shift={latent_shift}, scale={latent_scale}")

        # Create LatentSpace with detected/configured parameters
        space = LatentSpace(
            compression_model,
            device,
            deterministic=True,
            spatial_dims=3 if not slicewise_encoding else 2,  # Model is 2D for slicewise
            compression_type=comp_type,
            scale_factor=scale_factor,
            depth_scale_factor=depth_scale_factor,
            latent_channels=latent_channels,
            slicewise_encoding=slicewise_encoding,
            latent_shift=latent_shift,
            latent_scale=latent_scale,
        )

    else:
        # Pixel-space training using ModeFactory
        logger.info("=== Pixel-Space Mode ===")

        # Create train dataloader using unified factory (mode_config defined at top of function)
        train_loader, train_dataset = ModeFactory.create_train_dataloader(cfg, mode_config)
        logger.info(f"Train dataset: {len(train_dataset)} volumes")

        # Create validation dataloader
        val_result = ModeFactory.create_val_dataloader(cfg, mode_config)
        if val_result is not None:
            val_loader, val_dataset = val_result
            logger.info(f"Val dataset: {len(val_dataset)} volumes")
        else:
            val_loader = None
            logger.warning("No validation dataset found")

        # Check for space-to-depth or wavelet rearrangement
        s2d_cfg = cfg.get('space_to_depth', {})
        wavelet_cfg = cfg.get('wavelet', {})

        if s2d_cfg.get('enabled', False):
            spatial_factor = s2d_cfg.get('spatial_factor', 2)
            depth_factor = s2d_cfg.get('depth_factor', 2)
            rescale = cfg.training.get('rescale_data', False)
            space = SpaceToDepthSpace(
                spatial_factor=spatial_factor,
                depth_factor=depth_factor,
                rescale=rescale,
            )
            logger.info(
                f"Space-to-depth: {space.latent_channels}x channels, "
                f"spatial {spatial_factor}x, depth {depth_factor}x"
                f"{', rescale [-1,1]' if rescale else ''}"
            )
        elif wavelet_cfg.get('enabled', False):
            rescale = wavelet_cfg.get('rescale', True)
            if wavelet_cfg.get('normalize', True):
                stats = WaveletSpace.compute_subband_stats(train_loader, rescale=rescale)
                wavelet_shift = stats.get('wavelet_shift')
                wavelet_scale = stats.get('wavelet_scale')
                if wavelet_shift:
                    names = ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']
                    for i, name in enumerate(names):
                        if i < len(wavelet_shift):
                            logger.info(f"  {name}: shift={wavelet_shift[i]:.4f}, scale={wavelet_scale[i]:.4f}")
                space = WaveletSpace(shift=wavelet_shift, scale=wavelet_scale, rescale=rescale)
                logger.info(f"Wavelet space: Haar 3D 2x2x2, 8x channels, per-subband normalized{', rescale [-1,1]' if rescale else ''}")
            else:
                space = WaveletSpace(rescale=rescale)
                logger.info(f"Wavelet space: Haar 3D 2x2x2, 8x channels, raw coefficients{', rescale [-1,1]' if rescale else ''}")
        else:
            rescale = cfg.training.get('rescale_data', False)
            space = PixelSpace(rescale=rescale)

        # Pixel-space training: no separate loaders needed for reference features
        pixel_train_loader = None
        pixel_val_loader = None

    # Create and setup trainer
    trainer = DiffusionTrainer.create_3d(cfg, space=space)
    trainer.setup_model(train_dataset)

    # Resume from checkpoint if specified
    start_epoch = 0
    resume_from = cfg.training.get('resume_from', None)
    if resume_from:
        start_epoch = trainer.load_checkpoint(resume_from)

    # Log model info
    logger.info(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    logger.info(f"Space: {type(space).__name__}")
    if hasattr(space, 'scale_factor'):
        logger.info(f"Space scale factor: {space.scale_factor}x")

    # Set pixel-space loaders for latent diffusion reference features
    trainer.pixel_train_loader = pixel_train_loader
    trainer.pixel_val_loader = pixel_val_loader

    # Train
    trainer.train(
        train_loader, train_dataset,
        val_loader=val_loader,
        start_epoch=start_epoch,
    )

    # On wall-time signal: skip test eval, let SLURM script handle resubmission
    if getattr(trainer, '_sigterm_received', False):
        logger.info("Wall time signal received — skipping test evaluation")
        trainer.close_writer()
        return

    # Test evaluation (if test_new set exists and enabled)
    run_test = cfg.training.get('test_after_training', True)
    test_dir = os.path.join(cfg.paths.data_dir, 'test_new')
    use_latent = cfg.get('latent', {}).get('enabled', False)

    if run_test and os.path.exists(test_dir):
        logger.info("=== Test Evaluation ===")

        # Create test dataloader based on mode and latent/pixel space
        test_result = None
        try:
            if use_latent:
                # Latent diffusion: use latent test dataloader
                # Note: cache_dir is the local variable set earlier during latent training setup
                from medgen.data.loaders.latent import create_latent_test_dataloader
                test_result = create_latent_test_dataloader(cfg, cache_dir, mode)
                if test_result is None:
                    logger.warning("No test cache found for latent diffusion")
            else:
                # Pixel-space test dataloader using ModeFactory
                # Note: mode_config was set in the pixel-space training branch
                test_result = ModeFactory.create_test_dataloader(cfg, mode_config)
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            logger.warning(f"Could not create test dataloader: {e}")

        if test_result is not None:
            test_loader, test_dataset = test_result
            logger.info(f"Test dataset: {len(test_dataset)} volumes")

            # Evaluate best checkpoint
            trainer.evaluate_test_set(test_loader, checkpoint_name='best')

            # Evaluate latest checkpoint
            trainer.evaluate_test_set(test_loader, checkpoint_name='latest')
        else:
            logger.warning("No test dataset found or could not create dataloader")
    else:
        if run_test:
            logger.info("No test_new/ directory found - skipping test evaluation")

    # Close TensorBoard writer
    trainer.close_writer()

    # Mark training complete for SLURM job chaining
    from pathlib import Path
    (Path(trainer.save_dir) / '.training_complete').touch()


if __name__ == "__main__":
    main()
