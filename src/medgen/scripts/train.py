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

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from medgen.core import (
    DEFAULT_DUAL_IMAGE_KEYS,
    ModeType,
    get_modality_for_mode,
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
from medgen.diffusion import PixelSpace, LatentSpace

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

        # Create LatentSpace wrapper with detected/configured parameters
        space = LatentSpace(
            compression_model, device,
            deterministic=True,
            compression_type=compression_type,
            scale_factor=scale_factor,
            latent_channels=latent_channels,
        )
        space_name = f"latent ({compression_type}, {scale_factor}x)"

        # Determine cache directory - include checkpoint hash to avoid conflicts
        # between different compression models with same type
        cache_dir = latent_cfg.get('cache_dir')
        if cache_dir is None:
            checkpoint_hash = LatentCacheBuilder.compute_checkpoint_hash(compression_checkpoint)
            cache_dir = f"{cfg.paths.data_dir}-latents-{compression_type}-{checkpoint_hash}"

        log.info(f"Loaded {compression_type} compression model from {compression_checkpoint}")
        log.info(f"Scale factor: {scale_factor}x, Latent channels: {latent_channels}")
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
            if not cache_builder.validate_cache(
                val_cache_dir, compression_checkpoint, require_seg_mask=require_seg_mask
            ):
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
        log.info(f"Training dataset (latent): {len(train_dataset)} samples")

        # Create pixel-space loaders for reference feature caching
        # FID/KID metrics are computed in pixel space, not latent space
        image_type = 'seg' if mode == ModeType.SEG else 'bravo'
        log.info(f"Creating pixel-space loaders for reference features (type: {image_type})")
        pixel_train_loader, _ = create_dataloader(cfg=cfg, image_type=image_type, augment=False)
        pixel_val_result = create_validation_dataloader(cfg=cfg, image_type=image_type)
        pixel_val_loader = pixel_val_result[0] if pixel_val_result else None

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
        # Pixel-space training: no separate loaders needed for reference features
        pixel_train_loader = None
        pixel_val_loader = None

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
    trainer.train(
        dataloader, train_dataset,
        val_loader=val_loader,
        pixel_train_loader=pixel_train_loader,
        pixel_val_loader=pixel_val_loader,
    )

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


def _train_3d(cfg: DictConfig) -> None:
    """3D volumetric training entry point.

    Called when model.spatial_dims=3 is set in config.

    Args:
        cfg: Hydra configuration object.
    """
    from medgen.data.loaders.latent import (
        load_compression_model,
        LatentCacheBuilder3D,
        create_latent_3d_dataloader,
        create_latent_3d_validation_dataloader,
    )
    from medgen.data.loaders.volume_3d import (
        create_vae_3d_dataloader,
        create_vae_3d_validation_dataloader,
        SingleModality3DDatasetWithSeg,
    )
    from medgen.data.loaders.seg import (
        create_seg_dataloader,
        create_seg_validation_dataloader,
    )
    from torch.utils.data import DataLoader

    mode = cfg.mode.name
    strategy = cfg.strategy.name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log.info("")
    log.info("=" * 60)
    log.info("3D Volumetric Diffusion Training")
    log.info(f"Mode: {mode} | Strategy: {strategy}")
    log.info(f"Volume: {cfg.volume.depth}x{cfg.volume.height}x{cfg.volume.width}")
    log.info("=" * 60)
    log.info("")

    # Check latent diffusion config
    latent_cfg = cfg.get('latent', {})
    use_latent = latent_cfg.get('enabled', False)

    if use_latent:
        log.info("=== Latent Diffusion Mode ===")

        # Get checkpoint path
        checkpoint_path = latent_cfg.get('compression_checkpoint')
        if checkpoint_path is None:
            raise ValueError("latent.compression_checkpoint required when latent.enabled=true")

        compression_type = latent_cfg.get('compression_type', 'auto')
        spatial_dims_cfg = latent_cfg.get('spatial_dims', 'auto')

        # Load 3D compression model (returns model, type, dims, scale_factor, latent_channels)
        log.info(f"Loading compression model from: {checkpoint_path}")
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
            log.info("Slicewise encoding: 2D encoder will be applied slice-by-slice to 3D volumes")
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

        log.info(f"Loaded {comp_type} compression model ({detected_dims}D)")
        log.info(f"Scale factor: {scale_factor}x (depth: {depth_scale_factor}x), Latent channels: {latent_channels}")

        # Load separate seg compression model if specified (dual-encoder setup)
        seg_compression_model = None
        seg_checkpoint_path = latent_cfg.get('seg_compression_checkpoint')
        if seg_checkpoint_path is not None:
            log.info(f"Loading SEG compression model from: {seg_checkpoint_path}")
            seg_compression_model, seg_comp_type, _, _, _ = load_compression_model(
                seg_checkpoint_path,
                compression_type,  # Use same type as bravo encoder
                device,
                spatial_dims=spatial_dims_cfg,
            )
            log.info(f"Loaded SEG encoder: {seg_comp_type}")

        # Check for slicewise encoding (2D encoder applied slice-by-slice for 3D)
        slicewise_encoding = latent_cfg.get('slicewise_encoding', False)
        if slicewise_encoding:
            log.info("Using slicewise encoding: 2D encoder applied slice-by-slice")

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
        log.info(f"Cache directory: {cache_dir}")

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
                        log.info(f"No {split} cache found (optional)")
                        continue

                log.info(f"Building {split} cache (this may take a while)...")

                # Map mode to modality for pixel dataset loading
                # bravo_seg_cond uses bravo.nii.gz, seg_conditioned uses seg.nii.gz, etc.
                pixel_modality = get_modality_for_mode(mode)

                # Create pixel-space dataset for encoding
                if split == 'train':
                    pixel_loader, pixel_dataset = create_vae_3d_dataloader(cfg, pixel_modality)
                elif split == 'val':
                    result = create_vae_3d_validation_dataloader(cfg, pixel_modality)
                    if result is None:
                        log.warning(f"No {split} data found, skipping")
                        continue
                    pixel_loader, pixel_dataset = result
                else:  # test_new
                    # Check if test_new directory exists
                    test_dir = os.path.join(cfg.paths.data_dir, 'test_new')
                    if not os.path.exists(test_dir):
                        log.info("No test_new data found, skipping test cache")
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
        log.info(f"Train dataset (latent): {len(train_dataset)} volumes")

        val_result = create_latent_3d_validation_dataloader(cfg, cache_dir, mode)
        if val_result is not None:
            val_loader, val_dataset = val_result
            log.info(f"Val dataset (latent): {len(val_dataset)} volumes")
        else:
            val_loader = None
            log.warning("No validation dataset found")

        # Create pixel-space loaders for reference feature caching
        # FID/KID metrics are computed in pixel space, not latent space
        pixel_modality = get_modality_for_mode(mode)
        log.info(f"Creating pixel-space loaders for reference features (modality: {pixel_modality})")
        pixel_train_loader, _ = create_vae_3d_dataloader(cfg, pixel_modality)
        pixel_val_result = create_vae_3d_validation_dataloader(cfg, pixel_modality)
        pixel_val_loader = pixel_val_result[0] if pixel_val_result else None

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
        )

    else:
        # Pixel-space training
        log.info("=== Pixel-Space Mode ===")

        # Use mode-specific dataloader
        if mode in ('seg', 'seg_conditioned', 'seg_conditioned_input'):
            log.info("Using 3D seg dataloader (3D connected components)")
            train_loader, train_dataset = create_seg_dataloader(cfg)
            val_result = create_seg_validation_dataloader(cfg)
        elif mode == 'bravo':
            # Bravo mode: needs seg mask for conditioning
            log.info("Using 3D bravo dataloader with seg mask conditioning")
            train_dir = os.path.join(cfg.paths.data_dir, 'train')
            train_dataset = SingleModality3DDatasetWithSeg(
                data_dir=train_dir,
                modality='bravo',
                height=cfg.volume.height,
                width=cfg.volume.width,
                pad_depth_to=cfg.volume.pad_depth_to,
                pad_mode=cfg.volume.get('pad_mode', 'replicate'),
                slice_step=cfg.volume.get('slice_step', 1),
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=cfg.training.get('num_workers', 4),
                pin_memory=True,
                drop_last=True,
            )

            val_dir = os.path.join(cfg.paths.data_dir, 'val')
            if os.path.exists(val_dir):
                val_dataset = SingleModality3DDatasetWithSeg(
                    data_dir=val_dir,
                    modality='bravo',
                    height=cfg.volume.height,
                    width=cfg.volume.width,
                    pad_depth_to=cfg.volume.pad_depth_to,
                    pad_mode=cfg.volume.get('pad_mode', 'replicate'),
                    slice_step=cfg.volume.get('slice_step', 1),
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                    num_workers=cfg.training.get('num_workers', 4),
                    pin_memory=True,
                )
                val_result = (val_loader, val_dataset)
            else:
                val_result = None
        else:
            # Other modes - map mode to modality for file loading
            pixel_modality = get_modality_for_mode(mode)
            train_loader, train_dataset = create_vae_3d_dataloader(cfg, pixel_modality)
            val_result = create_vae_3d_validation_dataloader(cfg, pixel_modality)

        log.info(f"Train dataset: {len(train_dataset)} volumes")

        if val_result is not None:
            val_loader, val_dataset = val_result
            log.info(f"Val dataset: {len(val_dataset)} volumes")
        else:
            val_loader = None
            log.warning("No validation dataset found")

        space = PixelSpace()
        # Pixel-space training: no separate loaders needed for reference features
        pixel_train_loader = None
        pixel_val_loader = None

    # Create and setup trainer
    log.info("=== Creating 3D Trainer ===")
    trainer = DiffusionTrainer.create_3d(cfg, space=space)
    trainer.setup_model(train_dataset)

    # Log model info
    log.info(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    log.info(f"Space: {type(space).__name__}")
    if hasattr(space, 'scale_factor'):
        log.info(f"Space scale factor: {space.scale_factor}x")

    # Train
    log.info("=== Starting Training ===")
    trainer.train(
        train_loader, train_dataset,
        val_loader=val_loader,
        pixel_train_loader=pixel_train_loader,
        pixel_val_loader=pixel_val_loader,
    )

    # Test evaluation (if test_new set exists and enabled)
    run_test = cfg.training.get('test_after_training', True)
    test_dir = os.path.join(cfg.paths.data_dir, 'test_new')
    use_latent = cfg.latent.get('enabled', False)

    if run_test and os.path.exists(test_dir):
        log.info("=== Test Evaluation ===")

        # Create test dataloader based on mode and latent/pixel space
        test_result = None
        try:
            if use_latent:
                # Latent diffusion: use latent test dataloader
                # Note: cache_dir is the local variable set earlier during latent training setup
                from medgen.data.loaders.latent import create_latent_test_dataloader
                test_result = create_latent_test_dataloader(cfg, cache_dir, mode)
                if test_result is None:
                    log.warning("No test cache found for latent diffusion")
            elif mode in ('bravo', 'bravo_seg_cond'):
                # Bravo or bravo_seg_cond: both use SingleModality3DDatasetWithSeg
                test_dataset = SingleModality3DDatasetWithSeg(
                    data_dir=test_dir,
                    modality='bravo',
                    height=cfg.volume.height,
                    width=cfg.volume.width,
                    pad_depth_to=cfg.volume.pad_depth_to,
                    pad_mode=cfg.volume.get('pad_mode', 'replicate'),
                    slice_step=cfg.volume.get('slice_step', 1),
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                    num_workers=cfg.training.get('num_workers', 4),
                    pin_memory=True,
                )
                test_result = (test_loader, test_dataset)
            elif mode == 'seg':
                from medgen.data.loaders.seg import SegDataset
                size_bin_cfg = cfg.mode.get('size_bins', {})
                bin_edges = list(size_bin_cfg.get('edges', [0, 3, 6, 10, 15, 20, 30]))
                num_bins = int(size_bin_cfg.get('num_bins', 7))
                default_spacing = 240.0 / cfg.volume.height
                voxel_spacing = tuple(size_bin_cfg.get('voxel_spacing', [1.0, default_spacing, default_spacing]))
                test_dataset = SegDataset(
                    data_dir=test_dir,
                    bin_edges=bin_edges,
                    num_bins=num_bins,
                    voxel_spacing=voxel_spacing,
                    height=cfg.volume.height,
                    width=cfg.volume.width,
                    pad_depth_to=cfg.volume.pad_depth_to,
                    pad_mode=cfg.volume.get('pad_mode', 'replicate'),
                    slice_step=cfg.volume.get('slice_step', 1),
                    positive_only=False,
                    cfg_dropout_prob=0.0,
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                    num_workers=cfg.training.get('num_workers', 4),
                    pin_memory=True,
                )
                test_result = (test_loader, test_dataset)
        except Exception as e:
            log.warning(f"Could not create test dataloader: {e}")

        if test_result is not None:
            test_loader, test_dataset = test_result
            log.info(f"Test dataset: {len(test_dataset)} volumes")

            # Evaluate best checkpoint
            trainer.evaluate_test_set(test_loader, checkpoint_name='best')

            # Evaluate latest checkpoint
            trainer.evaluate_test_set(test_loader, checkpoint_name='latest')
        else:
            log.warning("No test dataset found or could not create dataloader")
    else:
        if run_test:
            log.info("No test_new/ directory found - skipping test evaluation")

    # Close TensorBoard writer
    trainer.close_writer()


if __name__ == "__main__":
    main()
