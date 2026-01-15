"""
Training entry point for 3D volumetric diffusion models.

Supports both pixel-space and latent-space training:
- Pixel-space: Train directly on 3D volumes [B, C, D, H, W]
- Latent-space: Train on compressed representations using 3D VAE/VQ-VAE

Usage:
    # Pixel-space training
    python -m medgen.scripts.train_diffusion_3d mode=bravo strategy=rflow

    # Latent-space training (auto-encodes dataset if cache missing)
    python -m medgen.scripts.train_diffusion_3d mode=bravo strategy=rflow \
        latent.enabled=true \
        latent.compression_checkpoint=runs/compression_3d/.../checkpoint_best.pt

    # View resolved config
    python -m medgen.scripts.train_diffusion_3d --cfg job
"""
import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="diffusion_3d")
def main(cfg: DictConfig) -> None:
    """Main 3D diffusion training entry point."""

    # Log resolved config
    logger.info("=== 3D Diffusion Training ===")
    logger.info(f"Mode: {cfg.mode.name}")
    logger.info(f"Strategy: {cfg.strategy.name}")
    logger.info(f"Volume: {cfg.volume.depth}x{cfg.volume.height}x{cfg.volume.width}")

    # Import training components
    from medgen.pipeline.diffusion_3d_trainer import Diffusion3DTrainer
    from medgen.pipeline.spaces import LatentSpace, PixelSpace
    from medgen.data.loaders.latent import load_compression_model
    from medgen.data.loaders.latent_3d import (
        LatentCacheBuilder3D,
        create_latent_3d_dataloader,
        create_latent_3d_validation_dataloader,
    )
    from medgen.data.loaders.vae_3d import (
        create_vae_3d_dataloader,
        create_vae_3d_validation_dataloader,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

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
        spatial_dims = latent_cfg.get('spatial_dims', 'auto')

        # Load 3D compression model
        logger.info(f"Loading compression model from: {checkpoint_path}")
        compression_model, comp_type, detected_dims = load_compression_model(
            checkpoint_path,
            compression_type,
            device,
            spatial_dims=spatial_dims,
        )

        # Validate it's 3D
        if detected_dims != 3:
            raise ValueError(
                f"Expected 3D compression model for train_diffusion_3d, "
                f"got {detected_dims}D. Use train.py for 2D."
            )

        logger.info(f"Loaded {comp_type} compression model ({detected_dims}D)")

        # Setup cache directory
        cache_dir = latent_cfg.get('cache_dir')
        if cache_dir is None:
            cache_dir = f"{cfg.paths.data_dir}-latents-{comp_type}-3d"
        logger.info(f"Cache directory: {cache_dir}")

        # Create cache builder
        cache_builder = LatentCacheBuilder3D(
            compression_model=compression_model,
            device=device,
            mode=cfg.mode.name,
            volume_shape=(cfg.volume.depth, cfg.volume.height, cfg.volume.width),
            compression_type=comp_type,
        )

        # Build cache for train/val if needed
        auto_encode = latent_cfg.get('auto_encode', True)
        validate_cache = latent_cfg.get('validate_cache', True)

        for split in ['train', 'val']:
            split_cache = os.path.join(cache_dir, split)

            needs_encoding = not os.path.exists(split_cache)
            if not needs_encoding and validate_cache:
                needs_encoding = not cache_builder.validate_cache(split_cache, checkpoint_path)

            if needs_encoding:
                if not auto_encode:
                    raise ValueError(
                        f"Cache missing/invalid for {split} and auto_encode=false. "
                        f"Set latent.auto_encode=true to build cache automatically."
                    )

                logger.info(f"Building {split} cache (this may take a while)...")

                # Create pixel-space dataset for encoding
                if split == 'train':
                    pixel_loader, pixel_dataset = create_vae_3d_dataloader(cfg, cfg.mode.name)
                else:
                    result = create_vae_3d_validation_dataloader(cfg, cfg.mode.name)
                    if result is None:
                        logger.warning(f"No {split} data found, skipping")
                        continue
                    pixel_loader, pixel_dataset = result

                # Build cache
                cache_builder.build_cache(pixel_dataset, split_cache, checkpoint_path)

        # Create latent dataloaders
        train_loader, train_dataset = create_latent_3d_dataloader(
            cfg, cache_dir, 'train', cfg.mode.name
        )
        logger.info(f"Train dataset: {len(train_dataset)} volumes")

        val_result = create_latent_3d_validation_dataloader(cfg, cache_dir, cfg.mode.name)
        if val_result is not None:
            val_loader, val_dataset = val_result
            logger.info(f"Val dataset: {len(val_dataset)} volumes")
        else:
            val_loader = None
            logger.warning("No validation dataset found")

        # Create LatentSpace
        space = LatentSpace(
            compression_model,
            device,
            deterministic=True,
            spatial_dims=3,
        )

    else:
        # Pixel-space training
        logger.info("=== Pixel-Space Mode ===")

        train_loader, train_dataset = create_vae_3d_dataloader(cfg, cfg.mode.name)
        logger.info(f"Train dataset: {len(train_dataset)} volumes")

        val_result = create_vae_3d_validation_dataloader(cfg, cfg.mode.name)
        if val_result is not None:
            val_loader, val_dataset = val_result
            logger.info(f"Val dataset: {len(val_dataset)} volumes")
        else:
            val_loader = None
            logger.warning("No validation dataset found")

        space = PixelSpace()

    # Create and setup trainer
    logger.info("=== Creating Trainer ===")
    trainer = Diffusion3DTrainer(cfg, space=space)
    trainer.setup_model(train_dataset)

    # Log model info
    logger.info(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    logger.info(f"Space: {type(space).__name__}")
    if hasattr(space, 'scale_factor'):
        logger.info(f"Space scale factor: {space.scale_factor}x")

    # Train
    logger.info("=== Starting Training ===")
    trainer.train(train_loader, train_dataset, val_loader=val_loader)


if __name__ == "__main__":
    main()
