"""Unified training script for compression models (VAE, VQ-VAE, DC-AE).

Supports both 2D and 3D training. The trainer type and spatial dimensions
are detected automatically from the Hydra config.

Usage:
    # VAE 2D
    python -m medgen.scripts.train_compression --config-name=vae

    # VAE 3D
    python -m medgen.scripts.train_compression --config-name=vae_3d

    # VQ-VAE 2D
    python -m medgen.scripts.train_compression --config-name=vqvae

    # VQ-VAE 3D
    python -m medgen.scripts.train_compression --config-name=vqvae_3d

    # DC-AE 2D
    python -m medgen.scripts.train_compression --config-name=dcae

    # DC-AE 3D
    python -m medgen.scripts.train_compression --config-name=dcae_3d

    # With overrides
    python -m medgen.scripts.train_compression --config-name=vae mode=bravo
    python -m medgen.scripts.train_compression --config-name=vae_3d paths=cluster
"""
import logging
from collections.abc import Callable

import hydra
from omegaconf import DictConfig, OmegaConf

from medgen.core import (
    run_validation,
    setup_cuda_optimizations,
    validate_common_config,
    validate_model_config,
    validate_vae_config,
    validate_vqvae_config,
)
from medgen.data import (
    create_multi_modality_dataloader,
    create_multi_modality_test_dataloader,
    create_multi_modality_validation_dataloader,
    create_single_modality_validation_loader,
    # 3D dataloaders
    create_vae_3d_dataloader,
    create_vae_3d_multi_modality_dataloader,
    create_vae_3d_multi_modality_test_dataloader,
    create_vae_3d_multi_modality_validation_dataloader,
    create_vae_3d_single_modality_validation_loader,
    create_vae_3d_test_dataloader,
    create_vae_3d_validation_dataloader,
    # 2D dataloaders
    create_vae_dataloader,
    create_vae_test_dataloader,
    create_vae_validation_dataloader,
)
from medgen.data.loaders.builder_2d import create_seg_compression_loader
from medgen.pipeline import DCAETrainer, VAETrainer, VQVAETrainer

from .common import (
    create_per_modality_val_loaders,
    create_per_modality_val_loaders_3d,
    get_image_keys,
    override_vae_channels,
    run_test_evaluation,
)

# Enable CUDA optimizations at module import
setup_cuda_optimizations()

logger = logging.getLogger(__name__)


# =============================================================================
# Config Validators
# =============================================================================

def validate_vae_3d_config(cfg: DictConfig) -> list:
    """Validate 3D VAE configuration."""
    errors = []

    if 'volume' not in cfg:
        errors.append("Missing 'volume' config section for 3D VAE")
        return errors

    volume = cfg.volume
    if volume.get('depth', 0) <= 0:
        errors.append("volume.depth must be positive")
    if volume.get('height', 0) <= 0:
        errors.append("volume.height must be positive")
    if volume.get('width', 0) <= 0:
        errors.append("volume.width must be positive")

    if volume.get('depth', 0) % 8 != 0:
        errors.append(f"volume.depth ({volume.get('depth')}) must be divisible by 8")

    if 'vae_3d' not in cfg:
        errors.append("Missing 'vae_3d' config section")

    return errors


def validate_vqvae_3d_config(cfg: DictConfig) -> list:
    """Validate 3D VQ-VAE configuration."""
    errors = []

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

    n_downsamples = len(cfg.vqvae_3d.get('channels', [64, 128]))
    divisor = 2 ** n_downsamples
    if volume.get('depth', 0) % divisor != 0:
        errors.append(
            f"volume.depth ({volume.get('depth')}) must be divisible by {divisor}"
        )

    if 'vqvae_3d' not in cfg:
        errors.append("Missing 'vqvae_3d' config section")

    return errors


def validate_dcae_config(cfg: DictConfig) -> list:
    """Validate 2D DC-AE configuration."""
    errors = []

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


def validate_dcae_3d_config(cfg: DictConfig) -> list:
    """Validate 3D DC-AE configuration."""
    import math

    errors = []

    if 'volume' not in cfg:
        errors.append("Missing 'volume' config section for 3D DC-AE")
        return errors

    volume = cfg.volume
    if volume.get('depth', 0) <= 0:
        errors.append("volume.depth must be positive")
    if volume.get('height', 0) <= 0:
        errors.append("volume.height must be positive")
    if volume.get('width', 0) <= 0:
        errors.append("volume.width must be positive")

    if 'dcae_3d' not in cfg:
        errors.append("Missing 'dcae_3d' config section")
        return errors

    dcae = cfg.dcae_3d
    n_stages = len(dcae.encoder_block_out_channels)
    n_down_blocks = n_stages - 1
    spatial_compression = 2 ** n_down_blocks
    depth_factors = dcae.depth_factors
    depth_compression = math.prod(depth_factors)

    if len(depth_factors) != n_down_blocks:
        errors.append(
            f"depth_factors ({len(depth_factors)}) must have {n_down_blocks} elements"
        )

    if volume.get('height', 0) % spatial_compression != 0:
        errors.append(
            f"volume.height ({volume.get('height')}) must be divisible by {spatial_compression}"
        )
    if volume.get('width', 0) % spatial_compression != 0:
        errors.append(
            f"volume.width ({volume.get('width')}) must be divisible by {spatial_compression}"
        )
    if volume.get('depth', 0) % depth_compression != 0:
        errors.append(
            f"volume.depth ({volume.get('depth')}) must be divisible by {depth_compression}"
        )

    return errors


# =============================================================================
# Trainer Registry
# =============================================================================

class TrainerConfig:
    """Configuration for a compression trainer type."""

    def __init__(
        self,
        trainer_class: type,
        validator: Callable[[DictConfig], list],
        config_section: str,
        display_name: str,
        spatial_dims: int,
    ):
        self.trainer_class = trainer_class
        self.validator = validator
        self.config_section = config_section
        self.display_name = display_name
        self.spatial_dims = spatial_dims


# Registry: config_section -> TrainerConfig
# Order matters for detection (check 3D sections before 2D)
TRAINER_REGISTRY: dict[str, TrainerConfig] = {
    # 3D trainers (check first)
    'vae_3d': TrainerConfig(
        trainer_class=VAETrainer,
        validator=validate_vae_3d_config,
        config_section='vae_3d',
        display_name='3D VAE',
        spatial_dims=3,
    ),
    'vqvae_3d': TrainerConfig(
        trainer_class=VQVAETrainer,
        validator=validate_vqvae_3d_config,
        config_section='vqvae_3d',
        display_name='3D VQ-VAE',
        spatial_dims=3,
    ),
    'dcae_3d': TrainerConfig(
        trainer_class=DCAETrainer,
        validator=validate_dcae_3d_config,
        config_section='dcae_3d',
        display_name='3D DC-AE',
        spatial_dims=3,
    ),
    # 2D trainers
    'vae': TrainerConfig(
        trainer_class=VAETrainer,
        validator=validate_vae_config,
        config_section='vae',
        display_name='VAE',
        spatial_dims=2,
    ),
    'vqvae': TrainerConfig(
        trainer_class=VQVAETrainer,
        validator=validate_vqvae_config,
        config_section='vqvae',
        display_name='VQ-VAE',
        spatial_dims=2,
    ),
    'dcae': TrainerConfig(
        trainer_class=DCAETrainer,
        validator=validate_dcae_config,
        config_section='dcae',
        display_name='DC-AE',
        spatial_dims=2,
    ),
}


def detect_trainer_type(cfg: DictConfig) -> str:
    """Detect trainer type from config sections.

    Checks for presence of config sections like 'vae', 'vae_3d', 'dcae', etc.
    Returns the first matching trainer type.
    """
    # Check 3D first (they have more specific sections)
    for section in ['vae_3d', 'vqvae_3d', 'dcae_3d', 'vae', 'vqvae', 'dcae']:
        if section in cfg:
            return section

    raise ValueError(
        "Could not detect trainer type. Config must have one of: "
        "vae, vae_3d, vqvae, vqvae_3d, dcae, dcae_3d"
    )


# =============================================================================
# Training Functions
# =============================================================================

def _train_2d(cfg: DictConfig, trainer_config: TrainerConfig) -> None:
    """2D compression training."""
    mode = cfg.mode.name
    use_multi_gpu = cfg.training.get('use_multi_gpu', False)
    is_multi_modality = mode == 'multi_modality'
    is_seg_mode = cfg.get(trainer_config.config_section, {}).get('seg_mode', False)

    # Override in_channels
    in_channels = override_vae_channels(cfg, mode)

    # Log resolved config
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Build extra info for header
    extra_info = _build_extra_info(trainer_config, cfg)

    # Log training header
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Training {trainer_config.display_name} for {mode} mode{extra_info}")
    logger.info(f"Channels: {in_channels} | Image size: {cfg.model.image_size}")
    logger.info(f"Batch size: {cfg.training.batch_size} | Epochs: {cfg.training.epochs}")
    logger.info(f"Multi-GPU: {use_multi_gpu} | EMA: {cfg.training.use_ema}")
    logger.info("=" * 60)
    logger.info("")

    # Create trainer
    trainer = trainer_config.trainer_class(cfg)
    logger.info(f"Validation: every epoch, figures at interval {trainer.figure_interval}")

    # Create dataloaders
    augment = cfg.training.get('augment', True)
    per_modality_val_loaders = {}

    if is_seg_mode or mode == 'seg_compression':
        # Segmentation mask compression
        seg_image_size = cfg.get('dcae', cfg.get('model', {})).get('image_size', 256)
        seg_batch_size = cfg.training.batch_size
        train_loader, train_dataset = create_seg_compression_loader(
            cfg=cfg, split='train',
            image_size=seg_image_size, batch_size=seg_batch_size, augment=augment,
        )
        val_result = create_seg_compression_loader(
            cfg=cfg, split='val',
            image_size=seg_image_size, batch_size=seg_batch_size,
        )
        val_loader = val_result[0] if val_result else None
    elif is_multi_modality:
        image_keys = get_image_keys(cfg)
        train_loader, train_dataset = create_multi_modality_dataloader(
            cfg=cfg,
            image_keys=image_keys,
            image_size=cfg.model.image_size,
            batch_size=cfg.training.batch_size,
            use_distributed=use_multi_gpu,
            rank=trainer.rank if use_multi_gpu else 0,
            world_size=trainer.world_size if use_multi_gpu else 1,
            augment=augment,
        )
        logger.info(f"Training on multi_modality mode (modalities: {image_keys})")

        val_result = create_multi_modality_validation_dataloader(
            cfg=cfg,
            image_keys=image_keys,
            image_size=cfg.model.image_size,
            batch_size=cfg.training.batch_size,
        )
        val_loader = val_result[0] if val_result else None

        # Per-modality validation loaders
        per_modality_val_loaders = create_per_modality_val_loaders(
            cfg=cfg,
            image_keys=image_keys,
            create_loader_fn=create_single_modality_validation_loader,
            image_size=cfg.model.image_size,
            batch_size=cfg.training.batch_size,
        )
    else:
        # Single/dual modality
        train_loader, train_dataset = create_vae_dataloader(
            cfg=cfg,
            modality=mode,
            use_distributed=use_multi_gpu,
            rank=trainer.rank if use_multi_gpu else 0,
            world_size=trainer.world_size if use_multi_gpu else 1,
            augment=augment,
        )
        val_result = create_vae_validation_dataloader(cfg=cfg, modality=mode)
        val_loader = val_result[0] if val_result else None

    logger.info(f"Training dataset: {len(train_dataset)} samples")
    if val_loader:
        logger.info(f"Validation batches: {len(val_loader)}")
    else:
        logger.info("No val/ directory found - using train samples for validation")

    # Setup model
    pretrained_checkpoint = cfg.get('pretrained_checkpoint', None)
    if pretrained_checkpoint:
        logger.info(f"Loading pretrained weights from: {pretrained_checkpoint}")
    trainer.setup_model(pretrained_checkpoint=pretrained_checkpoint)

    # Resume from checkpoint if specified
    start_epoch = 0
    resume_from = cfg.training.get('resume_from', None)
    if resume_from:
        start_epoch = trainer.load_checkpoint(resume_from)

    # Train
    trainer.train(
        train_loader,
        train_dataset,
        val_loader=val_loader,
        per_modality_val_loaders=per_modality_val_loaders if per_modality_val_loaders else None,
        start_epoch=start_epoch,
    )

    # On wall-time signal: skip test eval, let SLURM script handle resubmission
    if getattr(trainer, '_sigterm_received', False):
        logger.info("Wall time signal received — skipping test evaluation")
        trainer.close_writer()
        return

    # Test evaluation
    if is_seg_mode or mode == 'seg_compression':
        test_result = create_seg_compression_loader(
            cfg=cfg, split='test',
            image_size=seg_image_size, batch_size=seg_batch_size,
        )
    elif is_multi_modality:
        test_result = create_multi_modality_test_dataloader(
            cfg=cfg,
            image_keys=image_keys,
            image_size=cfg.model.image_size,
            batch_size=cfg.training.batch_size,
        )
    else:
        test_result = create_vae_test_dataloader(cfg=cfg, modality=mode)

    run_test_evaluation(trainer, test_result)

    trainer.close_writer()

    # Mark training complete for SLURM job chaining
    from pathlib import Path
    (Path(trainer.save_dir) / '.training_complete').touch()


def _train_3d(cfg: DictConfig, trainer_config: TrainerConfig) -> None:
    """3D compression training."""
    mode = cfg.mode.name
    use_multi_gpu = cfg.training.get('use_multi_gpu', False)
    is_multi_modality = mode == 'multi_modality'

    # Override in_channels
    in_channels = override_vae_channels(cfg, mode)

    # Log resolved config
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Build extra info
    extra_info = _build_extra_info_3d(trainer_config, cfg)

    # Log training header
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Training {trainer_config.display_name} for {mode} mode")
    logger.info(f"Channels: {in_channels}")
    logger.info(f"Volume: {cfg.volume.width}x{cfg.volume.height}x{cfg.volume.depth}")
    logger.info(extra_info)
    logger.info(f"Batch size: {cfg.training.batch_size} | Epochs: {cfg.training.epochs}")
    logger.info(f"Multi-GPU: {use_multi_gpu}")
    if is_multi_modality:
        image_keys = get_image_keys(cfg, is_3d=True)
        logger.info(f"Modalities: {image_keys}")
    logger.info("=" * 60)
    logger.info("")

    # Create trainer using .create_3d() factory
    trainer = trainer_config.trainer_class.create_3d(cfg)
    logger.info(f"Validation: every epoch, figures at interval {trainer.figure_interval}")

    # Create 3D dataloaders
    if is_multi_modality:
        train_loader, train_dataset = create_vae_3d_multi_modality_dataloader(
            cfg=cfg,
            use_distributed=use_multi_gpu,
            rank=trainer.rank if use_multi_gpu else 0,
            world_size=trainer.world_size if use_multi_gpu else 1,
        )
    else:
        train_loader, train_dataset = create_vae_3d_dataloader(
            cfg=cfg,
            modality=mode,
            use_distributed=use_multi_gpu,
            rank=trainer.rank if use_multi_gpu else 0,
            world_size=trainer.world_size if use_multi_gpu else 1,
        )
    logger.info(f"Training dataset: {len(train_dataset)} volumes")

    # Validation dataloader
    if is_multi_modality:
        val_result = create_vae_3d_multi_modality_validation_dataloader(cfg=cfg)
    else:
        val_result = create_vae_3d_validation_dataloader(cfg=cfg, modality=mode)

    val_loader = None
    if val_result is not None:
        val_loader, val_dataset = val_result
        logger.info(f"Validation dataset: {len(val_dataset)} volumes")
    else:
        logger.info("No val/ directory found - using train samples for validation")

    # Per-modality validation loaders
    per_modality_val_loaders = {}
    if is_multi_modality:
        image_keys = get_image_keys(cfg, is_3d=True)
        per_modality_val_loaders = create_per_modality_val_loaders_3d(
            cfg, image_keys, create_vae_3d_single_modality_validation_loader
        )

    # Setup model
    pretrained_checkpoint = cfg.get('pretrained_checkpoint', None)
    if pretrained_checkpoint:
        logger.info(f"Loading pretrained weights from: {pretrained_checkpoint}")
    trainer.setup_model(pretrained_checkpoint=pretrained_checkpoint)

    # Resume from checkpoint if specified
    start_epoch = 0
    resume_from = cfg.training.get('resume_from', None)
    if resume_from:
        start_epoch = trainer.load_checkpoint(resume_from)

    # Train
    trainer.train(
        train_loader,
        train_dataset,
        val_loader=val_loader,
        per_modality_val_loaders=per_modality_val_loaders if per_modality_val_loaders else None,
        start_epoch=start_epoch,
    )

    # On wall-time signal: skip test eval, let SLURM script handle resubmission
    if getattr(trainer, '_sigterm_received', False):
        logger.info("Wall time signal received — skipping test evaluation")
        trainer.close_writer()
        return

    # Test evaluation
    if is_multi_modality:
        test_result = create_vae_3d_multi_modality_test_dataloader(cfg=cfg)
    else:
        test_result = create_vae_3d_test_dataloader(cfg=cfg, modality=mode)

    run_test_evaluation(trainer, test_result, eval_method="evaluate_test_set")

    trainer.close_writer()

    # Mark training complete for SLURM job chaining
    from pathlib import Path
    (Path(trainer.save_dir) / '.training_complete').touch()


def _build_extra_info(trainer_config: TrainerConfig, cfg: DictConfig) -> str:
    """Build trainer-specific extra info for 2D logging header."""
    section = trainer_config.config_section
    if section == 'vqvae':
        disable_gan = cfg.vqvae.get('disable_gan', False)
        return f" (GAN: {'disabled' if disable_gan else 'enabled'})"
    elif section == 'dcae':
        compression = cfg.dcae.get('compression_ratio', 32)
        return f" ({compression}× compression)"
    return ""


def _build_extra_info_3d(trainer_config: TrainerConfig, cfg: DictConfig) -> str:
    """Build trainer-specific extra info for 3D logging header."""
    import math

    section = trainer_config.config_section
    if section == 'vae_3d':
        return f"Latent channels: {cfg.vae_3d.latent_channels}"
    elif section == 'vqvae_3d':
        n_downsamples = len(cfg.vqvae_3d.channels)
        latent_h = cfg.volume.height // (2 ** n_downsamples)
        latent_w = cfg.volume.width // (2 ** n_downsamples)
        latent_d = cfg.volume.depth // (2 ** n_downsamples)
        return (
            f"Latent: {latent_w}x{latent_h}x{latent_d} ({n_downsamples}x compression) | "
            f"Codebook: {cfg.vqvae_3d.num_embeddings} x {cfg.vqvae_3d.embedding_dim}"
        )
    elif section == 'dcae_3d':
        dcae = cfg.dcae_3d
        n_stages = len(dcae.encoder_block_out_channels)
        n_down_blocks = n_stages - 1
        spatial_comp = 2 ** n_down_blocks
        depth_comp = math.prod(dcae.depth_factors)
        latent_h = cfg.volume.height // spatial_comp
        latent_w = cfg.volume.width // spatial_comp
        latent_d = cfg.volume.depth // depth_comp
        return (
            f"Latent: {latent_w}x{latent_h}x{latent_d}x{dcae.latent_channels} | "
            f"Compression: {spatial_comp}x spatial, {depth_comp}x depth"
        )
    return ""


# =============================================================================
# Main Entry Point
# =============================================================================

@hydra.main(version_base=None, config_path="../../../configs", config_name="vae")
def main(cfg: DictConfig) -> None:
    """Unified compression model training entry point.

    The trainer type (VAE, VQ-VAE, DC-AE) and spatial dimensions (2D/3D)
    are automatically detected from the config.

    Use --config-name to select different configs:
        python -m medgen.scripts.train_compression --config-name=vae
        python -m medgen.scripts.train_compression --config-name=vae_3d
        python -m medgen.scripts.train_compression --config-name=dcae
    """
    # Detect trainer type from config
    trainer_type = detect_trainer_type(cfg)
    trainer_config = TRAINER_REGISTRY[trainer_type]

    logger.info(f"Detected trainer type: {trainer_type} ({trainer_config.display_name})")

    # Validate configuration
    run_validation(cfg, [
        validate_common_config,
        validate_model_config,
        trainer_config.validator,
    ])

    # Route to 2D or 3D training
    if trainer_config.spatial_dims == 3:
        _train_3d(cfg, trainer_config)
    else:
        _train_2d(cfg, trainer_config)


if __name__ == "__main__":
    main()
