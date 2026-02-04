"""Factory functions for creating 3D volume dataloaders.

This module consolidates all 3D dataloader creation functions into a unified
factory pattern, reducing redundancy from the original 16+ functions to a
clean factory API with backward-compatible aliases.

Primary API:
    create_3d_dataloader(mode, data_dir, split, batch_size, ...)

Backward-compatible aliases (re-exported from volume_3d.py):
    create_vae_3d_dataloader()
    create_vae_3d_validation_dataloader()
    etc.
"""
import logging
import os
from dataclasses import dataclass

from torch.utils.data import DataLoader, Dataset

from .common import DataLoaderConfig, DistributedArgs, create_dataloader
from .volume_3d_datasets import (
    DualVolume3DDataset,
    MultiModality3DDataset,
    SingleModality3DDatasetWithSeg,
    SingleModality3DDatasetWithSegDropout,
    Volume3DDataset,
    build_3d_augmentation,
)

logger = logging.getLogger(__name__)


@dataclass
class VolumeConfig:
    """Configuration extracted from Hydra config for 3D volume loading.

    Supports separate train/val resolutions for efficient training:
    - Train at lower resolution (e.g., 128x128) for speed
    - Validate at full resolution (e.g., 256x256) for accurate metrics
    """
    height: int
    width: int
    pad_depth_to: int
    pad_mode: str
    slice_step: int
    batch_size: int
    load_seg: bool
    image_keys: list
    # Optional training resolution (defaults to height/width)
    _train_height: int | None = None
    _train_width: int | None = None
    # DataLoader config (stored as DataLoaderConfig)
    _loader_config: DataLoaderConfig = None

    @classmethod
    def from_cfg(cls, cfg) -> 'VolumeConfig':
        """Extract volume configuration from Hydra config object."""
        logging_cfg = cfg.training.get('logging', {})
        loader_config = DataLoaderConfig.from_cfg(cfg)

        # Get optional train resolution (null in config becomes None)
        train_height = cfg.volume.get('train_height', None)
        train_width = cfg.volume.get('train_width', None)

        return cls(
            height=cfg.volume.height,
            width=cfg.volume.width,
            pad_depth_to=cfg.volume.pad_depth_to,
            pad_mode=cfg.volume.get('pad_mode', 'replicate'),
            slice_step=cfg.volume.get('slice_step', 1),
            batch_size=cfg.training.batch_size,
            load_seg=logging_cfg.get('regional_losses', False),
            image_keys=cfg.mode.get('image_keys', ['bravo', 'flair', 't1_pre', 't1_gd']),
            _train_height=train_height,
            _train_width=train_width,
            _loader_config=loader_config,
        )

    @property
    def train_height(self) -> int:
        """Height for training (may be lower resolution)."""
        return self._train_height if self._train_height is not None else self.height

    @property
    def train_width(self) -> int:
        """Width for training (may be lower resolution)."""
        return self._train_width if self._train_width is not None else self.width

    @property
    def uses_reduced_train_resolution(self) -> bool:
        """Whether training uses lower resolution than validation."""
        return self.train_height != self.height or self.train_width != self.width

    @property
    def loader_config(self) -> DataLoaderConfig:
        """Get DataLoaderConfig for create_dataloader()."""
        return self._loader_config

    # Legacy properties for backwards compatibility
    @property
    def num_workers(self) -> int:
        return self._loader_config.num_workers

    @property
    def prefetch_factor(self) -> int | None:
        return self._loader_config.prefetch_factor

    @property
    def pin_memory(self) -> bool:
        return self._loader_config.pin_memory

    @property
    def persistent_workers(self) -> bool:
        return self._loader_config.persistent_workers


def _create_single_dual_dataset(
    data_dir: str,
    modality: str,
    vcfg: VolumeConfig,
    height: int | None = None,
    width: int | None = None,
) -> Dataset:
    """Create Volume3DDataset or DualVolume3DDataset based on modality.

    Args:
        data_dir: Path to data directory (train/val/test_new).
        modality: 'dual' for DualVolume3DDataset, else Volume3DDataset.
        vcfg: Volume configuration.
        height: Override height (defaults to vcfg.height).
        width: Override width (defaults to vcfg.width).

    Returns:
        Appropriate dataset instance.
    """
    h = height if height is not None else vcfg.height
    w = width if width is not None else vcfg.width

    if modality == 'dual':
        return DualVolume3DDataset(
            data_dir=data_dir,
            height=h,
            width=w,
            pad_depth_to=vcfg.pad_depth_to,
            pad_mode=vcfg.pad_mode,
            slice_step=vcfg.slice_step,
            load_seg=vcfg.load_seg,
        )
    return Volume3DDataset(
        data_dir=data_dir,
        modality=modality,
        height=h,
        width=w,
        pad_depth_to=vcfg.pad_depth_to,
        pad_mode=vcfg.pad_mode,
        slice_step=vcfg.slice_step,
        load_seg=vcfg.load_seg,
    )


def _create_multi_modality_dataset(
    data_dir: str,
    vcfg: VolumeConfig,
    height: int | None = None,
    width: int | None = None,
    augment: bool = False,
    seg_mode: bool = False,
) -> MultiModality3DDataset:
    """Create MultiModality3DDataset with config.

    Args:
        data_dir: Path to data directory.
        vcfg: Volume configuration.
        height: Override height (defaults to vcfg.height).
        width: Override width (defaults to vcfg.width).
        augment: Whether to apply 3D augmentation.
        seg_mode: Whether this is for segmentation masks (binary).
    """
    h = height if height is not None else vcfg.height
    w = width if width is not None else vcfg.width

    aug = build_3d_augmentation(seg_mode=seg_mode) if augment else None

    return MultiModality3DDataset(
        data_dir=data_dir,
        image_keys=vcfg.image_keys,
        height=h,
        width=w,
        pad_depth_to=vcfg.pad_depth_to,
        pad_mode=vcfg.pad_mode,
        slice_step=vcfg.slice_step,
        load_seg=vcfg.load_seg,
        augmentation=aug,
    )


def _create_loader(
    dataset: Dataset,
    vcfg: VolumeConfig,
    shuffle: bool = True,
    drop_last: bool = False,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Create DataLoader with standard settings using shared create_dataloader().

    Args:
        dataset: Dataset to wrap.
        vcfg: Volume configuration containing batch_size and DataLoader settings.
        shuffle: Whether to shuffle (ignored if distributed).
        drop_last: Whether to drop last incomplete batch.
        use_distributed: Use DistributedSampler.
        rank: Process rank for distributed.
        world_size: Total processes for distributed.

    Returns:
        Configured DataLoader.
    """
    distributed_args = DistributedArgs(
        use_distributed=use_distributed,
        rank=rank,
        world_size=world_size,
    )

    return create_dataloader(
        dataset=dataset,
        batch_size=vcfg.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        distributed_args=distributed_args,
        loader_config=vcfg.loader_config,
        scale_batch_for_distributed=False,  # 3D volumes: batch_size=1-2, don't divide
    )


# =============================================================================
# VAE 3D Dataloaders (Single/Dual Modality)
# =============================================================================

def create_vae_3d_dataloader(
    cfg,
    modality: str,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, Dataset]:
    """Create 3D VAE training dataloader.

    Uses train_height/train_width if configured, allowing training at lower
    resolution than validation for faster iteration.

    Args:
        cfg: Hydra configuration object.
        modality: Modality name or 'dual' for dual-channel.
        use_distributed: Whether to use distributed sampling.
        rank: Process rank for distributed training.
        world_size: Total number of processes.

    Returns:
        Tuple of (DataLoader, Dataset).
    """
    vcfg = VolumeConfig.from_cfg(cfg)
    data_dir = os.path.join(cfg.paths.data_dir, 'train')

    # Use training resolution (may be lower than validation resolution)
    dataset = _create_single_dual_dataset(
        data_dir, modality, vcfg,
        height=vcfg.train_height,
        width=vcfg.train_width,
    )

    if vcfg.uses_reduced_train_resolution:
        logger.info(
            f"Training at {vcfg.train_height}x{vcfg.train_width}, "
            f"validation at {vcfg.height}x{vcfg.width}"
        )

    loader = _create_loader(
        dataset, vcfg, shuffle=True, drop_last=True,
        use_distributed=use_distributed, rank=rank, world_size=world_size
    )
    return loader, dataset


def create_vae_3d_validation_dataloader(
    cfg,
    modality: str,
) -> tuple[DataLoader, Dataset] | None:
    """Create 3D VAE validation dataloader.

    Args:
        cfg: Hydra configuration object.
        modality: Modality name or 'dual' for dual-channel.

    Returns:
        Tuple of (DataLoader, Dataset) or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, 'val')
    if not os.path.exists(val_dir):
        return None

    vcfg = VolumeConfig.from_cfg(cfg)
    dataset = _create_single_dual_dataset(val_dir, modality, vcfg)
    loader = _create_loader(dataset, vcfg, shuffle=False)
    return loader, dataset


def create_vae_3d_test_dataloader(
    cfg,
    modality: str,
) -> tuple[DataLoader, Dataset] | None:
    """Create 3D VAE test dataloader.

    Args:
        cfg: Hydra configuration object.
        modality: Modality name or 'dual' for dual-channel.

    Returns:
        Tuple of (DataLoader, Dataset) or None if test_new/ doesn't exist.
    """
    test_dir = os.path.join(cfg.paths.data_dir, 'test_new')
    if not os.path.exists(test_dir):
        return None

    vcfg = VolumeConfig.from_cfg(cfg)
    dataset = _create_single_dual_dataset(test_dir, modality, vcfg)
    loader = _create_loader(dataset, vcfg, shuffle=False)

    return loader, dataset


# =============================================================================
# VAE 3D Multi-Modality Dataloaders
# =============================================================================

def create_vae_3d_multi_modality_dataloader(
    cfg,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, Dataset]:
    """Create 3D VAE multi-modality training dataloader.

    Uses train_height/train_width if configured, allowing training at lower
    resolution than validation for faster iteration.

    Args:
        cfg: Hydra configuration object.
        use_distributed: Whether to use distributed sampling.
        rank: Process rank for distributed training.
        world_size: Total number of processes.

    Returns:
        Tuple of (DataLoader, Dataset).
    """
    vcfg = VolumeConfig.from_cfg(cfg)
    data_dir = os.path.join(cfg.paths.data_dir, 'train')

    # Check if augmentation is enabled
    augment = getattr(cfg.training, 'augment', False)
    # Check if this is seg mode (for binary mask handling)
    seg_mode = getattr(cfg.mode, 'name', '') == 'seg'

    # Use training resolution (may be lower than validation resolution)
    dataset = _create_multi_modality_dataset(
        data_dir, vcfg,
        height=vcfg.train_height,
        width=vcfg.train_width,
        augment=augment,
        seg_mode=seg_mode,
    )

    if vcfg.uses_reduced_train_resolution:
        logger.info(
            f"Training at {vcfg.train_height}x{vcfg.train_width}, "
            f"validation at {vcfg.height}x{vcfg.width}"
        )

    loader = _create_loader(
        dataset, vcfg, shuffle=True, drop_last=True,
        use_distributed=use_distributed, rank=rank, world_size=world_size
    )
    return loader, dataset


def create_vae_3d_multi_modality_validation_dataloader(
    cfg,
) -> tuple[DataLoader, Dataset] | None:
    """Create 3D VAE multi-modality validation dataloader.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Tuple of (DataLoader, Dataset) or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, 'val')
    if not os.path.exists(val_dir):
        return None

    vcfg = VolumeConfig.from_cfg(cfg)
    dataset = _create_multi_modality_dataset(val_dir, vcfg)
    loader = _create_loader(dataset, vcfg, shuffle=False)
    return loader, dataset


def create_vae_3d_multi_modality_test_dataloader(
    cfg,
) -> tuple[DataLoader, Dataset] | None:
    """Create 3D VAE multi-modality test dataloader.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Tuple of (DataLoader, Dataset) or None if test_new/ doesn't exist.
    """
    test_dir = os.path.join(cfg.paths.data_dir, 'test_new')
    if not os.path.exists(test_dir):
        return None

    vcfg = VolumeConfig.from_cfg(cfg)
    dataset = _create_multi_modality_dataset(test_dir, vcfg)
    loader = _create_loader(dataset, vcfg, shuffle=False)
    return loader, dataset


def create_vae_3d_single_modality_validation_loader(
    cfg,
    modality: str,
) -> DataLoader | None:
    """Create 3D validation loader for a single modality (for per-modality metrics).

    Includes seg masks paired with each volume for regional metrics tracking.

    Args:
        cfg: Hydra configuration object.
        modality: Single modality to load (e.g., 'bravo', 't1_pre', 't1_gd').

    Returns:
        DataLoader for single modality or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, 'val')
    if not os.path.exists(val_dir):
        return None

    # Check if modality exists in any patient directory
    has_modality = False
    for patient in os.listdir(val_dir):
        patient_dir = os.path.join(val_dir, patient)
        if os.path.isdir(patient_dir):
            modality_path = os.path.join(patient_dir, f"{modality}.nii.gz")
            if os.path.exists(modality_path):
                has_modality = True
                break

    if not has_modality:
        logger.warning(f"Modality {modality} not found in {val_dir}")
        return None

    vcfg = VolumeConfig.from_cfg(cfg)

    try:
        dataset = SingleModality3DDatasetWithSeg(
            data_dir=val_dir,
            modality=modality,
            height=vcfg.height,
            width=vcfg.width,
            pad_depth_to=vcfg.pad_depth_to,
            pad_mode=vcfg.pad_mode,
            slice_step=vcfg.slice_step,
        )
    except ValueError as e:
        logger.warning(f"Could not create dataset for {modality}: {e}")
        return None

    loader = _create_loader(dataset, vcfg, shuffle=False)

    return loader


# =============================================================================
# 3D Diffusion Dataloaders (seg mode, bravo mode with conditioning)
# =============================================================================

def create_segmentation_dataloader(
    cfg,
    vol_cfg: VolumeConfig,
    augment: bool = False,
) -> tuple[DataLoader, Dataset]:
    """Create 3D dataloader for unconditional segmentation training.

    Loads seg masks only (no conditioning).

    Args:
        cfg: Hydra configuration object.
        vol_cfg: Volume configuration.
        augment: Whether to apply augmentation.

    Returns:
        Tuple of (DataLoader, Dataset).
    """
    data_dir = os.path.join(cfg.paths.data_dir, 'train')

    aug = build_3d_augmentation(seg_mode=True) if augment else None

    dataset = Volume3DDataset(
        data_dir=data_dir,
        modality='seg',
        height=vol_cfg.train_height,
        width=vol_cfg.train_width,
        pad_depth_to=vol_cfg.pad_depth_to,
        pad_mode=vol_cfg.pad_mode,
        slice_step=vol_cfg.slice_step,
        load_seg=False,  # No separate seg needed - we ARE loading seg as image
        augmentation=aug,
    )

    logger.info(f"Created 3D seg dataset: {len(dataset)} volumes")

    loader = _create_loader(dataset, vol_cfg, shuffle=True, drop_last=True)
    return loader, dataset


def create_segmentation_validation_dataloader(
    cfg,
    vol_cfg: VolumeConfig,
) -> tuple[DataLoader, Dataset] | None:
    """Create 3D validation dataloader for unconditional segmentation.

    Args:
        cfg: Hydra configuration object.
        vol_cfg: Volume configuration.

    Returns:
        Tuple of (DataLoader, Dataset) or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, 'val')
    if not os.path.exists(val_dir):
        return None

    dataset = Volume3DDataset(
        data_dir=val_dir,
        modality='seg',
        height=vol_cfg.height,
        width=vol_cfg.width,
        pad_depth_to=vol_cfg.pad_depth_to,
        pad_mode=vol_cfg.pad_mode,
        slice_step=vol_cfg.slice_step,
        load_seg=False,
    )

    loader = _create_loader(dataset, vol_cfg, shuffle=False)
    return loader, dataset


def create_single_modality_dataloader_with_seg(
    cfg,
    vol_cfg: VolumeConfig,
    modality: str = 'bravo',
    augment: bool = False,
) -> tuple[DataLoader, Dataset]:
    """Create 3D dataloader for single modality conditioned on seg mask.

    Used for 3D bravo mode where bravo generation is conditioned on seg mask.
    Includes CFG dropout to randomly zero seg mask during training.

    Args:
        cfg: Hydra configuration object.
        vol_cfg: Volume configuration.
        modality: Modality to load (default: 'bravo').
        augment: Whether to apply augmentation.

    Returns:
        Tuple of (DataLoader, Dataset).
    """
    data_dir = os.path.join(cfg.paths.data_dir, 'train')

    # CFG dropout for 3D - get from config (default 15%)
    cfg_dropout_prob = float(cfg.mode.get('cfg_dropout_prob', 0.15))

    # include_seg=True ensures both image and seg are augmented consistently
    aug = build_3d_augmentation(seg_mode=False, include_seg=True) if augment else None

    dataset = SingleModality3DDatasetWithSegDropout(
        data_dir=data_dir,
        modality=modality,
        height=vol_cfg.train_height,
        width=vol_cfg.train_width,
        pad_depth_to=vol_cfg.pad_depth_to,
        pad_mode=vol_cfg.pad_mode,
        slice_step=vol_cfg.slice_step,
        cfg_dropout_prob=cfg_dropout_prob,
        augmentation=aug,
    )

    logger.info(f"Created 3D {modality} dataset with seg conditioning: {len(dataset)} volumes, "
                f"cfg_dropout={cfg_dropout_prob}")

    loader = _create_loader(dataset, vol_cfg, shuffle=True, drop_last=True)
    return loader, dataset


def create_single_modality_validation_dataloader_with_seg(
    cfg,
    vol_cfg: VolumeConfig,
    modality: str = 'bravo',
) -> tuple[DataLoader, Dataset] | None:
    """Create 3D validation dataloader for single modality with seg conditioning.

    No CFG dropout during validation.

    Args:
        cfg: Hydra configuration object.
        vol_cfg: Volume configuration.
        modality: Modality to load (default: 'bravo').

    Returns:
        Tuple of (DataLoader, Dataset) or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, 'val')
    if not os.path.exists(val_dir):
        return None

    # No CFG dropout during validation
    dataset = SingleModality3DDatasetWithSegDropout(
        data_dir=val_dir,
        modality=modality,
        height=vol_cfg.height,
        width=vol_cfg.width,
        pad_depth_to=vol_cfg.pad_depth_to,
        pad_mode=vol_cfg.pad_mode,
        slice_step=vol_cfg.slice_step,
        cfg_dropout_prob=0.0,  # No dropout during validation
    )

    loader = _create_loader(dataset, vol_cfg, shuffle=False)
    return loader, dataset


def create_segmentation_conditioned_dataloader(
    cfg,
    vol_cfg: VolumeConfig,
    size_bin_config: dict,
    augment: bool = False,
) -> tuple[DataLoader, Dataset]:
    """Create 3D dataloader for size-conditioned segmentation training.

    Wrapper that routes to seg.py's create_seg_dataloader.

    Args:
        cfg: Hydra configuration object.
        vol_cfg: Volume configuration.
        size_bin_config: Size bin configuration dict.
        augment: Whether to apply augmentation.

    Returns:
        Tuple of (DataLoader, Dataset).
    """
    from .seg import create_seg_dataloader
    return create_seg_dataloader(cfg)


def create_segmentation_conditioned_validation_dataloader(
    cfg,
    vol_cfg: VolumeConfig,
    size_bin_config: dict,
) -> tuple[DataLoader, Dataset] | None:
    """Create 3D validation dataloader for size-conditioned segmentation.

    Wrapper that routes to seg.py's create_seg_validation_dataloader.

    Args:
        cfg: Hydra configuration object.
        vol_cfg: Volume configuration.
        size_bin_config: Size bin configuration dict.

    Returns:
        Tuple of (DataLoader, Dataset) or None if val/ doesn't exist.
    """
    from .seg import create_seg_validation_dataloader
    return create_seg_validation_dataloader(cfg)
