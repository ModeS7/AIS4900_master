"""Unified dataloader factory for 2D and 3D diffusion and compression training.

This module provides a single entry point for creating dataloaders that work
consistently across 2D/3D training for both diffusion and compression (VAE/VQVAE/DC-AE).
All loaders return dict-format batches.

Usage:
    >>> from medgen.data.loaders.unified import create_dataloader
    >>>
    >>> # Diffusion 2D
    >>> loader, _ = create_dataloader(
    ...     cfg=cfg, task='diffusion', mode='bravo', spatial_dims=2
    ... )
    >>>
    >>> # Compression 2D
    >>> loader, _ = create_dataloader(
    ...     cfg=cfg, task='compression', mode='bravo', spatial_dims=2
    ... )
    >>>
    >>> # 3D loader (same interface)
    >>> loader, _ = create_dataloader(
    ...     cfg=cfg, task='diffusion', mode='bravo', spatial_dims=3
    ... )
    >>>
    >>> # All loaders return dict format:
    >>> batch = next(iter(loader))
    >>> batch['image'].shape  # [B, C, H, W] or [B, C, D, H, W]
    >>> batch.get('seg')      # [B, 1, H, W] or None

Batch Format:
    All loaders return dict with keys:
    - 'image': Tensor [B, C, H, W] or [B, C, D, H, W]
    - 'seg': Tensor | None - conditioning (diffusion) or metrics (compression)
    - 'mode_id': Tensor | None - multi-modality mode ID
    - 'size_bins': Tensor | None - seg_conditioned only

Diffusion Modes:
    - 'seg': Unconditional segmentation mask generation
    - 'bravo': BRAVO image generation conditioned on seg mask
    - 'dual': T1_pre + T1_gd generation conditioned on seg mask
    - 'multi': Multi-modality (mode embedding for different MR sequences)
    - 'seg_conditioned': Seg generation conditioned on tumor size bins

Compression Modes:
    - 'seg_compression': DC-AE seg mask compression
    - 'bravo', 't1_pre', 't1_gd', 'flair': Single modality compression
    - 'dual': Dual modality (t1_pre + t1_gd) compression
    - 'multi_modality': Multi-modality pooled compression
"""

import logging
from typing import Any, Dict, Literal, Optional, Tuple

from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from .base import dict_collate_fn, DictDatasetWrapper

logger = logging.getLogger(__name__)

Task = Literal['diffusion', 'compression']
Mode = Literal['seg', 'bravo', 'dual', 'multi', 'seg_conditioned']
Split = Literal['train', 'val', 'test']


def create_dataloader(
    cfg: DictConfig,
    task: Task,
    mode: str,
    spatial_dims: int = 2,
    split: Split = 'train',
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: Optional[bool] = None,
    batch_size: Optional[int] = None,
) -> Tuple[DataLoader, Dataset]:
    """Unified dataloader factory for diffusion and compression.

    This is the main entry point for creating dataloaders. It routes to the
    appropriate loader based on task type (diffusion vs compression) and
    spatial dimensions (2D vs 3D).

    Args:
        cfg: Hydra configuration object.
        task: Training task ('diffusion' or 'compression').
        mode: Training mode (e.g., 'bravo', 'dual', 'multi_modality').
        spatial_dims: 2 for 2D images, 3 for 3D volumes.
        split: Data split ('train', 'val', 'test').
        use_distributed: Enable distributed training sampler.
        rank: Process rank for distributed training.
        world_size: Total number of processes.
        augment: Override augmentation setting (None = use config).
        batch_size: Override batch size (None = use config).

    Returns:
        Tuple of (DataLoader, Dataset).

    Raises:
        ValueError: If task or spatial_dims is invalid.

    Example:
        >>> # Diffusion training
        >>> loader, dataset = create_dataloader(
        ...     cfg, task='diffusion', mode='bravo', spatial_dims=2
        ... )
        >>>
        >>> # Compression training
        >>> loader, dataset = create_dataloader(
        ...     cfg, task='compression', mode='multi_modality', spatial_dims=2
        ... )
    """
    if spatial_dims not in (2, 3):
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

    if task == 'diffusion':
        return create_diffusion_dataloader(
            cfg=cfg,
            mode=mode,  # type: ignore
            spatial_dims=spatial_dims,
            split=split,
            use_distributed=use_distributed,
            rank=rank,
            world_size=world_size,
            augment=augment,
            batch_size=batch_size,
        )
    elif task == 'compression':
        return _create_compression_dataloader(
            cfg=cfg,
            mode=mode,
            spatial_dims=spatial_dims,
            split=split,
            use_distributed=use_distributed,
            rank=rank,
            world_size=world_size,
            augment=augment,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unknown task: {task}. Expected 'diffusion' or 'compression'.")


def create_diffusion_dataloader(
    cfg: DictConfig,
    mode: Mode,
    spatial_dims: int = 2,
    split: Split = 'train',
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: Optional[bool] = None,
    batch_size: Optional[int] = None,
) -> Tuple[DataLoader, Dataset]:
    """Create unified dataloader for diffusion training.

    This factory function creates dataloaders that return consistent dict-format
    batches for both 2D and 3D training.

    Args:
        cfg: Hydra configuration object.
        mode: Training mode ('seg', 'bravo', 'dual', 'multi', 'seg_conditioned').
        spatial_dims: 2 for 2D images, 3 for 3D volumes.
        split: Data split ('train', 'val', 'test').
        use_distributed: Enable distributed training sampler.
        rank: Process rank for distributed training.
        world_size: Total number of processes.
        augment: Override augmentation setting (None = use config).
        batch_size: Override batch size (None = use config).

    Returns:
        Tuple of (DataLoader, Dataset).

    Raises:
        ValueError: If mode is not supported or spatial_dims is invalid.

    Example:
        >>> loader, dataset = create_diffusion_dataloader(
        ...     cfg, mode='bravo', spatial_dims=2, split='train'
        ... )
        >>> for batch in loader:
        ...     images = batch['image']  # [B, 1, H, W]
        ...     seg = batch.get('seg')   # [B, 1, H, W] or None
    """
    if spatial_dims not in (2, 3):
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

    if mode not in ('seg', 'bravo', 'dual', 'multi', 'seg_conditioned'):
        raise ValueError(f"Unsupported mode: {mode}")

    # Determine augmentation setting
    if augment is None:
        augment = cfg.training.get('augment', False) if split == 'train' else False

    # Route to appropriate loader
    if spatial_dims == 2:
        return _create_2d_loader(cfg, mode, split, use_distributed, rank, world_size, augment, batch_size)
    else:
        return _create_3d_loader(cfg, mode, split, use_distributed, rank, world_size, augment, batch_size)


def _create_2d_loader(
    cfg: DictConfig,
    mode: str,
    split: str,
    use_distributed: bool,
    rank: int,
    world_size: int,
    augment: bool,
    batch_size: Optional[int],
) -> Tuple[DataLoader, Dataset]:
    """Create 2D dataloader based on mode.

    Wraps existing loaders to return dict format batches.
    """
    from medgen.data.loaders.common import DataLoaderConfig

    augment_type = cfg.training.get('augment_type', 'diffusion')

    # Get the underlying loader and dataset
    loader, raw_dataset = _get_raw_2d_loader(
        cfg, mode, split, use_distributed, rank, world_size, augment, augment_type, batch_size
    )

    # Wrap dataset to return dict format
    wrapped_dataset = DictDatasetWrapper(raw_dataset, output_format=mode, spatial_dims=2)

    # Create new DataLoader with dict_collate_fn
    dl_cfg = DataLoaderConfig.from_cfg(cfg)
    effective_batch_size = batch_size if batch_size else loader.batch_size

    # Preserve distributed sampler if present
    if hasattr(loader, 'sampler') and loader.sampler is not None:
        sampler = loader.sampler
        shuffle = False
    else:
        sampler = None
        shuffle = (split == 'train')

    new_loader = DataLoader(
        wrapped_dataset,
        batch_size=effective_batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        collate_fn=dict_collate_fn,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor if dl_cfg.num_workers > 0 else None,
        persistent_workers=dl_cfg.persistent_workers if dl_cfg.num_workers > 0 else False,
    )

    return new_loader, wrapped_dataset


def _get_raw_2d_loader(
    cfg: DictConfig,
    mode: str,
    split: str,
    use_distributed: bool,
    rank: int,
    world_size: int,
    augment: bool,
    augment_type: str,
    batch_size: Optional[int],
) -> Tuple[DataLoader, Dataset]:
    """Get raw 2D loader (may return tuple/tensor format)."""
    from medgen.data.loaders import single, dual, multi_diffusion, seg_conditioned

    if mode == 'seg':
        if split == 'train':
            return single.create_dataloader(
                cfg, 'seg', use_distributed, rank, world_size, augment, augment_type
            )
        elif split == 'val':
            result = single.create_validation_dataloader(cfg, 'seg', batch_size, world_size)
            if result is None:
                raise ValueError("No validation data found")
            return result
        else:  # test
            result = single.create_test_dataloader(cfg, 'seg', batch_size)
            if result is None:
                raise ValueError("No test data found")
            return result

    elif mode == 'bravo':
        if split == 'train':
            return single.create_dataloader(
                cfg, 'bravo', use_distributed, rank, world_size, augment, augment_type
            )
        elif split == 'val':
            result = single.create_validation_dataloader(cfg, 'bravo', batch_size, world_size)
            if result is None:
                raise ValueError("No validation data found")
            return result
        else:  # test
            result = single.create_test_dataloader(cfg, 'bravo', batch_size)
            if result is None:
                raise ValueError("No test data found")
            return result

    elif mode == 'dual':
        image_keys = cfg.mode.get('image_keys', ['t1_pre', 't1_gd'])
        conditioning = cfg.mode.get('conditioning', 'seg')

        if split == 'train':
            return dual.create_dual_image_dataloader(
                cfg, image_keys, conditioning, use_distributed, rank, world_size, augment, augment_type
            )
        elif split == 'val':
            result = dual.create_dual_image_validation_dataloader(cfg, image_keys, conditioning, batch_size, world_size)
            if result is None:
                raise ValueError("No validation data found")
            return result
        else:  # test
            result = dual.create_dual_image_test_dataloader(cfg, image_keys, conditioning, batch_size)
            if result is None:
                raise ValueError("No test data found")
            return result

    elif mode == 'multi':
        if split == 'train':
            return multi_diffusion.create_multi_modality_dataloader(
                cfg, use_distributed, rank, world_size, augment
            )
        else:
            # Multi-modality validation/test not yet supported
            raise ValueError(f"Multi-modality {split} loader not yet supported")

    elif mode == 'seg_conditioned':
        size_bin_config = dict(cfg.mode.get('size_bins', {}))

        if split == 'train':
            return seg_conditioned.create_seg_conditioned_dataloader(
                cfg, size_bin_config, use_distributed, rank, world_size, augment
            )
        elif split == 'val':
            result = seg_conditioned.create_seg_conditioned_validation_dataloader(cfg, size_bin_config, batch_size, world_size)
            if result is None:
                raise ValueError("No validation data found")
            return result
        else:
            raise ValueError("seg_conditioned test loader not yet supported")

    raise ValueError(f"Unknown mode: {mode}")


def _create_3d_loader(
    cfg: DictConfig,
    mode: str,
    split: str,
    use_distributed: bool,
    rank: int,
    world_size: int,
    augment: bool,
    batch_size: Optional[int],
) -> Tuple[DataLoader, Dataset]:
    """Create 3D dataloader based on mode.

    Note: 3D loaders use different implementations from 2D due to:
    - Volume padding and depth handling
    - Memory constraints (smaller batch sizes)
    - Different augmentation strategies
    """
    from medgen.data.loaders.volume_3d import VolumeConfig

    # Get volume config
    vol_cfg = VolumeConfig.from_cfg(cfg)

    # 3D loaders are in volume_3d.py - they work for both diffusion and compression
    # since loaders just load data; trainers decide how to use it (concat vs separate)
    if mode == 'seg':
        return _create_3d_seg_loader(cfg, vol_cfg, split, augment, batch_size)
    elif mode == 'bravo':
        return _create_3d_bravo_loader(cfg, vol_cfg, split, augment, batch_size)
    elif mode == 'dual':
        raise ValueError("3D dual mode not yet supported")
    elif mode == 'multi':
        raise ValueError("3D multi mode not yet supported")
    elif mode == 'seg_conditioned':
        return _create_3d_seg_conditioned_loader(cfg, vol_cfg, split, augment, batch_size)

    raise ValueError(f"Unknown 3D mode: {mode}")


def _create_3d_seg_loader(
    cfg: DictConfig,
    vol_cfg: Any,
    split: str,
    augment: bool,
    batch_size: Optional[int],
) -> Tuple[DataLoader, Dataset]:
    """Create 3D segmentation mode loader."""
    from medgen.data.loaders.volume_3d import (
        create_segmentation_dataloader,
        create_segmentation_validation_dataloader,
    )

    if split == 'train':
        return create_segmentation_dataloader(cfg, vol_cfg, augment=augment)
    elif split == 'val':
        result = create_segmentation_validation_dataloader(cfg, vol_cfg)
        if result is None:
            raise ValueError("No 3D validation data found")
        return result
    else:
        raise ValueError("3D seg test loader not yet supported")


def _create_3d_bravo_loader(
    cfg: DictConfig,
    vol_cfg: Any,
    split: str,
    augment: bool,
    batch_size: Optional[int],
) -> Tuple[DataLoader, Dataset]:
    """Create 3D BRAVO mode loader (BRAVO conditioned on seg)."""
    from medgen.data.loaders.volume_3d import (
        create_single_modality_dataloader_with_seg,
        create_single_modality_validation_dataloader_with_seg,
    )

    if split == 'train':
        return create_single_modality_dataloader_with_seg(
            cfg, vol_cfg, modality='bravo', augment=augment
        )
    elif split == 'val':
        result = create_single_modality_validation_dataloader_with_seg(
            cfg, vol_cfg, modality='bravo'
        )
        if result is None:
            raise ValueError("No 3D validation data found")
        return result
    else:
        raise ValueError("3D bravo test loader not yet supported")


def _create_3d_seg_conditioned_loader(
    cfg: DictConfig,
    vol_cfg: Any,
    split: str,
    augment: bool,
    batch_size: Optional[int],
) -> Tuple[DataLoader, Dataset]:
    """Create 3D seg_conditioned mode loader."""
    from medgen.data.loaders.volume_3d import (
        create_segmentation_conditioned_dataloader,
        create_segmentation_conditioned_validation_dataloader,
    )

    size_bin_config = dict(cfg.mode.get('size_bins', {}))

    if split == 'train':
        return create_segmentation_conditioned_dataloader(
            cfg, vol_cfg, size_bin_config, augment=augment
        )
    elif split == 'val':
        result = create_segmentation_conditioned_validation_dataloader(
            cfg, vol_cfg, size_bin_config
        )
        if result is None:
            raise ValueError("No 3D validation data found")
        return result
    else:
        raise ValueError("3D seg_conditioned test loader not yet supported")


def get_dataloader_info(loader: DataLoader) -> Dict[str, Any]:
    """Get information about a dataloader for logging.

    Args:
        loader: DataLoader to inspect.

    Returns:
        Dict with loader information.
    """
    dataset = loader.dataset
    sample = dataset[0] if len(dataset) > 0 else {}

    info = {
        'num_samples': len(dataset),
        'batch_size': loader.batch_size,
        'num_batches': len(loader),
    }

    if isinstance(sample, dict):
        info['batch_format'] = 'dict'
        info['keys'] = list(sample.keys())
        if 'image' in sample:
            info['image_shape'] = tuple(sample['image'].shape)
    else:
        info['batch_format'] = type(sample).__name__

    return info


# ==============================================================================
# Compression Dataloader Implementation
# ==============================================================================


def _get_compression_output_format(mode: str) -> str:
    """Map compression mode to DictDatasetWrapper output_format.

    Args:
        mode: Compression mode name.

    Returns:
        Output format string for DictDatasetWrapper.
    """
    if mode == 'seg_compression':
        return 'compression_seg'
    elif mode == 'dual':
        return 'compression_dual'
    elif mode == 'multi_modality':
        return 'compression_multi'
    else:
        # Single modality: bravo, t1_pre, t1_gd, flair, seg
        return 'compression_single'


def _create_compression_dataloader(
    cfg: DictConfig,
    mode: str,
    spatial_dims: int,
    split: Split,
    use_distributed: bool,
    rank: int,
    world_size: int,
    augment: Optional[bool],
    batch_size: Optional[int],
) -> Tuple[DataLoader, Dataset]:
    """Create compression dataloader (VAE/VQVAE/DC-AE).

    Routes to 2D or 3D compression loader based on spatial_dims.
    """
    if augment is None:
        augment = cfg.training.get('augment', False) if split == 'train' else False

    if spatial_dims == 2:
        return _create_compression_2d_loader(
            cfg, mode, split, use_distributed, rank, world_size, augment, batch_size
        )
    else:
        return _create_compression_3d_loader(
            cfg, mode, split, use_distributed, rank, world_size, augment, batch_size
        )


def _create_compression_2d_loader(
    cfg: DictConfig,
    mode: str,
    split: str,
    use_distributed: bool,
    rank: int,
    world_size: int,
    augment: bool,
    batch_size: Optional[int],
) -> Tuple[DataLoader, Dataset]:
    """Create 2D compression dataloader.

    Routes to the appropriate underlying loader based on mode,
    then wraps with DictDatasetWrapper for consistent dict output.
    """
    from medgen.data.loaders.common import DataLoaderConfig

    # Get raw loader and dataset from underlying implementation
    loader, raw_dataset = _get_raw_compression_2d_loader(
        cfg, mode, split, use_distributed, rank, world_size, augment, batch_size
    )

    # Wrap dataset to return dict format
    output_format = _get_compression_output_format(mode)
    wrapped_dataset = DictDatasetWrapper(raw_dataset, output_format=output_format, spatial_dims=2)

    # Create new DataLoader with dict_collate_fn
    dl_cfg = DataLoaderConfig.from_cfg(cfg)
    effective_batch_size = batch_size if batch_size else loader.batch_size

    # Preserve distributed sampler if present
    if hasattr(loader, 'sampler') and loader.sampler is not None:
        sampler = loader.sampler
        shuffle = False
    else:
        sampler = None
        shuffle = (split == 'train')

    new_loader = DataLoader(
        wrapped_dataset,
        batch_size=effective_batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        collate_fn=dict_collate_fn,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor if dl_cfg.num_workers > 0 else None,
        persistent_workers=dl_cfg.persistent_workers if dl_cfg.num_workers > 0 else False,
    )

    return new_loader, wrapped_dataset


def _get_raw_compression_2d_loader(
    cfg: DictConfig,
    mode: str,
    split: str,
    use_distributed: bool,
    rank: int,
    world_size: int,
    augment: bool,
    batch_size: Optional[int],
) -> Tuple[DataLoader, Dataset]:
    """Get raw 2D compression loader (returns tuple/tensor format)."""
    from medgen.data.loaders import vae, seg_compression, multi_modality

    if mode == 'seg_compression':
        # DC-AE seg mask compression
        image_size = cfg.model.get('image_size', 256)
        effective_batch_size = batch_size if batch_size else cfg.training.batch_size

        if split == 'train':
            return seg_compression.create_seg_compression_dataloader(
                cfg, use_distributed, rank, world_size, augment
            )
        elif split == 'val':
            result = seg_compression.create_seg_compression_validation_dataloader(
                cfg, image_size, effective_batch_size
            )
            if result is None:
                raise ValueError("No validation data found for seg_compression")
            return result
        else:  # test
            result = seg_compression.create_seg_compression_test_dataloader(
                cfg, image_size, effective_batch_size
            )
            if result is None:
                raise ValueError("No test data found for seg_compression")
            return result

    elif mode == 'multi_modality':
        # Multi-modality pooled VAE
        image_keys = cfg.model.get('image_keys', ['bravo', 'flair', 't1_pre', 't1_gd'])
        image_size = cfg.model.get('image_size', 256)
        effective_batch_size = batch_size if batch_size else cfg.training.batch_size

        if split == 'train':
            return multi_modality.create_multi_modality_dataloader(
                cfg, image_keys, image_size, effective_batch_size,
                use_distributed, rank, world_size, augment
            )
        elif split == 'val':
            result = multi_modality.create_multi_modality_validation_dataloader(
                cfg, image_keys, image_size, effective_batch_size
            )
            if result is None:
                raise ValueError("No validation data found for multi_modality")
            return result
        else:  # test
            result = multi_modality.create_multi_modality_test_dataloader(
                cfg, image_keys, image_size, effective_batch_size
            )
            if result is None:
                raise ValueError("No test data found for multi_modality")
            return result

    elif mode == 'dual':
        # Dual modality VAE (t1_pre + t1_gd, 2 channels)
        if split == 'train':
            return vae.create_vae_dataloader(
                cfg, modality='dual', use_distributed=use_distributed,
                rank=rank, world_size=world_size, augment=augment
            )
        elif split == 'val':
            result = vae.create_vae_validation_dataloader(cfg, 'dual', batch_size)
            if result is None:
                raise ValueError("No validation data found for dual VAE")
            return result
        else:  # test
            result = vae.create_vae_test_dataloader(cfg, 'dual', batch_size)
            if result is None:
                raise ValueError("No test data found for dual VAE")
            return result

    else:
        # Single modality: bravo, t1_pre, t1_gd, flair, seg
        if split == 'train':
            return vae.create_vae_dataloader(
                cfg, modality=mode, use_distributed=use_distributed,
                rank=rank, world_size=world_size, augment=augment
            )
        elif split == 'val':
            result = vae.create_vae_validation_dataloader(cfg, mode, batch_size)
            if result is None:
                raise ValueError(f"No validation data found for {mode} VAE")
            return result
        else:  # test
            result = vae.create_vae_test_dataloader(cfg, mode, batch_size)
            if result is None:
                raise ValueError(f"No test data found for {mode} VAE")
            return result


def _create_compression_3d_loader(
    cfg: DictConfig,
    mode: str,
    split: str,
    use_distributed: bool,
    rank: int,
    world_size: int,
    augment: bool,
    batch_size: Optional[int],
) -> Tuple[DataLoader, Dataset]:
    """Create 3D compression dataloader.

    3D loaders already return dict format, so minimal wrapping needed.
    """
    from medgen.data.loaders import volume_3d
    from medgen.data.loaders.volume_3d import VolumeConfig

    vol_cfg = VolumeConfig.from_cfg(cfg)

    if mode == 'seg_compression':
        raise ValueError("3D seg_compression mode not yet supported")

    elif mode == 'multi_modality':
        # Multi-modality 3D VAE
        if split == 'train':
            return volume_3d.create_vae_3d_multi_modality_dataloader(
                cfg, vol_cfg, augment=augment
            )
        elif split == 'val':
            result = volume_3d.create_vae_3d_multi_modality_validation_dataloader(cfg, vol_cfg)
            if result is None:
                raise ValueError("No 3D validation data found for multi_modality")
            return result
        else:  # test
            result = volume_3d.create_vae_3d_multi_modality_test_dataloader(cfg, vol_cfg)
            if result is None:
                raise ValueError("No 3D test data found for multi_modality")
            return result

    elif mode == 'dual':
        # Dual modality 3D VAE
        if split == 'train':
            return volume_3d.create_vae_3d_dataloader(
                cfg, vol_cfg, modality='dual', augment=augment
            )
        elif split == 'val':
            result = volume_3d.create_vae_3d_validation_dataloader(cfg, vol_cfg, 'dual')
            if result is None:
                raise ValueError("No 3D validation data found for dual VAE")
            return result
        else:  # test
            result = volume_3d.create_vae_3d_test_dataloader(cfg, vol_cfg, 'dual')
            if result is None:
                raise ValueError("No 3D test data found for dual VAE")
            return result

    else:
        # Single modality: bravo, t1_pre, t1_gd, flair, seg
        if split == 'train':
            return volume_3d.create_vae_3d_dataloader(
                cfg, vol_cfg, modality=mode, augment=augment
            )
        elif split == 'val':
            result = volume_3d.create_vae_3d_validation_dataloader(cfg, vol_cfg, mode)
            if result is None:
                raise ValueError(f"No 3D validation data found for {mode} VAE")
            return result
        else:  # test
            result = volume_3d.create_vae_3d_test_dataloader(cfg, vol_cfg, mode)
            if result is None:
                raise ValueError(f"No 3D test data found for {mode} VAE")
            return result
