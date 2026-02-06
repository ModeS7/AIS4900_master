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
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from .base import DictDatasetWrapper, dict_collate_fn

logger = logging.getLogger(__name__)

Task = Literal['diffusion', 'compression']
Mode = Literal['seg', 'bravo', 'dual', 'multi', 'seg_conditioned']
Split = Literal['train', 'val', 'test']


# =============================================================================
# Mode Configuration Registry
# =============================================================================

@dataclass
class ModeConfig:
    """Configuration for a specific dataloader mode.

    This dataclass captures all the mode-specific parameters needed to create
    dataloaders, allowing us to define modes declaratively rather than with
    verbose if/elif chains.
    """
    output_format: str  # Format string for DictDatasetWrapper
    needs_seg: bool = False  # Whether this mode requires seg masks
    cfg_dropout_prob: float = 0.0  # CFG dropout probability (train only)
    extra_config: dict[str, Any] = field(default_factory=dict)


# Registry of supported diffusion modes
DIFFUSION_MODES = {
    'seg', 'bravo', 'bravo_seg_cond', 'dual', 'multi',
    'seg_conditioned', 'seg_conditioned_input'
}

# Registry of supported compression modes
COMPRESSION_MODES = {
    'seg_compression', 'bravo', 'flair', 't1_pre', 't1_gd', 'seg',
    'dual', 'multi_modality'
}


def _get_split_dir(base_dir: str, split: str) -> str:
    """Get directory path for a data split.

    Args:
        base_dir: Base data directory.
        split: Data split ('train', 'val', 'test').

    Returns:
        Path to split directory.
    """
    split_dirs = {'train': 'train', 'val': 'val', 'test': 'test_new'}
    return os.path.join(base_dir, split_dirs.get(split, split))


def _wrap_with_dict_loader(
    raw_dataset: Dataset,
    cfg: DictConfig,
    output_format: str,
    spatial_dims: int,
    split: str,
    batch_size: int | None,
    sampler: Any | None = None,
) -> tuple[DataLoader, Dataset]:
    """Wrap a raw dataset with DictDatasetWrapper and create a new DataLoader.

    This utility function handles the common pattern of:
    1. Wrapping a raw dataset to return dict format
    2. Creating a DataLoader with dict_collate_fn
    3. Preserving distributed sampler if present

    Args:
        raw_dataset: The underlying dataset to wrap.
        cfg: Hydra configuration.
        output_format: Format string for DictDatasetWrapper.
        spatial_dims: 2 for 2D images, 3 for 3D volumes.
        split: Data split ('train', 'val', 'test').
        batch_size: Override batch size (None = use config).
        sampler: Optional distributed sampler to preserve.

    Returns:
        Tuple of (DataLoader, wrapped_dataset).
    """
    from .common import DataLoaderConfig

    # Wrap dataset to return dict format
    wrapped_dataset = DictDatasetWrapper(
        raw_dataset, output_format=output_format, spatial_dims=spatial_dims
    )

    # Get DataLoader settings
    dl_cfg = DataLoaderConfig.from_cfg(cfg)
    effective_batch_size = batch_size if batch_size else cfg.training.batch_size

    # Determine shuffle based on split and sampler
    if sampler is not None:
        shuffle = False  # Sampler handles shuffling
    else:
        shuffle = (split == 'train')

    new_loader = DataLoader(
        wrapped_dataset,
        batch_size=effective_batch_size,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=dict_collate_fn,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor if dl_cfg.num_workers > 0 else None,
        persistent_workers=dl_cfg.persistent_workers if dl_cfg.num_workers > 0 else False,
    )

    return new_loader, wrapped_dataset


def create_dataloader(
    cfg: DictConfig,
    task: Task,
    mode: str,
    spatial_dims: int = 2,
    split: Split = 'train',
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool | None = None,
    batch_size: int | None = None,
) -> tuple[DataLoader, Dataset]:
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
    augment: bool | None = None,
    batch_size: int | None = None,
) -> tuple[DataLoader, Dataset]:
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

    if mode not in ('seg', 'bravo', 'dual', 'multi', 'seg_conditioned', 'seg_conditioned_input', 'bravo_seg_cond'):
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
    batch_size: int | None,
) -> tuple[DataLoader, Dataset]:
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
    batch_size: int | None,
) -> tuple[DataLoader, Dataset]:
    """Get raw 2D loader (may return tuple/tensor format).

    Dispatches to build_2d_loader via spec factories instead of verbose
    per-mode if/elif chains.
    """
    from medgen.data.loaders.builder_2d import (
        build_2d_loader,
        dual_spec,
        multi_diffusion_spec,
        seg_conditioned_spec,
        single_spec,
    )

    spec = None
    is_train = split == 'train'

    if mode in ('seg', 'bravo', 'bravo_seg_cond'):
        # bravo_seg_cond: same pixel loader as bravo
        image_type = 'bravo' if mode in ('bravo', 'bravo_seg_cond') else 'seg'
        spec = single_spec(image_type, augment_type)

    elif mode == 'dual':
        image_keys = cfg.mode.get('image_keys', ['t1_pre', 't1_gd'])
        conditioning = cfg.mode.get('conditioning', 'seg')
        spec = dual_spec(image_keys, conditioning, augment_type)

    elif mode == 'multi':
        if not is_train:
            raise ValueError(f"Multi-modality {split} loader not yet supported")
        spec = multi_diffusion_spec(cfg.mode.image_keys)

    elif mode == 'seg_conditioned':
        size_bin_config = dict(cfg.mode.get('size_bins', {}))
        spec = seg_conditioned_spec(cfg, size_bin_config, is_train=is_train)

    elif mode == 'seg_conditioned_input':
        size_bin_config = dict(cfg.mode.get('size_bins', {}))
        size_bin_config['return_bin_maps'] = True
        size_bin_config['cfg_dropout_prob'] = cfg.mode.get('cfg_dropout_prob', 0.15)
        spec = seg_conditioned_spec(cfg, size_bin_config, is_train=is_train)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    result = build_2d_loader(
        spec, cfg, split,
        use_distributed=use_distributed, rank=rank, world_size=world_size,
        augment=augment, batch_size=batch_size,
    )
    if result is None:
        raise ValueError(f"No {split} data found")
    return result


def _create_3d_loader(
    cfg: DictConfig,
    mode: str,
    split: str,
    use_distributed: bool,
    rank: int,
    world_size: int,
    augment: bool,
    batch_size: int | None,
) -> tuple[DataLoader, Dataset]:
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
    elif mode == 'bravo_seg_cond':
        # bravo_seg_cond: same pixel loader as bravo, latent handling is separate
        return _create_3d_bravo_loader(cfg, vol_cfg, split, augment, batch_size)
    elif mode == 'dual':
        raise ValueError("3D dual mode not yet supported")
    elif mode == 'multi':
        raise ValueError("3D multi mode not yet supported")
    elif mode == 'seg_conditioned':
        return _create_3d_seg_conditioned_loader(cfg, vol_cfg, split, augment, batch_size)
    elif mode == 'seg_conditioned_input':
        # Input conditioning mode: returns bin_maps for concatenation with noise
        # Updates the size_bin_config to enable return_bin_maps in SegDataset
        return _create_3d_seg_conditioned_input_loader(cfg, vol_cfg, split, augment, batch_size)

    raise ValueError(f"Unknown 3D mode: {mode}")


def _create_3d_seg_loader(
    cfg: DictConfig,
    vol_cfg: Any,
    split: str,
    augment: bool,
    batch_size: int | None,
) -> tuple[DataLoader, Dataset]:
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
    batch_size: int | None,
) -> tuple[DataLoader, Dataset]:
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
    batch_size: int | None,
) -> tuple[DataLoader, Dataset]:
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


def _create_3d_seg_conditioned_input_loader(
    cfg: DictConfig,
    vol_cfg: Any,
    split: str,
    augment: bool,
    batch_size: int | None,
) -> tuple[DataLoader, Dataset]:
    """Create 3D seg_conditioned_input mode loader.

    This mode uses input channel conditioning where size bins are converted
    to spatial volumes and concatenated with the noise input. This provides
    stronger conditioning than FiLM (which only modulates the timestep embedding).

    Returns batches with:
        - 'image': [B, 1, D, H, W] segmentation volume
        - 'size_bins': [B, num_bins] size bin counts
        - 'bin_maps': [B, num_bins, D, H, W] spatial conditioning maps
    """
    from medgen.data.loaders import seg
    from medgen.data.loaders.common import DataLoaderConfig

    # Configure to enable bin_maps output
    size_bin_config = dict(cfg.mode.get('size_bins', {}))
    size_bin_config['return_bin_maps'] = True  # Enable spatial bin maps
    cfg_dropout_prob = cfg.mode.get('cfg_dropout_prob', 0.15)

    # Create a modified cfg with the updated size_bin_config
    from omegaconf import OmegaConf
    cfg_modified = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_modified.mode.size_bins = size_bin_config
    cfg_modified.mode.cfg_dropout_prob = cfg_dropout_prob

    if split == 'train':
        loader, raw_dataset = seg.create_seg_dataloader(cfg_modified)
    elif split == 'val':
        result = seg.create_seg_validation_dataloader(cfg_modified)
        if result is None:
            raise ValueError("No 3D validation data found")
        loader, raw_dataset = result
    else:
        result = seg.create_seg_test_dataloader(cfg_modified)
        if result is None:
            raise ValueError("No 3D test data found")
        loader, raw_dataset = result

    # Wrap dataset to return dict format with bin_maps
    wrapped_dataset = DictDatasetWrapper(
        raw_dataset, output_format='seg_conditioned_input', spatial_dims=3
    )

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


def get_dataloader_info(loader: DataLoader) -> dict[str, Any]:
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
    augment: bool | None,
    batch_size: int | None,
) -> tuple[DataLoader, Dataset]:
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
    batch_size: int | None,
) -> tuple[DataLoader, Dataset]:
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
    batch_size: int | None,
) -> tuple[DataLoader, Dataset]:
    """Get raw 2D compression loader (returns tuple/tensor format).

    Dispatches to build_2d_loader via spec factories.
    """
    from medgen.data.loaders.builder_2d import (
        build_2d_loader,
        multi_modality_spec,
        multi_modality_val_spec,
        seg_compression_spec,
        vae_dual_spec,
        vae_single_spec,
    )

    image_size = cfg.model.get('image_size', 256)
    effective_batch_size = batch_size if batch_size else cfg.training.batch_size

    if mode == 'seg_compression':
        spec = seg_compression_spec()
    elif mode == 'multi_modality':
        image_keys = cfg.model.get('image_keys', ['bravo', 'flair', 't1_pre', 't1_gd'])
        if split == 'train':
            spec = multi_modality_spec(image_keys)
        else:
            spec = multi_modality_val_spec(image_keys)
    elif mode == 'dual':
        spec = vae_dual_spec()
    else:
        # Single modality: bravo, t1_pre, t1_gd, flair, seg
        spec = vae_single_spec(mode)

    result = build_2d_loader(
        spec, cfg, split,
        use_distributed=use_distributed, rank=rank, world_size=world_size,
        augment=augment, batch_size=effective_batch_size, image_size=image_size,
    )
    if result is None:
        raise ValueError(f"No {split} data found for {mode}")
    return result


def _create_compression_3d_loader(
    cfg: DictConfig,
    mode: str,
    split: str,
    use_distributed: bool,
    rank: int,
    world_size: int,
    augment: bool,  # Note: 3D loaders read from cfg.training.augment instead
    batch_size: int | None,
) -> tuple[DataLoader, Dataset]:
    """Create 3D compression dataloader.

    3D loaders already return dict format, so minimal wrapping needed.

    Note: The `augment` parameter is not passed to 3D loaders - they read
    augmentation settings directly from cfg.training.augment. This parameter
    is kept for interface consistency with 2D loaders.
    """
    from medgen.data.loaders import volume_3d
    from medgen.data.loaders.volume_3d import VolumeConfig

    vol_cfg = VolumeConfig.from_cfg(cfg)

    if mode == 'seg_compression':
        raise ValueError("3D seg_compression mode not yet supported")

    elif mode == 'multi_modality':
        # Multi-modality 3D VAE
        if split == 'train':
            return volume_3d.create_vae_3d_multi_modality_dataloader(cfg)
        elif split == 'val':
            result = volume_3d.create_vae_3d_multi_modality_validation_dataloader(cfg)
            if result is None:
                raise ValueError("No 3D validation data found for multi_modality")
            return result
        else:  # test
            result = volume_3d.create_vae_3d_multi_modality_test_dataloader(cfg)
            if result is None:
                raise ValueError("No 3D test data found for multi_modality")
            return result

    elif mode == 'dual':
        # Dual modality 3D VAE
        if split == 'train':
            return volume_3d.create_vae_3d_dataloader(cfg, modality='dual')
        elif split == 'val':
            result = volume_3d.create_vae_3d_validation_dataloader(cfg, 'dual')
            if result is None:
                raise ValueError("No 3D validation data found for dual VAE")
            return result
        else:  # test
            result = volume_3d.create_vae_3d_test_dataloader(cfg, 'dual')
            if result is None:
                raise ValueError("No 3D test data found for dual VAE")
            return result

    else:
        # Single modality: bravo, t1_pre, t1_gd, flair, seg
        if split == 'train':
            return volume_3d.create_vae_3d_dataloader(cfg, modality=mode)
        elif split == 'val':
            result = volume_3d.create_vae_3d_validation_dataloader(cfg, mode)
            if result is None:
                raise ValueError(f"No 3D validation data found for {mode} VAE")
            return result
        else:  # test
            result = volume_3d.create_vae_3d_test_dataloader(cfg, mode)
            if result is None:
                raise ValueError(f"No 3D test data found for {mode} VAE")
            return result
