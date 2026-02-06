"""Parameterized 2D dataloader builder.

Extracts the shared recipe from single.py, dual.py, vae.py, multi_modality.py,
multi_diffusion.py, seg_compression.py, and seg_conditioned.py into a single
build_2d_loader() function driven by a LoaderSpec dataclass.

Each original file becomes a thin wrapper that creates a LoaderSpec and calls
build_2d_loader().
"""
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from omegaconf import DictConfig
from torch.utils.data import Dataset as TorchDataset

from medgen.data.dataset import (
    NiFTIDataset,
    build_standard_transform,
    validate_modality_exists,
)
from medgen.data.loaders.common import (
    DataLoaderConfig,
    DistributedArgs,
    get_validated_split_dir,
    setup_distributed_sampler,
    validate_mode_requirements,
)
from medgen.data.loaders.common import (
    create_dataloader as create_dataloader_from_dataset,
)
from medgen.data.utils import (
    CFGDropoutDataset,
    extract_slices_dual,
    extract_slices_single,
    extract_slices_single_with_seg,
    merge_sequences,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LoaderSpec: captures the 'knobs' that differ across modes
# =============================================================================

@dataclass
class LoaderSpec:
    """Specification for a 2D dataloader mode.

    Captures all parameters that vary across the 7 loader files so that
    build_2d_loader() can implement the shared recipe once.

    Attributes:
        modalities: NiFTI sequences to load (e.g. ['seg'], ['bravo', 'seg']).
        aug_type: Augmentation type for training: 'diffusion', 'vae', 'seg',
            'seg_binarize', or None.
        extractor: How to extract 2D slices from 3D volumes:
            'single', 'dual', 'single_with_seg', 'seg_only', 'with_mode_id'.
        pool: Whether to pool slices from all modalities into one dataset.
        validation_mode: Mode name for validate_mode_requirements().
        require_seg: Whether seg is required for validation.
        optional_seg: Try to load seg for metrics even if not required.
        wrapper: Dataset wrapper after extraction:
            'cfg_dropout', 'augmented_seg', 'multi_diffusion',
            'seg_conditioned', None.
        wrapper_kwargs: Extra kwargs for the wrapper.
        collate: Collate function type: 'vae', 'seg', None.
        mode_id_map: mode_id mapping for multi_diffusion extractor.
        use_manual_dataloader: If True, build DataLoader manually
            (for multi_diffusion which uses DataLoaderConfig directly).
        image_keys_param: Extra image_keys for validate_mode_requirements.
        val_shuffle: Whether to shuffle validation loaders (default True).
        val_drop_last: Whether to drop last batch in validation (default False).
        test_shuffle: Whether to shuffle test loaders (default False).
    """
    modalities: list[str]
    aug_type: str | None = 'diffusion'
    extractor: str = 'single'

    pool: bool = False
    validation_mode: str = ''
    require_seg: bool = True
    optional_seg: bool = False

    wrapper: str | None = None
    wrapper_kwargs: dict = field(default_factory=dict)

    collate: str | None = None
    mode_id_map: dict | None = None
    use_manual_dataloader: bool = False

    image_keys_param: list[str] | None = None
    val_shuffle: bool = True
    val_drop_last: bool = False
    test_shuffle: bool = False


# =============================================================================
# Main builder function
# =============================================================================

def build_2d_loader(
    spec: LoaderSpec,
    cfg: DictConfig,
    split: str,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True,
    batch_size: int | None = None,
    image_size: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[DataLoader, TorchDataset] | None:
    """Build a 2D dataloader from a LoaderSpec.

    Implements the 9-step shared recipe:
    1. Resolve data directory
    2. Validate modalities
    3. Build transform
    4. Build augmentation (train only)
    5. Load NiFTI datasets
    6. Extract slices
    7. Wrap dataset
    8. Build collate function
    9. Create DataLoader

    Args:
        spec: LoaderSpec describing this mode's configuration.
        cfg: Hydra config.
        split: 'train', 'val', or 'test'.
        use_distributed: Whether to use distributed training.
        rank: Process rank.
        world_size: Total processes.
        augment: Whether to apply augmentation (train only).
        batch_size: Override batch size.
        image_size: Override image size (for progressive training).
        generator: Optional torch.Generator for reproducible shuffling.

    Returns:
        Tuple of (DataLoader, dataset) or None if val/test dir missing.
    """
    # --- Step 1: Resolve data directory ---
    data_dir = _resolve_data_dir(cfg, split)
    if data_dir is None:
        return None

    effective_image_size = image_size or cfg.model.image_size
    effective_batch_size = batch_size or cfg.training.batch_size

    # Reduce batch size for DDP validation/test (runs on single GPU)
    if split != 'train' and world_size > 1:
        effective_batch_size = max(1, effective_batch_size // world_size)

    # --- Step 2: Validate modalities ---
    is_train = split == 'train'
    if not _validate_modalities(data_dir, spec, raise_on_error=is_train):
        return None

    # --- Step 3: Build transform ---
    transform = build_standard_transform(effective_image_size)

    # --- Step 4: Build augmentation (train only) ---
    aug = _build_augmentation(spec.aug_type, enabled=(is_train and augment))

    # --- Step 5: Load NiFTI datasets ---
    datasets = _load_nifti_datasets(
        data_dir, spec.modalities, transform, spec.optional_seg
    )

    # --- Step 6: Extract slices ---
    slice_data = _extract_slices(spec, datasets, aug if is_train else None)

    # --- Step 7: Wrap dataset ---
    dataset = _wrap_dataset(
        spec, slice_data, aug if is_train else None, effective_image_size,
        is_train=is_train,
    )

    # --- Step 8: Build collate function ---
    collate_fn = _build_collate(spec, cfg)

    # --- Step 9: Create DataLoader ---
    if spec.use_manual_dataloader:
        dataloader = _create_manual_dataloader(
            dataset, cfg, effective_batch_size, split,
            use_distributed, rank, world_size,
        )
    else:
        shuffle = True if is_train else (
            spec.val_shuffle if split == 'val' else spec.test_shuffle
        )
        drop_last = spec.val_drop_last if split == 'val' else False

        dataloader = create_dataloader_from_dataset(
            dataset,
            cfg=cfg,
            batch_size=effective_batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            distributed_args=DistributedArgs(use_distributed, rank, world_size) if is_train else None,
            generator=generator,
        )

    return dataloader, dataset


# =============================================================================
# Private helpers
# =============================================================================

def _resolve_data_dir(cfg: DictConfig, split: str) -> str | None:
    """Resolve data directory for a split."""
    if split == 'train':
        return os.path.join(cfg.paths.data_dir, 'train')

    split_name = 'val' if split == 'val' else 'test_new'
    return get_validated_split_dir(cfg.paths.data_dir, split_name, logger)


def _validate_modalities(
    data_dir: str,
    spec: LoaderSpec,
    raise_on_error: bool,
) -> bool:
    """Validate that required modalities exist in data_dir."""
    try:
        if spec.validation_mode:
            kwargs: dict[str, Any] = {}
            if spec.image_keys_param is not None:
                kwargs['image_keys'] = spec.image_keys_param
            if not spec.require_seg:
                kwargs['require_seg'] = False
            validate_mode_requirements(
                data_dir, spec.validation_mode, validate_modality_exists,
                **kwargs,
            )
        else:
            # Validate each modality individually
            for mod in spec.modalities:
                if mod == 'seg' and spec.optional_seg:
                    continue  # Will be checked in _load_nifti_datasets
                validate_modality_exists(data_dir, mod)
        return True
    except ValueError as e:
        if raise_on_error:
            raise
        logger.warning(f"Data not available in {data_dir}: {e}")
        return False


def _build_augmentation(aug_type: str | None, enabled: bool) -> Any:
    """Build augmentation pipeline based on type."""
    if aug_type is None or not enabled:
        return None

    if aug_type == 'diffusion':
        from medgen.augmentation import build_diffusion_augmentation
        return build_diffusion_augmentation(enabled=True)
    elif aug_type == 'vae':
        from medgen.augmentation import build_vae_augmentation
        return build_vae_augmentation(enabled=True)
    elif aug_type == 'seg':
        from medgen.augmentation import build_seg_augmentation
        return build_seg_augmentation(enabled=True)
    elif aug_type == 'seg_binarize':
        from medgen.augmentation import build_seg_diffusion_augmentation_with_binarize
        return build_seg_diffusion_augmentation_with_binarize(enabled=True)
    else:
        raise ValueError(f"Unknown aug_type: {aug_type}")


def _load_nifti_datasets(
    data_dir: str,
    modalities: list[str],
    transform: Any,
    optional_seg: bool,
) -> dict[str, NiFTIDataset | None]:
    """Load NiFTI datasets for each modality."""
    datasets: dict[str, NiFTIDataset | None] = {}
    for mod in modalities:
        if mod == 'seg' and optional_seg:
            try:
                validate_modality_exists(data_dir, 'seg')
                datasets['seg'] = NiFTIDataset(
                    data_dir=data_dir, mr_sequence='seg', transform=transform
                )
            except ValueError as e:
                logger.debug(f"Seg not available (regional metrics disabled): {e}")
                datasets['seg'] = None
        else:
            datasets[mod] = NiFTIDataset(
                data_dir=data_dir, mr_sequence=mod, transform=transform
            )
    return datasets


def _extract_slices(
    spec: LoaderSpec,
    datasets: dict[str, NiFTIDataset | None],
    aug: Any,
) -> Any:
    """Extract 2D slices from 3D volumes using the specified extractor."""
    if spec.extractor == 'single':
        # Single modality: just one dataset
        ds = next(iter(datasets.values()))
        return extract_slices_single(ds, augmentation=aug)

    elif spec.extractor == 'dual':
        # Dual: merge multiple sequences, extract pairs
        has_seg = 'seg' in datasets and datasets['seg'] is not None
        merged = merge_sequences({k: v for k, v in datasets.items() if v is not None})
        return extract_slices_dual(merged, has_seg=has_seg, augmentation=aug)

    elif spec.extractor == 'single_with_seg':
        # Single modality + optional seg for regional metrics
        non_seg_keys = [k for k in datasets if k != 'seg']
        if not non_seg_keys:
            raise ValueError("No non-seg modalities found")
        img_ds = datasets[non_seg_keys[0]]
        seg_ds = datasets.get('seg')

        if spec.pool:
            # Pool across modalities
            all_slices: list = []
            for key in non_seg_keys:
                img_ds = datasets[key]
                if seg_ds is not None:
                    slices = extract_slices_single_with_seg(img_ds, seg_ds, augmentation=aug)
                else:
                    slices = extract_slices_single(img_ds, augmentation=aug)
                all_slices.extend(list(slices))
            return all_slices
        else:
            if seg_ds is not None:
                return extract_slices_single_with_seg(img_ds, seg_ds, augmentation=aug)
            else:
                return extract_slices_single(img_ds, augmentation=aug)

    elif spec.extractor == 'single_pool':
        # Pool single-channel slices from multiple modalities
        all_slices: list = []
        for key in spec.modalities:
            ds = datasets[key]
            slices = extract_slices_single(ds, augmentation=aug)
            all_slices.extend(list(slices))
        return all_slices

    elif spec.extractor == 'seg_only':
        # Seg-only extraction (for seg compression)
        from medgen.data.loaders.datasets import extract_seg_slices
        seg_ds = datasets['seg']
        return extract_seg_slices(seg_ds)

    elif spec.extractor == 'with_mode_id':
        # Multi-diffusion: extract with mode_id
        from medgen.data.loaders.datasets import extract_slices_with_seg_and_mode

        if spec.mode_id_map is None:
            raise ValueError("mode_id_map required for with_mode_id extractor")

        seg_ds = datasets['seg']
        all_samples: list[tuple[np.ndarray, np.ndarray, int]] = []

        for key in spec.modalities:
            if key == 'seg':
                continue
            mode_id = spec.mode_id_map.get(key)
            if mode_id is None:
                raise ValueError(
                    f"Unknown modality key '{key}'. "
                    f"Valid keys: {list(spec.mode_id_map.keys())}"
                )
            img_ds = datasets[key]
            slices = extract_slices_with_seg_and_mode(
                img_ds, seg_ds, mode_id, augmentation=aug
            )
            all_samples.extend(slices)
            logger.info(f"Extracted {len(slices)} slices from {key}")

        logger.info(f"Total training slices: {len(all_samples)}")
        return all_samples

    elif spec.extractor == 'single_no_aug':
        # Single modality, no augmentation during extraction (for seg_conditioned)
        ds = next(iter(datasets.values()))
        return extract_slices_single(ds, augmentation=None)

    else:
        raise ValueError(f"Unknown extractor: {spec.extractor}")


def _wrap_dataset(
    spec: LoaderSpec,
    slice_data: Any,
    aug: Any,
    image_size: int,
    is_train: bool = True,
) -> TorchDataset:
    """Wrap extracted slices into a Dataset with optional transformations."""
    wrapper = spec.wrapper

    if wrapper is None:
        # Default: wrap list into Dataset if needed
        if isinstance(slice_data, list):
            return Dataset(slice_data)
        return slice_data

    elif wrapper == 'cfg_dropout':
        # For bravo/dual with CFG dropout (train only)
        if is_train:
            return CFGDropoutDataset(slice_data, **spec.wrapper_kwargs)
        else:
            return slice_data

    elif wrapper == 'augmented_seg':
        # For seg compression (AugmentedSegDataset)
        from medgen.data.loaders.datasets import AugmentedSegDataset
        return AugmentedSegDataset(slice_data, augmentation=aug)

    elif wrapper == 'multi_diffusion':
        # For multi-diffusion (MultiDiffusionDataset)
        from medgen.data.loaders.datasets import MultiDiffusionDataset
        return MultiDiffusionDataset(slice_data)

    elif wrapper == 'seg_conditioned':
        # For seg_conditioned (SegConditionedDataset)
        from medgen.data.loaders.datasets import SegConditionedDataset
        kwargs = dict(spec.wrapper_kwargs)
        kwargs['image_size'] = image_size
        # Pass augmentation from the build step (train only)
        if aug is not None:
            kwargs['augmentation'] = aug
        return SegConditionedDataset(slice_data, **kwargs)

    elif wrapper == 'dataset_list':
        # Wrap a list into MONAI Dataset
        return Dataset(slice_data)

    else:
        raise ValueError(f"Unknown wrapper: {wrapper}")


def _build_collate(
    spec: LoaderSpec,
    cfg: DictConfig,
) -> Callable | None:
    """Build collate function based on spec."""
    if spec.collate is None:
        return None

    batch_aug_cfg = cfg.training.get('batch_augment', {})
    batch_aug_enabled = batch_aug_cfg.get('enabled', False)

    if spec.collate == 'vae':
        if not batch_aug_enabled:
            return None
        from medgen.augmentation import create_vae_collate_fn
        return create_vae_collate_fn(
            mixup_prob=batch_aug_cfg.get('mixup_prob', 0.2),
            cutmix_prob=batch_aug_cfg.get('cutmix_prob', 0.2),
        )

    elif spec.collate == 'seg':
        if not batch_aug_enabled:
            return None
        from medgen.augmentation import create_seg_collate_fn
        return create_seg_collate_fn(
            mosaic_prob=batch_aug_cfg.get('mosaic_prob', 0.2),
            cutmix_prob=batch_aug_cfg.get('cutmix_prob', 0.2),
            copy_paste_prob=batch_aug_cfg.get('copy_paste_prob', 0.3),
        )

    else:
        raise ValueError(f"Unknown collate type: {spec.collate}")


def _create_manual_dataloader(
    dataset: TorchDataset,
    cfg: DictConfig,
    batch_size: int,
    split: str,
    use_distributed: bool,
    rank: int,
    world_size: int,
) -> DataLoader:
    """Create DataLoader manually (for multi_diffusion which bypasses common.create_dataloader)."""
    dl_cfg = DataLoaderConfig.from_cfg(cfg)

    if split == 'train':
        sampler, batch_size_per_gpu, shuffle = setup_distributed_sampler(
            dataset, use_distributed, rank, world_size, batch_size, shuffle=True
        )
    else:
        sampler = None
        batch_size_per_gpu = batch_size
        shuffle = True  # multi_diffusion val/test shuffle

    return DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        shuffle=shuffle,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )


# =============================================================================
# Spec factory functions
# =============================================================================

def single_spec(
    image_type: str,
    augment_type: str = 'diffusion',
    cfg_dropout_prob: float = 0.15,
) -> LoaderSpec:
    """Create LoaderSpec for single-image diffusion (seg or bravo+seg)."""
    if image_type == 'seg':
        return LoaderSpec(
            modalities=['seg'],
            aug_type=augment_type,
            extractor='single',
            validation_mode='seg',
            val_drop_last=True,
        )
    elif image_type == 'bravo':
        spec = LoaderSpec(
            modalities=['bravo', 'seg'],
            aug_type=augment_type,
            extractor='dual',
            validation_mode='bravo',
            val_drop_last=True,
        )
        if cfg_dropout_prob > 0:
            spec.wrapper = 'cfg_dropout'
            spec.wrapper_kwargs = {'cfg_dropout_prob': cfg_dropout_prob}
        return spec
    else:
        raise ValueError(f"Unknown image_type: {image_type}")


def dual_spec(
    image_keys: list[str],
    conditioning: str | None = 'seg',
    augment_type: str = 'diffusion',
    cfg_dropout_prob: float = 0.15,
) -> LoaderSpec:
    """Create LoaderSpec for dual-image diffusion (t1_pre + t1_gd + seg)."""
    if len(image_keys) != 2:
        raise ValueError(
            f"Dual-image mode requires exactly 2 image types, got {len(image_keys)}: {image_keys}"
        )

    modalities = list(image_keys)
    if conditioning:
        modalities.append(conditioning)

    spec = LoaderSpec(
        modalities=modalities,
        aug_type=augment_type,
        extractor='dual',
        validation_mode='dual',
        image_keys_param=list(image_keys),
        test_shuffle=True,
    )

    if conditioning is not None and cfg_dropout_prob > 0:
        spec.wrapper = 'cfg_dropout'
        spec.wrapper_kwargs = {'cfg_dropout_prob': cfg_dropout_prob}

    return spec


def vae_single_spec(modality: str) -> LoaderSpec:
    """Create LoaderSpec for single-modality VAE training."""
    return LoaderSpec(
        modalities=[modality, 'seg'],
        aug_type='vae',
        extractor='single_with_seg',
        optional_seg=True,
        require_seg=False,
        collate='vae',
        val_drop_last=True,
        test_shuffle=True,
    )


def vae_dual_spec() -> LoaderSpec:
    """Create LoaderSpec for dual-modality VAE training (t1_pre + t1_gd)."""
    return LoaderSpec(
        modalities=['t1_pre', 't1_gd', 'seg'],
        aug_type='vae',
        extractor='dual',
        validation_mode='dual',
        require_seg=False,
        optional_seg=True,
        collate='vae',
        val_drop_last=True,
        test_shuffle=True,
    )


def multi_modality_spec(image_keys: list[str]) -> LoaderSpec:
    """Create LoaderSpec for multi-modality VAE training (pooled single-channel)."""
    return LoaderSpec(
        modalities=list(image_keys),
        aug_type='vae',
        extractor='single_pool',
        pool=True,
        validation_mode='multi_modality',
        require_seg=False,
        image_keys_param=list(image_keys),
        collate='vae',
        wrapper='dataset_list',
        val_drop_last=True,
        test_shuffle=True,
    )


def multi_modality_val_spec(image_keys: list[str]) -> LoaderSpec:
    """Create LoaderSpec for multi-modality VAE validation (with optional seg)."""
    # Validation includes seg for metrics
    modalities = list(image_keys) + ['seg']
    return LoaderSpec(
        modalities=modalities,
        aug_type=None,
        extractor='single_with_seg',
        pool=True,
        optional_seg=True,
        validation_mode='multi_modality',
        require_seg=False,
        image_keys_param=list(image_keys),
        wrapper='dataset_list',
        val_drop_last=True,
        test_shuffle=True,
    )


def seg_compression_spec() -> LoaderSpec:
    """Create LoaderSpec for seg mask compression (DC-AE)."""
    return LoaderSpec(
        modalities=['seg'],
        aug_type='seg',
        extractor='seg_only',
        wrapper='augmented_seg',
        collate='seg',
        val_shuffle=True,
        val_drop_last=True,
        test_shuffle=True,
    )


def seg_conditioned_spec(
    cfg: DictConfig,
    size_bin_config: dict | None = None,
    is_train: bool = True,
) -> LoaderSpec:
    """Create LoaderSpec for seg_conditioned diffusion.

    Args:
        cfg: Hydra config with mode.size_bins settings.
        size_bin_config: Optional override for size bin settings.
        is_train: Whether this is for training (affects positive_only and dropout).
    """
    from medgen.data.loaders.datasets import DEFAULT_BIN_EDGES

    # Get size bin config from mode config, with optional override
    size_bin_cfg = cfg.mode.get('size_bins', {})
    if size_bin_config:
        size_bin_cfg = {**dict(size_bin_cfg), **size_bin_config}

    bin_edges = list(size_bin_cfg.get('edges', DEFAULT_BIN_EDGES))
    num_bins = int(size_bin_cfg.get('num_bins', len(bin_edges) - 1))
    fov_mm = float(size_bin_cfg.get('fov_mm', 240.0))
    return_bin_maps = bool(size_bin_cfg.get('return_bin_maps', False))
    max_count = int(size_bin_cfg.get('max_count', 10))

    if is_train:
        cfg_dropout_prob = float(
            size_bin_cfg.get('cfg_dropout_prob', cfg.mode.get('cfg_dropout_prob', 0.0))
        )
        aug_type = 'seg_binarize'
        positive_only = True
    else:
        cfg_dropout_prob = 0.0
        aug_type = None
        positive_only = False

    return LoaderSpec(
        modalities=['seg'],
        aug_type=aug_type,
        extractor='single_no_aug',
        validation_mode='seg_conditioned',
        wrapper='seg_conditioned',
        wrapper_kwargs={
            'bin_edges': bin_edges,
            'num_bins': num_bins,
            'fov_mm': fov_mm,
            'positive_only': positive_only,
            'cfg_dropout_prob': cfg_dropout_prob,
            'return_bin_maps': return_bin_maps,
            'max_count': max_count,
            # augmentation is set dynamically in build_2d_loader via _wrap_dataset
        },
        val_shuffle=True,
        val_drop_last=True,
    )


def multi_diffusion_spec(image_keys: list[str]) -> LoaderSpec:
    """Create LoaderSpec for multi-modality diffusion with mode embedding."""
    from medgen.models.wrappers import MODE_ID_MAP

    modalities = list(image_keys) + ['seg']
    return LoaderSpec(
        modalities=modalities,
        aug_type='diffusion',
        extractor='with_mode_id',
        mode_id_map=dict(MODE_ID_MAP),
        wrapper='multi_diffusion',
        use_manual_dataloader=True,
    )


# =============================================================================
# Convenience functions (replace thin wrapper files)
# =============================================================================

def create_single_loader(
    cfg: DictConfig,
    image_type: str,
    split: str = 'train',
    augment_type: str = 'diffusion',
    cfg_dropout_prob: float = 0.15,
    **kwargs,
) -> tuple[DataLoader, TorchDataset] | None:
    """Create single-image (seg/bravo) loader."""
    spec = single_spec(image_type, augment_type, cfg_dropout_prob)
    return build_2d_loader(spec, cfg, split, **kwargs)


def create_dual_loader(
    cfg: DictConfig,
    image_keys: list[str],
    conditioning: str | None = 'seg',
    split: str = 'train',
    augment_type: str = 'diffusion',
    cfg_dropout_prob: float = 0.15,
    **kwargs,
) -> tuple[DataLoader, TorchDataset] | None:
    """Create dual-image (t1_pre + t1_gd + seg) loader."""
    spec = dual_spec(image_keys, conditioning, augment_type, cfg_dropout_prob)
    return build_2d_loader(spec, cfg, split, **kwargs)


def create_vae_loader(
    cfg: DictConfig,
    modality: str,
    split: str = 'train',
    **kwargs,
) -> tuple[DataLoader, TorchDataset] | None:
    """Create VAE single/dual modality loader."""
    if modality == 'dual':
        spec = vae_dual_spec()
    else:
        spec = vae_single_spec(modality)
    return build_2d_loader(spec, cfg, split, **kwargs)


def create_multi_modality_loader(
    cfg: DictConfig,
    image_keys: list[str],
    split: str = 'train',
    **kwargs,
) -> tuple[DataLoader, TorchDataset] | None:
    """Create multi-modality VAE loader (pooled single-channel)."""
    if split == 'train':
        spec = multi_modality_spec(image_keys)
    else:
        spec = multi_modality_val_spec(image_keys)
    return build_2d_loader(spec, cfg, split, **kwargs)


def create_single_modality_validation_loader(
    cfg: DictConfig,
    modality: str,
    image_size: int,
    batch_size: int,
) -> DataLoader | None:
    """Create validation loader for a single modality (for per-modality metrics).

    Includes seg masks paired with each slice for regional metrics tracking.
    Batches are tuples of (image [B,1,H,W], seg [B,1,H,W]).

    Returns:
        DataLoader for single modality or None if val/ doesn't exist.
    """
    spec = vae_single_spec(modality)
    result = build_2d_loader(
        spec, cfg, 'val',
        batch_size=batch_size, image_size=image_size,
    )
    if result is None:
        return None
    return result[0]  # Return just the DataLoader


def create_multi_diffusion_loader(
    cfg: DictConfig,
    image_keys: list[str],
    split: str = 'train',
    **kwargs,
) -> tuple[DataLoader, TorchDataset] | None:
    """Create multi-modality diffusion loader with mode embedding."""
    spec = multi_diffusion_spec(image_keys)
    return build_2d_loader(spec, cfg, split, **kwargs)


def create_seg_compression_loader(
    cfg: DictConfig,
    split: str = 'train',
    **kwargs,
) -> tuple[DataLoader, TorchDataset] | None:
    """Create seg mask compression (DC-AE) loader."""
    spec = seg_compression_spec()
    return build_2d_loader(spec, cfg, split, **kwargs)


def create_seg_conditioned_loader(
    cfg: DictConfig,
    split: str = 'train',
    size_bin_config: dict | None = None,
    **kwargs,
) -> tuple[DataLoader, TorchDataset] | None:
    """Create seg_conditioned diffusion loader."""
    is_train = split == 'train'
    spec = seg_conditioned_spec(cfg, size_bin_config, is_train=is_train)
    if split == 'val':
        # Deterministic validation
        kwargs.setdefault('generator', torch.Generator().manual_seed(42))
    return build_2d_loader(spec, cfg, split, **kwargs)


def create_single_modality_diffusion_val_loader(
    cfg: DictConfig,
    modality: str,
) -> DataLoader | None:
    """Create validation loader for a single modality (for per-modality diffusion metrics).

    Returns:
        DataLoader for single modality or None if val/ doesn't exist.
    """
    from medgen.data.loaders.common import (
        DataLoaderConfig,
        get_validated_split_dir,
    )
    from medgen.data.dataset import (
        NiFTIDataset,
        build_standard_transform,
        validate_modality_exists,
    )
    from medgen.data.loaders.datasets import (
        MultiDiffusionDataset,
        extract_slices_with_seg_and_mode,
    )
    from medgen.models.wrappers import MODE_ID_MAP

    val_dir = get_validated_split_dir(cfg.paths.data_dir, "val", logger)
    if val_dir is None:
        return None

    image_size = cfg.model.image_size
    batch_size = cfg.training.batch_size

    try:
        validate_modality_exists(val_dir, modality)
        validate_modality_exists(val_dir, 'seg')
    except ValueError as e:
        logger.warning(f"Modality {modality} not found in val/: {e}")
        return None

    transform = build_standard_transform(image_size)

    image_dataset = NiFTIDataset(
        data_dir=val_dir, mr_sequence=modality, transform=transform
    )
    seg_dataset = NiFTIDataset(
        data_dir=val_dir, mr_sequence='seg', transform=transform
    )

    mode_id = MODE_ID_MAP.get(modality, 0)
    slices = extract_slices_with_seg_and_mode(
        image_dataset, seg_dataset, mode_id, augmentation=None
    )

    val_dataset = MultiDiffusionDataset(slices)

    dl_cfg = DataLoaderConfig.from_cfg(cfg)
    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )

    return dataloader
