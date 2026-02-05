"""
VAE dataloaders for training and evaluation.

Provides dataloaders for VAE training with single or dual modality,
with separate handling for training, validation, and test sets.

Uses aggressive VAE-specific augmentation and optional batch-level
augmentations (mixup, cutmix) via custom collate function.
"""
import logging
import os
from collections.abc import Callable

from monai.data import DataLoader, Dataset
from omegaconf import DictConfig

from medgen.augmentation import build_vae_augmentation, create_vae_collate_fn
from medgen.data.dataset import (
    NiFTIDataset,
    build_standard_transform,
    validate_modality_exists,
)
from medgen.data.loaders.common import (
    DistributedArgs,
    create_dataloader,
    get_validated_split_dir,
    validate_mode_requirements,
)
from medgen.data.utils import (
    extract_slices_dual,
    extract_slices_single,
    extract_slices_single_with_seg,
    merge_sequences,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Helpers
# =============================================================================

def _validate_vae_modality(
    data_dir: str,
    modality: str,
    context: str = "VAE",
    raise_on_error: bool = False,
) -> bool:
    """Validate VAE modality requirements exist in directory.

    For dual mode: requires t1_pre and t1_gd (no seg).
    For single mode: requires the specified modality.

    Args:
        data_dir: Directory to check.
        modality: 'dual', 'bravo', 't1_pre', 't1_gd', 'flair', or 'seg'.
        context: Context for log message (e.g., "VAE validation").
        raise_on_error: If True, raises ValueError. If False, logs warning.

    Returns:
        True if valid, False otherwise.

    Raises:
        ValueError: If raise_on_error=True and validation fails.
    """
    try:
        if modality == 'dual':
            validate_mode_requirements(
                data_dir, 'dual', validate_modality_exists, require_seg=False
            )
        else:
            validate_modality_exists(data_dir, modality)
        return True
    except ValueError as e:
        if raise_on_error:
            raise
        logger.warning(f"{context} data for {modality} mode not available in {data_dir}: {e}")
        return False


# =============================================================================
# Helper Functions for Seg Loading
# =============================================================================

def _try_load_seg_dataset(
    data_dir: str,
    transform,
    context: str = "VAE"
) -> NiFTIDataset | None:
    """Try to load segmentation dataset for regional metrics.

    Args:
        data_dir: Directory containing NIfTI files.
        transform: Transform to apply to data.
        context: Context string for log message (e.g., "dual VAE training").

    Returns:
        NiFTIDataset if seg exists, None otherwise.
    """
    try:
        validate_modality_exists(data_dir, 'seg')
        return NiFTIDataset(data_dir=data_dir, mr_sequence='seg', transform=transform)
    except ValueError as e:
        logger.debug(f"Seg not available for {context} (regional metrics disabled): {e}")
        return None


def _create_single_modality_dataset(
    data_dir: str,
    modality: str,
    transform,
    context: str = "VAE",
    augmentation=None
) -> Dataset:
    """Create single-modality dataset with optional segmentation.

    Args:
        data_dir: Directory containing NIfTI files.
        modality: MR sequence name (e.g., 'bravo', 't1_pre').
        transform: Transform to apply to data.
        context: Context string for log message.
        augmentation: Optional augmentation to apply.

    Returns:
        Dataset with optional seg for regional metrics.
    """
    nifti_dataset = NiFTIDataset(data_dir=data_dir, mr_sequence=modality, transform=transform)
    seg_dataset = _try_load_seg_dataset(data_dir, transform, f"single {context}")

    if seg_dataset is not None:
        return extract_slices_single_with_seg(nifti_dataset, seg_dataset, augmentation=augmentation)
    else:
        return extract_slices_single(nifti_dataset, augmentation=augmentation)


def _create_dual_modality_dataset(
    data_dir: str,
    transform,
    context: str = "VAE",
    augmentation=None
) -> tuple[Dataset, bool]:
    """Create dual-modality (t1_pre + t1_gd) dataset with optional segmentation.

    Args:
        data_dir: Directory containing NIfTI files.
        transform: Transform to apply to data.
        context: Context string for log message.
        augmentation: Optional augmentation to apply.

    Returns:
        Tuple of (dataset, has_seg).
    """
    datasets_dict: dict[str, NiFTIDataset] = {}
    for key in ['t1_pre', 't1_gd']:
        datasets_dict[key] = NiFTIDataset(data_dir=data_dir, mr_sequence=key, transform=transform)

    seg_dataset = _try_load_seg_dataset(data_dir, transform, f"dual {context}")
    has_seg = seg_dataset is not None
    if has_seg:
        datasets_dict['seg'] = seg_dataset

    merged = merge_sequences(datasets_dict)
    dataset = extract_slices_dual(merged, has_seg=has_seg, augmentation=augmentation)
    return dataset, has_seg


def create_vae_dataloader(
    cfg: DictConfig,
    modality: str,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True
) -> tuple[DataLoader, Dataset]:
    """Create dataloader for VAE training - correct single/dual modality handling.

    For VAE training, we never concatenate seg with images.
    - Single modality (bravo, seg, t1_pre, t1_gd): 1 channel
    - Dual mode (t1_pre + t1_gd): 2 channels, NO seg

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        modality: Modality to train on ('bravo', 'seg', 't1_pre', 't1_gd', 'dual').
        use_distributed: Whether to use distributed training.
        rank: Process rank for distributed training.
        world_size: Total number of processes for distributed training.
        augment: Whether to apply data augmentation.

    Returns:
        Tuple of (DataLoader, train_dataset).
    """
    data_dir = os.path.join(cfg.paths.data_dir, "train")
    image_size = cfg.model.image_size

    # Validate modalities exist before loading (no seg required for VAE)
    _validate_vae_modality(data_dir, modality, "VAE training", raise_on_error=True)

    transform = build_standard_transform(image_size)
    aug = build_vae_augmentation(enabled=augment)

    if modality == 'dual':
        train_dataset, _ = _create_dual_modality_dataset(
            data_dir, transform, context="VAE training", augmentation=aug
        )
    else:
        train_dataset = _create_single_modality_dataset(
            data_dir, modality, transform, context="VAE training", augmentation=aug
        )

    # Get batch augmentation settings
    batch_aug_cfg = cfg.training.get('batch_augment', {})
    batch_aug_enabled = batch_aug_cfg.get('enabled', False)

    # Create collate function with batch augmentations if enabled
    collate_fn: Callable | None = None
    if batch_aug_enabled:
        mixup_prob = batch_aug_cfg.get('mixup_prob', 0.2)
        cutmix_prob = batch_aug_cfg.get('cutmix_prob', 0.2)
        collate_fn = create_vae_collate_fn(mixup_prob=mixup_prob, cutmix_prob=cutmix_prob)

    # Use unified dataloader creation
    dataloader = create_dataloader(
        train_dataset,
        cfg,
        shuffle=True,
        collate_fn=collate_fn,
        distributed_args=DistributedArgs(
            use_distributed=use_distributed,
            rank=rank,
            world_size=world_size,
        ),
    )

    return dataloader, train_dataset


def create_vae_validation_dataloader(
    cfg: DictConfig,
    modality: str,
    batch_size: int | None = None
) -> tuple[DataLoader, Dataset] | None:
    """Create validation dataloader for VAE training from val/ directory.

    Loads data from the val/ subdirectory if it exists. Returns None if
    val/ directory doesn't exist (training will use train dataset sampling
    for validation visualizations).

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        modality: Modality to validate on ('bravo', 'seg', 't1_pre', 't1_gd', 'dual').
        batch_size: Optional batch size override. Defaults to training batch size.

    Returns:
        Tuple of (DataLoader, val_dataset) or None if val/ doesn't exist.
    """
    val_dir = get_validated_split_dir(cfg.paths.data_dir, "val", logger)
    if val_dir is None:
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Validate modalities exist in val directory (no seg required for VAE)
    if not _validate_vae_modality(val_dir, modality, "Validation"):
        return None

    transform = build_standard_transform(image_size)

    if modality == 'dual':
        val_dataset, _ = _create_dual_modality_dataset(
            val_dir, transform, context="VAE validation"
        )
    else:
        val_dataset = _create_single_modality_dataset(
            val_dir, modality, transform, context="VAE validation"
        )

    # Validation loader: shuffle for diverse sampling, drop_last for consistent batch sizes
    dataloader = create_dataloader(
        val_dataset,
        cfg,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # Ensure consistent batch sizes for worst_batch visualization
    )

    return dataloader, val_dataset


def create_vae_test_dataloader(
    cfg: DictConfig,
    modality: str,
    batch_size: int | None = None
) -> tuple[DataLoader, Dataset] | None:
    """Create test dataloader for VAE evaluation from test_new/ directory.

    Loads data from the test_new/ subdirectory if it exists. Returns None if
    test_new/ directory doesn't exist.

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        modality: Modality to test on ('bravo', 'seg', 't1_pre', 't1_gd', 'dual').
        batch_size: Optional batch size override. Defaults to training batch size.

    Returns:
        Tuple of (DataLoader, test_dataset) or None if test_new/ doesn't exist.
    """
    test_dir = get_validated_split_dir(cfg.paths.data_dir, "test_new", logger)
    if test_dir is None:
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Validate modalities exist in test directory (no seg required for VAE)
    if not _validate_vae_modality(test_dir, modality, "Test"):
        return None

    transform = build_standard_transform(image_size)

    if modality == 'dual':
        test_dataset, _ = _create_dual_modality_dataset(
            test_dir, transform, context="VAE test"
        )
    else:
        test_dataset = _create_single_modality_dataset(
            test_dir, modality, transform, context="VAE test"
        )

    # Test loader: shuffle for diverse visualization samples
    dataloader = create_dataloader(
        test_dataset,
        cfg,
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader, test_dataset


# NOTE: VolumeDataset and DualVolumeDataset have been consolidated into datasets.py.
# Import from there for backward compatibility.
from medgen.data.loaders.datasets import DualVolumeDataset, VolumeDataset


def create_vae_volume_validation_dataloader(
    cfg: DictConfig,
    modality: str,
    data_split: str = 'val',
) -> tuple[DataLoader, Dataset] | None:
    """Create dataloader that returns full 3D volumes for volume-level metrics.

    Unlike slice-based loaders, this returns [C, H, W, D] volumes without
    slice extraction. Used for computing 3D MS-SSIM on 2D model reconstructions.

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        modality: Modality ('bravo', 'seg', 't1_pre', 't1_gd', 'dual').
        data_split: Which split to load ('val' or 'test_new').

    Returns:
        Tuple of (DataLoader, volume_dataset) or None if directory doesn't exist.
    """
    data_dir = get_validated_split_dir(cfg.paths.data_dir, data_split, logger)
    if data_dir is None:
        return None

    image_size = cfg.model.image_size

    # Validate modalities exist (no seg required for VAE)
    if not _validate_vae_modality(data_dir, modality, "Volume"):
        return None

    transform = build_standard_transform(image_size)

    if modality == 'dual':
        # Dual mode: stack t1_pre + t1_gd as 2 channels
        t1_pre_dataset = NiFTIDataset(data_dir, 't1_pre', transform)
        t1_gd_dataset = NiFTIDataset(data_dir, 't1_gd', transform)
        seg_dataset = _try_load_seg_dataset(data_dir, transform, "dual volume validation")
        volume_dataset = DualVolumeDataset(t1_pre_dataset, t1_gd_dataset, seg_dataset)
    else:
        # Single modality
        image_dataset = NiFTIDataset(data_dir, modality, transform)
        seg_dataset = _try_load_seg_dataset(data_dir, transform, "single volume validation")
        volume_dataset = VolumeDataset(image_dataset, seg_dataset)

    # Volume-level loader: batch_size=1, no shuffle (process each volume once)
    dataloader = DataLoader(
        volume_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Simple loader, no need for multi-processing
        pin_memory=True,
    )

    return dataloader, volume_dataset
