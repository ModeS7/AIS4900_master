"""
Multi-modality dataloaders for diffusion training with mode embedding.

Provides dataloaders that pool multiple MR modalities, each paired with seg mask
and mode_id for training a single model on all modalities.

Uses eager loading like other loaders - all slices extracted upfront into memory.
"""
import logging
import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from omegaconf import DictConfig

from medgen.augmentation import build_diffusion_augmentation, apply_augmentation
from medgen.data.loaders.common import DataLoaderConfig, setup_distributed_sampler
from medgen.data.dataset import (
    NiFTIDataset,
    build_standard_transform,
    validate_modality_exists,
)
from medgen.models.wrappers import MODE_ID_MAP
from medgen.core.constants import BINARY_THRESHOLD_GT

logger = logging.getLogger(__name__)


def extract_slices_with_seg_and_mode(
    image_dataset: Dataset,
    seg_dataset: Dataset,
    mode_id: int,
    augmentation: Optional[Callable] = None,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Extract 2D slices with paired seg masks and mode_id.

    Each slice is returned as a tuple (image, seg, mode_id) where:
    - image: [1, H, W] single-channel image
    - seg: [1, H, W] binary segmentation mask
    - mode_id: int (0=bravo, 1=flair, 2=t1_pre, 3=t1_gd)

    Args:
        image_dataset: Dataset of 3D image volumes with shape [1, H, W, D].
        seg_dataset: Dataset of 3D seg volumes with shape [1, H, W, D].
        mode_id: Integer mode ID for this modality.
        augmentation: Optional albumentations Compose for data augmentation.

    Returns:
        List of tuples (image_slice, seg_slice, mode_id).
    """
    all_slices: List[Tuple[np.ndarray, np.ndarray, int]] = []

    if len(image_dataset) != len(seg_dataset):
        raise ValueError(
            f"Image dataset ({len(image_dataset)}) and seg dataset ({len(seg_dataset)}) "
            "must have same number of patients"
        )

    for i in range(len(image_dataset)):
        image_volume, image_name = image_dataset[i]  # Shape: [1, H, W, D]
        seg_volume, seg_name = seg_dataset[i]  # Shape: [1, H, W, D]

        # Convert to numpy if tensor
        if isinstance(image_volume, torch.Tensor):
            image_volume = image_volume.numpy()
        if isinstance(seg_volume, torch.Tensor):
            seg_volume = seg_volume.numpy()

        # Verify same patient
        if image_name != seg_name:
            raise ValueError(f"Patient mismatch: {image_name} vs {seg_name}")

        # Extract non-empty slices along depth dimension (axis 3)
        for k in range(image_volume.shape[3]):
            image_slice = image_volume[:, :, :, k].copy()
            seg_slice = seg_volume[:, :, :, k].copy()

            if np.sum(image_slice) > 1.0:
                if augmentation is not None:
                    # Transpose to [H, W, C] for albumentations
                    img_hwc = np.transpose(image_slice, (1, 2, 0))
                    seg_hwc = np.transpose(seg_slice, (1, 2, 0))

                    transformed = augmentation(image=img_hwc, mask=seg_hwc)
                    img_aug = transformed['image']
                    seg_aug = transformed['mask']

                    # Transpose back to [C, H, W]
                    image_slice = np.transpose(img_aug, (2, 0, 1))
                    seg_slice = np.transpose(seg_aug, (2, 0, 1))

                # Binarize seg mask
                seg_slice = (seg_slice > BINARY_THRESHOLD_GT).astype(np.float32)

                all_slices.append((image_slice, seg_slice, mode_id))

    return all_slices


class MultiDiffusionDataset(Dataset):
    """Dataset that returns (image, seg, mode_id) tuples.

    Wraps a list of pre-extracted slices for efficient training.
    """

    def __init__(self, samples: List[Tuple[np.ndarray, np.ndarray, int]]):
        """Initialize dataset.

        Args:
            samples: List of (image, seg, mode_id) tuples.
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, seg, mode_id = self.samples[idx]
        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(seg).float(),
            torch.tensor(mode_id, dtype=torch.long),
        )


def create_multi_diffusion_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True,
) -> Tuple[DataLoader, Dataset]:
    """Create dataloader for multi-modality diffusion training.

    Loads multiple MR sequences, each paired with seg mask and mode_id.
    Each batch contains mixed slices from all modalities with their mode IDs.

    Args:
        cfg: Hydra configuration with paths.
        image_keys: List of MR sequences (e.g., ['bravo', 'flair', 't1_pre', 't1_gd']).
        use_distributed: Whether to use distributed training.
        rank: Process rank for distributed training.
        world_size: Total number of processes for distributed training.
        augment: Whether to apply data augmentation.

    Returns:
        Tuple of (DataLoader, train_dataset).
        Batches are tuples of (image [B,1,H,W], seg [B,1,H,W], mode_id [B]).
    """
    data_dir = os.path.join(cfg.paths.data_dir, "train")
    image_size = cfg.model.image_size
    batch_size = cfg.training.batch_size

    # Validate all modalities exist
    for key in image_keys:
        validate_modality_exists(data_dir, key)
    validate_modality_exists(data_dir, 'seg')

    transform = build_standard_transform(image_size)
    aug = build_diffusion_augmentation(enabled=augment)

    # Load seg dataset (shared across all modalities)
    seg_dataset = NiFTIDataset(
        data_dir=data_dir, mr_sequence='seg', transform=transform
    )

    # Extract all slices from all modalities
    all_samples: List[Tuple[np.ndarray, np.ndarray, int]] = []

    for key in image_keys:
        mode_id = MODE_ID_MAP.get(key)
        if mode_id is None:
            raise ValueError(
                f"Unknown modality key '{key}'. "
                f"Valid keys: {list(MODE_ID_MAP.keys())}"
            )

        image_dataset = NiFTIDataset(
            data_dir=data_dir, mr_sequence=key, transform=transform
        )

        slices = extract_slices_with_seg_and_mode(
            image_dataset, seg_dataset, mode_id, augmentation=aug
        )
        all_samples.extend(slices)
        logger.info(f"Extracted {len(slices)} slices from {key}")

    train_dataset = MultiDiffusionDataset(all_samples)
    logger.info(f"Total training slices: {len(train_dataset)}")

    # Setup distributed sampler and batch size
    sampler, batch_size_per_gpu, shuffle = setup_distributed_sampler(
        train_dataset, use_distributed, rank, world_size, batch_size, shuffle=True
    )

    # Get DataLoader settings from config
    dl_cfg = DataLoaderConfig.from_cfg(cfg)

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        shuffle=shuffle,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )

    return dataloader, train_dataset


def create_multi_diffusion_validation_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    world_size: int = 1,
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create validation dataloader for multi-modality diffusion.

    Args:
        cfg: Hydra configuration with paths.
        image_keys: List of MR sequences to load.
        world_size: Number of GPUs for DDP (for batch size adjustment).

    Returns:
        Tuple of (DataLoader, val_dataset) or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, "val")
    if not os.path.exists(val_dir):
        return None

    image_size = cfg.model.image_size
    batch_size = cfg.training.batch_size

    # Reduce batch size for DDP (validation runs on single GPU)
    if world_size > 1:
        batch_size = max(1, batch_size // world_size)

    # Validate modalities exist
    try:
        for key in image_keys:
            validate_modality_exists(val_dir, key)
        validate_modality_exists(val_dir, 'seg')
    except ValueError as e:
        logger.warning(f"Validation directory misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    # Load seg dataset
    seg_dataset = NiFTIDataset(
        data_dir=val_dir, mr_sequence='seg', transform=transform
    )

    # Extract all slices (no augmentation for validation)
    all_samples: List[Tuple[np.ndarray, np.ndarray, int]] = []

    for key in image_keys:
        mode_id = MODE_ID_MAP.get(key)
        if mode_id is None:
            continue

        image_dataset = NiFTIDataset(
            data_dir=val_dir, mr_sequence=key, transform=transform
        )

        slices = extract_slices_with_seg_and_mode(
            image_dataset, seg_dataset, mode_id, augmentation=None
        )
        all_samples.extend(slices)

    val_dataset = MultiDiffusionDataset(all_samples)

    # Get DataLoader settings
    dl_cfg = DataLoaderConfig.from_cfg(cfg)

    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )

    return dataloader, val_dataset


def create_multi_diffusion_test_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    world_size: int = 1,
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create test dataloader for multi-modality diffusion.

    Args:
        cfg: Hydra configuration with paths.
        image_keys: List of MR sequences to load.
        world_size: Number of GPUs for DDP (for batch size adjustment).

    Returns:
        Tuple of (DataLoader, test_dataset) or None if test_new/ doesn't exist.
    """
    test_dir = os.path.join(cfg.paths.data_dir, "test_new")
    if not os.path.exists(test_dir):
        return None

    image_size = cfg.model.image_size
    batch_size = cfg.training.batch_size

    # Reduce batch size for DDP (test runs on single GPU)
    if world_size > 1:
        batch_size = max(1, batch_size // world_size)

    # Validate modalities exist
    try:
        for key in image_keys:
            validate_modality_exists(test_dir, key)
        validate_modality_exists(test_dir, 'seg')
    except ValueError as e:
        logger.warning(f"Test directory misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    # Load seg dataset
    seg_dataset = NiFTIDataset(
        data_dir=test_dir, mr_sequence='seg', transform=transform
    )

    # Extract all slices (no augmentation for test)
    all_samples: List[Tuple[np.ndarray, np.ndarray, int]] = []

    for key in image_keys:
        mode_id = MODE_ID_MAP.get(key)
        if mode_id is None:
            continue

        image_dataset = NiFTIDataset(
            data_dir=test_dir, mr_sequence=key, transform=transform
        )

        slices = extract_slices_with_seg_and_mode(
            image_dataset, seg_dataset, mode_id, augmentation=None
        )
        all_samples.extend(slices)

    test_dataset = MultiDiffusionDataset(all_samples)

    # Get DataLoader settings
    dl_cfg = DataLoaderConfig.from_cfg(cfg)

    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )

    return dataloader, test_dataset


def create_single_modality_diffusion_val_loader(
    cfg: DictConfig,
    modality: str,
) -> Optional[DataLoader]:
    """Create validation loader for a single modality (for per-modality metrics).

    Args:
        cfg: Hydra configuration with paths.
        modality: Single modality to load (e.g., 'bravo', 't1_pre').

    Returns:
        DataLoader for single modality or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, "val")
    if not os.path.exists(val_dir):
        return None

    image_size = cfg.model.image_size
    batch_size = cfg.training.batch_size

    # Validate modality exists
    try:
        validate_modality_exists(val_dir, modality)
        validate_modality_exists(val_dir, 'seg')
    except ValueError as e:
        logger.warning(f"Modality {modality} not found in val/: {e}")
        return None

    transform = build_standard_transform(image_size)

    # Load datasets
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

    # Get DataLoader settings
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
