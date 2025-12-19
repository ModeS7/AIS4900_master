"""
Data loading and processing utilities for diffusion model training.

This module provides dataset classes and utility functions for loading,
processing, and preparing medical image data (NIfTI format) for training
diffusion models on brain MRI sequences.
"""
import logging
import os

logger = logging.getLogger(__name__)
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    Resize,
    ScaleIntensity,
    ToTensor,
)
from omegaconf import DictConfig
from torch.utils.data.distributed import DistributedSampler

from medgen.core.constants import BINARY_THRESHOLD_GT, DEFAULT_NUM_WORKERS
from medgen.data.augmentation import apply_augmentation, build_augmentation

try:
    import albumentations as A
except ImportError:
    A = None  # type: ignore


def build_standard_transform(image_size: int) -> Compose:
    """Build standard transform pipeline for medical images.

    Creates a MONAI Compose transform that:
    - Loads NIfTI images
    - Ensures channel-first format
    - Converts to PyTorch tensor
    - Scales intensity to [0, 1]
    - Resizes spatial dimensions (preserves depth)

    Args:
        image_size: Target size for H and W dimensions.

    Returns:
        Composed transform pipeline.
    """
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ToTensor(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        Resize(spatial_size=(image_size, image_size, -1))
    ])


def validate_modality_exists(data_dir: str, modality: str) -> None:
    """Validate that modality files exist in the dataset.

    Checks the first patient directory to verify the modality file exists.

    Args:
        data_dir: Training data directory containing patient subdirectories.
        modality: MR sequence name to check (e.g., 'bravo', 't1_pre', 'seg').

    Raises:
        ValueError: If data directory is empty or modality file not found.
    """
    patients = sorted(os.listdir(data_dir))
    if not patients:
        raise ValueError(f"Data directory is empty: {data_dir}")

    sample_patient = patients[0]
    expected_file = os.path.join(data_dir, sample_patient, f"{modality}.nii.gz")

    if not os.path.exists(expected_file):
        available_files = os.listdir(os.path.join(data_dir, sample_patient))
        nifti_files = [f.replace('.nii.gz', '') for f in available_files if f.endswith('.nii.gz')]
        raise ValueError(
            f"Modality '{modality}' not found in dataset.\n"
            f"Expected: {expected_file}\n"
            f"Available modalities in {sample_patient}: {nifti_files}"
        )


class NiFTIDataset(Dataset):
    """Dataset for loading NIfTI medical images by MR sequence.

    Loads 3D NIfTI volumes from a directory structure where each patient
    has a subdirectory containing sequence-specific files.

    Args:
        data_dir: Root directory containing patient subdirectories.
        mr_sequence: MR sequence name (e.g., 'bravo', 't1_pre', 'seg').
        transform: Optional MONAI transform to apply to loaded images.

    Example:
        >>> transform = Compose([LoadImage(), EnsureChannelFirst()])
        >>> dataset = NiFTIDataset('/data/train', 'bravo', transform)
        >>> volume, patient_name = dataset[0]
    """

    def __init__(
        self,
        data_dir: str,
        mr_sequence: str,
        transform: Optional[Compose] = None
    ) -> None:
        self.data_dir: str = data_dir
        self.data: List[str] = sorted(os.listdir(data_dir))
        self.mr_sequence: str = mr_sequence
        self.transform: Optional[Compose] = transform

    def __len__(self) -> int:
        """Return number of patients in dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, str]:
        """Load and return a single patient volume.

        Args:
            index: Patient index.

        Returns:
            Tuple of (transformed_volume, patient_name).
        """
        patient_name = self.data[index]
        nifti_path = os.path.join(
            self.data_dir, patient_name, f"{self.mr_sequence}.nii.gz"
        )

        if self.transform is not None:
            volume = self.transform(nifti_path)
        else:
            # Load NIfTI file directly if no transform provided
            loader = LoadImage(image_only=True)
            volume = loader(nifti_path)

        return volume, patient_name


def make_binary(image: np.ndarray, threshold: float = BINARY_THRESHOLD_GT) -> np.ndarray:
    """Convert image to binary mask using threshold.

    Args:
        image: Input image array.
        threshold: Threshold value for binarization.

    Returns:
        Binary mask with values 0.0 or 1.0.
    """
    return np.where(image > threshold, 1.0, 0.0)


def extract_slices_single(
    nifti_dataset: Dataset,
    augmentation: Optional["A.Compose"] = None
) -> Dataset:
    """Extract 2D slices from 3D volumes for single-sequence training.

    Processes each 3D volume and extracts non-empty 2D slices along
    the depth dimension.

    Args:
        nifti_dataset: Dataset of 3D volumes with shape [C, H, W, D].
        augmentation: Optional albumentations Compose for data augmentation.

    Returns:
        Dataset of 2D slices with shape [C, H, W].
    """
    all_slices: List[np.ndarray] = []

    for i in range(len(nifti_dataset)):
        volume, patient_name = nifti_dataset[i]  # Shape: [1, H, W, D]

        # Extract non-empty slices along depth dimension (axis 3)
        for k in range(volume.shape[3]):
            slice_data = volume[:, :, :, k]
            if np.sum(slice_data) > 1.0:
                # Apply augmentation (no mask for single-channel)
                slice_data = apply_augmentation(slice_data, augmentation, has_mask=False)
                all_slices.append(slice_data)

    return Dataset(all_slices)


def extract_slices_dual(
    merged_dataset: Dataset,
    has_seg: bool = True,
    augmentation: Optional["A.Compose"] = None
) -> Dataset:
    """Extract 2D slices from merged 3D volumes for multi-sequence training.

    For dual mode, ensures ALL image channels have content (not black).
    Optionally binarizes segmentation masks.

    Args:
        merged_dataset: Dataset with volumes of shape [C, H, W, D]
            where C = 2 (bravo+seg) or C = 3 (pre+gd+seg).
        has_seg: Whether last channel is segmentation mask to binarize.
        augmentation: Optional albumentations Compose for data augmentation.

    Returns:
        Dataset of 2D slices with shape [C, H, W].
    """
    all_slices: List[np.ndarray] = []

    for i in range(len(merged_dataset)):
        volume = merged_dataset[i]  # Shape: [C, H, W, D]

        # Extract slices along depth dimension
        for k in range(volume.shape[3]):
            slice_data = volume[:, :, :, k]

            if has_seg:
                # Get image channels (all except last which is seg)
                image_channels = slice_data[:-1, :, :]
                num_image_channels = image_channels.shape[0]

                # Check EACH image channel independently
                all_images_have_content = True
                for ch in range(num_image_channels):
                    channel_sum = np.sum(image_channels[ch, :, :])
                    if channel_sum <= 1.0:
                        all_images_have_content = False
                        break

                # Only keep slice if ALL image channels have content
                if all_images_have_content:
                    slice_data_copy = slice_data.copy()
                    # Apply augmentation (with mask at last channel)
                    slice_data_copy = apply_augmentation(
                        slice_data_copy, augmentation, has_mask=True, mask_channel=-1
                    )
                    # Binarize seg AFTER augmentation to avoid interpolation artifacts
                    slice_data_copy[-1, :, :] = make_binary(
                        slice_data_copy[-1, :, :], threshold=BINARY_THRESHOLD_GT
                    )
                    all_slices.append(slice_data_copy)
            else:
                # No seg mask, keep all non-empty slices
                if np.sum(slice_data) > 1.0:
                    slice_data_copy = slice_data.copy()
                    # Apply augmentation (no mask)
                    slice_data_copy = apply_augmentation(
                        slice_data_copy, augmentation, has_mask=False
                    )
                    all_slices.append(slice_data_copy)

    return Dataset(all_slices)


def extract_slices_multi_modality(
    merged_dataset: Dataset,
    augmentation: Optional["A.Compose"] = None
) -> Dataset:
    """Extract 2D slices ensuring ALL modality channels have content.

    For multi-modality VAE training where we need slices that have
    valid data across all input modalities.

    Args:
        merged_dataset: Dataset with volumes of shape [C, H, W, D]
            where C = number of modalities.
        augmentation: Optional albumentations Compose for data augmentation.

    Returns:
        Dataset of 2D slices with shape [C, H, W].
    """
    all_slices: List[np.ndarray] = []

    for i in range(len(merged_dataset)):
        volume = merged_dataset[i]  # Shape: [C, H, W, D]
        num_channels = volume.shape[0]

        # Extract slices along depth dimension
        for k in range(volume.shape[3]):
            slice_data = volume[:, :, :, k]

            # Check EACH channel independently has content
            all_channels_have_content = True
            for ch in range(num_channels):
                channel_sum = np.sum(slice_data[ch, :, :])
                if channel_sum <= 1.0:
                    all_channels_have_content = False
                    break

            # Only keep slice if ALL channels have content
            if all_channels_have_content:
                slice_data_copy = slice_data.copy()
                # Apply augmentation (no mask for multi-modality VAE)
                slice_data_copy = apply_augmentation(
                    slice_data_copy, augmentation, has_mask=False
                )
                all_slices.append(slice_data_copy)

    return Dataset(all_slices)


def merge_sequences(datasets_dict: Dict[str, NiFTIDataset]) -> Dataset:
    """Merge multiple MR sequences from same patients into single dataset.

    Concatenates volumes from different sequences along channel dimension,
    ensuring patient alignment across all sequences.

    Args:
        datasets_dict: Dictionary mapping sequence names to NiFTIDatasets.
            Example: {'t1_pre': dataset1, 't1_gd': dataset2, 'seg': dataset3}

    Returns:
        Dataset with merged volumes, shape [C, H, W, D] where C = num sequences.

    Raises:
        AssertionError: If datasets have different lengths or patient mismatch.
    """
    sequence_keys = list(datasets_dict.keys())
    num_patients = len(datasets_dict[sequence_keys[0]])

    # Verify all datasets have same length
    for key, dataset in datasets_dict.items():
        if len(dataset) != num_patients:
            raise ValueError(
                f"Dataset {key} has {len(dataset)} patients, expected {num_patients}"
            )

    merged_data: List[np.ndarray] = []

    for patient_idx in range(num_patients):
        patient_volumes: List[np.ndarray] = []
        patient_name: Optional[str] = None

        for seq_key in sequence_keys:
            volume, name = datasets_dict[seq_key][patient_idx]
            patient_volumes.append(volume)

            # Verify all sequences are from same patient
            if patient_name is None:
                patient_name = name
            elif name != patient_name:
                raise ValueError(
                    f"Patient name mismatch: {patient_name} vs {name} "
                    f"for sequence {seq_key}"
                )

        # Concatenate along channel dimension (axis 0)
        merged_volume = np.concatenate(patient_volumes, axis=0)
        merged_data.append(merged_volume)

    return Dataset(merged_data)


def create_dataloader(
    cfg: DictConfig,
    image_type: str,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True
) -> Tuple[DataLoader, Dataset]:
    """Create dataloader for single-image training (seg or bravo+seg).

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        image_type: Image type ('seg' or 'bravo').
        use_distributed: Whether to use distributed training.
        rank: Process rank for distributed training.
        world_size: Total number of processes for distributed training.
        augment: Whether to apply data augmentation.

    Returns:
        Tuple of (DataLoader, train_dataset).
    """
    data_dir = os.path.join(cfg.paths.data_dir, "train")
    image_size = cfg.model.image_size
    batch_size = cfg.training.batch_size

    # Validate modalities exist before loading
    if image_type == 'seg':
        validate_modality_exists(data_dir, 'seg')
    elif image_type == 'bravo':
        validate_modality_exists(data_dir, 'bravo')
        validate_modality_exists(data_dir, 'seg')

    transform = build_standard_transform(image_size)
    aug = build_augmentation(enabled=augment)

    if image_type == 'seg':
        # Load only segmentation masks
        seg_dataset = NiFTIDataset(
            data_dir=data_dir, mr_sequence="seg", transform=transform
        )
        train_dataset = extract_slices_single(seg_dataset, augmentation=aug)

    elif image_type == 'bravo':
        # Load bravo + seg for conditioning
        bravo_dataset = NiFTIDataset(
            data_dir=data_dir, mr_sequence="bravo", transform=transform
        )
        seg_dataset = NiFTIDataset(
            data_dir=data_dir, mr_sequence="seg", transform=transform
        )

        datasets_dict = {'bravo': bravo_dataset, 'seg': seg_dataset}
        merged = merge_sequences(datasets_dict)
        train_dataset = extract_slices_dual(merged, has_seg=True, augmentation=aug)
    else:
        raise ValueError(f"Unknown image_type: {image_type}")

    # Setup sampler
    sampler: Optional[DistributedSampler] = None
    shuffle: Optional[bool] = True

    if use_distributed:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        shuffle = None
        batch_size_per_gpu = batch_size // world_size
    else:
        batch_size_per_gpu = batch_size

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=DEFAULT_NUM_WORKERS,
        persistent_workers=DEFAULT_NUM_WORKERS > 0
    )

    return dataloader, train_dataset


def create_dual_image_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    conditioning: Optional[str],
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True
) -> Tuple[DataLoader, Dataset]:
    """Create dataloader for dual-image training (T1 pre + T1 gd).

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        image_keys: List of two sequences to train (e.g., ['t1_pre', 't1_gd']).
        conditioning: Conditioning sequence name (e.g., 'seg') or None.
        use_distributed: Whether to use distributed training.
        rank: Process rank for distributed training.
        world_size: Total number of processes for distributed training.
        augment: Whether to apply data augmentation.

    Returns:
        Tuple of (DataLoader, train_dataset).
        Batches have shape [B, 3, H, W] = [t1_pre, t1_gd, seg].

    Raises:
        ValueError: If image_keys does not contain exactly 2 items.
    """
    if len(image_keys) != 2:
        raise ValueError(f"Dual-image mode requires exactly 2 image types, got {len(image_keys)}: {image_keys}")

    data_dir = os.path.join(cfg.paths.data_dir, "train")
    image_size = cfg.model.image_size
    batch_size = cfg.training.batch_size

    # Validate all modalities exist before loading
    for key in image_keys:
        validate_modality_exists(data_dir, key)
    if conditioning:
        validate_modality_exists(data_dir, conditioning)

    transform = build_standard_transform(image_size)
    aug = build_augmentation(enabled=augment)

    # Load all required datasets
    datasets_dict: Dict[str, NiFTIDataset] = {}

    for key in image_keys:
        datasets_dict[key] = NiFTIDataset(
            data_dir=data_dir, mr_sequence=key, transform=transform
        )

    # Load conditioning (segmentation)
    if conditioning:
        datasets_dict[conditioning] = NiFTIDataset(
            data_dir=data_dir, mr_sequence=conditioning, transform=transform
        )

    # Merge all sequences
    merged = merge_sequences(datasets_dict)
    train_dataset = extract_slices_dual(merged, has_seg=(conditioning is not None), augmentation=aug)

    # Setup sampler
    sampler: Optional[DistributedSampler] = None
    shuffle: Optional[bool] = True

    if use_distributed:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        shuffle = None
        batch_size_per_gpu = batch_size // world_size
    else:
        batch_size_per_gpu = batch_size

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=DEFAULT_NUM_WORKERS,
        persistent_workers=DEFAULT_NUM_WORKERS > 0
    )

    return dataloader, train_dataset


def create_multi_modality_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    image_size: int,
    batch_size: int,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True
) -> Tuple[DataLoader, Dataset]:
    """Create dataloader for multi-modality VAE training.

    Loads multiple MR sequences as individual single-channel images and
    combines them into one dataset. Each batch contains mixed slices from
    all modalities, giving 4x more training data than single-modality training.

    Args:
        cfg: Hydra configuration with paths.
        image_keys: List of MR sequences to load (e.g., ['bravo', 'flair', 't1_pre', 't1_gd']).
        image_size: Target image size (passed explicitly for progressive training).
        batch_size: Batch size (passed explicitly for progressive training).
        use_distributed: Whether to use distributed training.
        rank: Process rank for distributed training.
        world_size: Total number of processes for distributed training.
        augment: Whether to apply data augmentation.

    Returns:
        Tuple of (DataLoader, train_dataset).
        Batches have shape [B, 1, H, W] - single channel images from mixed modalities.
    """
    data_dir = os.path.join(cfg.paths.data_dir, "train")

    # Validate all modalities exist before loading
    for key in image_keys:
        validate_modality_exists(data_dir, key)

    transform = build_standard_transform(image_size)
    aug = build_augmentation(enabled=augment)

    # Collect all slices from all modalities into one list
    all_slices: List[np.ndarray] = []

    for key in image_keys:
        dataset = NiFTIDataset(
            data_dir=data_dir, mr_sequence=key, transform=transform
        )
        # Extract 2D slices from this modality
        slices = extract_slices_single(dataset, augmentation=aug)
        all_slices.extend(list(slices))

    train_dataset = Dataset(all_slices)

    # Setup sampler
    sampler: Optional[DistributedSampler] = None
    shuffle: Optional[bool] = True

    if use_distributed:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        shuffle = None
        batch_size_per_gpu = batch_size // world_size
    else:
        batch_size_per_gpu = batch_size

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=DEFAULT_NUM_WORKERS,
        persistent_workers=DEFAULT_NUM_WORKERS > 0
    )

    return dataloader, train_dataset


# =============================================================================
# Diffusion Validation Dataloaders
# =============================================================================


def create_validation_dataloader(
    cfg: DictConfig,
    image_type: str,
    batch_size: Optional[int] = None
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create validation dataloader for single-image diffusion from val/ directory.

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        image_type: Image type ('seg' or 'bravo').
        batch_size: Optional batch size override. Defaults to training batch size.

    Returns:
        Tuple of (DataLoader, val_dataset) or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, "val")

    if not os.path.exists(val_dir):
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Validate modalities exist
    try:
        if image_type == 'seg':
            validate_modality_exists(val_dir, 'seg')
        elif image_type == 'bravo':
            validate_modality_exists(val_dir, 'bravo')
            validate_modality_exists(val_dir, 'seg')
        else:
            return None
    except ValueError as e:
        logger.warning(f"Validation directory exists but is misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    if image_type == 'seg':
        seg_dataset = NiFTIDataset(
            data_dir=val_dir, mr_sequence="seg", transform=transform
        )
        val_dataset = extract_slices_single(seg_dataset)

    elif image_type == 'bravo':
        bravo_dataset = NiFTIDataset(
            data_dir=val_dir, mr_sequence="bravo", transform=transform
        )
        seg_dataset = NiFTIDataset(
            data_dir=val_dir, mr_sequence="seg", transform=transform
        )
        datasets_dict = {'bravo': bravo_dataset, 'seg': seg_dataset}
        merged = merge_sequences(datasets_dict)
        val_dataset = extract_slices_dual(merged, has_seg=True)

    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=DEFAULT_NUM_WORKERS,
        persistent_workers=DEFAULT_NUM_WORKERS > 0
    )

    return dataloader, val_dataset


def create_dual_image_validation_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    conditioning: Optional[str] = 'seg',
    batch_size: Optional[int] = None
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create validation dataloader for dual-image diffusion from val/ directory.

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        image_keys: List of two sequences (e.g., ['t1_pre', 't1_gd']).
        conditioning: Conditioning sequence name (e.g., 'seg') or None.
        batch_size: Optional batch size override. Defaults to training batch size.

    Returns:
        Tuple of (DataLoader, val_dataset) or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, "val")

    if not os.path.exists(val_dir):
        return None

    if len(image_keys) != 2:
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Validate modalities exist
    try:
        for key in image_keys:
            validate_modality_exists(val_dir, key)
        if conditioning:
            validate_modality_exists(val_dir, conditioning)
    except ValueError as e:
        logger.warning(f"Validation directory exists but is misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    datasets_dict: Dict[str, NiFTIDataset] = {}
    for key in image_keys:
        datasets_dict[key] = NiFTIDataset(
            data_dir=val_dir, mr_sequence=key, transform=transform
        )

    if conditioning:
        datasets_dict[conditioning] = NiFTIDataset(
            data_dir=val_dir, mr_sequence=conditioning, transform=transform
        )

    merged = merge_sequences(datasets_dict)
    val_dataset = extract_slices_dual(merged, has_seg=(conditioning is not None))

    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=DEFAULT_NUM_WORKERS,
        persistent_workers=DEFAULT_NUM_WORKERS > 0
    )

    return dataloader, val_dataset


# =============================================================================
# Diffusion Test Dataloaders
# =============================================================================


def create_test_dataloader(
    cfg: DictConfig,
    image_type: str,
    batch_size: Optional[int] = None
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create test dataloader for single-image diffusion from test_new/ directory.

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        image_type: Image type ('seg' or 'bravo').
        batch_size: Optional batch size override. Defaults to training batch size.

    Returns:
        Tuple of (DataLoader, test_dataset) or None if test_new/ doesn't exist.
    """
    test_dir = os.path.join(cfg.paths.data_dir, "test_new")

    if not os.path.exists(test_dir):
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Validate modalities exist
    try:
        if image_type == 'seg':
            validate_modality_exists(test_dir, 'seg')
        elif image_type == 'bravo':
            validate_modality_exists(test_dir, 'bravo')
            validate_modality_exists(test_dir, 'seg')
        else:
            return None
    except ValueError as e:
        logger.warning(f"Test directory exists but is misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    if image_type == 'seg':
        seg_dataset = NiFTIDataset(
            data_dir=test_dir, mr_sequence="seg", transform=transform
        )
        test_dataset = extract_slices_single(seg_dataset)

    elif image_type == 'bravo':
        bravo_dataset = NiFTIDataset(
            data_dir=test_dir, mr_sequence="bravo", transform=transform
        )
        seg_dataset = NiFTIDataset(
            data_dir=test_dir, mr_sequence="seg", transform=transform
        )
        datasets_dict = {'bravo': bravo_dataset, 'seg': seg_dataset}
        merged = merge_sequences(datasets_dict)
        test_dataset = extract_slices_dual(merged, has_seg=True)

    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=DEFAULT_NUM_WORKERS,
        persistent_workers=DEFAULT_NUM_WORKERS > 0
    )

    return dataloader, test_dataset


def create_dual_image_test_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    conditioning: Optional[str] = 'seg',
    batch_size: Optional[int] = None
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create test dataloader for dual-image diffusion from test_new/ directory.

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        image_keys: List of two sequences (e.g., ['t1_pre', 't1_gd']).
        conditioning: Conditioning sequence name (e.g., 'seg') or None.
        batch_size: Optional batch size override. Defaults to training batch size.

    Returns:
        Tuple of (DataLoader, test_dataset) or None if test_new/ doesn't exist.
    """
    test_dir = os.path.join(cfg.paths.data_dir, "test_new")

    if not os.path.exists(test_dir):
        return None

    if len(image_keys) != 2:
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Validate modalities exist
    try:
        for key in image_keys:
            validate_modality_exists(test_dir, key)
        if conditioning:
            validate_modality_exists(test_dir, conditioning)
    except ValueError as e:
        logger.warning(f"Test directory exists but is misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    datasets_dict: Dict[str, NiFTIDataset] = {}
    for key in image_keys:
        datasets_dict[key] = NiFTIDataset(
            data_dir=test_dir, mr_sequence=key, transform=transform
        )

    if conditioning:
        datasets_dict[conditioning] = NiFTIDataset(
            data_dir=test_dir, mr_sequence=conditioning, transform=transform
        )

    merged = merge_sequences(datasets_dict)
    test_dataset = extract_slices_dual(merged, has_seg=(conditioning is not None))

    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=DEFAULT_NUM_WORKERS,
        persistent_workers=DEFAULT_NUM_WORKERS > 0
    )

    return dataloader, test_dataset


# =============================================================================
# VAE Dataloaders
# =============================================================================


def create_vae_dataloader(
    cfg: DictConfig,
    modality: str,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True
) -> Tuple[DataLoader, Dataset]:
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
    batch_size = cfg.training.batch_size

    # Validate modalities exist before loading
    if modality == 'dual':
        for key in ['t1_pre', 't1_gd']:
            validate_modality_exists(data_dir, key)
    else:
        validate_modality_exists(data_dir, modality)

    transform = build_standard_transform(image_size)
    aug = build_augmentation(enabled=augment)

    if modality == 'dual':
        # Dual mode: t1_pre + t1_gd as 2 channels (NO seg)
        image_keys = ['t1_pre', 't1_gd']
        datasets_dict: Dict[str, NiFTIDataset] = {}
        for key in image_keys:
            datasets_dict[key] = NiFTIDataset(
                data_dir=data_dir, mr_sequence=key, transform=transform
            )
        merged = merge_sequences(datasets_dict)
        # Extract slices ensuring both channels have content (no seg binarization)
        train_dataset = extract_slices_dual(merged, has_seg=False, augmentation=aug)
    else:
        # Single modality: 1 channel
        nifti_dataset = NiFTIDataset(
            data_dir=data_dir, mr_sequence=modality, transform=transform
        )
        train_dataset = extract_slices_single(nifti_dataset, augmentation=aug)

    # Setup sampler
    sampler: Optional[DistributedSampler] = None
    shuffle: Optional[bool] = True

    if use_distributed:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        shuffle = None
        batch_size_per_gpu = batch_size // world_size
    else:
        batch_size_per_gpu = batch_size

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=DEFAULT_NUM_WORKERS,
        persistent_workers=DEFAULT_NUM_WORKERS > 0
    )

    return dataloader, train_dataset


def create_vae_validation_dataloader(
    cfg: DictConfig,
    modality: str,
    batch_size: Optional[int] = None
) -> Optional[Tuple[DataLoader, Dataset]]:
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
    val_dir = os.path.join(cfg.paths.data_dir, "val")

    # Check if validation directory exists
    if not os.path.exists(val_dir):
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Validate modalities exist in val directory
    try:
        if modality == 'dual':
            for key in ['t1_pre', 't1_gd']:
                validate_modality_exists(val_dir, key)
        else:
            validate_modality_exists(val_dir, modality)
    except ValueError as e:
        logger.warning(f"Validation directory exists but is misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    if modality == 'dual':
        # Dual mode: t1_pre + t1_gd as 2 channels (NO seg)
        image_keys = ['t1_pre', 't1_gd']
        datasets_dict: Dict[str, NiFTIDataset] = {}
        for key in image_keys:
            datasets_dict[key] = NiFTIDataset(
                data_dir=val_dir, mr_sequence=key, transform=transform
            )
        merged = merge_sequences(datasets_dict)
        val_dataset = extract_slices_dual(merged, has_seg=False)
    else:
        # Single modality: 1 channel
        nifti_dataset = NiFTIDataset(
            data_dir=val_dir, mr_sequence=modality, transform=transform
        )
        val_dataset = extract_slices_single(nifti_dataset)

    # Validation loader: shuffle enabled for diverse batch sampling
    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=DEFAULT_NUM_WORKERS,
        persistent_workers=DEFAULT_NUM_WORKERS > 0
    )

    return dataloader, val_dataset


def create_multi_modality_validation_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    image_size: int,
    batch_size: int
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create validation dataloader for multi-modality VAE training.

    Loads data from the val/ subdirectory if it exists. Returns None if
    val/ directory doesn't exist.

    Args:
        cfg: Hydra configuration with paths.
        image_keys: List of MR sequences to load.
        image_size: Target image size.
        batch_size: Batch size.

    Returns:
        Tuple of (DataLoader, val_dataset) or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, "val")

    # Check if validation directory exists
    if not os.path.exists(val_dir):
        return None

    # Validate modalities exist in val directory
    try:
        for key in image_keys:
            validate_modality_exists(val_dir, key)
    except ValueError as e:
        logger.warning(f"Validation directory exists but is misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    # Collect all slices from all modalities into one list
    all_slices: List[np.ndarray] = []

    for key in image_keys:
        dataset = NiFTIDataset(
            data_dir=val_dir, mr_sequence=key, transform=transform
        )
        slices = extract_slices_single(dataset)
        all_slices.extend(list(slices))

    val_dataset = Dataset(all_slices)

    # Validation loader: shuffle enabled for diverse batch sampling
    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=DEFAULT_NUM_WORKERS,
        persistent_workers=DEFAULT_NUM_WORKERS > 0
    )

    return dataloader, val_dataset


def create_vae_test_dataloader(
    cfg: DictConfig,
    modality: str,
    batch_size: Optional[int] = None
) -> Optional[Tuple[DataLoader, Dataset]]:
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
    test_dir = os.path.join(cfg.paths.data_dir, "test_new")

    # Check if test directory exists
    if not os.path.exists(test_dir):
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Validate modalities exist in test directory
    try:
        if modality == 'dual':
            for key in ['t1_pre', 't1_gd']:
                validate_modality_exists(test_dir, key)
        else:
            validate_modality_exists(test_dir, modality)
    except ValueError as e:
        logger.warning(f"Test directory exists but is misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    if modality == 'dual':
        # Dual mode: t1_pre + t1_gd as 2 channels (NO seg)
        image_keys = ['t1_pre', 't1_gd']
        datasets_dict: Dict[str, NiFTIDataset] = {}
        for key in image_keys:
            datasets_dict[key] = NiFTIDataset(
                data_dir=test_dir, mr_sequence=key, transform=transform
            )
        merged = merge_sequences(datasets_dict)
        test_dataset = extract_slices_dual(merged, has_seg=False)
    else:
        # Single modality: 1 channel
        nifti_dataset = NiFTIDataset(
            data_dir=test_dir, mr_sequence=modality, transform=transform
        )
        test_dataset = extract_slices_single(nifti_dataset)

    # Test loader: shuffle for diverse visualization samples
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=DEFAULT_NUM_WORKERS,
        persistent_workers=DEFAULT_NUM_WORKERS > 0
    )

    return dataloader, test_dataset


def create_multi_modality_test_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    image_size: int,
    batch_size: int
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create test dataloader for multi-modality VAE evaluation.

    Loads data from the test_new/ subdirectory if it exists. Returns None if
    test_new/ directory doesn't exist.

    Args:
        cfg: Hydra configuration with paths.
        image_keys: List of MR sequences to load.
        image_size: Target image size.
        batch_size: Batch size.

    Returns:
        Tuple of (DataLoader, test_dataset) or None if test_new/ doesn't exist.
    """
    test_dir = os.path.join(cfg.paths.data_dir, "test_new")

    # Check if test directory exists
    if not os.path.exists(test_dir):
        return None

    # Validate modalities exist in test directory
    try:
        for key in image_keys:
            validate_modality_exists(test_dir, key)
    except ValueError as e:
        logger.warning(f"Test directory exists but is misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    # Collect all slices from all modalities into one list
    all_slices: List[np.ndarray] = []

    for key in image_keys:
        dataset = NiFTIDataset(
            data_dir=test_dir, mr_sequence=key, transform=transform
        )
        slices = extract_slices_single(dataset)
        all_slices.extend(list(slices))

    test_dataset = Dataset(all_slices)

    # Test loader: shuffle for diverse visualization samples
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=DEFAULT_NUM_WORKERS,
        persistent_workers=DEFAULT_NUM_WORKERS > 0
    )

    return dataloader, test_dataset
