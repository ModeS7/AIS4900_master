"""
Data loading and processing utilities for diffusion model training.

This module provides dataset classes and utility functions for loading,
processing, and preparing medical image data (NIfTI format) for training
diffusion models on brain MRI sequences.
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
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


def extract_slices_single(nifti_dataset: Dataset) -> Dataset:
    """Extract 2D slices from 3D volumes for single-sequence training.

    Processes each 3D volume and extracts non-empty 2D slices along
    the depth dimension.

    Args:
        nifti_dataset: Dataset of 3D volumes with shape [C, H, W, D].

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
                all_slices.append(slice_data)

    return Dataset(all_slices)


def extract_slices_dual(
    merged_dataset: Dataset,
    has_seg: bool = True
) -> Dataset:
    """Extract 2D slices from merged 3D volumes for multi-sequence training.

    For dual mode, ensures ALL image channels have content (not black).
    Optionally binarizes segmentation masks.

    Args:
        merged_dataset: Dataset with volumes of shape [C, H, W, D]
            where C = 2 (bravo+seg) or C = 3 (pre+gd+seg).
        has_seg: Whether last channel is segmentation mask to binarize.

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
                    seg_channel = slice_data_copy[-1, :, :]
                    slice_data_copy[-1, :, :] = make_binary(
                        seg_channel, threshold=BINARY_THRESHOLD_GT
                    )
                    all_slices.append(slice_data_copy)
            else:
                # No seg mask, keep all non-empty slices
                if np.sum(slice_data) > 1.0:
                    all_slices.append(slice_data)

    return Dataset(all_slices)


def extract_slices_multi_modality(merged_dataset: Dataset) -> Dataset:
    """Extract 2D slices ensuring ALL modality channels have content.

    For multi-modality VAE training where we need slices that have
    valid data across all input modalities.

    Args:
        merged_dataset: Dataset with volumes of shape [C, H, W, D]
            where C = number of modalities.

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
                all_slices.append(slice_data.copy())

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
    world_size: int = 1
) -> Tuple[DataLoader, Dataset]:
    """Create dataloader for single-image training (seg or bravo+seg).

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        image_type: Image type ('seg' or 'bravo').
        use_distributed: Whether to use distributed training.
        rank: Process rank for distributed training.
        world_size: Total number of processes for distributed training.

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

    transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ToTensor(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        Resize(spatial_size=(image_size, image_size, -1))
    ])

    if image_type == 'seg':
        # Load only segmentation masks
        seg_dataset = NiFTIDataset(
            data_dir=data_dir, mr_sequence="seg", transform=transform
        )
        train_dataset = extract_slices_single(seg_dataset)

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
        train_dataset = extract_slices_dual(merged, has_seg=True)
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
        num_workers=DEFAULT_NUM_WORKERS
    )

    return dataloader, train_dataset


def create_dual_image_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    conditioning: Optional[str],
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1
) -> Tuple[DataLoader, Dataset]:
    """Create dataloader for dual-image training (T1 pre + T1 gd).

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        image_keys: List of two sequences to train (e.g., ['t1_pre', 't1_gd']).
        conditioning: Conditioning sequence name (e.g., 'seg') or None.
        use_distributed: Whether to use distributed training.
        rank: Process rank for distributed training.
        world_size: Total number of processes for distributed training.

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

    transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ToTensor(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        Resize(spatial_size=(image_size, image_size, -1))
    ])

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
    train_dataset = extract_slices_dual(merged, has_seg=(conditioning is not None))

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
        num_workers=DEFAULT_NUM_WORKERS
    )

    return dataloader, train_dataset


def create_multi_modality_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    image_size: int,
    batch_size: int,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1
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

    Returns:
        Tuple of (DataLoader, train_dataset).
        Batches have shape [B, 1, H, W] - single channel images from mixed modalities.
    """
    data_dir = os.path.join(cfg.paths.data_dir, "train")

    # Validate all modalities exist before loading
    for key in image_keys:
        validate_modality_exists(data_dir, key)

    transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ToTensor(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        Resize(spatial_size=(image_size, image_size, -1))
    ])

    # Collect all slices from all modalities into one list
    all_slices: List[np.ndarray] = []

    for key in image_keys:
        dataset = NiFTIDataset(
            data_dir=data_dir, mr_sequence=key, transform=transform
        )
        # Extract 2D slices from this modality
        slices = extract_slices_single(dataset)
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
        num_workers=DEFAULT_NUM_WORKERS
    )

    return dataloader, train_dataset


def create_vae_dataloader(
    cfg: DictConfig,
    modality: str,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1
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

    transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ToTensor(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        Resize(spatial_size=(image_size, image_size, -1))
    ])

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
        train_dataset = extract_slices_dual(merged, has_seg=False)
    else:
        # Single modality: 1 channel
        nifti_dataset = NiFTIDataset(
            data_dir=data_dir, mr_sequence=modality, transform=transform
        )
        train_dataset = extract_slices_single(nifti_dataset)

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
        num_workers=DEFAULT_NUM_WORKERS
    )

    return dataloader, train_dataset
