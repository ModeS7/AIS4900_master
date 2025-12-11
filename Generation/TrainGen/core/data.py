"""
Data loading and processing utilities for diffusion model training.

This module provides dataset classes and utility functions for loading,
processing, and preparing medical image data (NIfTI format) for training
diffusion models on brain MRI sequences.
"""
import os
from typing import Any, Dict, List, Optional, Tuple, Union

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
from torch.utils.data.distributed import DistributedSampler

from config import PathConfig


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
            volume = nifti_path

        return volume, patient_name


def make_binary(image: np.ndarray, threshold: float = 0.01) -> np.ndarray:
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
                        seg_channel, threshold=0.01
                    )
                    all_slices.append(slice_data_copy)
            else:
                # No seg mask, keep all non-empty slices
                if np.sum(slice_data) > 1.0:
                    all_slices.append(slice_data)

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
        assert len(dataset) == num_patients, (
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
            else:
                assert name == patient_name, (
                    f"Patient name mismatch: {patient_name} vs {name} "
                    f"for sequence {seq_key}"
                )

        # Concatenate along channel dimension (axis 0)
        merged_volume = np.concatenate(patient_volumes, axis=0)
        merged_data.append(merged_volume)

    return Dataset(merged_data)


def create_dataloader(
    path_config: PathConfig,
    image_type: str,
    image_size: int,
    batch_size: int,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1
) -> Tuple[DataLoader, Dataset]:
    """Create dataloader for single-image training (seg or bravo+seg).

    Args:
        path_config: Path configuration instance.
        image_type: Image type ('seg' or 'bravo').
        image_size: Target image size for resizing.
        batch_size: Total batch size (divided by world_size if distributed).
        use_distributed: Whether to use distributed training.
        rank: Process rank for distributed training.
        world_size: Total number of processes for distributed training.

    Returns:
        Tuple of (DataLoader, train_dataset).
    """
    data_dir = str(path_config.brainmet_train_dir)

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
        num_workers=4
    )

    return dataloader, train_dataset


def create_dual_image_dataloader(
    path_config: PathConfig,
    image_keys: List[str],
    conditioning: Optional[str],
    image_size: int,
    batch_size: int,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1
) -> Tuple[DataLoader, Dataset]:
    """Create dataloader for dual-image training (T1 pre + T1 gd).

    Args:
        path_config: Path configuration instance.
        image_keys: List of two sequences to train (e.g., ['t1_pre', 't1_gd']).
        conditioning: Conditioning sequence name (e.g., 'seg') or None.
        image_size: Target image size for resizing.
        batch_size: Total batch size (divided by world_size if distributed).
        use_distributed: Whether to use distributed training.
        rank: Process rank for distributed training.
        world_size: Total number of processes for distributed training.

    Returns:
        Tuple of (DataLoader, train_dataset).
        Batches have shape [B, 3, H, W] = [t1_pre, t1_gd, seg].

    Raises:
        AssertionError: If image_keys does not contain exactly 2 items.
    """
    assert len(image_keys) == 2, "Dual-image mode requires exactly 2 image types"

    data_dir = str(path_config.brainmet_train_dir)

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
        num_workers=4
    )

    return dataloader, train_dataset
