"""
Slice extraction and sequence merging utilities.

This module provides functions for extracting 2D slices from 3D NIfTI volumes
and merging multiple MR sequences from the same patients.
"""
from typing import Dict, List, Optional

import numpy as np
from monai.data import Dataset

from medgen.core.constants import BINARY_THRESHOLD_GT
from medgen.data.augmentation import apply_augmentation
from medgen.data.dataset import NiFTIDataset

try:
    import albumentations as A
except ImportError:
    A = None  # type: ignore


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
        volume, _ = nifti_dataset[i]  # Shape: [1, H, W, D]

        # Convert to numpy if tensor
        if hasattr(volume, 'numpy'):
            volume = volume.numpy()

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

        # Convert to numpy if tensor
        if hasattr(volume, 'numpy'):
            volume = volume.numpy()

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


def extract_slices_single_with_seg(
    image_dataset: Dataset,
    seg_dataset: Dataset,
    augmentation: Optional["A.Compose"] = None
) -> Dataset:
    """Extract 2D slices with paired segmentation masks for regional metrics.

    Each slice is returned as a tuple (image, seg) where:
    - image: [1, H, W] single-channel image
    - seg: [1, H, W] binary segmentation mask

    Args:
        image_dataset: Dataset of 3D image volumes with shape [1, H, W, D].
        seg_dataset: Dataset of 3D seg volumes with shape [1, H, W, D].
        augmentation: Optional albumentations Compose for data augmentation.

    Returns:
        Dataset of tuples (image_slice, seg_slice).
    """
    all_slices: List[tuple] = []

    if len(image_dataset) != len(seg_dataset):
        raise ValueError(
            f"Image dataset ({len(image_dataset)}) and seg dataset ({len(seg_dataset)}) "
            "must have same number of patients"
        )

    for i in range(len(image_dataset)):
        image_volume, image_name = image_dataset[i]  # Shape: [1, H, W, D]
        seg_volume, seg_name = seg_dataset[i]  # Shape: [1, H, W, D]

        # Convert to numpy if tensor
        if hasattr(image_volume, 'numpy'):
            image_volume = image_volume.numpy()
        if hasattr(seg_volume, 'numpy'):
            seg_volume = seg_volume.numpy()

        # Verify same patient
        if image_name != seg_name:
            raise ValueError(f"Patient mismatch: {image_name} vs {seg_name}")

        # Extract non-empty slices along depth dimension (axis 3)
        for k in range(image_volume.shape[3]):
            image_slice = image_volume[:, :, :, k]
            seg_slice = seg_volume[:, :, :, k]

            if np.sum(image_slice) > 1.0:
                if augmentation is not None:
                    # Stack for joint augmentation, then split
                    combined = np.concatenate([image_slice, seg_slice], axis=0)
                    combined = apply_augmentation(
                        combined, augmentation, has_mask=True, mask_channel=-1
                    )
                    image_slice = combined[:1, :, :]
                    seg_slice = combined[1:, :, :]

                # Binarize seg
                seg_slice = make_binary(seg_slice, threshold=BINARY_THRESHOLD_GT)
                all_slices.append((image_slice, seg_slice))

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
        ValueError: If datasets have different lengths or patient mismatch.
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
