"""
NiFTI dataset class and core transform utilities.

This module provides the NiFTIDataset class for loading NIfTI medical images
and utility functions for building transform pipelines.
"""
import os
from typing import Any

from monai.data import Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    Resize,
    ScaleIntensity,
    ToTensor,
)


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
        EnsureChannelFirst(channel_dim="no_channel"),
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
        ValueError: If data directory does not exist, is empty, or modality file not found.
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")

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
        transform: Compose | None = None
    ) -> None:
        self.data_dir: str = data_dir
        self.data: list[str] = sorted(
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        )
        self.mr_sequence: str = mr_sequence
        self.transform: Compose | None = transform
        # Cache loader to avoid recreating on every __getitem__ call
        self._loader: LoadImage = LoadImage(image_only=True)

    def __len__(self) -> int:
        """Return number of patients in dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Any, str]:
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
            volume = self._loader(nifti_path)

        return volume, patient_name
