"""3D volume dataset classes.

This module contains all dataset classes for loading 3D NIfTI volumes:
- Base3DVolumeDataset: Abstract base class with shared utilities
- Volume3DDataset: Single-modality 3D volumes
- DualVolume3DDataset: Dual-modality (t1_pre + t1_gd)
- MultiModality3DDataset: Multi-modality pooled samples
- SingleModality3DDatasetWithSeg: Single modality with seg masks
- SingleModality3DDatasetWithSegDropout: With CFG dropout support

These classes handle:
- NIfTI file loading via MONAI transforms
- Depth padding (replicate or constant mode)
- Optional 3D augmentation (flips, rotations)
- Segmentation mask loading for regional metrics
"""
import logging
import os
from collections.abc import Callable

import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    RandFlipd,
    RandRotate90d,
    Resize,
    ScaleIntensity,
)
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def build_3d_transform(height: int, width: int) -> Compose:
    """Build transform pipeline for 3D volumes.

    Args:
        height: Target height.
        width: Target width.

    Returns:
        MONAI Compose transform.
    """
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(channel_dim="no_channel"),  # NIfTI has no channel dim
        ScaleIntensity(minv=0.0, maxv=1.0),
        Resize(spatial_size=(height, width, -1)),  # Preserve depth
    ])


def build_3d_augmentation(seg_mode: bool = False, include_seg: bool = False) -> Callable:
    """Build 3D augmentation pipeline using MONAI transforms.

    Args:
        seg_mode: If True, augmentations are for binary segmentation masks only.
                  Uses nearest-neighbor interpolation to preserve binary values.
        include_seg: If True, augmentations apply to both 'image' and 'seg' keys.
                     Used for conditional modes (bravo, dual) where both must be
                     augmented consistently.

    Returns:
        MONAI Compose transform that operates on dict with 'image' key
        (and optionally 'seg' key if include_seg=True).
    """
    # Determine which keys to augment
    if include_seg:
        keys = ['image', 'seg']
    else:
        keys = ['image']

    # For seg masks, we need to preserve binary values
    # RandFlip and RandRotate90 don't interpolate, so they're safe for binary masks
    transforms = [
        # Random flips along each axis (p=0.5 each)
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),  # Flip along depth
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),  # Flip along height
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),  # Flip along width
        # Random 90 rotations in axial plane (H, W)
        RandRotate90d(keys=keys, prob=0.5, spatial_axes=(1, 2)),
    ]

    return Compose(transforms)


class Base3DVolumeDataset(Dataset):
    """Base class for 3D volume datasets with shared utilities.

    Provides common functionality:
    - Transform setup for 3D volumes
    - Depth padding (replicate or constant mode)
    - Volume loading and processing
    - Optional 3D augmentation (flips, rotations)
    """

    def __init__(
        self,
        data_dir: str,
        height: int = 256,
        width: int = 256,
        pad_depth_to: int = 160,
        pad_mode: str = 'replicate',
        slice_step: int = 1,
        load_seg: bool = False,
        augmentation: Callable | None = None,
    ) -> None:
        # Validate data directory exists
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"Data directory not found: {data_dir}")

        # Validate parameter ranges
        if height <= 0 or width <= 0:
            raise ValueError(f"height and width must be > 0, got height={height}, width={width}")
        if pad_depth_to <= 0:
            raise ValueError(f"pad_depth_to must be > 0, got {pad_depth_to}")
        if slice_step <= 0:
            raise ValueError(f"slice_step must be > 0, got {slice_step}")
        if pad_mode not in ('replicate', 'constant', 'reflect'):
            raise ValueError(f"pad_mode must be 'replicate', 'constant', or 'reflect', got '{pad_mode}'")

        self.data_dir = data_dir
        self.height = height
        self.width = width
        self.pad_depth_to = pad_depth_to
        self.pad_mode = pad_mode
        self.slice_step = slice_step
        self.load_seg = load_seg
        self.augmentation = augmentation

        # Track padding statistics for logging
        self._padding_stats = {'volumes_padded': 0, 'total_slices_padded': 0}

        self.transform = build_3d_transform(height, width)

    def _pad_depth(self, volume: torch.Tensor) -> torch.Tensor:
        """Pad volume depth to target size.

        Args:
            volume: Tensor of shape [C, D, H, W].

        Returns:
            Padded tensor with depth >= pad_depth_to.

        Raises:
            ValueError: If volume is not 4D.
        """
        if volume.ndim != 4:
            raise ValueError(
                f"Expected 4D volume [C, D, H, W], got {volume.ndim}D with shape {volume.shape}"
            )
        current_depth = volume.shape[1]
        if current_depth < self.pad_depth_to:
            pad_total = self.pad_depth_to - current_depth
            # Track padding statistics
            self._padding_stats['volumes_padded'] += 1
            self._padding_stats['total_slices_padded'] += pad_total
            logger.debug(
                f"Padding volume from {current_depth} to {self.pad_depth_to} slices "
                f"(+{pad_total} slices, mode={self.pad_mode})"
            )
            if self.pad_mode == 'replicate':
                last_slice = volume[:, -1:, :, :]
                padding = last_slice.repeat(1, pad_total, 1, 1)
                volume = torch.cat([volume, padding], dim=1)
            else:
                volume = F.pad(volume, (0, 0, 0, 0, 0, pad_total), mode='constant', value=0)
        return volume

    def get_padding_summary(self) -> str:
        """Get summary of depth padding applied to loaded volumes.

        Returns:
            Human-readable summary string.
        """
        n_padded = self._padding_stats['volumes_padded']
        total_slices = self._padding_stats['total_slices_padded']
        if n_padded == 0:
            return "No volumes required depth padding"
        avg_padding = total_slices / n_padded
        return f"{n_padded} volumes padded (avg +{avg_padding:.1f} slices each)"

    def _load_volume(self, nifti_path: str) -> torch.Tensor:
        """Load and preprocess a 3D volume from NIfTI file.

        Args:
            nifti_path: Path to NIfTI file.

        Returns:
            Tensor of shape [C, D, H, W] with depth padding applied.
        """
        volume = self.transform(nifti_path)

        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume).float()

        # MONAI loads as [C, H, W, D], we need [C, D, H, W] for 3D conv
        volume = volume.permute(0, 3, 1, 2)

        # Subsample slices if slice_step > 1
        if self.slice_step > 1:
            volume = volume[:, ::self.slice_step, :, :]

        # Pad depth
        volume = self._pad_depth(volume)

        return volume

    def _apply_augmentation(self, result: dict) -> dict:
        """Apply augmentation if configured.

        Args:
            result: Dict containing 'image' tensor and optionally 'seg'.

        Returns:
            Augmented dict (in-place modification).
        """
        if self.augmentation is None:
            return result

        # MONAI transforms expect dict format
        aug_result = self.augmentation(result)

        # Ensure tensors are contiguous after transforms
        if 'image' in aug_result:
            result['image'] = aug_result['image'].contiguous()
        if 'seg' in aug_result:
            result['seg'] = aug_result['seg'].contiguous()

        return result

    def _load_seg(self, patient_dir: str) -> torch.Tensor | None:
        """Load and preprocess segmentation mask if it exists.

        Args:
            patient_dir: Patient directory path.

        Returns:
            Binarized seg tensor of shape [C, D, H, W] or None.
        """
        seg_path = os.path.join(patient_dir, "seg.nii.gz")
        if not os.path.exists(seg_path):
            return None

        seg = self._load_volume(seg_path)
        seg = (seg > 0.5).float()  # Binarize
        return seg


class Volume3DDataset(Base3DVolumeDataset):
    """Dataset that loads single-modality 3D volumes with depth padding.

    Args:
        data_dir: Directory containing patient subdirectories.
        modality: MR sequence name (e.g., 'bravo', 't1_pre').
        height: Target height dimension.
        width: Target width dimension.
        pad_depth_to: Target depth after padding.
        pad_mode: Padding mode ('replicate' or 'constant').
        slice_step: Take every nth slice (1=all, 2=every 2nd, 3=every 3rd).
        load_seg: Whether to load segmentation masks for regional metrics.
        augmentation: Optional MONAI augmentation transform.
    """

    def __init__(
        self,
        data_dir: str,
        modality: str,
        height: int = 256,
        width: int = 256,
        pad_depth_to: int = 160,
        pad_mode: str = 'replicate',
        slice_step: int = 1,
        load_seg: bool = False,
        augmentation: Callable | None = None,
    ) -> None:
        super().__init__(data_dir, height, width, pad_depth_to, pad_mode, slice_step, load_seg, augmentation)
        self.modality = modality

        # List patient directories
        self.patients = sorted([
            p for p in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, p))
        ])

        if not self.patients:
            raise ValueError(f"No patient directories found in {data_dir}")

        logger.info(f"Found {len(self.patients)} patients for {modality}")

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict:
        patient = self.patients[idx]
        patient_dir = os.path.join(self.data_dir, patient)
        nifti_path = os.path.join(patient_dir, f"{self.modality}.nii.gz")

        volume = self._load_volume(nifti_path)
        result = {'image': volume, 'patient': patient}

        if self.load_seg:
            seg = self._load_seg(patient_dir)
            if seg is not None:
                result['seg'] = seg

        # Apply augmentation if configured
        result = self._apply_augmentation(result)

        return result


class DualVolume3DDataset(Base3DVolumeDataset):
    """Dataset that loads dual-modality 3D volumes (t1_pre + t1_gd).

    Args:
        data_dir: Directory containing patient subdirectories.
        height: Target height dimension.
        width: Target width dimension.
        pad_depth_to: Target depth after padding.
        pad_mode: Padding mode ('replicate' or 'constant').
        slice_step: Take every nth slice (1=all, 2=every 2nd, 3=every 3rd).
        load_seg: Whether to load segmentation masks for regional metrics.
        augmentation: Optional MONAI augmentation transform.
    """

    def __init__(
        self,
        data_dir: str,
        height: int = 256,
        width: int = 256,
        pad_depth_to: int = 160,
        pad_mode: str = 'replicate',
        slice_step: int = 1,
        load_seg: bool = False,
        augmentation: Callable | None = None,
    ) -> None:
        super().__init__(data_dir, height, width, pad_depth_to, pad_mode, slice_step, load_seg, augmentation)

        # List patient directories
        self.patients = sorted([
            p for p in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, p))
        ])

        if not self.patients:
            raise ValueError(f"No patient directories found in {data_dir}")

        logger.info(f"Found {len(self.patients)} patients for dual mode")

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict:
        patient = self.patients[idx]
        patient_dir = os.path.join(self.data_dir, patient)

        # Load both modalities using base class helper
        t1_pre = self._load_volume(os.path.join(patient_dir, "t1_pre.nii.gz"))
        t1_gd = self._load_volume(os.path.join(patient_dir, "t1_gd.nii.gz"))

        # Stack as 2 channels: [2, D, H, W]
        volume = torch.cat([t1_pre, t1_gd], dim=0)

        result = {'image': volume, 'patient': patient}

        if self.load_seg:
            seg = self._load_seg(patient_dir)
            if seg is not None:
                result['seg'] = seg

        # Apply augmentation if configured
        result = self._apply_augmentation(result)

        return result


class MultiModality3DDataset(Base3DVolumeDataset):
    """Dataset that loads 3D volumes from multiple modalities.

    Pools all modalities (bravo, flair, t1_pre, t1_gd) as separate samples.
    Each sample is a single-channel volume.

    Args:
        data_dir: Directory containing patient subdirectories.
        image_keys: List of modality names to load.
        height: Target height dimension.
        width: Target width dimension.
        pad_depth_to: Target depth after padding.
        pad_mode: Padding mode ('replicate' or 'constant').
        slice_step: Take every nth slice (1=all, 2=every 2nd, 3=every 3rd).
        load_seg: Whether to load segmentation masks for regional metrics.
        augmentation: Optional MONAI augmentation transform.
    """

    def __init__(
        self,
        data_dir: str,
        image_keys: list,
        height: int = 256,
        width: int = 256,
        pad_depth_to: int = 160,
        pad_mode: str = 'replicate',
        slice_step: int = 1,
        load_seg: bool = False,
        augmentation: Callable | None = None,
    ) -> None:
        super().__init__(data_dir, height, width, pad_depth_to, pad_mode, slice_step, load_seg, augmentation)
        self.image_keys = image_keys

        # List patient directories
        self.patients = sorted([
            p for p in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, p))
        ])

        if not self.patients:
            raise ValueError(f"No patient directories found in {data_dir}")

        # Build index: (patient_idx, modality)
        self.samples = []
        for p_idx, patient in enumerate(self.patients):
            patient_dir = os.path.join(data_dir, patient)
            for modality in image_keys:
                nifti_path = os.path.join(patient_dir, f"{modality}.nii.gz")
                if os.path.exists(nifti_path):
                    self.samples.append((p_idx, modality))

        logger.info(f"Found {len(self.samples)} volumes from {len(self.patients)} patients "
                    f"({len(image_keys)} modalities)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        p_idx, modality = self.samples[idx]
        patient = self.patients[p_idx]
        patient_dir = os.path.join(self.data_dir, patient)
        nifti_path = os.path.join(patient_dir, f"{modality}.nii.gz")

        volume = self._load_volume(nifti_path)
        result = {'image': volume, 'patient': patient, 'modality': modality}

        if self.load_seg:
            seg = self._load_seg(patient_dir)
            if seg is not None:
                result['seg'] = seg

        # Apply augmentation if configured
        result = self._apply_augmentation(result)

        return result


class SingleModality3DDatasetWithSeg(Base3DVolumeDataset):
    """3D Dataset that loads single modality with segmentation masks.

    Used for per-modality validation with regional metrics (tumor tracking).
    Returns volume and segmentation mask pairs.

    Args:
        data_dir: Directory containing patient subdirectories.
        modality: MR sequence name (e.g., 'bravo', 't1_pre').
        height: Target height dimension.
        width: Target width dimension.
        pad_depth_to: Target depth after padding.
        pad_mode: Padding mode ('replicate' or 'constant').
        slice_step: Take every nth slice (1=all, 2=every 2nd, 3=every 3rd).
    """

    def __init__(
        self,
        data_dir: str,
        modality: str,
        height: int = 256,
        width: int = 256,
        pad_depth_to: int = 160,
        pad_mode: str = 'replicate',
        slice_step: int = 1,
    ) -> None:
        super().__init__(data_dir, height, width, pad_depth_to, pad_mode, slice_step, load_seg=True)
        self.modality = modality

        # Build index of patients that have modality (track which have seg)
        self.samples = []
        patients = sorted([
            p for p in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, p))
        ])

        for patient in patients:
            patient_dir = os.path.join(data_dir, patient)
            modality_path = os.path.join(patient_dir, f"{modality}.nii.gz")
            seg_path = os.path.join(patient_dir, "seg.nii.gz")

            if os.path.exists(modality_path):
                has_seg = os.path.exists(seg_path)
                self.samples.append((patient, has_seg))

        if not self.samples:
            raise ValueError(f"No patients with {modality} found in {data_dir}")

        n_with_seg = sum(1 for _, has_seg in self.samples if has_seg)
        logger.info(f"SingleModality3DDatasetWithSeg: {len(self.samples)} volumes for {modality}, "
                    f"{n_with_seg} with segmentation masks")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        patient, has_seg = self.samples[idx]
        patient_dir = os.path.join(self.data_dir, patient)
        modality_path = os.path.join(patient_dir, f"{self.modality}.nii.gz")

        volume = self._load_volume(modality_path)
        result = {'image': volume, 'patient': patient, 'modality': self.modality}

        if has_seg:
            seg = self._load_seg(patient_dir)
            if seg is not None:
                result['seg'] = seg

        return result


class SingleModality3DDatasetWithSegDropout(Base3DVolumeDataset):
    """3D Dataset that loads single modality with seg mask and CFG dropout.

    Used for 3D bravo mode training where bravo is conditioned on seg mask.
    Supports classifier-free guidance dropout (randomly zeroing seg mask).

    Args:
        data_dir: Directory containing patient subdirectories.
        modality: MR sequence name (e.g., 'bravo', 't1_pre').
        height: Target height dimension.
        width: Target width dimension.
        pad_depth_to: Target depth after padding.
        pad_mode: Padding mode ('replicate' or 'constant').
        slice_step: Take every nth slice (1=all, 2=every 2nd, 3=every 3rd).
        cfg_dropout_prob: Probability of zeroing seg mask for CFG (default: 0.0).
        augmentation: Optional MONAI augmentation transform.
    """

    def __init__(
        self,
        data_dir: str,
        modality: str,
        height: int = 256,
        width: int = 256,
        pad_depth_to: int = 160,
        pad_mode: str = 'replicate',
        slice_step: int = 1,
        cfg_dropout_prob: float = 0.0,
        augmentation: Callable | None = None,
    ) -> None:
        super().__init__(data_dir, height, width, pad_depth_to, pad_mode, slice_step, load_seg=True, augmentation=augmentation)
        self.modality = modality
        self.cfg_dropout_prob = cfg_dropout_prob

        # Build index of patients that have both modality and seg
        self.samples = []
        patients = sorted([
            p for p in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, p))
        ])

        for patient in patients:
            patient_dir = os.path.join(data_dir, patient)
            modality_path = os.path.join(patient_dir, f"{modality}.nii.gz")
            seg_path = os.path.join(patient_dir, "seg.nii.gz")

            # For conditioning mode, require BOTH modality and seg
            if os.path.exists(modality_path) and os.path.exists(seg_path):
                self.samples.append(patient)

        if not self.samples:
            raise ValueError(f"No patients with both {modality} and seg found in {data_dir}")

        logger.info(f"SingleModality3DDatasetWithSegDropout: {len(self.samples)} volumes for {modality}, "
                    f"cfg_dropout_prob={cfg_dropout_prob}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        patient = self.samples[idx]
        patient_dir = os.path.join(self.data_dir, patient)
        modality_path = os.path.join(patient_dir, f"{self.modality}.nii.gz")

        volume = self._load_volume(modality_path)
        seg = self._load_seg(patient_dir)

        result = {'image': volume, 'seg': seg, 'patient': patient, 'modality': self.modality}

        # Apply augmentation if configured (to both image and seg together)
        result = self._apply_augmentation(result)

        # CFG dropout: randomly zero out seg mask
        if self.cfg_dropout_prob > 0 and torch.rand(1).item() < self.cfg_dropout_prob:
            result['seg'] = torch.zeros_like(result['seg'])

        return result
