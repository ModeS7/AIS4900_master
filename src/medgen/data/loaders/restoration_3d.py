"""3D paired volume dataset for restoration training.

Loads paired (degraded, clean) volumes from pre-computed SDEdit degradation
pairs. Each patient has one clean volume and multiple degraded variants;
one degraded variant is randomly selected per __getitem__ call.

Augmentation (flips, rotations) is applied identically to all volumes
(clean, degraded, seg) via MONAI dict-based transforms.

Directory structure expected:
    data_dir/
        patient_001/
            clean_bravo.nii.gz
            seg.nii.gz
            degraded_001.nii.gz
            degraded_002.nii.gz
            ...
        patient_002/
            ...
"""
import logging
import os
import random
from collections.abc import Callable
from typing import Any

import torch
from monai.transforms import Compose, RandFlipd, RandRotate90d

from .volume_3d import Base3DVolumeDataset

logger = logging.getLogger(__name__)


def build_restoration_augmentation() -> Callable:
    """Build augmentation that applies identical transforms to all three keys.

    Returns:
        MONAI Compose transform operating on dict with 'image', 'degraded', 'seg'.
    """
    keys = ['image', 'degraded', 'seg']
    return Compose([
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        RandRotate90d(keys=keys, prob=0.5, spatial_axes=(1, 2)),
    ])


class Restoration3DDataset(Base3DVolumeDataset):
    """Paired (degraded, clean) volume dataset for restoration training.

    Each __getitem__ returns a dict with:
        image: clean volume [1, D, H, W]
        degraded: degraded volume [1, D, H, W] (random variant)
        seg: binary seg mask [1, D, H, W]
        patient: patient ID string

    Args:
        data_dir: Directory with patient subdirs containing clean/degraded/seg.
        height: Target height.
        width: Target width.
        pad_depth_to: Target depth after padding.
        pad_mode: Padding mode.
        slice_step: Subsample slices.
        augment: Whether to apply augmentation.
    """

    def __init__(
        self,
        data_dir: str,
        height: int = 256,
        width: int = 256,
        pad_depth_to: int = 160,
        pad_mode: str = 'replicate',
        slice_step: int = 1,
        augment: bool = True,
    ) -> None:
        augmentation = build_restoration_augmentation() if augment else None
        super().__init__(
            data_dir=data_dir,
            height=height,
            width=width,
            pad_depth_to=pad_depth_to,
            pad_mode=pad_mode,
            slice_step=slice_step,
            load_seg=True,
            augmentation=augmentation,
        )

        # Discover patients and their degraded variants
        self.patients: list[str] = []
        self._degraded_files: dict[str, list[str]] = {}

        for patient in sorted(os.listdir(data_dir)):
            patient_dir = os.path.join(data_dir, patient)
            if not os.path.isdir(patient_dir):
                continue

            clean_path = os.path.join(patient_dir, "clean_bravo.nii.gz")
            seg_path = os.path.join(patient_dir, "seg.nii.gz")
            if not os.path.exists(clean_path) or not os.path.exists(seg_path):
                continue

            # Find all degraded variants
            degraded = sorted([
                f for f in os.listdir(patient_dir)
                if f.startswith("degraded_") and f.endswith(".nii.gz")
            ])

            if not degraded:
                logger.warning(f"No degraded files for {patient}, skipping")
                continue

            self.patients.append(patient)
            self._degraded_files[patient] = [
                os.path.join(patient_dir, f) for f in degraded
            ]

        if not self.patients:
            raise ValueError(f"No valid restoration pairs found in {data_dir}")

        total_degraded = sum(len(v) for v in self._degraded_files.values())
        logger.info(
            f"Restoration3DDataset: {len(self.patients)} patients, "
            f"{total_degraded} degraded variants total "
            f"(avg {total_degraded / len(self.patients):.1f} per patient)"
        )

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        patient = self.patients[idx]
        patient_dir = os.path.join(self.data_dir, patient)

        # Load clean volume
        clean_path = os.path.join(patient_dir, "clean_bravo.nii.gz")
        clean = self._load_volume(clean_path)

        # Randomly pick one degraded variant
        degraded_path = random.choice(self._degraded_files[patient])
        degraded = self._load_volume(degraded_path)

        # Load seg mask
        seg = self._load_seg(patient_dir)
        if seg is None:
            # Fallback: zeros if seg missing (shouldn't happen)
            seg = torch.zeros_like(clean)

        result: dict[str, Any] = {
            'image': clean,
            'degraded': degraded,
            'seg': seg,
            'patient': patient,
        }

        # Apply augmentation (identical transforms to all three)
        result = self._apply_augmentation(result)

        return result

    def _apply_augmentation(self, result: dict) -> dict:
        """Apply augmentation to clean, degraded, and seg simultaneously."""
        if self.augmentation is None:
            return result

        aug_result = self.augmentation(result)

        for key in ('image', 'degraded', 'seg'):
            if key in aug_result:
                result[key] = aug_result[key].contiguous()

        return result
