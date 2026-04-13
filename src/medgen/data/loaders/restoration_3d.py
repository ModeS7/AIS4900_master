"""3D paired volume dataset for restoration training.

Supports two modes:
- **Full volume**: Returns complete 256×256×160 paired volumes (original behavior)
- **Patch-based**: Random 3D crops from paired volumes, much more training samples

Patch-based training matches IR-SDE paper (Luo et al., ICML 2023) which uses
128×128 random crops with batch_size=16.

Directory structure expected:
    data_dir/
        patient_001/
            clean_bravo.nii.gz
            seg.nii.gz
            degraded_001.nii.gz
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
    """Build augmentation for all three keys (image, degraded, seg)."""
    keys = ['image', 'degraded', 'seg']
    return Compose([
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        RandRotate90d(keys=keys, prob=0.5, spatial_axes=(1, 2)),
    ])


class Restoration3DDataset(Base3DVolumeDataset):
    """Paired (degraded, clean) volume dataset for restoration training.

    Full-volume mode: returns complete volumes, __len__ = num_patients.
    Patch mode: random 3D crops, __len__ = samples_per_epoch (configurable).

    Args:
        data_dir: Directory with patient subdirs.
        height: Target height.
        width: Target width.
        pad_depth_to: Target depth.
        pad_mode: Padding mode.
        slice_step: Subsample slices.
        augment: Apply augmentation.
        patch_size: Tuple (D, H, W) for random 3D crops. None = full volume.
        samples_per_epoch: Number of random patches per epoch (only for patch mode).
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
        patch_size: tuple[int, int, int] | None = None,
        samples_per_epoch: int = 10000,
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

        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch

        # Discover patients and degraded variants
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

            degraded = sorted([
                f for f in os.listdir(patient_dir)
                if f.startswith("degraded_") and f.endswith(".nii.gz")
            ])
            if not degraded:
                continue

            self.patients.append(patient)
            self._degraded_files[patient] = [
                os.path.join(patient_dir, f) for f in degraded
            ]

        if not self.patients:
            raise ValueError(f"No valid restoration pairs found in {data_dir}")

        total_degraded = sum(len(v) for v in self._degraded_files.values())
        mode_str = f"patch {patch_size}, {samples_per_epoch}/epoch" if patch_size else "full volume"
        logger.info(
            f"Restoration3DDataset: {len(self.patients)} patients, "
            f"{total_degraded} degraded variants ({mode_str})"
        )

    def __len__(self) -> int:
        if self.patch_size is not None:
            return self.samples_per_epoch
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Pick random patient (patch mode) or sequential (full volume mode)
        if self.patch_size is not None:
            p_idx = random.randint(0, len(self.patients) - 1)
        else:
            p_idx = idx

        patient = self.patients[p_idx]
        patient_dir = os.path.join(self.data_dir, patient)

        # Load volumes
        clean = self._load_volume(os.path.join(patient_dir, "clean_bravo.nii.gz"))
        degraded_path = random.choice(self._degraded_files[patient])
        degraded = self._load_volume(degraded_path)
        seg = self._load_seg(patient_dir)
        if seg is None:
            seg = torch.zeros_like(clean)

        result: dict[str, Any] = {
            'image': clean,
            'degraded': degraded,
            'seg': seg,
            'patient': patient,
        }

        # Apply augmentation first (on full volume)
        result = self._apply_augmentation(result)

        # Then crop patch if in patch mode
        if self.patch_size is not None:
            result = self._random_crop(result)

        return result

    def _random_crop(self, result: dict[str, Any]) -> dict[str, Any]:
        """Random 3D crop of all volumes at the same location."""
        pd, ph, pw = self.patch_size  # type: ignore[misc]
        # Volumes are [C, D, H, W]
        _, d, h, w = result['image'].shape

        d0 = random.randint(0, max(0, d - pd))
        h0 = random.randint(0, max(0, h - ph))
        w0 = random.randint(0, max(0, w - pw))

        for key in ('image', 'degraded', 'seg'):
            result[key] = result[key][:, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw].contiguous()

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
