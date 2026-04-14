"""3D paired volume dataset for restoration training.

Supports three modes:
- **Full volume**: Returns complete 256×256×160 paired volumes
- **Patch-based**: Random 3D crops, many patches per volume
- **2D slice**: Random axial slices with 2D crops

For patch/slice modes, all volumes are cached in RAM at init to avoid
repeated NIfTI loading (the main bottleneck for small-patch training).

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

    For patch/slice modes, volumes are pre-loaded into RAM at init time
    to avoid the NIfTI I/O bottleneck. With 105 patients × 4 degradations
    × ~40MB per volume ≈ 17GB RAM — fits easily in 48-128GB nodes.

    Args:
        data_dir: Directory with patient subdirs.
        height, width, pad_depth_to: Volume dimensions.
        pad_mode, slice_step: Volume processing params.
        augment: Apply augmentation.
        patch_size: (D, H, W) for random 3D crops. None = full volume.
        samples_per_epoch: Patches/slices per epoch (patch/slice mode).
        slice_2d: Extract random 2D axial slices.
        patch_size_2d: (H, W) for 2D crops from slices.
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
        slice_2d: bool = False,
        patch_size_2d: tuple[int, int] | None = None,
    ) -> None:
        augmentation = build_restoration_augmentation() if (augment and not slice_2d) else None
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
        self.slice_2d = slice_2d
        self.patch_size_2d = patch_size_2d
        self._use_cache = patch_size is not None or slice_2d

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
        if self.slice_2d:
            crop_str = f" crop {patch_size_2d}" if patch_size_2d else ""
            mode_str = f"2D slices{crop_str}, {samples_per_epoch}/epoch"
        elif self.patch_size:
            mode_str = f"patch {patch_size}, {samples_per_epoch}/epoch"
        else:
            mode_str = "full volume"

        logger.info(
            f"Restoration3DDataset: {len(self.patients)} patients, "
            f"{total_degraded} degraded variants ({mode_str})"
        )

        # Pre-load all volumes into RAM for patch/slice mode
        self._cache: list[dict[str, torch.Tensor]] = []
        if self._use_cache:
            logger.info("Caching all volumes in RAM...")
            for patient in self.patients:
                patient_dir = os.path.join(data_dir, patient)
                clean = self._load_volume(os.path.join(patient_dir, "clean_bravo.nii.gz"))
                seg = self._load_seg(patient_dir)
                if seg is None:
                    seg = torch.zeros_like(clean)

                for deg_path in self._degraded_files[patient]:
                    degraded = self._load_volume(deg_path)
                    self._cache.append({
                        'image': clean,
                        'degraded': degraded,
                        'seg': seg,
                    })

            logger.info(f"Cached {len(self._cache)} volume pairs "
                        f"({len(self._cache) * clean.nelement() * 4 / 1e9:.1f}GB)")

    def __len__(self) -> int:
        if self._use_cache:
            return self.samples_per_epoch
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._use_cache:
            # Pick random cached pair
            pair = self._cache[random.randint(0, len(self._cache) - 1)]
            result = {
                'image': pair['image'],
                'degraded': pair['degraded'],
                'seg': pair['seg'],
                'patient': '',
            }

            if self.slice_2d:
                return self._random_slice_2d(result)

            # 3D patch: augment then crop
            result = self._apply_augmentation(result)
            if self.patch_size is not None:
                result = self._random_crop(result)
            return result

        # Full volume mode (no cache)
        patient = self.patients[idx]
        patient_dir = os.path.join(self.data_dir, patient)

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
        result = self._apply_augmentation(result)
        return result

    def _random_slice_2d(self, result: dict[str, Any]) -> dict[str, Any]:
        """Extract random axial slice and optional 2D crop. Returns [C, H, W]."""
        _, d, h, w = result['image'].shape
        max_slice = max(1, d - 10)
        s = random.randint(0, max_slice - 1)

        for key in ('image', 'degraded', 'seg'):
            result[key] = result[key][:, s, :, :]

        if self.patch_size_2d is not None:
            ph, pw = self.patch_size_2d
            _, h, w = result['image'].shape
            h0 = random.randint(0, max(0, h - ph))
            w0 = random.randint(0, max(0, w - pw))
            for key in ('image', 'degraded', 'seg'):
                result[key] = result[key][:, h0:h0 + ph, w0:w0 + pw].contiguous()

        if random.random() > 0.5:
            for key in ('image', 'degraded', 'seg'):
                result[key] = result[key].flip(-1)
        if random.random() > 0.5:
            for key in ('image', 'degraded', 'seg'):
                result[key] = result[key].flip(-2)
        if random.random() > 0.5:
            for key in ('image', 'degraded', 'seg'):
                result[key] = torch.rot90(result[key], k=1, dims=(-2, -1))

        return result

    def _random_crop(self, result: dict[str, Any]) -> dict[str, Any]:
        """Random 3D crop of all volumes at the same location."""
        pd, ph, pw = self.patch_size  # type: ignore[misc]
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
