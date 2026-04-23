"""3D paired volume dataset for restoration training.

Supports three modes:
- **Full volume**: Returns complete 256×256×160 paired volumes
- **Patch-based**: Random 3D crops, many patches per volume
- **2D slice**: Random axial slices with 2D crops

For patch/slice modes, all volumes are cached in RAM at init to avoid
repeated NIfTI loading (the main bottleneck for small-patch training).

Two degradation sources:
- **precomputed** (default): expects `degraded_*.nii.gz` files alongside
  `clean_bravo.nii.gz` in each patient dir.
- **synthetic**: generates degraded volumes on-the-fly from the clean
  volume via random Gaussian blur + additive noise. Requires only
  `bravo.nii.gz` (or `clean_bravo.nii.gz`); no paired files needed.

Precomputed directory structure:
    data_dir/
        patient_001/
            clean_bravo.nii.gz
            seg.nii.gz
            degraded_001.nii.gz
            ...

Synthetic directory structure (same as BRAVO training data):
    data_dir/
        patient_001/
            bravo.nii.gz       # (or clean_bravo.nii.gz)
            seg.nii.gz
"""
import logging
import os
import random
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandBiasFieldd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
)

from .volume_3d import Base3DVolumeDataset

logger = logging.getLogger(__name__)


def build_restoration_augmentation(level: str = 'basic') -> Callable:
    """Build augmentation transform for (clean, degraded, seg) triples.

    Levels:
      basic  — flips (×3 axes) + axial 90° rotations. Original behaviour.
      medium — basic + affine (small rot/translate/scale) + intensity jitter
               (contrast, brightness). Applied consistently to all keys so the
               (clean, degraded) pairing is preserved.
      heavy  — medium + MRI bias field + mild noise. Best anti-overfitting
               option for small-dataset restoration training. Noise is paired
               so the network sees degraded-plus-noise with the same noise
               profile on clean — avoids breaking the pair semantics.

    Intensity transforms are applied WITH THE SAME RANDOM DRAW to all keys
    (via Compose) so brightness shifts match clean/degraded consistently.
    """
    keys = ['image', 'degraded', 'seg']
    transforms: list = [
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        RandRotate90d(keys=keys, prob=0.5, spatial_axes=(1, 2)),
    ]
    if level in ('medium', 'heavy'):
        # Intensity transforms only on image+degraded (seg is binary, must stay
        # binary — applying intensity ops on it would destroy the mask).
        img_keys = ['image', 'degraded']
        transforms.extend([
            RandAffined(
                keys=keys,
                prob=0.3,
                rotate_range=(0.05, 0.05, 0.05),    # ±~3°
                translate_range=(5, 5, 5),          # ±5 voxels
                scale_range=(0.05, 0.05, 0.05),     # 0.95–1.05
                mode=['bilinear', 'bilinear', 'nearest'],
                padding_mode='border',
            ),
            RandScaleIntensityd(keys=img_keys, prob=0.3, factors=0.15),       # ±15%
            RandAdjustContrastd(keys=img_keys, prob=0.3, gamma=(0.85, 1.15)),
        ])
    if level == 'heavy':
        img_keys = ['image', 'degraded']
        transforms.extend([
            RandBiasFieldd(keys=img_keys, prob=0.2, coeff_range=(0.0, 0.2)),
            RandGaussianNoised(keys=img_keys, prob=0.2, mean=0.0, std=0.01),
        ])
    return Compose(transforms)


def _gaussian_kernel_1d(sigma: float, radius: int | None = None) -> torch.Tensor:
    """1D Gaussian kernel for separable 3D convolution."""
    if radius is None:
        radius = max(1, int(3.0 * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _gaussian_smooth_3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 3D Gaussian smooth applied independently along D, H, W.

    Much cheaper than a full 3D kernel (O(3k) vs O(k^3)).

    Args:
        x: [C, D, H, W]
        sigma: Gaussian std in voxel units
    Returns:
        Same shape, smoothed.
    """
    kernel = _gaussian_kernel_1d(sigma)
    radius = (kernel.numel() - 1) // 2
    kernel = kernel.to(x.device)
    c = x.shape[0]
    # [C, 1, 1, 1, K] for conv along W (last axis)
    out = x.unsqueeze(0)  # [1, C, D, H, W]
    for axis in (2, 3, 4):
        # Reshape kernel to the right axis
        shape = [1, 1, 1, 1, 1]
        shape[axis] = kernel.numel()
        k = kernel.view(*shape).expand(c, 1, *shape[2:])
        pad = [0, 0, 0, 0, 0, 0]
        # F.conv3d pad layout: (W_left, W_right, H_top, H_bot, D_front, D_back)
        # axis 2 (D) -> indices 4,5 ; axis 3 (H) -> 2,3 ; axis 4 (W) -> 0,1
        pad_idx = {2: (4, 5), 3: (2, 3), 4: (0, 1)}[axis]
        pad[pad_idx[0]] = radius
        pad[pad_idx[1]] = radius
        out = F.pad(out, pad, mode='replicate')
        out = F.conv3d(out, k, groups=c)
    return out.squeeze(0)


def _downsample_upsample_3d(x: torch.Tensor, factor: float = 2.0) -> torch.Tensor:
    """Downsample then upsample — kills some HF content in a more aggressive
    way than pure blur. Used occasionally during synthetic degradation.
    """
    x5 = x.unsqueeze(0)
    down = F.interpolate(x5, scale_factor=1.0 / factor, mode='trilinear', align_corners=False)
    up = F.interpolate(down, size=x.shape[1:], mode='trilinear', align_corners=False)
    return up.squeeze(0)


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
        degradation_cfg: dict | None = None,
        augmentation_level: str = 'basic',
    ) -> None:
        augmentation = (
            build_restoration_augmentation(level=augmentation_level)
            if (augment and not slice_2d) else None
        )
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
        self.degradation_cfg = degradation_cfg or {}
        self.degradation_type = self.degradation_cfg.get('type', 'precomputed')

        # Discover patients and degraded variants
        self.patients: list[str] = []
        self._degraded_files: dict[str, list[str]] = {}
        self._clean_filenames: dict[str, str] = {}   # patient -> clean filename

        for patient in sorted(os.listdir(data_dir)):
            patient_dir = os.path.join(data_dir, patient)
            if not os.path.isdir(patient_dir):
                continue

            # Accept either naming convention
            clean_name = None
            for candidate in ("clean_bravo.nii.gz", "bravo.nii.gz"):
                if os.path.exists(os.path.join(patient_dir, candidate)):
                    clean_name = candidate
                    break
            seg_path = os.path.join(patient_dir, "seg.nii.gz")
            if clean_name is None or not os.path.exists(seg_path):
                continue

            if self.degradation_type == 'precomputed':
                degraded = sorted([
                    f for f in os.listdir(patient_dir)
                    if f.startswith("degraded_") and f.endswith(".nii.gz")
                ])
                if not degraded:
                    continue
                self._degraded_files[patient] = [
                    os.path.join(patient_dir, f) for f in degraded
                ]

            self.patients.append(patient)
            self._clean_filenames[patient] = clean_name

        if not self.patients:
            raise ValueError(
                f"No valid restoration inputs found in {data_dir} "
                f"(degradation_type={self.degradation_type})"
            )

        if self.degradation_type == 'precomputed':
            total_degraded = sum(len(v) for v in self._degraded_files.values())
            deg_str = f"{total_degraded} precomputed degraded variants"
        else:
            deg_str = f"synthetic degradation ({self.degradation_cfg})"
        if self.slice_2d:
            crop_str = f" crop {patch_size_2d}" if patch_size_2d else ""
            mode_str = f"2D slices{crop_str}, {samples_per_epoch}/epoch"
        elif self.patch_size:
            mode_str = f"patch {patch_size}, {samples_per_epoch}/epoch"
        else:
            mode_str = "full volume"

        logger.info(
            f"Restoration3DDataset: {len(self.patients)} patients, "
            f"{deg_str} ({mode_str})"
        )

        # Pre-load all volumes into RAM for patch/slice mode
        self._cache: list[dict[str, torch.Tensor]] = []
        if self._use_cache:
            logger.info("Caching all volumes in RAM...")
            for patient in self.patients:
                patient_dir = os.path.join(data_dir, patient)
                clean = self._load_volume(os.path.join(patient_dir, self._clean_filenames[patient]))
                seg = self._load_seg(patient_dir)
                if seg is None:
                    seg = torch.zeros_like(clean)

                if self.degradation_type == 'precomputed':
                    for deg_path in self._degraded_files[patient]:
                        degraded = self._load_volume(deg_path)
                        self._cache.append({
                            'image': clean,
                            'degraded': degraded,
                            'seg': seg,
                        })
                else:
                    # Synthetic: cache only clean; degradation generated per __getitem__
                    self._cache.append({
                        'image': clean,
                        'degraded': None,
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
                'degraded': pair['degraded']
                            if pair['degraded'] is not None
                            else self._apply_synthetic_degradation(pair['image']),
                'seg': pair['seg'],
                'patient': '',
            }

            if self.slice_2d:
                return self._random_slice_2d(result)

            # 3D patch: crop FIRST (small patch), then augment (cheap on small tensor)
            if self.patch_size is not None:
                result = self._random_crop(result)
            result = self._apply_augmentation(result)
            return result

        # Full volume mode (no cache)
        patient = self.patients[idx]
        patient_dir = os.path.join(self.data_dir, patient)

        clean = self._load_volume(os.path.join(patient_dir, self._clean_filenames[patient]))
        if self.degradation_type == 'precomputed':
            degraded_path = random.choice(self._degraded_files[patient])
            degraded = self._load_volume(degraded_path)
        else:
            degraded = self._apply_synthetic_degradation(clean)
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

    def _apply_synthetic_degradation(self, clean: torch.Tensor) -> torch.Tensor:
        """Generate a synthetic degraded volume from a clean one.

        Mimics the MSE-posterior-mean-blur phenotype: Gaussian blur + mild
        additive noise. Optionally a downsample-upsample pass for lower-frequency
        information loss.

        Args:
            clean: [C, D, H, W] float32 tensor in [0, 1].
        Returns:
            Same shape, with random per-call blur sigma and noise sigma.
        """
        blur_range = self.degradation_cfg.get('blur_sigma', [0.8, 1.6])
        noise_range = self.degradation_cfg.get('noise_sigma', [0.005, 0.02])
        ds_prob = float(self.degradation_cfg.get('downsample_upsample_prob', 0.0))

        sigma = random.uniform(float(blur_range[0]), float(blur_range[1]))
        noise_sigma = random.uniform(float(noise_range[0]), float(noise_range[1]))

        # 3D Gaussian smooth via separable 1D kernels (much cheaper than a full 3D kernel).
        degraded = _gaussian_smooth_3d(clean, sigma=sigma)

        if ds_prob > 0 and random.random() < ds_prob:
            degraded = _downsample_upsample_3d(degraded)

        degraded = degraded + noise_sigma * torch.randn_like(degraded)
        return degraded

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
