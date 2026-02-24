"""Downstream segmentation data loading.

Provides dataloaders for three training scenarios:
- baseline: Real data only (control)
- synthetic: Generated data only
- mixed: Real + synthetic data with configurable ratio

Supports both 2D and 3D segmentation.

Reuses infrastructure from medgen.data:
- NiFTIDataset for volume loading
- build_standard_transform / build_3d_transform for transforms
- create_dataloader for DataLoader creation
"""
import logging
import os
from dataclasses import dataclass
from typing import Any

import albumentations as A
import nibabel as nib
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from medgen.data.dataset import NiFTIDataset, build_standard_transform, validate_modality_exists
from medgen.data.loaders.common import DataLoaderConfig, create_dataloader
from medgen.data.loaders.volume_3d import build_3d_transform
from medgen.data.utils import make_binary

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


@dataclass
class SegmentationConfig:
    """Configuration extracted from Hydra config for segmentation data loading."""
    modality: str | list[str]
    image_size: int
    volume_depth: int
    batch_size: int
    augment: bool
    augmentation_strength: str
    real_dir: str
    synthetic_dir: str | None
    synthetic_ratio: float

    @classmethod
    def from_cfg(cls, cfg: DictConfig, spatial_dims: int, split: str = 'train') -> 'SegmentationConfig':
        """Extract segmentation config from Hydra config."""
        return cls(
            modality=cfg.data.get('modality', 'bravo'),
            image_size=cfg.model.image_size,
            volume_depth=cfg.volume.get('pad_depth_to', 160) if spatial_dims == 3 else 160,
            batch_size=cfg.training.get('batch_size_3d', 2) if spatial_dims == 3 else cfg.training.batch_size,
            augment=cfg.data.get('augment', True) and split == 'train',
            augmentation_strength=cfg.data.get('augmentation_strength', 'standard'),
            real_dir=cfg.data.real_dir,
            synthetic_dir=cfg.data.get('synthetic_dir'),
            synthetic_ratio=cfg.data.get('synthetic_ratio', 0.5),
        )


def _to_tensor(data: Any) -> torch.Tensor:
    """Convert numpy array or tensor to float tensor."""
    if isinstance(data, torch.Tensor):
        return data.float()
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    return torch.from_numpy(np.array(data)).float()


def _binarize_seg(seg: torch.Tensor) -> torch.Tensor:
    """Binarize segmentation mask."""
    return _to_tensor(make_binary(seg.numpy()))


def _build_transform(spatial_dims: int, image_size: int):
    """Build transform based on spatial dimensions."""
    if spatial_dims == 2:
        return build_standard_transform(image_size)
    return build_3d_transform(image_size, image_size)


# =============================================================================
# Augmentation Pipelines
# =============================================================================


def build_seg_downstream_augmentation_2d(strength: str = 'standard') -> A.Compose | None:
    """Build 2D augmentation pipeline for downstream segmentation.

    Uses albumentations with ``additional_targets`` for paired image-mask transforms.

    Args:
        strength: 'light' (flips only) or 'standard' (full pipeline).

    Returns:
        Albumentations Compose, or None if strength is invalid.
    """
    if strength == 'light':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ], additional_targets={'mask': 'mask'})

    # Standard: spatial + intensity
    return A.Compose([
        # Spatial (applied identically to image and mask)
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=0.5,
        ),
        # Intensity (image only — mask is unaffected by default)
        A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3,
        ),
    ], additional_targets={'mask': 'mask'})


def build_seg_downstream_augmentation_3d(strength: str = 'standard'):
    """Build 3D augmentation pipeline for downstream segmentation.

    Uses MONAI dict transforms operating on ``{'image': ..., 'seg': ...}``.

    Args:
        strength: 'light' (flips only) or 'standard' (full pipeline).

    Returns:
        MONAI Compose pipeline.
    """
    from monai.transforms import (
        Compose,
        RandAdjustContrastd,
        RandAffined,
        RandFlipd,
        RandGaussianNoised,
        RandRotate90d,
    )

    keys = ['image', 'seg']

    if strength == 'light':
        return Compose([
            RandFlipd(keys=keys, spatial_axis=0, prob=0.5),
            RandFlipd(keys=keys, spatial_axis=1, prob=0.5),
            RandFlipd(keys=keys, spatial_axis=2, prob=0.5),
            RandRotate90d(keys=keys, spatial_axes=(1, 2), prob=0.5),
        ])

    # Standard: spatial + intensity
    return Compose([
        RandFlipd(keys=keys, spatial_axis=0, prob=0.5),
        RandFlipd(keys=keys, spatial_axis=1, prob=0.5),
        RandFlipd(keys=keys, spatial_axis=2, prob=0.5),
        RandRotate90d(keys=keys, spatial_axes=(1, 2), prob=0.5),
        RandAffined(
            keys=keys,
            rotate_range=(0.26, 0.26, 0.26),  # ~15 degrees
            scale_range=(0.1, 0.1, 0.1),       # ±10%
            mode=('bilinear', 'nearest'),       # bilinear for image, nearest for seg
            padding_mode='zeros',
            prob=0.5,
        ),
        # Intensity (image only)
        RandGaussianNoised(keys=['image'], std=0.03, prob=0.3),
        RandAdjustContrastd(keys=['image'], gamma=(0.9, 1.1), prob=0.3),
    ])


def _apply_augmentation_2d(
    image: torch.Tensor,
    seg: torch.Tensor,
    aug: A.Compose,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D albumentations augmentation to image-seg pair.

    Args:
        image: [C, H, W] tensor.
        seg: [1, H, W] tensor.
        aug: Albumentations Compose with ``additional_targets={'mask': 'mask'}``.

    Returns:
        Augmented (image, seg) tensors.
    """
    # Transpose to HWC for albumentations
    img_np = image.permute(1, 2, 0).numpy()  # [H, W, C]
    seg_np = seg[0].numpy()  # [H, W]

    result = aug(image=img_np, mask=seg_np)

    img_out = torch.from_numpy(result['image']).permute(2, 0, 1).float()
    seg_out = torch.from_numpy((result['mask'] > 0.5).astype(np.float32)).unsqueeze(0)

    return img_out, seg_out


def _apply_augmentation_3d(
    image: torch.Tensor,
    seg: torch.Tensor,
    aug,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 3D MONAI augmentation to image-seg pair.

    Args:
        image: [C, D, H, W] tensor.
        seg: [1, D, H, W] tensor.
        aug: MONAI Compose pipeline.

    Returns:
        Augmented (image, seg) tensors.
    """
    data = {'image': image, 'seg': seg}
    result = aug(data)
    img_out = result['image'].float()
    seg_out = (result['seg'] > 0.5).float()
    return img_out, seg_out


# =============================================================================
# Datasets
# =============================================================================


class SegmentationDataset(Dataset):
    """Dataset for paired image-segmentation loading.

    Loads images and corresponding segmentation masks from NIfTI files.
    Returns dict format: {'image': [C, H, W], 'seg': [1, H, W]} where C is num modalities.

    For 2D: Extracts tumor-positive slices from 3D volumes.
    For 3D: Returns full volumes with depth padding.

    Args:
        data_dir: Directory containing patient subdirectories.
        modality: Input modality name(s). Can be a single string (e.g., 'bravo')
            or a list of modalities (e.g., ['t1_pre', 't1_gd']) for multi-channel input.
        image_size: Target image size (2D: H=W, 3D: H=W).
        spatial_dims: 2 for 2D slices, 3 for 3D volumes.
        augment: Whether to apply data augmentation.
        augmentation_strength: 'light' (flips only) or 'standard' (full pipeline).
        volume_depth: Target depth for 3D volumes.
    """

    def __init__(
        self,
        data_dir: str,
        modality: str | list[str] = 'bravo',
        image_size: int = 256,
        spatial_dims: int = 2,
        augment: bool = False,
        augmentation_strength: str = 'standard',
        volume_depth: int = 160,
    ) -> None:
        self.data_dir = data_dir
        self.modalities = [modality] if isinstance(modality, str) else list(modality)
        self.image_size = image_size
        self.spatial_dims = spatial_dims
        self.augment = augment
        self.volume_depth = volume_depth

        # Validate modalities exist
        for mod in self.modalities:
            validate_modality_exists(data_dir, mod)
        validate_modality_exists(data_dir, 'seg')

        # Build transform
        self.transform = _build_transform(spatial_dims, image_size)

        # Build augmentation pipeline
        self._aug = None
        if augment:
            if spatial_dims == 2:
                self._aug = build_seg_downstream_augmentation_2d(augmentation_strength)
            else:
                self._aug = build_seg_downstream_augmentation_3d(augmentation_strength)

        # Load datasets using NiFTIDataset
        self._image_datasets = {
            mod: NiFTIDataset(data_dir, mod, self.transform)
            for mod in self.modalities
        }
        self._seg_dataset = NiFTIDataset(data_dir, 'seg', self.transform)
        self.patients = self._seg_dataset.data

        # For 2D: build slice index mapping (tumor-positive only)
        self.slice_indices = self._build_slice_indices() if spatial_dims == 2 else None

        logger.info(
            f"SegmentationDataset: {len(self.patients)} patients, "
            f"modalities={self.modalities}, spatial_dims={spatial_dims}, "
            f"augment={augment}, strength={augmentation_strength}"
        )

    def _build_slice_indices(self) -> list[tuple[int, int]]:
        """Build mapping from linear index to (patient_idx, slice_idx).

        Only includes slices with tumor pixels (positive examples).
        """
        indices = []
        for patient_idx in range(len(self._seg_dataset)):
            seg_volume, _ = self._seg_dataset[patient_idx]
            seg_np = seg_volume.numpy() if isinstance(seg_volume, torch.Tensor) else seg_volume

            for slice_idx in range(seg_np.shape[-1]):
                if seg_np[..., slice_idx].sum() > 0:
                    indices.append((patient_idx, slice_idx))

        logger.info(f"Built slice indices: {len(indices)} tumor-positive slices")
        return indices

    def __len__(self) -> int:
        if self.spatial_dims == 2:
            return len(self.slice_indices)
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.spatial_dims == 2:
            return self._get_slice(idx)
        return self._get_volume(idx)

    def _load_modalities(self, patient_idx: int) -> torch.Tensor:
        """Load and concatenate all modalities for a patient."""
        images = [
            _to_tensor(self._image_datasets[mod][patient_idx][0])
            for mod in self.modalities
        ]
        return torch.cat(images, dim=0)

    def _get_slice(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a 2D slice."""
        patient_idx, slice_idx = self.slice_indices[idx]

        image = self._load_modalities(patient_idx)  # [C, H, W, D]
        seg, _ = self._seg_dataset[patient_idx]
        seg = _to_tensor(seg)

        # Extract slice
        image_slice = image[..., slice_idx]  # [C, H, W]
        seg_slice = _binarize_seg(seg[..., slice_idx])  # [1, H, W]

        if self._aug is not None:
            image_slice, seg_slice = _apply_augmentation_2d(image_slice, seg_slice, self._aug)

        return {'image': image_slice, 'seg': seg_slice}

    def _get_volume(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a 3D volume."""
        image = self._load_modalities(idx)  # [C, H, W, D]
        seg, _ = self._seg_dataset[idx]
        seg = _binarize_seg(_to_tensor(seg))

        # Transpose to [C, D, H, W] for 3D convolutions
        image = image.permute(0, 3, 1, 2)
        seg = seg.permute(0, 3, 1, 2)

        # Pad depth
        image = self._pad_depth(image)
        seg = self._pad_depth(seg)

        if self._aug is not None:
            image, seg = _apply_augmentation_3d(image, seg, self._aug)

        return {'image': image, 'seg': seg}

    def _pad_depth(self, volume: torch.Tensor) -> torch.Tensor:
        """Pad volume depth to target size using replication."""
        current_depth = volume.shape[1]  # [C, D, H, W]
        if current_depth < self.volume_depth:
            pad_total = self.volume_depth - current_depth
            padding = volume[:, -1:, :, :].repeat(1, pad_total, 1, 1)
            volume = torch.cat([volume, padding], dim=1)
        return volume


class SyntheticDataset(Dataset):
    """Dataset for loading generated synthetic data from ``generate.py``.

    Auto-detects the output format:

    **2D stacked format** — flat NIfTI files ``{id:05d}.nii.gz`` with
    channels stacked along the last axis ``[H, W, C]``. The last channel
    is the segmentation mask; preceding channels are image modalities.

    **3D subdirectory format** — subdirectories ``{id:05d}/`` containing
    separate files: ``seg.nii.gz`` + ``{modality}.nii.gz``.

    Args:
        data_dir: Directory containing generated samples.
        image_size: Target image size (H=W) for resizing.
        spatial_dims: 2 for 2D, 3 for 3D.
        volume_depth: Target depth for 3D volumes.
        modality: Modality file name for 3D subdirectory format (e.g., 'bravo').
        augment: Whether to apply augmentation.
        augmentation_strength: 'light' or 'standard'.
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        spatial_dims: int = 2,
        volume_depth: int = 160,
        modality: str = 'bravo',
        augment: bool = False,
        augmentation_strength: str = 'standard',
    ) -> None:
        self.data_dir = data_dir
        self.spatial_dims = spatial_dims
        self.image_size = image_size
        self.volume_depth = volume_depth
        self.modality = modality

        # Detect format and find samples
        self.format, self.samples = self._find_samples()
        if not self.samples:
            raise ValueError(f"No synthetic samples found in {data_dir}")

        # Build augmentation pipeline
        self._aug = None
        if augment:
            if spatial_dims == 2:
                self._aug = build_seg_downstream_augmentation_2d(augmentation_strength)
            else:
                self._aug = build_seg_downstream_augmentation_3d(augmentation_strength)

        logger.info(
            f"SyntheticDataset: {len(self.samples)} samples from {data_dir} "
            f"(format={self.format}, modality={modality})"
        )

    def _find_samples(self) -> tuple[str, list]:
        """Detect format and find all samples.

        Returns:
            Tuple of (format_name, sample_list) where:
            - '2d_stacked': sample_list is list of NIfTI file paths
            - '3d_subdir': sample_list is list of subdirectory paths
        """
        entries = sorted(os.listdir(self.data_dir))

        # Check for subdirectories with seg.nii.gz (3D format)
        subdirs = []
        for entry in entries:
            entry_path = os.path.join(self.data_dir, entry)
            if os.path.isdir(entry_path):
                seg_path = os.path.join(entry_path, 'seg.nii.gz')
                if os.path.exists(seg_path):
                    subdirs.append(entry_path)

        if subdirs:
            return '3d_subdir', subdirs

        # Check for flat NIfTI files (2D stacked format)
        nifti_files = [
            os.path.join(self.data_dir, f)
            for f in entries
            if f.endswith('.nii.gz') and not f.startswith('.')
        ]

        if nifti_files:
            return '2d_stacked', nifti_files

        return 'unknown', []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.format == '2d_stacked':
            return self._load_2d_stacked(idx)
        elif self.format == '3d_subdir':
            return self._load_3d_subdir(idx)
        else:
            raise RuntimeError(f"Unknown synthetic data format: {self.format}")

    def _load_nifti(self, path: str) -> np.ndarray:
        """Load a NIfTI file and return float32 array."""
        return nib.load(path).get_fdata().astype(np.float32)

    def _load_2d_stacked(self, idx: int) -> dict[str, torch.Tensor]:
        """Load a 2D stacked NIfTI file.

        Format: ``[H, W, C]`` where last channel is seg, rest are image.
        """
        path = self.samples[idx]
        data = self._load_nifti(path)  # [H, W, C]

        # Split: last channel is seg, rest is image
        seg_np = data[..., -1]  # [H, W]
        image_np = data[..., :-1]  # [H, W, C_img]

        # Normalize image to [0, 1]
        img_max = image_np.max()
        if img_max > 0:
            image_np = image_np / img_max

        # Binarize seg
        seg_np = (seg_np > 0.5).astype(np.float32)

        # Resize if needed
        if image_np.shape[0] != self.image_size or image_np.shape[1] != self.image_size:
            from skimage.transform import resize as skimage_resize
            image_np = skimage_resize(
                image_np, (self.image_size, self.image_size, image_np.shape[-1]),
                order=1, preserve_range=True,
            ).astype(np.float32)
            seg_np = skimage_resize(
                seg_np, (self.image_size, self.image_size),
                order=0, preserve_range=True,
            ).astype(np.float32)

        # To tensors: [C, H, W]
        if image_np.ndim == 2:
            image = torch.from_numpy(image_np).unsqueeze(0).float()
        else:
            image = torch.from_numpy(image_np).permute(2, 0, 1).float()  # [C, H, W]
        seg = torch.from_numpy(seg_np).unsqueeze(0).float()  # [1, H, W]

        if self._aug is not None:
            image, seg = _apply_augmentation_2d(image, seg, self._aug)

        return {'image': image, 'seg': seg}

    def _load_3d_subdir(self, idx: int) -> dict[str, torch.Tensor]:
        """Load a 3D sample from subdirectory.

        Format: ``{id}/seg.nii.gz`` + ``{id}/{modality}.nii.gz``.
        """
        subdir = self.samples[idx]

        seg_path = os.path.join(subdir, 'seg.nii.gz')
        img_path = os.path.join(subdir, f'{self.modality}.nii.gz')

        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Modality file not found: {img_path}. "
                f"Available files: {os.listdir(subdir)}"
            )

        seg_np = self._load_nifti(seg_path)    # [H, W, D]
        image_np = self._load_nifti(img_path)  # [H, W, D]

        # Normalize image to [0, 1]
        img_max = image_np.max()
        if img_max > 0:
            image_np = image_np / img_max

        # Binarize seg
        seg_np = (seg_np > 0.5).astype(np.float32)

        # Resize H, W (preserve depth)
        if image_np.shape[0] != self.image_size or image_np.shape[1] != self.image_size:
            from skimage.transform import resize as skimage_resize
            h, w = self.image_size, self.image_size
            d = image_np.shape[2]
            image_np = skimage_resize(
                image_np, (h, w, d), order=1, preserve_range=True,
            ).astype(np.float32)
            seg_np = skimage_resize(
                seg_np, (h, w, d), order=0, preserve_range=True,
            ).astype(np.float32)

        # To tensors [C, D, H, W]  (NIfTI is [H, W, D])
        image = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()  # [1, D, H, W]
        seg = torch.from_numpy(seg_np).permute(2, 0, 1).unsqueeze(0).float()      # [1, D, H, W]

        # Pad depth
        image = self._pad_depth(image)
        seg = self._pad_depth(seg)

        if self._aug is not None:
            image, seg = _apply_augmentation_3d(image, seg, self._aug)

        return {'image': image, 'seg': seg}

    def _pad_depth(self, volume: torch.Tensor) -> torch.Tensor:
        """Pad volume depth to target size using replication."""
        current_depth = volume.shape[1]  # [C, D, H, W]
        if current_depth < self.volume_depth:
            pad_total = self.volume_depth - current_depth
            padding = volume[:, -1:, :, :].repeat(1, pad_total, 1, 1)
            volume = torch.cat([volume, padding], dim=1)
        return volume


# =============================================================================
# Dataloader factories
# =============================================================================


def _create_eval_dataloader(
    cfg: DictConfig,
    split: str,
    spatial_dims: int,
) -> tuple[DataLoader, Dataset] | None:
    """Create evaluation (val/test) dataloader for segmentation.

    Args:
        cfg: Hydra configuration object.
        split: 'val' or 'test_new'.
        spatial_dims: 2 for 2D, 3 for 3D.

    Returns:
        Tuple of (DataLoader, Dataset) or None if directory doesn't exist.
    """
    seg_cfg = SegmentationConfig.from_cfg(cfg, spatial_dims, split='eval')
    data_dir = os.path.join(seg_cfg.real_dir, split)

    if not os.path.exists(data_dir):
        logger.warning(f"Directory not found: {data_dir}")
        return None

    dataset = SegmentationDataset(
        data_dir=data_dir,
        modality=seg_cfg.modality,
        image_size=seg_cfg.image_size,
        spatial_dims=spatial_dims,
        augment=False,
        volume_depth=seg_cfg.volume_depth,
    )

    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=seg_cfg.batch_size,
        shuffle=False,
        drop_last=False,
        loader_config=DataLoaderConfig.from_cfg(cfg),
    )

    logger.info(f"Segmentation {split} dataloader: {len(dataset)} samples")
    return dataloader, dataset


def create_segmentation_dataloader(
    cfg: DictConfig,
    scenario: str,
    split: str,
    spatial_dims: int = 2,
) -> tuple[DataLoader | None, Dataset | None]:
    """Create dataloader for downstream segmentation training.

    Args:
        cfg: Hydra configuration object.
        scenario: Training scenario - 'baseline', 'synthetic', or 'mixed'.
        split: Data split - 'train', 'val', or 'test'.
        spatial_dims: 2 for 2D, 3 for 3D.

    Returns:
        Tuple of (DataLoader, Dataset).
    """
    if scenario not in ('baseline', 'synthetic', 'mixed'):
        raise ValueError(f"Invalid scenario: {scenario}. Must be 'baseline', 'synthetic', or 'mixed'")

    seg_cfg = SegmentationConfig.from_cfg(cfg, spatial_dims, split)
    datasets = []

    # Load real data for baseline/mixed
    if scenario in ('baseline', 'mixed'):
        data_split_dir = os.path.join(seg_cfg.real_dir, split)
        if not os.path.exists(data_split_dir):
            if split == 'train':
                raise ValueError(f"Real training data not found: {data_split_dir}")
            logger.warning(f"Real {split} data not found: {data_split_dir}")
            return None, None

        real_dataset = SegmentationDataset(
            data_dir=data_split_dir,
            modality=seg_cfg.modality,
            image_size=seg_cfg.image_size,
            spatial_dims=spatial_dims,
            augment=seg_cfg.augment,
            augmentation_strength=seg_cfg.augmentation_strength,
            volume_depth=seg_cfg.volume_depth,
        )
        datasets.append(('real', real_dataset))

    # Load synthetic data for synthetic/mixed
    if scenario in ('synthetic', 'mixed'):
        if seg_cfg.synthetic_dir is None:
            raise ValueError(
                f"synthetic_dir must be specified for scenario='{scenario}'. "
                "Use data.synthetic_dir=/path/to/generated"
            )
        if not os.path.exists(seg_cfg.synthetic_dir):
            raise ValueError(f"Synthetic data directory not found: {seg_cfg.synthetic_dir}")

        synthetic_dataset = SyntheticDataset(
            data_dir=seg_cfg.synthetic_dir,
            image_size=seg_cfg.image_size,
            spatial_dims=spatial_dims,
            volume_depth=seg_cfg.volume_depth,
            modality=seg_cfg.modality if isinstance(seg_cfg.modality, str) else seg_cfg.modality[0],
            augment=seg_cfg.augment,
            augmentation_strength=seg_cfg.augmentation_strength,
        )
        datasets.append(('synthetic', synthetic_dataset))

    # Combine datasets
    if scenario in ('baseline', 'synthetic'):
        dataset = datasets[0][1]
    else:  # mixed
        real_dataset = datasets[0][1]
        synthetic_dataset = datasets[1][1]

        n_real = len(real_dataset)
        n_synthetic_available = len(synthetic_dataset)
        n_synthetic_target = int(n_real * seg_cfg.synthetic_ratio / (1 - seg_cfg.synthetic_ratio))
        n_synthetic = min(n_synthetic_target, n_synthetic_available)

        if n_synthetic < n_synthetic_available:
            indices = torch.randperm(n_synthetic_available)[:n_synthetic].tolist()
            synthetic_subset = Subset(synthetic_dataset, indices)
            dataset = ConcatDataset([real_dataset, synthetic_subset])
        else:
            dataset = ConcatDataset([real_dataset, synthetic_dataset])

        logger.info(
            f"Mixed dataset: {n_real} real + {n_synthetic} synthetic "
            f"(ratio: {n_synthetic / (n_real + n_synthetic):.2%})"
        )

    # Create DataLoader
    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=seg_cfg.batch_size,
        shuffle=(split == 'train'),
        drop_last=(split == 'train'),
        loader_config=DataLoaderConfig.from_cfg(cfg),
    )

    logger.info(
        f"Segmentation dataloader ({split}): {len(dataset)} samples, "
        f"batch_size={seg_cfg.batch_size}, spatial_dims={spatial_dims}"
    )

    return dataloader, dataset


def create_segmentation_val_dataloader(
    cfg: DictConfig,
    spatial_dims: int = 2,
) -> tuple[DataLoader, Dataset] | None:
    """Create validation dataloader for segmentation."""
    return _create_eval_dataloader(cfg, 'val', spatial_dims)


def create_segmentation_test_dataloader(
    cfg: DictConfig,
    spatial_dims: int = 2,
) -> tuple[DataLoader, Dataset] | None:
    """Create test dataloader for segmentation."""
    return _create_eval_dataloader(cfg, 'test_new', spatial_dims)
