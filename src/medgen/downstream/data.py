"""
Downstream segmentation data loading.

Provides dataloaders for three training scenarios:
- baseline: Real data only (control)
- synthetic: Generated data only
- mixed: Real + synthetic data with configurable ratio

Supports both 2D and 3D segmentation.
"""
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    RandFlip,
    RandRotate90,
    Resize,
    ScaleIntensity,
    ToTensor,
)
from omegaconf import DictConfig

from medgen.data.loaders.common import DataLoaderConfig

logger = logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    """Dataset for paired image-segmentation loading.

    Loads images and corresponding segmentation masks from NIfTI files.
    Returns dict format: {'image': [1, H, W], 'seg': [1, H, W]}

    For 2D: Extracts slices from 3D volumes.
    For 3D: Returns full volumes.

    Args:
        data_dir: Directory containing patient subdirectories.
        modality: Input modality name (e.g., 'bravo', 't1_gd').
        image_size: Target image size (2D: H=W, 3D: H=W).
        spatial_dims: 2 for 2D slices, 3 for 3D volumes.
        augment: Whether to apply data augmentation.
        volume_depth: Target depth for 3D volumes.
    """

    def __init__(
        self,
        data_dir: str,
        modality: str = 'bravo',
        image_size: int = 256,
        spatial_dims: int = 2,
        augment: bool = False,
        volume_depth: int = 160,
    ) -> None:
        self.data_dir = data_dir
        self.modality = modality
        self.image_size = image_size
        self.spatial_dims = spatial_dims
        self.augment = augment
        self.volume_depth = volume_depth

        # Find all patients
        self.patients = sorted([
            p for p in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, p))
        ])

        if len(self.patients) == 0:
            raise ValueError(f"No patient directories found in {data_dir}")

        # Build transforms
        self.transform = self._build_transform()

        # For 2D: build slice index mapping
        if spatial_dims == 2:
            self.slice_indices = self._build_slice_indices()
        else:
            self.slice_indices = None

        logger.info(
            f"SegmentationDataset: {len(self.patients)} patients, "
            f"spatial_dims={spatial_dims}, augment={augment}"
        )

    def _build_transform(self) -> Compose:
        """Build MONAI transform pipeline."""
        transforms = [
            LoadImage(image_only=True),
            EnsureChannelFirst(channel_dim="no_channel"),
            ToTensor(),
            ScaleIntensity(minv=0.0, maxv=1.0),
        ]

        if self.spatial_dims == 2:
            # For 2D, resize spatial dimensions but keep depth
            transforms.append(Resize(spatial_size=(self.image_size, self.image_size, -1)))
        else:
            # For 3D, resize all dimensions
            transforms.append(Resize(spatial_size=(
                self.image_size, self.image_size, self.volume_depth
            )))

        return Compose(transforms)

    def _build_slice_indices(self) -> List[Tuple[int, int]]:
        """Build mapping from linear index to (patient_idx, slice_idx).

        Skips slices with no tumor (empty segmentation).
        """
        indices = []
        loader = LoadImage(image_only=True)

        for patient_idx, patient in enumerate(self.patients):
            seg_path = os.path.join(self.data_dir, patient, 'seg.nii.gz')
            if not os.path.exists(seg_path):
                continue

            # Load seg to count slices and check for tumors
            seg = loader(seg_path)
            n_slices = seg.shape[-1] if len(seg.shape) == 3 else seg.shape[2]

            for slice_idx in range(n_slices):
                # Include slice if it has tumor pixels
                slice_data = seg[..., slice_idx] if len(seg.shape) == 3 else seg[:, :, slice_idx]
                if slice_data.sum() > 0:
                    indices.append((patient_idx, slice_idx))

        return indices

    def __len__(self) -> int:
        if self.spatial_dims == 2:
            return len(self.slice_indices)
        return len(self.patients)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.spatial_dims == 2:
            return self._get_slice(idx)
        return self._get_volume(idx)

    def _get_slice(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a 2D slice."""
        patient_idx, slice_idx = self.slice_indices[idx]
        patient = self.patients[patient_idx]

        # Load image and seg
        img_path = os.path.join(self.data_dir, patient, f'{self.modality}.nii.gz')
        seg_path = os.path.join(self.data_dir, patient, 'seg.nii.gz')

        image = self.transform(img_path)  # [1, H, W, D]
        seg = self.transform(seg_path)    # [1, H, W, D]

        # Extract slice
        image_slice = image[..., slice_idx]  # [1, H, W]
        seg_slice = seg[..., slice_idx]      # [1, H, W]

        # Binarize segmentation
        seg_slice = (seg_slice > 0.5).float()

        # Apply augmentation
        if self.augment:
            image_slice, seg_slice = self._apply_augmentation(image_slice, seg_slice)

        return {'image': image_slice, 'seg': seg_slice}

    def _get_volume(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a 3D volume."""
        patient = self.patients[idx]

        # Load image and seg
        img_path = os.path.join(self.data_dir, patient, f'{self.modality}.nii.gz')
        seg_path = os.path.join(self.data_dir, patient, 'seg.nii.gz')

        image = self.transform(img_path)  # [1, H, W, D]
        seg = self.transform(seg_path)    # [1, H, W, D]

        # Binarize segmentation
        seg = (seg > 0.5).float()

        # Transpose to [1, D, H, W] for 3D convolutions
        image = image.permute(0, 3, 1, 2)
        seg = seg.permute(0, 3, 1, 2)

        # Apply augmentation
        if self.augment:
            image, seg = self._apply_augmentation_3d(image, seg)

        return {'image': image, 'seg': seg}

    def _apply_augmentation(
        self,
        image: torch.Tensor,
        seg: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 2D augmentation to image and mask pair."""
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[-1])
            seg = torch.flip(seg, dims=[-1])

        # Random vertical flip
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[-2])
            seg = torch.flip(seg, dims=[-2])

        # Random 90-degree rotation
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[-2, -1])
            seg = torch.rot90(seg, k, dims=[-2, -1])

        return image, seg

    def _apply_augmentation_3d(
        self,
        image: torch.Tensor,
        seg: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 3D augmentation to image and mask pair."""
        # Random horizontal flip (W axis)
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[-1])
            seg = torch.flip(seg, dims=[-1])

        # Random vertical flip (H axis)
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[-2])
            seg = torch.flip(seg, dims=[-2])

        return image, seg


class SyntheticDataset(Dataset):
    """Dataset for loading generated synthetic data.

    Expects NIfTI files in the synthetic directory with naming:
    - {prefix}_image.nii.gz or image_{idx}.nii.gz
    - {prefix}_seg.nii.gz or seg_{idx}.nii.gz

    Args:
        data_dir: Directory containing synthetic NIfTI files.
        image_size: Target image size.
        spatial_dims: 2 for 2D, 3 for 3D.
        volume_depth: Target depth for 3D volumes.
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        spatial_dims: int = 2,
        volume_depth: int = 160,
    ) -> None:
        self.data_dir = data_dir
        self.image_size = image_size
        self.spatial_dims = spatial_dims
        self.volume_depth = volume_depth

        # Find all image-seg pairs
        self.pairs = self._find_pairs()

        if len(self.pairs) == 0:
            raise ValueError(f"No synthetic image-seg pairs found in {data_dir}")

        # Build transform
        self.transform = self._build_transform()

        logger.info(f"SyntheticDataset: {len(self.pairs)} pairs from {data_dir}")

    def _find_pairs(self) -> List[Tuple[str, str]]:
        """Find all image-seg pairs in the directory."""
        pairs = []
        files = os.listdir(self.data_dir)

        # Look for patterns like: sample_0_image.nii.gz, sample_0_seg.nii.gz
        # Or: image_0.nii.gz, seg_0.nii.gz
        seg_files = [f for f in files if 'seg' in f.lower() and f.endswith('.nii.gz')]

        for seg_file in seg_files:
            # Try to find matching image file
            if '_seg' in seg_file:
                # Pattern: prefix_seg.nii.gz -> prefix_image.nii.gz
                image_file = seg_file.replace('_seg', '_image')
            elif 'seg_' in seg_file:
                # Pattern: seg_0.nii.gz -> image_0.nii.gz
                image_file = seg_file.replace('seg_', 'image_')
            else:
                continue

            if image_file in files:
                pairs.append((
                    os.path.join(self.data_dir, image_file),
                    os.path.join(self.data_dir, seg_file),
                ))

        return pairs

    def _build_transform(self) -> Compose:
        """Build MONAI transform pipeline."""
        transforms = [
            LoadImage(image_only=True),
            EnsureChannelFirst(channel_dim="no_channel"),
            ToTensor(),
            ScaleIntensity(minv=0.0, maxv=1.0),
        ]

        if self.spatial_dims == 2:
            transforms.append(Resize(spatial_size=(self.image_size, self.image_size)))
        else:
            transforms.append(Resize(spatial_size=(
                self.image_size, self.image_size, self.volume_depth
            )))

        return Compose(transforms)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, seg_path = self.pairs[idx]

        image = self.transform(img_path)
        seg = self.transform(seg_path)

        # Binarize segmentation
        seg = (seg > 0.5).float()

        # For 3D, transpose to [1, D, H, W]
        if self.spatial_dims == 3 and len(image.shape) == 4:
            image = image.permute(0, 3, 1, 2)
            seg = seg.permute(0, 3, 1, 2)

        return {'image': image, 'seg': seg}


def create_segmentation_dataloader(
    cfg: DictConfig,
    scenario: str,
    split: str,
    spatial_dims: int = 2,
) -> Tuple[DataLoader, Dataset]:
    """Create dataloader for downstream segmentation training.

    Args:
        cfg: Hydra configuration object.
        scenario: Training scenario - 'baseline', 'synthetic', or 'mixed'.
        split: Data split - 'train', 'val', or 'test'.
        spatial_dims: 2 for 2D, 3 for 3D.

    Returns:
        Tuple of (DataLoader, Dataset).

    Raises:
        ValueError: If scenario is invalid or required directories don't exist.
    """
    if scenario not in ('baseline', 'synthetic', 'mixed'):
        raise ValueError(f"Invalid scenario: {scenario}. Must be 'baseline', 'synthetic', or 'mixed'")

    # Extract config values
    real_dir = cfg.data.real_dir
    synthetic_dir = cfg.data.get('synthetic_dir')
    synthetic_ratio = cfg.data.get('synthetic_ratio', 0.5)
    modality = cfg.data.get('modality', 'bravo')
    image_size = cfg.model.image_size
    augment = cfg.data.get('augment', True) and split == 'train'
    volume_depth = cfg.volume.get('pad_depth_to', 160) if spatial_dims == 3 else None

    # Determine batch size
    if spatial_dims == 3:
        batch_size = cfg.training.get('batch_size_3d', 2)
    else:
        batch_size = cfg.training.batch_size

    # Build dataset(s) based on scenario
    datasets = []

    if scenario in ('baseline', 'mixed'):
        # Load real data
        data_split_dir = os.path.join(real_dir, split)
        if not os.path.exists(data_split_dir):
            if split == 'train':
                raise ValueError(f"Real training data not found: {data_split_dir}")
            else:
                logger.warning(f"Real {split} data not found: {data_split_dir}")
                return None, None

        real_dataset = SegmentationDataset(
            data_dir=data_split_dir,
            modality=modality,
            image_size=image_size,
            spatial_dims=spatial_dims,
            augment=augment,
            volume_depth=volume_depth or 160,
        )
        datasets.append(('real', real_dataset))

    if scenario in ('synthetic', 'mixed'):
        # Load synthetic data
        if synthetic_dir is None:
            raise ValueError(
                f"synthetic_dir must be specified for scenario='{scenario}'. "
                "Use data.synthetic_dir=/path/to/generated"
            )

        if not os.path.exists(synthetic_dir):
            raise ValueError(f"Synthetic data directory not found: {synthetic_dir}")

        synthetic_dataset = SyntheticDataset(
            data_dir=synthetic_dir,
            image_size=image_size,
            spatial_dims=spatial_dims,
            volume_depth=volume_depth or 160,
        )
        datasets.append(('synthetic', synthetic_dataset))

    # Combine datasets based on scenario
    if scenario == 'baseline':
        dataset = datasets[0][1]
    elif scenario == 'synthetic':
        dataset = datasets[0][1]
    else:  # mixed
        real_dataset = datasets[0][1]
        synthetic_dataset = datasets[1][1]

        # Calculate how many synthetic samples to use based on ratio
        # ratio = synthetic / total -> synthetic = ratio * total = ratio * (real + synthetic)
        # synthetic = ratio * real / (1 - ratio)
        n_real = len(real_dataset)
        n_synthetic_available = len(synthetic_dataset)
        n_synthetic_target = int(n_real * synthetic_ratio / (1 - synthetic_ratio))
        n_synthetic = min(n_synthetic_target, n_synthetic_available)

        if n_synthetic < n_synthetic_available:
            # Subsample synthetic data
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
    dl_cfg = DataLoaderConfig.from_cfg(cfg)

    shuffle = split == 'train'
    drop_last = split == 'train'

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )

    logger.info(
        f"Segmentation dataloader ({split}): {len(dataset)} samples, "
        f"batch_size={batch_size}, spatial_dims={spatial_dims}"
    )

    return dataloader, dataset


def create_segmentation_val_dataloader(
    cfg: DictConfig,
    spatial_dims: int = 2,
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create validation dataloader for segmentation.

    Always uses real validation data regardless of training scenario.

    Args:
        cfg: Hydra configuration object.
        spatial_dims: 2 for 2D, 3 for 3D.

    Returns:
        Tuple of (DataLoader, Dataset) or None if val data doesn't exist.
    """
    real_dir = cfg.data.real_dir
    val_dir = os.path.join(real_dir, 'val')

    if not os.path.exists(val_dir):
        logger.warning(f"Validation directory not found: {val_dir}")
        return None

    modality = cfg.data.get('modality', 'bravo')
    image_size = cfg.model.image_size
    volume_depth = cfg.volume.get('pad_depth_to', 160) if spatial_dims == 3 else None

    # Determine batch size
    if spatial_dims == 3:
        batch_size = cfg.training.get('batch_size_3d', 2)
    else:
        batch_size = cfg.training.batch_size

    dataset = SegmentationDataset(
        data_dir=val_dir,
        modality=modality,
        image_size=image_size,
        spatial_dims=spatial_dims,
        augment=False,  # No augmentation for validation
        volume_depth=volume_depth or 160,
    )

    dl_cfg = DataLoaderConfig.from_cfg(cfg)

    # Use fixed seed generator for reproducible validation
    generator = torch.Generator().manual_seed(42)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        generator=generator,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )

    logger.info(f"Segmentation val dataloader: {len(dataset)} samples")

    return dataloader, dataset


def create_segmentation_test_dataloader(
    cfg: DictConfig,
    spatial_dims: int = 2,
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create test dataloader for segmentation.

    Uses real test data from test_new/ directory.

    Args:
        cfg: Hydra configuration object.
        spatial_dims: 2 for 2D, 3 for 3D.

    Returns:
        Tuple of (DataLoader, Dataset) or None if test data doesn't exist.
    """
    real_dir = cfg.data.real_dir
    test_dir = os.path.join(real_dir, 'test_new')

    if not os.path.exists(test_dir):
        logger.warning(f"Test directory not found: {test_dir}")
        return None

    modality = cfg.data.get('modality', 'bravo')
    image_size = cfg.model.image_size
    volume_depth = cfg.volume.get('pad_depth_to', 160) if spatial_dims == 3 else None

    # Determine batch size
    if spatial_dims == 3:
        batch_size = cfg.training.get('batch_size_3d', 2)
    else:
        batch_size = cfg.training.batch_size

    dataset = SegmentationDataset(
        data_dir=test_dir,
        modality=modality,
        image_size=image_size,
        spatial_dims=spatial_dims,
        augment=False,
        volume_depth=volume_depth or 160,
    )

    dl_cfg = DataLoaderConfig.from_cfg(cfg)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )

    logger.info(f"Segmentation test dataloader: {len(dataset)} samples")

    return dataloader, dataset
