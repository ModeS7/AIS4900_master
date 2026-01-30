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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from omegaconf import DictConfig

from medgen.data.dataset import NiFTIDataset, build_standard_transform, validate_modality_exists
from medgen.data.utils import make_binary
from medgen.data.loaders.common import DataLoaderConfig, create_dataloader
from medgen.data.loaders.volume_3d import build_3d_transform

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


@dataclass
class SegmentationConfig:
    """Configuration extracted from Hydra config for segmentation data loading."""
    modality: Union[str, List[str]]
    image_size: int
    volume_depth: int
    batch_size: int
    augment: bool
    real_dir: str
    synthetic_dir: Optional[str]
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
        volume_depth: Target depth for 3D volumes.
    """

    def __init__(
        self,
        data_dir: str,
        modality: Union[str, List[str]] = 'bravo',
        image_size: int = 256,
        spatial_dims: int = 2,
        augment: bool = False,
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
            f"modalities={self.modalities}, spatial_dims={spatial_dims}, augment={augment}"
        )

    def _build_slice_indices(self) -> List[Tuple[int, int]]:
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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

    def _get_slice(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a 2D slice."""
        patient_idx, slice_idx = self.slice_indices[idx]

        image = self._load_modalities(patient_idx)  # [C, H, W, D]
        seg, _ = self._seg_dataset[patient_idx]
        seg = _to_tensor(seg)

        # Extract slice
        image_slice = image[..., slice_idx]  # [C, H, W]
        seg_slice = _binarize_seg(seg[..., slice_idx])  # [1, H, W]

        if self.augment:
            image_slice, seg_slice = self._apply_augmentation(image_slice, seg_slice, rotate=True)

        return {'image': image_slice, 'seg': seg_slice}

    def _get_volume(self, idx: int) -> Dict[str, torch.Tensor]:
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

        if self.augment:
            image, seg = self._apply_augmentation(image, seg, rotate=False)

        return {'image': image, 'seg': seg}

    def _pad_depth(self, volume: torch.Tensor) -> torch.Tensor:
        """Pad volume depth to target size using replication."""
        current_depth = volume.shape[1]  # [C, D, H, W]
        if current_depth < self.volume_depth:
            pad_total = self.volume_depth - current_depth
            padding = volume[:, -1:, :, :].repeat(1, pad_total, 1, 1)
            volume = torch.cat([volume, padding], dim=1)
        return volume

    def _apply_augmentation(
        self, image: torch.Tensor, seg: torch.Tensor, rotate: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation to image and mask pair.

        Args:
            image: Image tensor.
            seg: Segmentation tensor.
            rotate: Whether to apply 90-degree rotation (2D only).
        """
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[-1])
            seg = torch.flip(seg, dims=[-1])

        # Random vertical flip
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[-2])
            seg = torch.flip(seg, dims=[-2])

        # Random 90-degree rotation (2D only)
        if rotate:
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                image = torch.rot90(image, k, dims=[-2, -1])
                seg = torch.rot90(seg, k, dims=[-2, -1])

        return image, seg


class SyntheticDataset(Dataset):
    """Dataset for loading generated synthetic data.

    Expects NIfTI files in the synthetic directory with naming:
    - {prefix}_image.nii.gz or image_{idx}.nii.gz
    - {prefix}_seg.nii.gz or seg_{idx}.nii.gz
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        spatial_dims: int = 2,
        volume_depth: int = 160,
    ) -> None:
        self.data_dir = data_dir
        self.spatial_dims = spatial_dims
        self.volume_depth = volume_depth

        self.pairs = self._find_pairs()
        if not self.pairs:
            raise ValueError(f"No synthetic image-seg pairs found in {data_dir}")

        self.transform = _build_transform(spatial_dims, image_size)
        logger.info(f"SyntheticDataset: {len(self.pairs)} pairs from {data_dir}")

    def _find_pairs(self) -> List[Tuple[str, str]]:
        """Find all image-seg pairs in the directory."""
        pairs = []
        files = os.listdir(self.data_dir)
        seg_files = [f for f in files if 'seg' in f.lower() and f.endswith('.nii.gz')]

        for seg_file in seg_files:
            if '_seg' in seg_file:
                image_file = seg_file.replace('_seg', '_image')
            elif 'seg_' in seg_file:
                image_file = seg_file.replace('seg_', 'image_')
            else:
                continue

            if image_file in files:
                pairs.append((
                    os.path.join(self.data_dir, image_file),
                    os.path.join(self.data_dir, seg_file),
                ))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, seg_path = self.pairs[idx]

        image = _to_tensor(self.transform(img_path))
        seg = _binarize_seg(_to_tensor(self.transform(seg_path)))

        # For 3D, transpose to [C, D, H, W]
        if self.spatial_dims == 3 and image.dim() == 4:
            image = image.permute(0, 3, 1, 2)
            seg = seg.permute(0, 3, 1, 2)

        return {'image': image, 'seg': seg}


# =============================================================================
# Dataloader factories
# =============================================================================


def _create_eval_dataloader(
    cfg: DictConfig,
    split: str,
    spatial_dims: int,
) -> Optional[Tuple[DataLoader, Dataset]]:
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
) -> Tuple[Optional[DataLoader], Optional[Dataset]]:
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
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create validation dataloader for segmentation."""
    return _create_eval_dataloader(cfg, 'val', spatial_dims)


def create_segmentation_test_dataloader(
    cfg: DictConfig,
    spatial_dims: int = 2,
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create test dataloader for segmentation."""
    return _create_eval_dataloader(cfg, 'test_new', spatial_dims)
