"""3D volume dataloaders for volumetric medical images.

Loads full 3D NIfTI volumes for training 3D models (both diffusion and compression).
Key differences from 2D:
- Returns full volumes instead of 2D slices
- Applies depth padding for clean compression
- Memory-efficient batch sizes (typically 1-2)

Data Augmentation:
    3D augmentation is supported via MONAI transforms. Use `augment=True` in
    dataloader creation to enable random flips and 90° rotations.

    Available augmentations:
    - RandFlipd: Random flips along each spatial axis (p=0.5 each)
    - RandRotate90d: Random 90° rotations in the axial plane

    For segmentation masks, augmentations preserve binary values (no interpolation).
"""
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    RandFlipd,
    RandRotate90d,
    ScaleIntensity,
)
from torch.utils.data import DataLoader, Dataset

from .common import DataLoaderConfig, DistributedArgs, create_dataloader, get_validated_split_dir


@dataclass
class VolumeConfig:
    """Configuration extracted from Hydra config for 3D volume loading.

    Supports separate train/val resolutions for efficient training:
    - Train at lower resolution (e.g., 128x128) for speed
    - Validate at full resolution (e.g., 256x256) for accurate metrics
    """
    height: int
    width: int
    pad_depth_to: int
    pad_mode: str
    slice_step: int
    batch_size: int
    load_seg: bool
    image_keys: list[str]
    # Optional training resolution (defaults to height/width)
    _train_height: int | None = None
    _train_width: int | None = None
    # DataLoader config (stored as DataLoaderConfig)
    _loader_config: DataLoaderConfig | None = None

    def __post_init__(self):
        """Validate configuration values."""
        if self.height <= 0:
            raise ValueError(f"height must be > 0, got {self.height}")
        if self.width <= 0:
            raise ValueError(f"width must be > 0, got {self.width}")
        if self.pad_depth_to <= 0:
            raise ValueError(f"pad_depth_to must be > 0, got {self.pad_depth_to}")
        if self.slice_step < 1:
            raise ValueError(f"slice_step must be >= 1, got {self.slice_step}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        valid_pad_modes = ('replicate', 'constant', 'reflect')
        if self.pad_mode not in valid_pad_modes:
            raise ValueError(f"pad_mode must be one of {valid_pad_modes}, got '{self.pad_mode}'")

    @classmethod
    def from_cfg(cls, cfg) -> 'VolumeConfig':
        """Extract volume configuration from Hydra config object."""
        logging_cfg = cfg.training.get('logging', {})
        loader_config = DataLoaderConfig.from_cfg(cfg)

        # Get optional train resolution (null in config becomes None)
        train_height = cfg.volume.get('train_height', None)
        train_width = cfg.volume.get('train_width', None)

        return cls(
            height=cfg.volume.height,
            width=cfg.volume.width,
            pad_depth_to=cfg.volume.pad_depth_to,
            pad_mode=cfg.volume.get('pad_mode', 'replicate'),
            slice_step=cfg.volume.get('slice_step', 1),
            batch_size=cfg.training.batch_size,
            load_seg=logging_cfg.get('regional_losses', False),
            image_keys=cfg.mode.image_keys,
            _train_height=train_height,
            _train_width=train_width,
            _loader_config=loader_config,
        )

    @property
    def train_height(self) -> int:
        """Height for training (may be lower resolution)."""
        return self._train_height if self._train_height is not None else self.height

    @property
    def train_width(self) -> int:
        """Width for training (may be lower resolution)."""
        return self._train_width if self._train_width is not None else self.width

    @property
    def uses_reduced_train_resolution(self) -> bool:
        """Whether training uses lower resolution than validation."""
        return self.train_height != self.height or self.train_width != self.width

    @property
    def loader_config(self) -> DataLoaderConfig:
        """Get DataLoaderConfig for create_dataloader()."""
        return self._loader_config

    @property
    def num_workers(self) -> int:
        return self._loader_config.num_workers

    @property
    def prefetch_factor(self) -> int | None:
        return self._loader_config.prefetch_factor

    @property
    def pin_memory(self) -> bool:
        return self._loader_config.pin_memory

    @property
    def persistent_workers(self) -> bool:
        return self._loader_config.persistent_workers

logger = logging.getLogger(__name__)


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
        # Random 90° rotations in axial plane (H, W)
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
        """Get summary of depth padding configuration.

        Note: Per-volume padding events are logged at DEBUG level during loading.
        Aggregate stats are not tracked because DataLoader workers get
        independent copies of the dataset object.
        """
        if self.pad_depth_to:
            return f"Depth padding enabled: pad to {self.pad_depth_to} slices (mode={self.pad_mode})"
        return "No depth padding configured"

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


def build_3d_transform(height: int, width: int) -> Compose:
    """Build transform pipeline for 3D volumes.

    Args:
        height: Target height.
        width: Target width.

    Returns:
        MONAI Compose transform.
    """
    from monai.transforms import Resize

    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(channel_dim="no_channel"),  # NIfTI has no channel dim
        ScaleIntensity(minv=0.0, maxv=1.0),
        Resize(spatial_size=(height, width, -1)),  # Preserve depth
    ])


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

    def __getitem__(self, idx: int) -> dict[str, Any]:
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

    def __getitem__(self, idx: int) -> dict[str, Any]:
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


def _create_single_dual_dataset(
    data_dir: str,
    modality: str,
    vcfg: VolumeConfig,
    height: int | None = None,
    width: int | None = None,
) -> Dataset:
    """Create Volume3DDataset or DualVolume3DDataset based on modality.

    Args:
        data_dir: Path to data directory (train/val/test_new).
        modality: 'dual' for DualVolume3DDataset, else Volume3DDataset.
        vcfg: Volume configuration.
        height: Override height (defaults to vcfg.height).
        width: Override width (defaults to vcfg.width).

    Returns:
        Appropriate dataset instance.
    """
    h = height if height is not None else vcfg.height
    w = width if width is not None else vcfg.width

    if modality == 'dual':
        return DualVolume3DDataset(
            data_dir=data_dir,
            height=h,
            width=w,
            pad_depth_to=vcfg.pad_depth_to,
            pad_mode=vcfg.pad_mode,
            slice_step=vcfg.slice_step,
            load_seg=vcfg.load_seg,
        )
    return Volume3DDataset(
        data_dir=data_dir,
        modality=modality,
        height=h,
        width=w,
        pad_depth_to=vcfg.pad_depth_to,
        pad_mode=vcfg.pad_mode,
        slice_step=vcfg.slice_step,
        load_seg=vcfg.load_seg,
    )


def _create_loader(
    dataset: Dataset,
    vcfg: VolumeConfig,
    shuffle: bool = True,
    drop_last: bool = False,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Create DataLoader with standard settings using shared create_dataloader().

    Args:
        dataset: Dataset to wrap.
        vcfg: Volume configuration containing batch_size and DataLoader settings.
        shuffle: Whether to shuffle (ignored if distributed).
        drop_last: Whether to drop last incomplete batch.
        use_distributed: Use DistributedSampler.
        rank: Process rank for distributed.
        world_size: Total processes for distributed.

    Returns:
        Configured DataLoader.
    """
    distributed_args = DistributedArgs(
        use_distributed=use_distributed,
        rank=rank,
        world_size=world_size,
    )

    return create_dataloader(
        dataset=dataset,
        batch_size=vcfg.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        distributed_args=distributed_args,
        loader_config=vcfg.loader_config,
        scale_batch_for_distributed=False,  # 3D volumes: batch_size=1-2, don't divide
    )


def create_vae_3d_dataloader(
    cfg,
    modality: str,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, Dataset]:
    """Create 3D VAE training dataloader.

    Uses train_height/train_width if configured, allowing training at lower
    resolution than validation for faster iteration.

    Args:
        cfg: Hydra configuration object.
        modality: Modality name or 'dual' for dual-channel.
        use_distributed: Whether to use distributed sampling.
        rank: Process rank for distributed training.
        world_size: Total number of processes.

    Returns:
        Tuple of (DataLoader, Dataset).
    """
    vcfg = VolumeConfig.from_cfg(cfg)
    data_dir = os.path.join(cfg.paths.data_dir, 'train')

    # Use training resolution (may be lower than validation resolution)
    dataset = _create_single_dual_dataset(
        data_dir, modality, vcfg,
        height=vcfg.train_height,
        width=vcfg.train_width,
    )

    if vcfg.uses_reduced_train_resolution:
        logger.info(
            f"Training at {vcfg.train_height}x{vcfg.train_width}, "
            f"validation at {vcfg.height}x{vcfg.width}"
        )

    loader = _create_loader(
        dataset, vcfg, shuffle=True, drop_last=True,
        use_distributed=use_distributed, rank=rank, world_size=world_size
    )
    return loader, dataset


def create_vae_3d_validation_dataloader(
    cfg,
    modality: str,
) -> tuple[DataLoader, Dataset] | None:
    """Create 3D VAE validation dataloader.

    Args:
        cfg: Hydra configuration object.
        modality: Modality name or 'dual' for dual-channel.

    Returns:
        Tuple of (DataLoader, Dataset) or None if val/ doesn't exist.
    """
    val_dir = get_validated_split_dir(cfg.paths.data_dir, 'val')
    if val_dir is None:
        return None

    vcfg = VolumeConfig.from_cfg(cfg)
    dataset = _create_single_dual_dataset(val_dir, modality, vcfg)
    loader = _create_loader(dataset, vcfg, shuffle=False)
    return loader, dataset


def create_vae_3d_test_dataloader(
    cfg,
    modality: str,
) -> tuple[DataLoader, Dataset] | None:
    """Create 3D VAE test dataloader.

    Args:
        cfg: Hydra configuration object.
        modality: Modality name or 'dual' for dual-channel.

    Returns:
        Tuple of (DataLoader, Dataset) or None if test_new/ doesn't exist.
    """
    test_dir = get_validated_split_dir(cfg.paths.data_dir, 'test_new')
    if test_dir is None:
        return None

    vcfg = VolumeConfig.from_cfg(cfg)
    dataset = _create_single_dual_dataset(test_dir, modality, vcfg)
    loader = _create_loader(dataset, vcfg, shuffle=False)

    return loader, dataset


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

    def __getitem__(self, idx: int) -> dict[str, Any]:
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


def _create_multi_modality_dataset(
    data_dir: str,
    vcfg: VolumeConfig,
    height: int | None = None,
    width: int | None = None,
    augment: bool = False,
    seg_mode: bool = False,
) -> MultiModality3DDataset:
    """Create MultiModality3DDataset with config.

    Args:
        data_dir: Path to data directory.
        vcfg: Volume configuration.
        height: Override height (defaults to vcfg.height).
        width: Override width (defaults to vcfg.width).
        augment: Whether to apply 3D augmentation.
        seg_mode: Whether this is for segmentation masks (binary).
    """
    h = height if height is not None else vcfg.height
    w = width if width is not None else vcfg.width

    aug = build_3d_augmentation(seg_mode=seg_mode) if augment else None

    return MultiModality3DDataset(
        data_dir=data_dir,
        image_keys=vcfg.image_keys,
        height=h,
        width=w,
        pad_depth_to=vcfg.pad_depth_to,
        pad_mode=vcfg.pad_mode,
        slice_step=vcfg.slice_step,
        load_seg=vcfg.load_seg,
        augmentation=aug,
    )


def create_vae_3d_multi_modality_dataloader(
    cfg,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, Dataset]:
    """Create 3D VAE multi-modality training dataloader.

    Uses train_height/train_width if configured, allowing training at lower
    resolution than validation for faster iteration.

    Args:
        cfg: Hydra configuration object.
        use_distributed: Whether to use distributed sampling.
        rank: Process rank for distributed training.
        world_size: Total number of processes.

    Returns:
        Tuple of (DataLoader, Dataset).
    """
    vcfg = VolumeConfig.from_cfg(cfg)
    data_dir = os.path.join(cfg.paths.data_dir, 'train')

    # Check if augmentation is enabled
    augment = getattr(cfg.training, 'augment', False)
    # Check if this is seg mode (for binary mask handling)
    seg_mode = getattr(cfg.mode, 'name', '') == 'seg'

    # Use training resolution (may be lower than validation resolution)
    dataset = _create_multi_modality_dataset(
        data_dir, vcfg,
        height=vcfg.train_height,
        width=vcfg.train_width,
        augment=augment,
        seg_mode=seg_mode,
    )

    if vcfg.uses_reduced_train_resolution:
        logger.info(
            f"Training at {vcfg.train_height}x{vcfg.train_width}, "
            f"validation at {vcfg.height}x{vcfg.width}"
        )

    loader = _create_loader(
        dataset, vcfg, shuffle=True, drop_last=True,
        use_distributed=use_distributed, rank=rank, world_size=world_size
    )
    return loader, dataset


def create_vae_3d_multi_modality_validation_dataloader(
    cfg,
) -> tuple[DataLoader, Dataset] | None:
    """Create 3D VAE multi-modality validation dataloader.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Tuple of (DataLoader, Dataset) or None if val/ doesn't exist.
    """
    val_dir = get_validated_split_dir(cfg.paths.data_dir, 'val')
    if val_dir is None:
        return None

    vcfg = VolumeConfig.from_cfg(cfg)
    dataset = _create_multi_modality_dataset(val_dir, vcfg)
    loader = _create_loader(dataset, vcfg, shuffle=False)
    return loader, dataset


def create_vae_3d_multi_modality_test_dataloader(
    cfg,
) -> tuple[DataLoader, Dataset] | None:
    """Create 3D VAE multi-modality test dataloader.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Tuple of (DataLoader, Dataset) or None if test_new/ doesn't exist.
    """
    test_dir = get_validated_split_dir(cfg.paths.data_dir, 'test_new')
    if test_dir is None:
        return None

    vcfg = VolumeConfig.from_cfg(cfg)
    dataset = _create_multi_modality_dataset(test_dir, vcfg)
    loader = _create_loader(dataset, vcfg, shuffle=False)
    return loader, dataset


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


def create_vae_3d_single_modality_validation_loader(
    cfg,
    modality: str,
) -> DataLoader | None:
    """Create 3D validation loader for a single modality (for per-modality metrics).

    Includes seg masks paired with each volume for regional metrics tracking.

    Args:
        cfg: Hydra configuration object.
        modality: Single modality to load (e.g., 'bravo', 't1_pre', 't1_gd').

    Returns:
        DataLoader for single modality or None if val/ doesn't exist.
    """
    val_dir = get_validated_split_dir(cfg.paths.data_dir, 'val')
    if val_dir is None:
        return None

    # Check if modality exists in any patient directory
    has_modality = False
    for patient in os.listdir(val_dir):
        patient_dir = os.path.join(val_dir, patient)
        if os.path.isdir(patient_dir):
            modality_path = os.path.join(patient_dir, f"{modality}.nii.gz")
            if os.path.exists(modality_path):
                has_modality = True
                break

    if not has_modality:
        logger.warning(f"Modality {modality} not found in {val_dir}")
        return None

    vcfg = VolumeConfig.from_cfg(cfg)

    try:
        dataset = SingleModality3DDatasetWithSeg(
            data_dir=val_dir,
            modality=modality,
            height=vcfg.height,
            width=vcfg.width,
            pad_depth_to=vcfg.pad_depth_to,
            pad_mode=vcfg.pad_mode,
            slice_step=vcfg.slice_step,
        )
    except ValueError as e:
        logger.warning(f"Could not create dataset for {modality}: {e}")
        return None

    loader = _create_loader(dataset, vcfg, shuffle=False)

    return loader


# ==============================================================================
# 3D Diffusion Dataloaders (seg mode, bravo mode with conditioning)
# ==============================================================================


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


def create_segmentation_dataloader(
    cfg,
    vol_cfg: VolumeConfig,
    augment: bool = False,
) -> tuple[DataLoader, Dataset]:
    """Create 3D dataloader for unconditional segmentation training.

    Loads seg masks only (no conditioning).

    Args:
        cfg: Hydra configuration object.
        vol_cfg: Volume configuration.
        augment: Whether to apply augmentation.

    Returns:
        Tuple of (DataLoader, Dataset).
    """
    data_dir = os.path.join(cfg.paths.data_dir, 'train')

    aug = build_3d_augmentation(seg_mode=True) if augment else None

    dataset = Volume3DDataset(
        data_dir=data_dir,
        modality='seg',
        height=vol_cfg.train_height,
        width=vol_cfg.train_width,
        pad_depth_to=vol_cfg.pad_depth_to,
        pad_mode=vol_cfg.pad_mode,
        slice_step=vol_cfg.slice_step,
        load_seg=False,  # No separate seg needed - we ARE loading seg as image
        augmentation=aug,
    )

    logger.info(f"Created 3D seg dataset: {len(dataset)} volumes")

    loader = _create_loader(dataset, vol_cfg, shuffle=True, drop_last=True)
    return loader, dataset


def create_segmentation_validation_dataloader(
    cfg,
    vol_cfg: VolumeConfig,
) -> tuple[DataLoader, Dataset] | None:
    """Create 3D validation dataloader for unconditional segmentation.

    Args:
        cfg: Hydra configuration object.
        vol_cfg: Volume configuration.

    Returns:
        Tuple of (DataLoader, Dataset) or None if val/ doesn't exist.
    """
    val_dir = get_validated_split_dir(cfg.paths.data_dir, 'val')
    if val_dir is None:
        return None

    dataset = Volume3DDataset(
        data_dir=val_dir,
        modality='seg',
        height=vol_cfg.height,
        width=vol_cfg.width,
        pad_depth_to=vol_cfg.pad_depth_to,
        pad_mode=vol_cfg.pad_mode,
        slice_step=vol_cfg.slice_step,
        load_seg=False,
    )

    loader = _create_loader(dataset, vol_cfg, shuffle=False)
    return loader, dataset


def create_single_modality_dataloader_with_seg(
    cfg,
    vol_cfg: VolumeConfig,
    modality: str = 'bravo',
    augment: bool = False,
) -> tuple[DataLoader, Dataset]:
    """Create 3D dataloader for single modality conditioned on seg mask.

    Used for 3D bravo mode where bravo generation is conditioned on seg mask.
    Includes CFG dropout to randomly zero seg mask during training.

    Args:
        cfg: Hydra configuration object.
        vol_cfg: Volume configuration.
        modality: Modality to load (default: 'bravo').
        augment: Whether to apply augmentation.

    Returns:
        Tuple of (DataLoader, Dataset).
    """
    data_dir = os.path.join(cfg.paths.data_dir, 'train')

    # CFG dropout for 3D - get from config (default 15%)
    cfg_dropout_prob = float(cfg.mode.get('cfg_dropout_prob', 0.15))

    # include_seg=True ensures both image and seg are augmented consistently
    aug = build_3d_augmentation(seg_mode=False, include_seg=True) if augment else None

    dataset = SingleModality3DDatasetWithSegDropout(
        data_dir=data_dir,
        modality=modality,
        height=vol_cfg.train_height,
        width=vol_cfg.train_width,
        pad_depth_to=vol_cfg.pad_depth_to,
        pad_mode=vol_cfg.pad_mode,
        slice_step=vol_cfg.slice_step,
        cfg_dropout_prob=cfg_dropout_prob,
        augmentation=aug,
    )

    logger.info(f"Created 3D {modality} dataset with seg conditioning: {len(dataset)} volumes, "
                f"cfg_dropout={cfg_dropout_prob}")

    loader = _create_loader(dataset, vol_cfg, shuffle=True, drop_last=True)
    return loader, dataset


def create_single_modality_validation_dataloader_with_seg(
    cfg,
    vol_cfg: VolumeConfig,
    modality: str = 'bravo',
    split: str = 'val',
) -> tuple[DataLoader, Dataset] | None:
    """Create 3D validation/test dataloader for single modality with seg conditioning.

    No CFG dropout during validation/test.

    Args:
        cfg: Hydra configuration object.
        vol_cfg: Volume configuration.
        modality: Modality to load (default: 'bravo').
        split: Data split ('val' or 'test_new').

    Returns:
        Tuple of (DataLoader, Dataset) or None if split directory doesn't exist.
    """
    val_dir = get_validated_split_dir(cfg.paths.data_dir, split)
    if val_dir is None:
        return None

    # No CFG dropout during validation
    dataset = SingleModality3DDatasetWithSegDropout(
        data_dir=val_dir,
        modality=modality,
        height=vol_cfg.height,
        width=vol_cfg.width,
        pad_depth_to=vol_cfg.pad_depth_to,
        pad_mode=vol_cfg.pad_mode,
        slice_step=vol_cfg.slice_step,
        cfg_dropout_prob=0.0,  # No dropout during validation
    )

    loader = _create_loader(dataset, vol_cfg, shuffle=False)
    return loader, dataset


def create_segmentation_conditioned_dataloader(
    cfg,
    vol_cfg: VolumeConfig,
    size_bin_config: dict,
    augment: bool = False,
) -> tuple[DataLoader, Dataset]:
    """Create 3D dataloader for size-conditioned segmentation training.

    Wrapper that routes to seg.py's create_seg_dataloader.

    Args:
        cfg: Hydra configuration object.
        vol_cfg: Volume configuration.
        size_bin_config: Size bin configuration dict.
        augment: Whether to apply augmentation.

    Returns:
        Tuple of (DataLoader, Dataset).
    """
    from .seg import create_seg_dataloader
    return create_seg_dataloader(cfg)


def create_segmentation_conditioned_validation_dataloader(
    cfg,
    vol_cfg: VolumeConfig,
    size_bin_config: dict,
) -> tuple[DataLoader, Dataset] | None:
    """Create 3D validation dataloader for size-conditioned segmentation.

    Wrapper that routes to seg.py's create_seg_validation_dataloader.

    Args:
        cfg: Hydra configuration object.
        vol_cfg: Volume configuration.
        size_bin_config: Size bin configuration dict.

    Returns:
        Tuple of (DataLoader, Dataset) or None if val/ doesn't exist.
    """
    from .seg import create_seg_validation_dataloader
    return create_seg_validation_dataloader(cfg)


# ==============================================================================
# 2D Volume-level validation (moved from vae.py)
# ==============================================================================


def _validate_vae_modality(
    data_dir: str,
    modality: str,
    context: str = "VAE",
    raise_on_error: bool = False,
) -> bool:
    """Validate VAE modality requirements exist in directory."""
    from medgen.data.dataset import validate_modality_exists

    from .common import validate_mode_requirements
    try:
        if modality == 'dual':
            validate_mode_requirements(
                data_dir, 'dual', validate_modality_exists, require_seg=False
            )
        else:
            validate_modality_exists(data_dir, modality)
        return True
    except ValueError as e:
        if raise_on_error:
            raise
        logger.warning(f"{context} data for {modality} mode not available in {data_dir}: {e}")
        return False


def _try_load_seg_dataset(
    data_dir: str,
    transform,
    context: str = "VAE",
):
    """Try to load segmentation dataset for regional metrics."""
    from medgen.data.dataset import NiFTIDataset, validate_modality_exists
    try:
        validate_modality_exists(data_dir, 'seg')
        return NiFTIDataset(data_dir=data_dir, mr_sequence='seg', transform=transform)
    except ValueError as e:
        logger.debug(f"Seg not available for {context} (regional metrics disabled): {e}")
        return None


def create_vae_volume_validation_dataloader(
    cfg,
    modality: str,
    data_split: str = 'val',
) -> tuple[DataLoader, Dataset] | None:
    """Create dataloader that returns full 3D volumes for volume-level metrics.

    Unlike slice-based loaders, this returns [C, H, W, D] volumes without
    slice extraction. Used for computing 3D MS-SSIM on 2D model reconstructions.

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        modality: Modality ('bravo', 'seg', 't1_pre', 't1_gd', 'dual').
        data_split: Which split to load ('val' or 'test_new').

    Returns:
        Tuple of (DataLoader, volume_dataset) or None if directory doesn't exist.
    """
    from medgen.data.dataset import NiFTIDataset, build_standard_transform
    from medgen.data.loaders.datasets import DualVolumeDataset, VolumeDataset

    data_dir = get_validated_split_dir(cfg.paths.data_dir, data_split, logger)
    if data_dir is None:
        return None

    image_size = cfg.model.image_size

    if not _validate_vae_modality(data_dir, modality, "Volume"):
        return None

    transform = build_standard_transform(image_size)

    if modality == 'dual':
        t1_pre_dataset = NiFTIDataset(data_dir, 't1_pre', transform)
        t1_gd_dataset = NiFTIDataset(data_dir, 't1_gd', transform)
        seg_dataset = _try_load_seg_dataset(data_dir, transform, "dual volume validation")
        volume_dataset = DualVolumeDataset(t1_pre_dataset, t1_gd_dataset, seg_dataset)
    else:
        image_dataset = NiFTIDataset(data_dir, modality, transform)
        seg_dataset = _try_load_seg_dataset(data_dir, transform, "single volume validation")
        volume_dataset = VolumeDataset(image_dataset, seg_dataset)

    dataloader = DataLoader(
        volume_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return dataloader, volume_dataset
