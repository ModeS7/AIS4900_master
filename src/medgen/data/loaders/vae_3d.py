"""3D VAE dataloaders for volumetric medical images.

Loads full 3D NIfTI volumes for training 3D AutoencoderKL models.
Key differences from 2D:
- Returns full volumes instead of 2D slices
- Applies depth padding for clean compression
- Memory-efficient batch sizes (typically 1-2)
"""
import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity
from torch.utils.data import DataLoader, Dataset

from ..dataset import NiFTIDataset

logger = logging.getLogger(__name__)


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


class Volume3DDataset(Dataset):
    """Dataset that loads 3D volumes with depth padding.

    Args:
        data_dir: Directory containing patient subdirectories.
        modality: MR sequence name (e.g., 'bravo', 't1_pre').
        height: Target height dimension.
        width: Target width dimension.
        pad_depth_to: Target depth after padding.
        pad_mode: Padding mode ('replicate' or 'constant').
        slice_step: Take every nth slice (1=all, 2=every 2nd, 3=every 3rd).
        load_seg: Whether to load segmentation masks for regional metrics.
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
    ) -> None:
        self.data_dir = data_dir
        self.modality = modality
        self.pad_depth_to = pad_depth_to
        self.pad_mode = pad_mode
        self.slice_step = slice_step
        self.load_seg = load_seg

        self.transform = build_3d_transform(height, width)

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

    def _pad_depth(self, volume: torch.Tensor) -> torch.Tensor:
        """Pad volume depth to target size."""
        current_depth = volume.shape[1]
        if current_depth < self.pad_depth_to:
            pad_total = self.pad_depth_to - current_depth
            if self.pad_mode == 'replicate':
                last_slice = volume[:, -1:, :, :]
                padding = last_slice.repeat(1, pad_total, 1, 1)
                volume = torch.cat([volume, padding], dim=1)
            else:
                volume = F.pad(volume, (0, 0, 0, 0, 0, pad_total), mode='constant', value=0)
        return volume

    def __getitem__(self, idx: int) -> dict:
        patient = self.patients[idx]
        nifti_path = os.path.join(self.data_dir, patient, f"{self.modality}.nii.gz")

        # Load and transform volume: [C, H, W, D]
        volume = self.transform(nifti_path)

        # Convert to tensor if not already
        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume).float()

        # MONAI loads as [C, H, W, D], we need [C, D, H, W] for 3D conv
        volume = volume.permute(0, 3, 1, 2)  # [C, D, H, W]

        # Subsample slices if slice_step > 1 (for quick testing)
        if self.slice_step > 1:
            volume = volume[:, ::self.slice_step, :, :]

        # Pad depth if needed
        volume = self._pad_depth(volume)

        result = {'image': volume, 'patient': patient}

        # Optionally load segmentation mask for regional metrics
        if self.load_seg:
            seg_path = os.path.join(self.data_dir, patient, "seg.nii.gz")
            if os.path.exists(seg_path):
                seg = self.transform(seg_path)
                if not isinstance(seg, torch.Tensor):
                    seg = torch.from_numpy(seg).float()
                seg = seg.permute(0, 3, 1, 2)
                if self.slice_step > 1:
                    seg = seg[:, ::self.slice_step, :, :]
                seg = self._pad_depth(seg)
                seg = (seg > 0.5).float()  # Binarize
                result['seg'] = seg

        return result


class DualVolume3DDataset(Dataset):
    """Dataset that loads dual-modality 3D volumes (t1_pre + t1_gd).

    Args:
        data_dir: Directory containing patient subdirectories.
        height: Target height dimension.
        width: Target width dimension.
        pad_depth_to: Target depth after padding.
        pad_mode: Padding mode ('replicate' or 'constant').
        slice_step: Take every nth slice (1=all, 2=every 2nd, 3=every 3rd).
        load_seg: Whether to load segmentation masks for regional metrics.
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
    ) -> None:
        self.data_dir = data_dir
        self.pad_depth_to = pad_depth_to
        self.pad_mode = pad_mode
        self.slice_step = slice_step
        self.load_seg = load_seg

        self.transform = build_3d_transform(height, width)

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

    def _pad_depth(self, volume: torch.Tensor) -> torch.Tensor:
        """Pad volume depth to target size."""
        current_depth = volume.shape[1]
        if current_depth < self.pad_depth_to:
            pad_total = self.pad_depth_to - current_depth
            if self.pad_mode == 'replicate':
                last_slice = volume[:, -1:, :, :]
                padding = last_slice.repeat(1, pad_total, 1, 1)
                volume = torch.cat([volume, padding], dim=1)
            else:
                volume = F.pad(volume, (0, 0, 0, 0, 0, pad_total), mode='constant', value=0)
        return volume

    def __getitem__(self, idx: int) -> dict:
        patient = self.patients[idx]

        # Load both modalities
        t1_pre_path = os.path.join(self.data_dir, patient, "t1_pre.nii.gz")
        t1_gd_path = os.path.join(self.data_dir, patient, "t1_gd.nii.gz")

        t1_pre = self.transform(t1_pre_path)
        t1_gd = self.transform(t1_gd_path)

        # Convert to tensors
        if not isinstance(t1_pre, torch.Tensor):
            t1_pre = torch.from_numpy(t1_pre).float()
        if not isinstance(t1_gd, torch.Tensor):
            t1_gd = torch.from_numpy(t1_gd).float()

        # Permute to [C, D, H, W]
        t1_pre = t1_pre.permute(0, 3, 1, 2)
        t1_gd = t1_gd.permute(0, 3, 1, 2)

        # Subsample slices if slice_step > 1 (for quick testing)
        if self.slice_step > 1:
            t1_pre = t1_pre[:, ::self.slice_step, :, :]
            t1_gd = t1_gd[:, ::self.slice_step, :, :]

        # Pad depth
        t1_pre = self._pad_depth(t1_pre)
        t1_gd = self._pad_depth(t1_gd)

        # Stack as 2 channels: [2, D, H, W]
        volume = torch.cat([t1_pre, t1_gd], dim=0)

        result = {'image': volume, 'patient': patient}

        # Optionally load segmentation mask for regional metrics
        if self.load_seg:
            seg_path = os.path.join(self.data_dir, patient, "seg.nii.gz")
            if os.path.exists(seg_path):
                seg = self.transform(seg_path)
                if not isinstance(seg, torch.Tensor):
                    seg = torch.from_numpy(seg).float()
                seg = seg.permute(0, 3, 1, 2)
                if self.slice_step > 1:
                    seg = seg[:, ::self.slice_step, :, :]
                seg = self._pad_depth(seg)
                # Binarize segmentation mask
                seg = (seg > 0.5).float()
                result['seg'] = seg

        return result


def create_vae_3d_dataloader(
    cfg,
    modality: str,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, Dataset]:
    """Create 3D VAE training dataloader.

    Args:
        cfg: Hydra configuration object.
        modality: Modality name or 'dual' for dual-channel.
        use_distributed: Whether to use distributed sampling.
        rank: Process rank for distributed training.
        world_size: Total number of processes.

    Returns:
        Tuple of (DataLoader, Dataset).
    """
    data_dir = os.path.join(cfg.paths.data_dir, 'train')
    height = cfg.volume.height
    width = cfg.volume.width
    pad_depth_to = cfg.volume.pad_depth_to
    pad_mode = cfg.volume.get('pad_mode', 'replicate')
    slice_step = cfg.volume.get('slice_step', 1)
    batch_size = cfg.training.batch_size

    # Check if regional losses are enabled (need seg for metrics)
    logging_cfg = cfg.training.get('logging', {})
    load_seg = logging_cfg.get('regional_losses', False)

    if modality == 'dual':
        dataset = DualVolume3DDataset(
            data_dir=data_dir,
            height=height,
            width=width,
            pad_depth_to=pad_depth_to,
            pad_mode=pad_mode,
            slice_step=slice_step,
            load_seg=load_seg,
        )
    else:
        dataset = Volume3DDataset(
            data_dir=data_dir,
            modality=modality,
            height=height,
            width=width,
            pad_depth_to=pad_depth_to,
            pad_mode=pad_mode,
            slice_step=slice_step,
            load_seg=load_seg,
        )

    if use_distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    return loader, dataset


def create_vae_3d_validation_dataloader(
    cfg,
    modality: str,
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create 3D VAE validation dataloader.

    Args:
        cfg: Hydra configuration object.
        modality: Modality name or 'dual' for dual-channel.

    Returns:
        Tuple of (DataLoader, Dataset) or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, 'val')
    if not os.path.exists(val_dir):
        return None

    height = cfg.volume.height
    width = cfg.volume.width
    pad_depth_to = cfg.volume.pad_depth_to
    pad_mode = cfg.volume.get('pad_mode', 'replicate')
    slice_step = cfg.volume.get('slice_step', 1)
    batch_size = cfg.training.batch_size

    # Check if regional losses are enabled (need seg for metrics)
    logging_cfg = cfg.training.get('logging', {})
    load_seg = logging_cfg.get('regional_losses', False)

    if modality == 'dual':
        dataset = DualVolume3DDataset(
            data_dir=val_dir,
            height=height,
            width=width,
            pad_depth_to=pad_depth_to,
            pad_mode=pad_mode,
            slice_step=slice_step,
            load_seg=load_seg,
        )
    else:
        dataset = Volume3DDataset(
            data_dir=val_dir,
            modality=modality,
            height=height,
            width=width,
            pad_depth_to=pad_depth_to,
            pad_mode=pad_mode,
            slice_step=slice_step,
            load_seg=load_seg,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return loader, dataset


def create_vae_3d_test_dataloader(
    cfg,
    modality: str,
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create 3D VAE test dataloader.

    Args:
        cfg: Hydra configuration object.
        modality: Modality name or 'dual' for dual-channel.

    Returns:
        Tuple of (DataLoader, Dataset) or None if test_new/ doesn't exist.
    """
    test_dir = os.path.join(cfg.paths.data_dir, 'test_new')
    if not os.path.exists(test_dir):
        return None

    height = cfg.volume.height
    width = cfg.volume.width
    pad_depth_to = cfg.volume.pad_depth_to
    pad_mode = cfg.volume.get('pad_mode', 'replicate')
    slice_step = cfg.volume.get('slice_step', 1)
    batch_size = cfg.training.batch_size

    # Check if regional losses are enabled (need seg for metrics)
    logging_cfg = cfg.training.get('logging', {})
    load_seg = logging_cfg.get('regional_losses', False)

    if modality == 'dual':
        dataset = DualVolume3DDataset(
            data_dir=test_dir,
            height=height,
            width=width,
            pad_depth_to=pad_depth_to,
            pad_mode=pad_mode,
            slice_step=slice_step,
            load_seg=load_seg,
        )
    else:
        dataset = Volume3DDataset(
            data_dir=test_dir,
            modality=modality,
            height=height,
            width=width,
            pad_depth_to=pad_depth_to,
            pad_mode=pad_mode,
            slice_step=slice_step,
            load_seg=load_seg,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return loader, dataset


class MultiModality3DDataset(Dataset):
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
    ) -> None:
        self.data_dir = data_dir
        self.image_keys = image_keys
        self.pad_depth_to = pad_depth_to
        self.pad_mode = pad_mode
        self.slice_step = slice_step
        self.load_seg = load_seg

        self.transform = build_3d_transform(height, width)

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

    def _pad_depth(self, volume: torch.Tensor) -> torch.Tensor:
        """Pad volume depth to target size."""
        current_depth = volume.shape[1]
        if current_depth < self.pad_depth_to:
            pad_total = self.pad_depth_to - current_depth
            if self.pad_mode == 'replicate':
                last_slice = volume[:, -1:, :, :]
                padding = last_slice.repeat(1, pad_total, 1, 1)
                volume = torch.cat([volume, padding], dim=1)
            else:
                volume = F.pad(volume, (0, 0, 0, 0, 0, pad_total), mode='constant', value=0)
        return volume

    def __getitem__(self, idx: int) -> dict:
        p_idx, modality = self.samples[idx]
        patient = self.patients[p_idx]
        nifti_path = os.path.join(self.data_dir, patient, f"{modality}.nii.gz")

        # Load and transform volume: [C, H, W, D]
        volume = self.transform(nifti_path)

        # Convert to tensor if not already
        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume).float()

        # MONAI loads as [C, H, W, D], we need [C, D, H, W] for 3D conv
        volume = volume.permute(0, 3, 1, 2)  # [C, D, H, W]

        # Subsample slices if slice_step > 1
        if self.slice_step > 1:
            volume = volume[:, ::self.slice_step, :, :]

        # Pad depth
        volume = self._pad_depth(volume)

        result = {'image': volume, 'patient': patient, 'modality': modality}

        # Optionally load segmentation mask for regional metrics
        if self.load_seg:
            seg_path = os.path.join(self.data_dir, patient, "seg.nii.gz")
            if os.path.exists(seg_path):
                seg = self.transform(seg_path)
                if not isinstance(seg, torch.Tensor):
                    seg = torch.from_numpy(seg).float()
                seg = seg.permute(0, 3, 1, 2)
                if self.slice_step > 1:
                    seg = seg[:, ::self.slice_step, :, :]
                seg = self._pad_depth(seg)
                seg = (seg > 0.5).float()  # Binarize
                result['seg'] = seg

        return result


def create_vae_3d_multi_modality_dataloader(
    cfg,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, Dataset]:
    """Create 3D VAE multi-modality training dataloader.

    Args:
        cfg: Hydra configuration object.
        use_distributed: Whether to use distributed sampling.
        rank: Process rank for distributed training.
        world_size: Total number of processes.

    Returns:
        Tuple of (DataLoader, Dataset).
    """
    data_dir = os.path.join(cfg.paths.data_dir, 'train')
    height = cfg.volume.height
    width = cfg.volume.width
    pad_depth_to = cfg.volume.pad_depth_to
    pad_mode = cfg.volume.get('pad_mode', 'replicate')
    slice_step = cfg.volume.get('slice_step', 1)
    batch_size = cfg.training.batch_size
    image_keys = cfg.mode.get('image_keys', ['bravo', 'flair', 't1_pre', 't1_gd'])

    # Check if regional losses are enabled (need seg for metrics)
    logging_cfg = cfg.training.get('logging', {})
    load_seg = logging_cfg.get('regional_losses', False)

    dataset = MultiModality3DDataset(
        data_dir=data_dir,
        image_keys=image_keys,
        height=height,
        width=width,
        pad_depth_to=pad_depth_to,
        pad_mode=pad_mode,
        slice_step=slice_step,
        load_seg=load_seg,
    )

    if use_distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    return loader, dataset


def create_vae_3d_multi_modality_validation_dataloader(
    cfg,
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create 3D VAE multi-modality validation dataloader.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Tuple of (DataLoader, Dataset) or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, 'val')
    if not os.path.exists(val_dir):
        return None

    height = cfg.volume.height
    width = cfg.volume.width
    pad_depth_to = cfg.volume.pad_depth_to
    pad_mode = cfg.volume.get('pad_mode', 'replicate')
    slice_step = cfg.volume.get('slice_step', 1)
    batch_size = cfg.training.batch_size
    image_keys = cfg.mode.get('image_keys', ['bravo', 'flair', 't1_pre', 't1_gd'])

    # Check if regional losses are enabled (need seg for metrics)
    logging_cfg = cfg.training.get('logging', {})
    load_seg = logging_cfg.get('regional_losses', False)

    dataset = MultiModality3DDataset(
        data_dir=val_dir,
        image_keys=image_keys,
        height=height,
        width=width,
        pad_depth_to=pad_depth_to,
        pad_mode=pad_mode,
        slice_step=slice_step,
        load_seg=load_seg,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return loader, dataset


def create_vae_3d_multi_modality_test_dataloader(
    cfg,
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create 3D VAE multi-modality test dataloader.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Tuple of (DataLoader, Dataset) or None if test_new/ doesn't exist.
    """
    test_dir = os.path.join(cfg.paths.data_dir, 'test_new')
    if not os.path.exists(test_dir):
        return None

    height = cfg.volume.height
    width = cfg.volume.width
    pad_depth_to = cfg.volume.pad_depth_to
    pad_mode = cfg.volume.get('pad_mode', 'replicate')
    slice_step = cfg.volume.get('slice_step', 1)
    batch_size = cfg.training.batch_size
    image_keys = cfg.mode.get('image_keys', ['bravo', 'flair', 't1_pre', 't1_gd'])

    # Check if regional losses are enabled (need seg for metrics)
    logging_cfg = cfg.training.get('logging', {})
    load_seg = logging_cfg.get('regional_losses', False)

    dataset = MultiModality3DDataset(
        data_dir=test_dir,
        image_keys=image_keys,
        height=height,
        width=width,
        pad_depth_to=pad_depth_to,
        pad_mode=pad_mode,
        slice_step=slice_step,
        load_seg=load_seg,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return loader, dataset


class SingleModality3DDatasetWithSeg(Dataset):
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
        self.data_dir = data_dir
        self.modality = modality
        self.pad_depth_to = pad_depth_to
        self.pad_mode = pad_mode
        self.slice_step = slice_step

        self.transform = build_3d_transform(height, width)

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

    def _pad_depth(self, volume: torch.Tensor) -> torch.Tensor:
        """Pad volume depth to target size."""
        current_depth = volume.shape[1]
        if current_depth < self.pad_depth_to:
            pad_total = self.pad_depth_to - current_depth
            if self.pad_mode == 'replicate':
                last_slice = volume[:, -1:, :, :]
                padding = last_slice.repeat(1, pad_total, 1, 1)
                volume = torch.cat([volume, padding], dim=1)
            else:
                volume = F.pad(volume, (0, 0, 0, 0, 0, pad_total), mode='constant', value=0)
        return volume

    def __getitem__(self, idx: int) -> dict:
        patient, has_seg = self.samples[idx]
        modality_path = os.path.join(self.data_dir, patient, f"{self.modality}.nii.gz")

        # Load and transform volume: [C, H, W, D]
        volume = self.transform(modality_path)

        # Convert to tensor if not already
        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume).float()

        # Permute to [C, D, H, W]
        volume = volume.permute(0, 3, 1, 2)

        # Subsample slices if slice_step > 1
        if self.slice_step > 1:
            volume = volume[:, ::self.slice_step, :, :]

        # Pad depth
        volume = self._pad_depth(volume)

        result = {'image': volume, 'patient': patient, 'modality': self.modality}

        # Load segmentation if available
        if has_seg:
            seg_path = os.path.join(self.data_dir, patient, "seg.nii.gz")
            seg = self.transform(seg_path)
            if not isinstance(seg, torch.Tensor):
                seg = torch.from_numpy(seg).float()
            seg = seg.permute(0, 3, 1, 2)
            if self.slice_step > 1:
                seg = seg[:, ::self.slice_step, :, :]
            seg = self._pad_depth(seg)
            # Binarize segmentation (any non-zero is tumor)
            seg = (seg > 0.5).float()
            result['seg'] = seg

        return result


def create_vae_3d_single_modality_validation_loader(
    cfg,
    modality: str,
) -> Optional[DataLoader]:
    """Create 3D validation loader for a single modality (for per-modality metrics).

    Includes seg masks paired with each volume for regional metrics tracking.

    Args:
        cfg: Hydra configuration object.
        modality: Single modality to load (e.g., 'bravo', 't1_pre', 't1_gd').

    Returns:
        DataLoader for single modality or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, 'val')
    if not os.path.exists(val_dir):
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

    height = cfg.volume.height
    width = cfg.volume.width
    pad_depth_to = cfg.volume.pad_depth_to
    pad_mode = cfg.volume.get('pad_mode', 'replicate')
    slice_step = cfg.volume.get('slice_step', 1)
    batch_size = cfg.training.batch_size

    try:
        dataset = SingleModality3DDatasetWithSeg(
            data_dir=val_dir,
            modality=modality,
            height=height,
            width=width,
            pad_depth_to=pad_depth_to,
            pad_mode=pad_mode,
            slice_step=slice_step,
        )
    except ValueError as e:
        logger.warning(f"Could not create dataset for {modality}: {e}")
        return None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return loader
