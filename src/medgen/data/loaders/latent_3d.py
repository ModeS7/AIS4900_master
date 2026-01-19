"""
3D Latent space dataloaders for volumetric latent diffusion training.

Provides dataloaders that load pre-encoded 3D latent tensors from cache,
plus utilities for building and validating the 3D latent cache.
"""
import hashlib
import json
import logging
import os
from datetime import datetime
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import torch
from monai.data import DataLoader
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from medgen.data.loaders.common import DataLoaderConfig

logger = logging.getLogger(__name__)


class Latent3DDataset(Dataset):
    """Dataset that loads pre-encoded 3D latent volumes from cache.

    Each .pt file contains:
    - 'latent': Encoded volume tensor [C_latent, D_latent, H_latent, W_latent]
    - 'seg_mask': Original pixel-space segmentation mask [1, D, H, W] (if conditional)
    - 'patient_id': Patient identifier string

    Args:
        cache_dir: Directory containing pre-encoded .pt files.
        mode: Training mode ('bravo', 'dual', 'seg', 'multi', 'multi_modality').
    """

    def __init__(self, cache_dir: str, mode: str) -> None:
        self.cache_dir = cache_dir
        self.mode = mode

        # Find all .pt files in cache directory
        self.files = sorted(glob(os.path.join(cache_dir, "*.pt")))

        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in 3D cache directory: {cache_dir}")

        logger.info(f"Latent3DDataset: Found {len(self.files)} volumes in {cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a single 3D volume from cache.

        Returns:
            Dictionary with 'latent', optionally 'seg_mask', 'patient_id'.
        """
        data = torch.load(self.files[idx], weights_only=False)

        # Ensure latent is float32
        if 'latent' in data:
            data['latent'] = data['latent'].float()

        # Ensure seg_mask is float32 if present
        if 'seg_mask' in data and data['seg_mask'] is not None:
            data['seg_mask'] = data['seg_mask'].float()

        return data


class LatentCacheBuilder3D:
    """Builds and validates 3D latent cache from pixel-space volumes.

    Encodes 3D volumes using a compression model (VAE/VQ-VAE) and saves
    them as .pt files for fast loading during diffusion training.

    Unlike 2D, 3D processes one volume at a time (no batching due to memory).

    Args:
        compression_model: Trained 3D compression model with encode() method.
        device: Device for encoding.
        mode: Training mode for determining what to encode.
        volume_shape: Original volume shape (D, H, W).
        compression_type: Type of compression model ('vae', 'dcae', 'vqvae').
    """

    def __init__(
        self,
        compression_model: torch.nn.Module,
        device: torch.device,
        mode: str,
        volume_shape: Tuple[int, int, int],
        compression_type: str = "vae",
        verbose: bool = True,
    ) -> None:
        self.model = compression_model.eval()
        self.device = device
        self.mode = mode
        self.volume_shape = volume_shape  # (D, H, W)
        self.compression_type = compression_type
        self.verbose = verbose

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    @staticmethod
    def compute_checkpoint_hash(checkpoint_path: str) -> str:
        """Compute SHA256 hash of checkpoint file (first 16 chars).

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            First 16 characters of SHA256 hash.
        """
        sha256 = hashlib.sha256()
        with open(checkpoint_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]

    def validate_cache(self, cache_dir: str, checkpoint_path: str) -> bool:
        """Check if cache exists and matches checkpoint hash.

        Args:
            cache_dir: Directory containing cached latents.
            checkpoint_path: Path to compression model checkpoint.

        Returns:
            True if cache is valid, False otherwise.
        """
        metadata_path = os.path.join(cache_dir, "metadata.json")

        if not os.path.exists(metadata_path):
            logger.info(f"3D cache metadata not found at {metadata_path}")
            return False

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read 3D cache metadata: {e}")
            return False

        # Check hash match
        expected_hash = self.compute_checkpoint_hash(checkpoint_path)
        cached_hash = metadata.get('checkpoint_hash', '')

        if cached_hash != expected_hash:
            logger.info(
                f"3D cache hash mismatch: cached={cached_hash}, "
                f"expected={expected_hash}"
            )
            return False

        # Check mode match
        cached_mode = metadata.get('mode', '')
        if cached_mode != self.mode:
            logger.info(f"3D cache mode mismatch: cached={cached_mode}, expected={self.mode}")
            return False

        # Check spatial dims
        cached_dims = metadata.get('spatial_dims', 2)
        if cached_dims != 3:
            logger.info(f"3D cache spatial_dims mismatch: cached={cached_dims}, expected=3")
            return False

        # Check number of samples
        num_samples = metadata.get('num_samples', 0)
        actual_files = len(glob(os.path.join(cache_dir, "*.pt")))
        if actual_files != num_samples:
            logger.info(
                f"3D cache sample count mismatch: metadata={num_samples}, "
                f"actual={actual_files}"
            )
            return False

        logger.info(f"3D cache valid: {cache_dir} ({num_samples} volumes)")
        return True

    def build_cache(
        self,
        volume_dataset: Dataset,
        cache_dir: str,
        checkpoint_path: str,
    ) -> None:
        """Encode entire 3D dataset and save as .pt files.

        Unlike 2D, 3D processes one volume at a time (no batching).

        Args:
            volume_dataset: Dataset returning 3D pixel-space volumes.
            cache_dir: Directory to save encoded latents.
            checkpoint_path: Path to compression checkpoint (for hash).
        """
        os.makedirs(cache_dir, exist_ok=True)

        latent_shape = None
        sample_idx = 0

        logger.info(f"Encoding {len(volume_dataset)} 3D volumes to {cache_dir}...")

        with torch.no_grad():
            for idx in tqdm(range(len(volume_dataset)), desc="Encoding 3D volumes", disable=not self.verbose):
                # Get single volume
                batch = volume_dataset[idx]

                # Parse batch format
                volume, seg_mask, patient_id = self._parse_batch(batch)

                # Move to device and add batch dimension
                volume = volume.unsqueeze(0).to(self.device)  # [1, C, D, H, W]

                # Encode
                latent = self._encode(volume)  # [1, C_lat, D', H', W']

                if latent_shape is None:
                    latent_shape = list(latent.shape[1:])

                # Save sample
                sample_data = {
                    'latent': latent.squeeze(0).cpu(),
                    'patient_id': patient_id if patient_id else f"volume_{idx}",
                }

                # Include seg mask if available (keep in pixel space)
                if seg_mask is not None:
                    sample_data['seg_mask'] = seg_mask.cpu()

                # Save to file
                filename = f"volume_{sample_idx:06d}.pt"
                torch.save(sample_data, os.path.join(cache_dir, filename))
                sample_idx += 1

        # Save metadata
        metadata = {
            'compression_checkpoint': checkpoint_path,
            'checkpoint_hash': self.compute_checkpoint_hash(checkpoint_path),
            'compression_type': self.compression_type,
            'spatial_dims': 3,
            'latent_shape': latent_shape,
            'pixel_shape': list(self.volume_shape),
            'mode': self.mode,
            'num_samples': sample_idx,
            'created_at': datetime.now().isoformat(),
        }

        with open(os.path.join(cache_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"3D cache built: {sample_idx} volumes, latent shape {latent_shape}")

    def _parse_batch(
        self, batch: Any
    ) -> Tuple[Tensor, Optional[Tensor], Optional[str]]:
        """Parse batch into volume, seg mask, and patient ID.

        Args:
            batch: Single sample from 3D dataset.

        Returns:
            Tuple of (volume, seg_mask, patient_id).
        """
        seg_mask = None
        patient_id = None

        if isinstance(batch, dict):
            volume = batch.get('image', batch.get('volume'))
            seg_mask = batch.get('seg_mask', batch.get('seg'))
            patient_id = batch.get('patient_id', batch.get('patient'))
        elif isinstance(batch, tuple):
            if len(batch) >= 2:
                volume, seg_mask = batch[0], batch[1]
                if len(batch) >= 3:
                    patient_id = batch[2]
            else:
                volume = batch[0]
        else:
            volume = batch

        return volume, seg_mask, patient_id

    def _encode(self, volume: Tensor) -> Tensor:
        """Encode 3D volume to latent space.

        Handles different compression model types.

        Args:
            volume: Volume [B, C, D, H, W].

        Returns:
            Latent representation.
        """
        if self.compression_type == 'vae':
            # VAE returns (mu, logvar) - use mu for deterministic encoding
            z_mu, _ = self.model.encode(volume)
            return z_mu
        elif self.compression_type == 'vqvae':
            # VQ-VAE returns quantized directly
            return self.model.encode(volume)
        elif self.compression_type == 'dcae':
            # DC-AE is deterministic
            return self.model.encode(volume)
        else:
            # Generic fallback
            result = self.model.encode(volume)
            if isinstance(result, tuple):
                return result[0]
            return result


def create_latent_3d_dataloader(
    cfg: DictConfig,
    cache_dir: str,
    split: str,
    mode: str,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[DataLoader, Latent3DDataset]:
    """Create dataloader for pre-encoded 3D latents.

    Args:
        cfg: Hydra configuration.
        cache_dir: Base cache directory (contains train/, val/ subdirs).
        split: Data split ('train', 'val', 'test_new').
        mode: Training mode.
        batch_size: Override batch size (default: cfg.training.batch_size).
        shuffle: Whether to shuffle data.

    Returns:
        Tuple of (DataLoader, Latent3DDataset).
    """
    split_cache_dir = os.path.join(cache_dir, split)

    if not os.path.exists(split_cache_dir):
        raise ValueError(f"3D cache directory not found: {split_cache_dir}")

    dataset = Latent3DDataset(split_cache_dir, mode)

    batch_size = batch_size or cfg.training.batch_size

    # Get DataLoader settings
    dl_cfg = DataLoaderConfig.from_cfg(cfg)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )

    return dataloader, dataset


def create_latent_3d_validation_dataloader(
    cfg: DictConfig,
    cache_dir: str,
    mode: str,
    batch_size: Optional[int] = None,
) -> Optional[Tuple[DataLoader, Latent3DDataset]]:
    """Create validation dataloader for pre-encoded 3D latents.

    Args:
        cfg: Hydra configuration.
        cache_dir: Base cache directory.
        mode: Training mode.
        batch_size: Override batch size.

    Returns:
        Tuple of (DataLoader, Latent3DDataset) or None if val cache doesn't exist.
    """
    val_cache_dir = os.path.join(cache_dir, 'val')

    if not os.path.exists(val_cache_dir):
        return None

    batch_size = batch_size or cfg.training.batch_size

    dataset = Latent3DDataset(val_cache_dir, mode)

    dl_cfg = DataLoaderConfig.from_cfg(cfg)

    # Fixed seed for reproducible validation
    val_generator = torch.Generator().manual_seed(42)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for diverse worst_batch
        drop_last=True,
        generator=val_generator,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )

    return dataloader, dataset
