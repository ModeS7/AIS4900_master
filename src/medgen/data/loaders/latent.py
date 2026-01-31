"""
Latent space dataloaders for latent diffusion training.

Provides dataloaders that load pre-encoded latent tensors from cache,
plus utilities for building and validating the latent cache.

Supports both 2D images and 3D volumes via spatial_dims parameter.
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

from medgen.data.loaders.common import DataLoaderConfig, setup_distributed_sampler

logger = logging.getLogger(__name__)


class LatentDataset(Dataset):
    """Dataset that loads pre-encoded latent tensors from cache.

    Supports both 2D and 3D latents:
    - 2D: [C_latent, H_latent, W_latent]
    - 3D: [C_latent, D_latent, H_latent, W_latent]

    Each .pt file contains:
    - 'latent': Encoded tensor
    - 'seg_mask': Original pixel-space segmentation mask (if conditional)
    - 'patient_id': Patient identifier string
    - 'slice_idx': Slice index within the volume (2D only)

    Args:
        cache_dir: Directory containing pre-encoded .pt files.
        mode: Training mode ('bravo', 'dual', 'seg', 'multi', 'multi_modality').
        spatial_dims: Number of spatial dimensions (2 or 3). Auto-detected if None.
    """

    def __init__(
        self,
        cache_dir: str,
        mode: str,
        spatial_dims: Optional[int] = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.mode = mode

        # Find all .pt files in cache directory
        self.files = sorted(glob(os.path.join(cache_dir, "*.pt")))

        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in cache directory: {cache_dir}")

        # Auto-detect spatial dims from metadata if not specified
        if spatial_dims is None:
            spatial_dims = self._detect_spatial_dims()
        self.spatial_dims = spatial_dims

        name = "LatentDataset" if spatial_dims == 2 else "Latent3DDataset"
        logger.info(f"{name}: Found {len(self.files)} samples in {cache_dir}")

    def _detect_spatial_dims(self) -> int:
        """Detect spatial dimensions from cache metadata or first sample."""
        metadata_path = os.path.join(self.cache_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return metadata.get('spatial_dims', 2)
            except (json.JSONDecodeError, IOError):
                pass

        # Fallback: check first sample's latent shape
        if self.files:
            data = torch.load(self.files[0], weights_only=False)
            if 'latent' in data:
                # 4D = [C, D, H, W] = 3D spatial
                # 3D = [C, H, W] = 2D spatial
                return 3 if data['latent'].dim() == 4 else 2

        return 2  # Default to 2D

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a single sample from cache.

        Returns:
            Dictionary with 'latent', optionally 'latent_seg', 'seg_mask', 'patient_id', 'slice_idx'.
        """
        data = torch.load(self.files[idx], weights_only=False)

        # Ensure latent is float32
        if 'latent' in data:
            data['latent'] = data['latent'].float()

        # Ensure latent_seg is float32 if present (for seg conditioning)
        if 'latent_seg' in data and data['latent_seg'] is not None:
            data['latent_seg'] = data['latent_seg'].float()

        # Ensure seg_mask is float32 if present (pixel-space for regional metrics)
        if 'seg_mask' in data and data['seg_mask'] is not None:
            data['seg_mask'] = data['seg_mask'].float()

        return data


# Backwards compatibility alias
Latent3DDataset = LatentDataset


class LatentCacheBuilder:
    """Builds and validates latent cache from pixel-space datasets.

    Encodes images/volumes using a compression model (VAE/DC-AE/VQ-VAE) and saves
    them as .pt files for fast loading during diffusion training.

    Supports both 2D (batched encoding) and 3D (single-volume encoding due to memory).

    Args:
        compression_model: Trained compression model with encode() method.
        device: Device for encoding.
        mode: Training mode for determining what to encode.
        spatial_dims: Number of spatial dimensions (2 or 3).
        image_size: Original image size for 2D (e.g., 256).
        volume_shape: Original volume shape (D, H, W) for 3D.
        compression_type: Type of compression model ('vae', 'dcae', 'vqvae').
        verbose: Whether to show progress bars.
    """

    def __init__(
        self,
        compression_model: torch.nn.Module,
        device: torch.device,
        mode: str,
        spatial_dims: int = 2,
        image_size: Optional[int] = None,
        volume_shape: Optional[Tuple[int, int, int]] = None,
        compression_type: str = "vae",
        verbose: bool = True,
    ) -> None:
        self.model = compression_model.eval()
        self.device = device
        self.mode = mode
        self.spatial_dims = spatial_dims
        self.image_size = image_size
        self.volume_shape = volume_shape
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
            # Read in chunks to handle large files
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
            logger.info(f"Cache metadata not found at {metadata_path}")
            return False

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read cache metadata: {e}")
            return False

        # Check hash match
        expected_hash = self.compute_checkpoint_hash(checkpoint_path)
        cached_hash = metadata.get('checkpoint_hash', '')

        if cached_hash != expected_hash:
            logger.info(
                f"Cache hash mismatch: cached={cached_hash}, "
                f"expected={expected_hash}"
            )
            return False

        # Check mode match
        cached_mode = metadata.get('mode', '')
        if cached_mode != self.mode:
            logger.info(f"Cache mode mismatch: cached={cached_mode}, expected={self.mode}")
            return False

        # Check spatial dims match
        cached_dims = metadata.get('spatial_dims', 2)
        if cached_dims != self.spatial_dims:
            logger.info(f"Cache spatial_dims mismatch: cached={cached_dims}, expected={self.spatial_dims}")
            return False

        # Check number of samples
        num_samples = metadata.get('num_samples', 0)
        actual_files = len(glob(os.path.join(cache_dir, "*.pt")))
        if actual_files != num_samples:
            logger.info(
                f"Cache sample count mismatch: metadata={num_samples}, "
                f"actual={actual_files}"
            )
            return False

        logger.info(f"Cache valid: {cache_dir} ({num_samples} samples)")
        return True

    def build_cache(
        self,
        pixel_dataset: Dataset,
        cache_dir: str,
        checkpoint_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        """Encode entire dataset and save as .pt files.

        For 2D: Uses batched encoding for efficiency.
        For 3D: Processes one volume at a time due to memory constraints.

        Args:
            pixel_dataset: Dataset returning pixel-space images/volumes.
            cache_dir: Directory to save encoded latents.
            checkpoint_path: Path to compression checkpoint (for hash).
            batch_size: Batch size for encoding (2D only).
            num_workers: Number of dataloader workers (2D only).
        """
        os.makedirs(cache_dir, exist_ok=True)

        if self.spatial_dims == 2:
            self._build_cache_2d(pixel_dataset, cache_dir, checkpoint_path, batch_size, num_workers)
        else:
            self._build_cache_3d(pixel_dataset, cache_dir, checkpoint_path)

    def _build_cache_2d(
        self,
        pixel_dataset: Dataset,
        cache_dir: str,
        checkpoint_path: str,
        batch_size: int,
        num_workers: int,
    ) -> None:
        """Build 2D cache with batched encoding."""
        # Create temporary dataloader for encoding
        temp_loader = DataLoader(
            pixel_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        sample_idx = 0
        latent_shape = None

        logger.info(f"Encoding {len(pixel_dataset)} 2D samples to {cache_dir}...")

        with torch.no_grad():
            for batch in tqdm(temp_loader, desc="Encoding latents", disable=not self.verbose):
                # Handle different batch formats
                images, seg_masks, patient_ids, slice_indices = self._parse_batch_2d(batch)

                # Move to device and encode
                images = images.to(self.device, non_blocking=True)

                # Encode images
                latents = self._encode(images)

                if latent_shape is None:
                    latent_shape = list(latents.shape[1:])  # [C, H, W]

                # Encode seg masks to latent if available
                latent_segs = None
                if seg_masks is not None:
                    seg_masks_device = seg_masks.to(self.device, non_blocking=True)
                    latent_segs = self._encode(seg_masks_device)

                # Save each sample
                for i in range(latents.shape[0]):
                    sample_data = {
                        'latent': latents[i].cpu(),
                        'patient_id': patient_ids[i] if patient_ids else f"sample_{sample_idx}",
                        'slice_idx': slice_indices[i] if slice_indices else sample_idx,
                    }

                    # Include seg mask if available
                    if seg_masks is not None:
                        # Keep pixel-space seg for regional metrics
                        sample_data['seg_mask'] = seg_masks[i].cpu()
                        # Also store latent seg for seg conditioning
                        sample_data['latent_seg'] = latent_segs[i].cpu()

                    # Save to file
                    filename = f"sample_{sample_idx:06d}.pt"
                    torch.save(sample_data, os.path.join(cache_dir, filename))
                    sample_idx += 1

        # Save metadata
        metadata = {
            'compression_checkpoint': checkpoint_path,
            'checkpoint_hash': self.compute_checkpoint_hash(checkpoint_path),
            'compression_type': self.compression_type,
            'spatial_dims': 2,
            'latent_shape': latent_shape,
            'mode': self.mode,
            'image_size': self.image_size,
            'num_samples': sample_idx,
            'created_at': datetime.now().isoformat(),
        }

        with open(os.path.join(cache_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"2D cache built: {sample_idx} samples, latent shape {latent_shape}")

    def _build_cache_3d(
        self,
        volume_dataset: Dataset,
        cache_dir: str,
        checkpoint_path: str,
    ) -> None:
        """Build 3D cache with single-volume encoding."""
        latent_shape = None
        sample_idx = 0

        logger.info(f"Encoding {len(volume_dataset)} 3D volumes to {cache_dir}...")

        with torch.no_grad():
            for idx in tqdm(range(len(volume_dataset)), desc="Encoding 3D volumes", disable=not self.verbose):
                # Get single volume
                batch = volume_dataset[idx]

                # Parse batch format
                volume, seg_mask, patient_id = self._parse_batch_3d(batch)

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

                # Include seg mask if available
                if seg_mask is not None:
                    # Keep pixel-space seg for regional metrics
                    sample_data['seg_mask'] = seg_mask.cpu()

                    # Also encode seg to latent space for seg conditioning
                    # Use same encoder - works even if not trained on seg
                    seg_input = seg_mask.unsqueeze(0).to(self.device)  # [1, 1, D, H, W]
                    latent_seg = self._encode(seg_input)  # [1, C_lat, D', H', W']
                    sample_data['latent_seg'] = latent_seg.squeeze(0).cpu()

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
            'pixel_shape': list(self.volume_shape) if self.volume_shape else None,
            'mode': self.mode,
            'num_samples': sample_idx,
            'created_at': datetime.now().isoformat(),
        }

        with open(os.path.join(cache_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"3D cache built: {sample_idx} volumes, latent shape {latent_shape}")

    def _parse_batch_2d(
        self, batch: Any
    ) -> Tuple[Tensor, Optional[Tensor], Optional[List[str]], Optional[List[int]]]:
        """Parse 2D batch into images, seg masks, and metadata."""
        import numpy as np

        patient_ids = None
        slice_indices = None
        seg_masks = None

        if isinstance(batch, tuple):
            if len(batch) == 2:
                images, seg_masks = batch
            elif len(batch) == 3:
                images, seg_masks, metadata = batch
                if isinstance(metadata, dict):
                    patient_ids = metadata.get('patient_id')
                    slice_indices = metadata.get('slice_idx')
            else:
                images = batch[0]
        elif isinstance(batch, dict):
            images = batch.get('image', batch.get('images'))
            seg_masks = batch.get('seg_mask', batch.get('seg'))
            patient_ids = batch.get('patient_id')
            slice_indices = batch.get('slice_idx')
        elif isinstance(batch, np.ndarray):
            images = torch.from_numpy(batch).float()
        else:
            images = batch

        # Ensure tensor format
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        if seg_masks is not None and isinstance(seg_masks, np.ndarray):
            seg_masks = torch.from_numpy(seg_masks).float()

        return images, seg_masks, patient_ids, slice_indices

    def _parse_batch_3d(
        self, batch: Any
    ) -> Tuple[Tensor, Optional[Tensor], Optional[str]]:
        """Parse 3D batch into volume, seg mask, and patient ID."""
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

    def _encode(self, images: Tensor) -> Tensor:
        """Encode images/volumes to latent space.

        Handles different compression model types.

        Args:
            images: Images [B, C, H, W] or volumes [B, C, D, H, W].

        Returns:
            Latent representation.
        """
        if self.compression_type == 'vae':
            # VAE returns (mu, logvar) - use mu for deterministic encoding
            z_mu, _ = self.model.encode(images)
            return z_mu
        elif self.compression_type == 'vqvae':
            # VQ-VAE returns quantized directly
            return self.model.encode(images)
        elif self.compression_type == 'dcae':
            # DC-AE is deterministic
            return self.model.encode(images)
        else:
            # Generic fallback
            result = self.model.encode(images)
            if isinstance(result, tuple):
                return result[0]
            return result


# Backwards compatibility alias
class LatentCacheBuilder3D(LatentCacheBuilder):
    """3D latent cache builder (backwards compatibility wrapper).

    Equivalent to LatentCacheBuilder(..., spatial_dims=3).
    """

    def __init__(self, compression_model, device, mode, volume_shape=None, **kwargs):
        kwargs['spatial_dims'] = 3
        kwargs['volume_shape'] = volume_shape
        super().__init__(compression_model, device, mode, **kwargs)


# =============================================================================
# Dataloader Factory Functions
# =============================================================================

def create_latent_dataloader(
    cfg: DictConfig,
    cache_dir: str,
    split: str,
    mode: str,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    spatial_dims: Optional[int] = None,
) -> Tuple[DataLoader, LatentDataset]:
    """Create dataloader for pre-encoded latents.

    Supports both 2D and 3D latents (auto-detected from cache metadata).

    Args:
        cfg: Hydra configuration.
        cache_dir: Base cache directory (contains train/, val/ subdirs).
        split: Data split ('train', 'val', 'test_new').
        mode: Training mode.
        batch_size: Override batch size (default: cfg.training.batch_size).
        shuffle: Whether to shuffle data.
        use_distributed: Whether to use distributed sampling.
        rank: Process rank for distributed training.
        world_size: Total number of processes.
        spatial_dims: Override spatial dimensions (auto-detected if None).

    Returns:
        Tuple of (DataLoader, LatentDataset).
    """
    split_cache_dir = os.path.join(cache_dir, split)

    if not os.path.exists(split_cache_dir):
        raise ValueError(f"Cache directory not found: {split_cache_dir}")

    dataset = LatentDataset(split_cache_dir, mode, spatial_dims=spatial_dims)

    batch_size = batch_size or cfg.training.batch_size

    # Setup distributed sampler
    sampler, batch_size_per_gpu, actual_shuffle = setup_distributed_sampler(
        dataset, use_distributed, rank, world_size, batch_size, shuffle=shuffle
    )

    # Get DataLoader settings
    dl_cfg = DataLoaderConfig.from_cfg(cfg)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        shuffle=actual_shuffle,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )

    return dataloader, dataset


def create_latent_validation_dataloader(
    cfg: DictConfig,
    cache_dir: str,
    mode: str,
    batch_size: Optional[int] = None,
    world_size: int = 1,
    spatial_dims: Optional[int] = None,
) -> Optional[Tuple[DataLoader, LatentDataset]]:
    """Create validation dataloader for pre-encoded latents.

    Args:
        cfg: Hydra configuration.
        cache_dir: Base cache directory.
        mode: Training mode.
        batch_size: Override batch size.
        world_size: Number of GPUs for DDP.
        spatial_dims: Override spatial dimensions (auto-detected if None).

    Returns:
        Tuple of (DataLoader, LatentDataset) or None if val cache doesn't exist.
    """
    val_cache_dir = os.path.join(cache_dir, 'val')

    if not os.path.exists(val_cache_dir):
        return None

    batch_size = batch_size or cfg.training.batch_size

    # Reduce batch size for DDP (validation runs on single GPU)
    if world_size > 1:
        batch_size = max(1, batch_size // world_size)

    dataset = LatentDataset(val_cache_dir, mode, spatial_dims=spatial_dims)

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


def create_latent_test_dataloader(
    cfg: DictConfig,
    cache_dir: str,
    mode: str,
    batch_size: Optional[int] = None,
    spatial_dims: Optional[int] = None,
) -> Optional[Tuple[DataLoader, LatentDataset]]:
    """Create test dataloader for pre-encoded latents.

    Args:
        cfg: Hydra configuration.
        cache_dir: Base cache directory.
        mode: Training mode.
        batch_size: Override batch size.
        spatial_dims: Override spatial dimensions (auto-detected if None).

    Returns:
        Tuple of (DataLoader, LatentDataset) or None if test cache doesn't exist.
    """
    test_cache_dir = os.path.join(cache_dir, 'test_new')

    if not os.path.exists(test_cache_dir):
        return None

    batch_size = batch_size or cfg.training.batch_size

    dataset = LatentDataset(test_cache_dir, mode, spatial_dims=spatial_dims)

    dl_cfg = DataLoaderConfig.from_cfg(cfg)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Test must be deterministic
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )

    return dataloader, dataset


# Backwards compatibility aliases for 3D
create_latent_3d_dataloader = create_latent_dataloader
create_latent_3d_validation_dataloader = create_latent_validation_dataloader


# =============================================================================
# Compression Model Utilities
# =============================================================================

def detect_compression_type(checkpoint_path: str) -> str:
    """Detect compression model type from checkpoint.

    Args:
        checkpoint_path: Path to compression checkpoint.

    Returns:
        Compression type: 'vae', 'dcae', or 'vqvae'.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Check for config in checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        # DC-AE checkpoints have dc_ae or dcae in config
        if 'dc_ae' in config or 'dcae' in config:
            return 'dcae'
        # VQ-VAE has num_embeddings
        if 'num_embeddings' in config or 'vqvae' in config:
            return 'vqvae'

    # Check model state dict keys
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

    # VQ-VAE has quantize layer
    if any('quantize' in k or 'embedding' in k for k in state_dict.keys()):
        return 'vqvae'

    # DC-AE has specific layer patterns
    if any('residual_autoencoding' in k.lower() for k in state_dict.keys()):
        return 'dcae'

    # Default to VAE
    return 'vae'


def detect_spatial_dims(checkpoint_path: str) -> int:
    """Detect spatial dimensions (2D or 3D) from checkpoint.

    Args:
        checkpoint_path: Path to compression checkpoint.

    Returns:
        Spatial dimensions: 2 or 3.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Check for spatial_dims in config
    if 'config' in checkpoint:
        config = checkpoint['config']
        if 'spatial_dims' in config:
            return config['spatial_dims']
        # Check nested configs
        for key in ['vae', 'vqvae', 'dcae', 'vae_3d', 'vqvae_3d', 'dcae_3d']:
            if key in config and isinstance(config[key], dict):
                if 'spatial_dims' in config[key]:
                    return config[key]['spatial_dims']

    # Check model state dict for 3D patterns (conv3d weights have 5D shape)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

    # Look for encoder conv layers - 3D convs have 5D weights [out, in, D, H, W]
    for key, value in state_dict.items():
        if 'conv' in key.lower() and 'weight' in key:
            if isinstance(value, torch.Tensor) and value.dim() == 5:
                logger.info(f"Detected 3D compression model from weight shape {value.shape}")
                return 3

    # Default to 2D
    return 2


def detect_scale_factor(checkpoint_path: str, compression_type: str = 'auto') -> int:
    """Detect spatial scale factor from compression checkpoint.

    Args:
        checkpoint_path: Path to compression checkpoint.
        compression_type: Type of model ('auto', 'vae', 'dcae', 'vqvae').

    Returns:
        Spatial scale factor (8 for VAE/VQ-VAE, 32/64 for DC-AE).
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})

    # DC-AE: check for spatial_compression_ratio or f{N} naming
    if compression_type == 'dcae' or (compression_type == 'auto' and 'dcae' in str(config).lower()):
        # Explicit spatial_compression_ratio
        if 'spatial_compression_ratio' in config:
            return config['spatial_compression_ratio']
        # Check nested dcae config
        if 'dcae' in config and isinstance(config['dcae'], dict):
            if 'spatial_compression_ratio' in config['dcae']:
                return config['dcae']['spatial_compression_ratio']
        # Check for f{N} naming pattern (e.g., 'dc-ae-f32c32')
        if 'name' in config:
            import re
            match = re.search(r'f(\d+)', str(config['name']))
            if match:
                return int(match.group(1))
        # DC-AE default
        return 32

    # VAE/VQ-VAE: count downsampling stages or use channels length
    if 'channels' in config:
        num_stages = len(config['channels'])
        return 2 ** num_stages  # Typically 2^3 = 8

    # Default for VAE/VQ-VAE
    return 8


def detect_latent_channels(checkpoint_path: str, compression_type: str = 'auto') -> int:
    """Detect latent channels from compression checkpoint.

    Args:
        checkpoint_path: Path to compression checkpoint.
        compression_type: Type of model ('auto', 'vae', 'dcae', 'vqvae').

    Returns:
        Number of latent channels.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})

    # Check common attribute names
    if 'latent_channels' in config:
        return config['latent_channels']
    if 'z_channels' in config:
        return config['z_channels']
    if 'embedding_dim' in config:  # VQ-VAE
        return config['embedding_dim']

    # Check nested configs
    for key in ['vae', 'vqvae', 'dcae']:
        if key in config and isinstance(config[key], dict):
            nested = config[key]
            if 'latent_channels' in nested:
                return nested['latent_channels']
            if 'z_channels' in nested:
                return nested['z_channels']

    # Default
    return 4


def load_compression_model(
    checkpoint_path: str,
    compression_type: str,
    device: torch.device,
    cfg: Optional[DictConfig] = None,
    spatial_dims: Any = 'auto',
) -> Tuple[torch.nn.Module, str, int, int, int]:
    """Load compression model from checkpoint.

    Args:
        checkpoint_path: Path to compression checkpoint.
        compression_type: Type of model ('auto', 'vae', 'dcae', 'vqvae').
        device: Device to load model to.
        cfg: Optional config for model architecture.
        spatial_dims: Spatial dimensions ('auto', 2, or 3).

    Returns:
        Tuple of (model, detected_type, spatial_dims, scale_factor, latent_channels).
    """
    # Auto-detect type if needed
    if compression_type == 'auto':
        compression_type = detect_compression_type(checkpoint_path)
        logger.info(f"Auto-detected compression type: {compression_type}")

    # Auto-detect spatial dimensions if needed
    if spatial_dims == 'auto':
        spatial_dims = detect_spatial_dims(checkpoint_path)
        logger.info(f"Auto-detected spatial dimensions: {spatial_dims}D")
    else:
        spatial_dims = int(spatial_dims)

    # Detect scale factor and latent channels
    scale_factor = detect_scale_factor(checkpoint_path, compression_type)
    latent_channels = detect_latent_channels(checkpoint_path, compression_type)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get('config', {})

    if compression_type == 'vae':
        from monai.networks.nets import AutoencoderKL

        model = AutoencoderKL(
            spatial_dims=spatial_dims,
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 1),
            channels=tuple(model_config.get('channels', [64, 128, 256])),
            attention_levels=tuple(model_config.get('attention_levels', [False, False, True])),
            latent_channels=model_config.get('latent_channels', 4),
            num_res_blocks=model_config.get('num_res_blocks', 2),
            norm_num_groups=model_config.get('norm_num_groups', 32),
        ).to(device)

    elif compression_type == 'vqvae':
        from monai.networks.nets import VQVAE

        # Get channels to compute default downsample/upsample parameters
        channels = tuple(model_config.get('channels', [64, 128, 256]))
        n_levels = len(channels)

        # Default downsample/upsample parameters if not in config
        # Format: (kernel_size, stride, padding, output_padding) for each level
        default_downsample = tuple((4, 2, 1) for _ in range(n_levels))
        default_upsample = tuple((4, 2, 1, 0) for _ in range(n_levels))

        # Get parameters from config, converting lists to tuples
        downsample_params = model_config.get('downsample_parameters', default_downsample)
        upsample_params = model_config.get('upsample_parameters', default_upsample)

        # Ensure they are tuples of tuples
        if isinstance(downsample_params, list):
            downsample_params = tuple(tuple(p) for p in downsample_params)
        if isinstance(upsample_params, list):
            upsample_params = tuple(tuple(p) for p in upsample_params)

        model = VQVAE(
            spatial_dims=spatial_dims,
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 1),
            channels=channels,
            num_res_layers=model_config.get('num_res_layers', 2),
            num_res_channels=tuple(model_config.get('num_res_channels', list(channels))),
            downsample_parameters=downsample_params,
            upsample_parameters=upsample_params,
            num_embeddings=model_config.get('num_embeddings', 512),
            embedding_dim=model_config.get('embedding_dim', 3),
            commitment_cost=model_config.get('commitment_cost', 0.25),
            decay=model_config.get('decay', 0.99),
            epsilon=model_config.get('epsilon', 1e-5),
        ).to(device)

    elif compression_type == 'dcae':
        # DC-AE model loading (handles 2D/3D internally)
        from medgen.models.dc_ae import create_dc_ae_from_config

        model = create_dc_ae_from_config(model_config).to(device)

    else:
        raise ValueError(f"Unknown compression type: {compression_type}")

    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

    # Strip 'model.' prefix if present (trainer saves with this prefix)
    keys_with_prefix = [k for k in state_dict.keys() if k.startswith('model.')]
    if keys_with_prefix:
        state_dict = {
            k.replace('model.', '', 1) if k.startswith('model.') else k: v
            for k, v in state_dict.items()
        }
        logger.debug(f"Stripped 'model.' prefix from {len(keys_with_prefix)} state_dict keys")

    model.load_state_dict(state_dict)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    logger.info(
        f"Loaded {compression_type} compression model ({spatial_dims}D) from {checkpoint_path} "
        f"[scale_factor={scale_factor}x, latent_channels={latent_channels}]"
    )

    return model, compression_type, spatial_dims, scale_factor, latent_channels
