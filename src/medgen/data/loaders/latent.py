"""
Latent space dataloaders for latent diffusion training.

Provides dataloaders that load pre-encoded latent tensors from cache,
plus utilities for building and validating the latent cache.
"""
import hashlib
import json
import logging
import os
from datetime import datetime
from glob import glob
from typing import Any, Dict, List, Optional, Tuple, Union

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

    Each .pt file contains:
    - 'latent': Encoded image tensor [C_latent, H_latent, W_latent]
    - 'seg_mask': Original pixel-space segmentation mask [1, H, W] (if conditional)
    - 'patient_id': Patient identifier string
    - 'slice_idx': Slice index within the volume

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
            raise ValueError(f"No .pt files found in cache directory: {cache_dir}")

        logger.info(f"LatentDataset: Found {len(self.files)} samples in {cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a single sample from cache.

        Returns:
            Dictionary with 'latent', optionally 'seg_mask', 'patient_id', 'slice_idx'.
        """
        data = torch.load(self.files[idx], weights_only=False)

        # Ensure latent is float32
        if 'latent' in data:
            data['latent'] = data['latent'].float()

        # Ensure seg_mask is float32 if present
        if 'seg_mask' in data and data['seg_mask'] is not None:
            data['seg_mask'] = data['seg_mask'].float()

        return data


class LatentCacheBuilder:
    """Builds and validates latent cache from pixel-space datasets.

    Encodes images using a compression model (VAE/DC-AE/VQ-VAE) and saves
    them as .pt files for fast loading during diffusion training.

    Args:
        compression_model: Trained compression model with encode() method.
        device: Device for encoding.
        mode: Training mode for determining what to encode.
        image_size: Original image size (for metadata).
        compression_type: Type of compression model ('vae', 'dcae', 'vqvae').
    """

    def __init__(
        self,
        compression_model: torch.nn.Module,
        device: torch.device,
        mode: str,
        image_size: int,
        compression_type: str = "vae",
        verbose: bool = True,
    ) -> None:
        self.model = compression_model.eval()
        self.device = device
        self.mode = mode
        self.image_size = image_size
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

        Args:
            pixel_dataset: Dataset returning pixel-space images.
            cache_dir: Directory to save encoded latents.
            checkpoint_path: Path to compression checkpoint (for hash).
            batch_size: Batch size for encoding.
            num_workers: Number of dataloader workers.
        """
        os.makedirs(cache_dir, exist_ok=True)

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

        logger.info(f"Encoding {len(pixel_dataset)} samples to {cache_dir}...")

        with torch.no_grad():
            for batch in tqdm(temp_loader, desc="Encoding latents", disable=not self.verbose):
                # Handle different batch formats
                images, seg_masks, patient_ids, slice_indices = self._parse_batch(batch)

                # Move to device and encode
                images = images.to(self.device, non_blocking=True)

                # Encode images
                latents = self._encode(images)

                if latent_shape is None:
                    latent_shape = list(latents.shape[1:])  # [C, H, W]

                # Save each sample
                for i in range(latents.shape[0]):
                    sample_data = {
                        'latent': latents[i].cpu(),
                        'patient_id': patient_ids[i] if patient_ids else f"sample_{sample_idx}",
                        'slice_idx': slice_indices[i] if slice_indices else sample_idx,
                    }

                    # Include seg mask if available (keep in pixel space)
                    if seg_masks is not None:
                        sample_data['seg_mask'] = seg_masks[i].cpu()

                    # Save to file
                    filename = f"sample_{sample_idx:06d}.pt"
                    torch.save(sample_data, os.path.join(cache_dir, filename))
                    sample_idx += 1

        # Save metadata
        metadata = {
            'compression_checkpoint': checkpoint_path,
            'checkpoint_hash': self.compute_checkpoint_hash(checkpoint_path),
            'compression_type': self.compression_type,
            'latent_shape': latent_shape,
            'mode': self.mode,
            'image_size': self.image_size,
            'num_samples': sample_idx,
            'created_at': datetime.now().isoformat(),
        }

        with open(os.path.join(cache_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Cache built: {sample_idx} samples, latent shape {latent_shape}")

    def _parse_batch(
        self, batch: Any
    ) -> Tuple[Tensor, Optional[Tensor], Optional[List[str]], Optional[List[int]]]:
        """Parse batch into images, seg masks, and metadata.

        Handles different dataset formats:
        - Tuple: (images, seg_masks) from extract_slices_dual
        - Tensor: just images from extract_slices_single
        - Array: numpy array to convert

        Args:
            batch: Batch from dataloader.

        Returns:
            Tuple of (images, seg_masks, patient_ids, slice_indices).
        """
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

    def _encode(self, images: Tensor) -> Tensor:
        """Encode images to latent space.

        Handles different compression model types.

        Args:
            images: Images [B, C, H, W].

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
) -> Tuple[DataLoader, LatentDataset]:
    """Create dataloader for pre-encoded latents.

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

    Returns:
        Tuple of (DataLoader, LatentDataset).
    """
    split_cache_dir = os.path.join(cache_dir, split)

    if not os.path.exists(split_cache_dir):
        raise ValueError(f"Cache directory not found: {split_cache_dir}")

    dataset = LatentDataset(split_cache_dir, mode)

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
) -> Optional[Tuple[DataLoader, LatentDataset]]:
    """Create validation dataloader for pre-encoded latents.

    Args:
        cfg: Hydra configuration.
        cache_dir: Base cache directory.
        mode: Training mode.
        batch_size: Override batch size.
        world_size: Number of GPUs for DDP.

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

    dataset = LatentDataset(val_cache_dir, mode)

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
) -> Optional[Tuple[DataLoader, LatentDataset]]:
    """Create test dataloader for pre-encoded latents.

    Args:
        cfg: Hydra configuration.
        cache_dir: Base cache directory.
        mode: Training mode.
        batch_size: Override batch size.

    Returns:
        Tuple of (DataLoader, LatentDataset) or None if test cache doesn't exist.
    """
    test_cache_dir = os.path.join(cache_dir, 'test_new')

    if not os.path.exists(test_cache_dir):
        return None

    batch_size = batch_size or cfg.training.batch_size

    dataset = LatentDataset(test_cache_dir, mode)

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


def load_compression_model(
    checkpoint_path: str,
    compression_type: str,
    device: torch.device,
    cfg: Optional[DictConfig] = None,
    spatial_dims: Any = 'auto',
) -> Tuple[torch.nn.Module, str, int]:
    """Load compression model from checkpoint.

    Args:
        checkpoint_path: Path to compression checkpoint.
        compression_type: Type of model ('auto', 'vae', 'dcae', 'vqvae').
        device: Device to load model to.
        cfg: Optional config for model architecture.
        spatial_dims: Spatial dimensions ('auto', 2, or 3).

    Returns:
        Tuple of (model, detected_type, spatial_dims).
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

        model = VQVAE(
            spatial_dims=spatial_dims,
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 1),
            channels=tuple(model_config.get('channels', [64, 128, 256])),
            num_res_channels=tuple(model_config.get('num_res_channels', [64, 128, 256])),
            num_embeddings=model_config.get('num_embeddings', 512),
            embedding_dim=model_config.get('embedding_dim', 3),
        ).to(device)

    elif compression_type == 'dcae':
        # DC-AE model loading (handles 2D/3D internally)
        from medgen.models.dc_ae import create_dc_ae_from_config

        model = create_dc_ae_from_config(model_config).to(device)

    else:
        raise ValueError(f"Unknown compression type: {compression_type}")

    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    model.load_state_dict(state_dict)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    logger.info(f"Loaded {compression_type} compression model ({spatial_dims}D) from {checkpoint_path}")

    return model, compression_type, spatial_dims
