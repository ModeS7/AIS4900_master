"""Latent diffusion configuration dataclass.

Provides type-safe access to latent space diffusion settings.
"""
from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass
class LatentConfig:
    """Latent diffusion configuration.

    Attributes:
        enabled: Whether latent diffusion is enabled.
        compression_checkpoint: Path to compression model checkpoint.
        seg_compression_checkpoint: Path to segmentation compression checkpoint.
        compression_type: Type of compression ('auto', 'vae', 'vqvae', 'dcae').
        slicewise_encoding: Whether to encode 3D volumes slice-by-slice.
        scale_factor: Spatial scale factor (e.g., 8 for 8x downsampling).
        depth_scale_factor: Depth scale factor for 3D (usually same as scale_factor).
        latent_channels: Number of latent channels.
        cache_dir: Directory for caching encoded latents.
        auto_encode: Whether to automatically encode data before training.
        validate_cache: Whether to validate cache integrity.
        num_workers: Number of workers for encoding.
        batch_size: Batch size for encoding.
    """
    enabled: bool = False
    compression_checkpoint: str | None = None
    seg_compression_checkpoint: str | None = None
    compression_type: str = 'auto'
    slicewise_encoding: bool = False
    scale_factor: float | None = None
    depth_scale_factor: float | None = None
    latent_channels: int | None = None
    cache_dir: str | None = None
    auto_encode: bool = True
    validate_cache: bool = True
    num_workers: int = 4
    batch_size: int = 32

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'LatentConfig':
        """Extract latent config from Hydra DictConfig.

        Args:
            cfg: Hydra configuration object.

        Returns:
            LatentConfig instance.
        """
        latent = cfg.get('latent', {})
        return cls(
            enabled=latent.get('enabled', False),
            compression_checkpoint=latent.get('compression_checkpoint'),
            seg_compression_checkpoint=latent.get('seg_compression_checkpoint'),
            compression_type=latent.get('compression_type', 'auto'),
            slicewise_encoding=latent.get('slicewise_encoding', False),
            scale_factor=latent.get('scale_factor'),
            depth_scale_factor=latent.get('depth_scale_factor'),
            latent_channels=latent.get('latent_channels'),
            cache_dir=latent.get('cache_dir'),
            auto_encode=latent.get('auto_encode', True),
            validate_cache=latent.get('validate_cache', True),
            num_workers=latent.get('num_workers', 4),
            batch_size=latent.get('batch_size', 32),
        )
