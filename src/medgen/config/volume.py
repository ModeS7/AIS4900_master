"""Volume configuration dataclass for 3D training.

Provides type-safe access to 3D volume dimensions and settings.
"""
from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass
class VolumeConfig:
    """3D volume configuration.

    Attributes:
        height: Volume height in voxels.
        width: Volume width in voxels.
        depth: Volume depth (number of slices).
        pad_depth_to: Depth to pad volumes to (for consistent tensor shapes).
        pad_mode: Padding mode ('replicate', 'constant', 'reflect').
        original_depth: Original unpadded depth (used for metrics).
        slice_step: Step size when selecting slices (for 2D from 3D).
    """
    height: int = 256
    width: int = 256
    depth: int = 160
    pad_depth_to: int = 160
    pad_mode: str = 'replicate'
    original_depth: int = 150
    slice_step: int = 1

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'VolumeConfig | None':
        """Extract volume config from Hydra DictConfig.

        Args:
            cfg: Hydra configuration object.

        Returns:
            VolumeConfig instance, or None if not 3D training.
        """
        # Check if this is 3D training
        spatial_dims = cfg.model.get('spatial_dims', 2)
        if spatial_dims != 3:
            return None

        # Check if volume config exists
        if not hasattr(cfg, 'volume') or cfg.volume is None:
            return None

        volume = cfg.volume
        return cls(
            height=volume.get('height', 256),
            width=volume.get('width', 256),
            depth=volume.get('depth', 160),
            pad_depth_to=volume.get('pad_depth_to', 160),
            pad_mode=volume.get('pad_mode', 'replicate'),
            original_depth=volume.get('original_depth', 150),
            slice_step=volume.get('slice_step', 1),
        )
