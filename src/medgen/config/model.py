"""Model configuration dataclass.

Provides type-safe access to model architecture configuration.
"""
from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Attributes:
        name: Model type ('unet' or 'dit').
        image_size: Image size for 2D training.
        spatial_dims: Spatial dimensions (2 or 3).
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        channels: Tuple of channel counts per resolution level.
        attention_levels: Tuple of bools indicating attention at each level.
        num_res_blocks: Number of residual blocks per level.
        num_head_channels: Number of channels per attention head.
        norm_num_groups: Number of groups for GroupNorm.
    """
    name: str
    image_size: int
    spatial_dims: int = 2
    in_channels: int = 1
    out_channels: int = 1
    channels: tuple[int, ...] = (128, 256, 256)
    attention_levels: tuple[bool, ...] = (False, True, True)
    num_res_blocks: int = 1
    num_head_channels: int = 256
    norm_num_groups: int = 32

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'ModelConfig':
        """Extract model config from Hydra DictConfig.

        Args:
            cfg: Hydra configuration object containing model section.

        Returns:
            ModelConfig instance with extracted values.
        """
        model = cfg.model
        return cls(
            name=model.get('type', 'unet'),
            image_size=model.image_size,
            spatial_dims=model.get('spatial_dims', 2),
            in_channels=model.get('in_channels', 1),
            out_channels=model.get('out_channels', 1),
            channels=tuple(model.get('channels', [128, 256, 256])),
            attention_levels=tuple(model.get('attention_levels', [False, True, True])),
            num_res_blocks=model.get('num_res_blocks', 1),
            num_head_channels=model.get('num_head_channels', 256),
            norm_num_groups=model.get('norm_num_groups', 32),
        )
