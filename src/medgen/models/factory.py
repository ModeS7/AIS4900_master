"""
Model factory for diffusion models.

Provides unified creation of diffusion backbones (UNet or DiT) based on config.
"""

import logging

import torch
import torch.nn as nn
from omegaconf import DictConfig

from medgen.pipeline.base_config import ModelConfig

logger = logging.getLogger(__name__)


def create_diffusion_model(
    cfg: DictConfig,
    device: torch.device,
    in_channels: int,
    out_channels: int,
) -> nn.Module:
    """Factory for creating diffusion model backbones.

    Args:
        cfg: Full configuration (needs model and mode sections).
        device: Device to place model on.
        in_channels: Input channels (from mode config, includes conditioning).
        out_channels: Output channels (from mode config).

    Returns:
        Initialized model on specified device.
    """
    mc = ModelConfig.from_hydra(cfg)

    if mc.type == 'unet':
        return _create_unet(cfg, mc, device, in_channels, out_channels)
    elif mc.type in ('dit', 'sit'):
        return _create_dit(cfg, mc, device, in_channels, out_channels)
    else:
        raise ValueError(f"Unknown model type: {mc.type}. Choose 'unet' or 'dit'")


def _create_unet(
    cfg: DictConfig,
    mc: ModelConfig,
    device: torch.device,
    in_channels: int,
    out_channels: int,
) -> nn.Module:
    """Create MONAI DiffusionModelUNet."""
    from monai.networks.nets import DiffusionModelUNet

    # Validate spatial_dims
    if mc.spatial_dims not in (2, 3):
        raise ValueError(
            f"spatial_dims must be 2 or 3, got {mc.spatial_dims}. "
            f"Use 2 for 2D images, 3 for 3D volumes."
        )

    model = DiffusionModelUNet(
        spatial_dims=mc.spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        num_channels=mc.channels,
        attention_levels=mc.attention_levels,
        num_res_blocks=mc.num_res_blocks,
        num_head_channels=mc.num_head_channels,
    )

    logger.info(
        f"Created UNet: spatial_dims={mc.spatial_dims}, "
        f"in_channels={in_channels}, out_channels={out_channels}, "
        f"channels={mc.channels}"
    )

    return model.to(device)


def _create_dit(
    cfg: DictConfig,
    mc: ModelConfig,
    device: torch.device,
    in_channels: int,
    out_channels: int,
) -> nn.Module:
    """Create DiT (Diffusion Transformer)."""
    from .dit import DIT_VARIANTS, create_dit

    # Validate spatial_dims
    if mc.spatial_dims not in (2, 3):
        raise ValueError(
            f"spatial_dims must be 2 or 3, got {mc.spatial_dims}. "
            f"Use 2 for 2D images, 3 for 3D volumes."
        )

    # Validate DiT-specific fields
    valid_variants = ('S', 'B', 'L', 'XL')
    if mc.variant not in valid_variants:
        raise ValueError(
            f"DiT variant must be one of {valid_variants}, got '{mc.variant}'"
        )

    valid_patch_sizes = (1, 2, 4, 8, 16)
    if mc.patch_size not in valid_patch_sizes:
        raise ValueError(
            f"DiT patch_size must be one of {valid_patch_sizes}, got {mc.patch_size}"
        )

    # For concat conditioning, in_channels already includes cond_channels from mode config
    # For cross_attn, we need to separate them
    if mc.conditioning == "cross_attn":
        # Separate input and conditioning channels
        cond_channels = getattr(cfg.mode, 'cond_channels', 1)
        model_in_channels = out_channels  # noisy target channels (same as output)
    else:
        # Concatenation mode: in_channels = target_channels + cond_channels
        cond_channels = in_channels - out_channels
        model_in_channels = in_channels

    # Get latent space scale factors
    # Check both old (vae) and new (latent) config locations
    latent_cfg = cfg.get('latent', {})
    is_latent_space = latent_cfg.get('enabled', False) or cfg.get('diffusion', {}).get('space', 'pixel') == 'latent'

    if is_latent_space:
        # Use latent config scale factors
        spatial_scale = latent_cfg.get('scale_factor') or cfg.get('vae', {}).get('spatial_scale', 8)
        # For slicewise encoding, depth is not compressed
        slicewise = latent_cfg.get('slicewise_encoding', False)
        depth_scale = 1 if slicewise else (latent_cfg.get('depth_scale_factor') or spatial_scale)
    else:
        spatial_scale = 1
        depth_scale = 1

    # Get input size - prefer volume config for 3D, fallback to model config
    # This avoids redundant specification of dimensions
    if mc.spatial_dims == 3 and 'volume' in cfg:
        # For 3D: derive from volume config (height/width should match)
        base_size = cfg.volume.get('height', mc.image_size)
        base_depth = cfg.volume.get('pad_depth_to', cfg.volume.get('depth', base_size))
    else:
        # For 2D or when volume config not available: use model config
        base_size = mc.image_size
        base_depth = getattr(cfg.model, 'depth_size', base_size)

    input_size = base_size // spatial_scale

    # 3D specific settings
    depth_size = None
    if mc.spatial_dims == 3:
        depth_size = base_depth // depth_scale

    model = create_dit(
        variant=mc.variant,
        spatial_dims=mc.spatial_dims,
        input_size=input_size,
        patch_size=mc.patch_size,
        in_channels=model_in_channels,
        out_channels=out_channels,  # Always output target channels only
        conditioning=mc.conditioning,
        cond_channels=cond_channels,
        learn_sigma=False,  # Not needed for flow matching
        depth_size=depth_size,
        mlp_ratio=mc.mlp_ratio,
        drop_rate=mc.drop_rate,
        drop_path_rate=mc.drop_path_rate,
    )

    variant_info = DIT_VARIANTS[mc.variant]
    num_params = sum(p.numel() for p in model.parameters()) / 1e6

    drop_path_str = f", drop_path={mc.drop_path_rate}" if mc.drop_path_rate > 0 else ""
    logger.info(
        f"Created DiT-{mc.variant}: spatial_dims={mc.spatial_dims}, input_size={input_size}, "
        f"patch_size={mc.patch_size}, hidden_size={variant_info['hidden_size']}, "
        f"depth={variant_info['depth']}, heads={variant_info['num_heads']}, "
        f"conditioning={mc.conditioning}, params={num_params:.1f}M{drop_path_str}"
    )

    return model.to(device)


def get_model_type(cfg: DictConfig) -> str:
    """Get model type from config."""
    mc = ModelConfig.from_hydra(cfg)
    return mc.type


def is_transformer_model(cfg: DictConfig) -> bool:
    """Check if model is transformer-based (DiT)."""
    model_type = get_model_type(cfg)
    return model_type in ('dit', 'sit')
