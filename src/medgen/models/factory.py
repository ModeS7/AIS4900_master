"""
Model factory for diffusion models.

Provides unified creation of diffusion backbones (UNet or SiT) based on config.
"""

import logging
from typing import Any, Dict
from omegaconf import DictConfig

import torch
import torch.nn as nn

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
    model_type = cfg.model.get('type', 'unet')

    if model_type == 'unet':
        return _create_unet(cfg, device, in_channels, out_channels)
    elif model_type == 'sit':
        return _create_sit(cfg, device, in_channels, out_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'unet' or 'sit'")


def _create_unet(
    cfg: DictConfig,
    device: torch.device,
    in_channels: int,
    out_channels: int,
) -> nn.Module:
    """Create MONAI DiffusionModelUNet."""
    from monai.networks.nets import DiffusionModelUNet

    spatial_dims = cfg.model.get('spatial_dims', 2)

    model = DiffusionModelUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        num_channels=cfg.model.channels,
        attention_levels=cfg.model.attention_levels,
        num_res_blocks=cfg.model.num_res_blocks,
        num_head_channels=cfg.model.num_head_channels,
    )

    logger.info(
        f"Created UNet: spatial_dims={spatial_dims}, "
        f"in_channels={in_channels}, out_channels={out_channels}, "
        f"channels={cfg.model.channels}"
    )

    return model.to(device)


def _create_sit(
    cfg: DictConfig,
    device: torch.device,
    in_channels: int,
    out_channels: int,
) -> nn.Module:
    """Create SiT (Scalable Interpolant Transformer)."""
    from .sit import create_sit, SIT_VARIANTS

    spatial_dims = cfg.model.get('spatial_dims', 2)
    variant = cfg.model.get('variant', 'B')
    patch_size = cfg.model.get('patch_size', 2)
    conditioning = cfg.model.get('conditioning', 'concat')
    mlp_ratio = cfg.model.get('mlp_ratio', 4.0)
    drop_rate = cfg.model.get('drop_rate', 0.0)

    # For concat conditioning, in_channels already includes cond_channels from mode config
    # For cross_attn, we need to separate them
    if conditioning == "cross_attn":
        # Separate input and conditioning channels
        cond_channels = cfg.mode.get('cond_channels', 1)
        model_in_channels = out_channels  # noisy target channels (same as output)
    else:
        # Concatenation mode: in_channels = target_channels + cond_channels
        cond_channels = in_channels - out_channels
        model_in_channels = in_channels

    # Get input size (depends on pixel vs latent space)
    if cfg.get('diffusion', {}).get('space', 'pixel') == 'latent':
        # Latent space: image_size / VAE downscale factor (typically 8)
        vae_scale = cfg.get('vae', {}).get('spatial_scale', 8)
        input_size = cfg.model.image_size // vae_scale
    else:
        input_size = cfg.model.image_size

    # 3D specific settings
    depth_size = None
    if spatial_dims == 3:
        if cfg.get('diffusion', {}).get('space', 'pixel') == 'latent':
            vae_scale = cfg.get('vae', {}).get('spatial_scale', 8)
            depth_size = cfg.model.get('depth_size', cfg.model.image_size) // vae_scale
        else:
            depth_size = cfg.model.get('depth_size', cfg.model.image_size)

    model = create_sit(
        variant=variant,
        spatial_dims=spatial_dims,
        input_size=input_size,
        patch_size=patch_size,
        in_channels=model_in_channels,
        out_channels=out_channels,  # Always output target channels only
        conditioning=conditioning,
        cond_channels=cond_channels,
        learn_sigma=False,  # Not needed for flow matching
        depth_size=depth_size,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
    )

    variant_info = SIT_VARIANTS[variant]
    num_params = sum(p.numel() for p in model.parameters()) / 1e6

    logger.info(
        f"Created SiT-{variant}: spatial_dims={spatial_dims}, input_size={input_size}, "
        f"patch_size={patch_size}, hidden_size={variant_info['hidden_size']}, "
        f"depth={variant_info['depth']}, heads={variant_info['num_heads']}, "
        f"conditioning={conditioning}, params={num_params:.1f}M"
    )

    return model.to(device)


def get_model_type(cfg: DictConfig) -> str:
    """Get model type from config."""
    return cfg.model.get('type', 'unet')


def is_transformer_model(cfg: DictConfig) -> bool:
    """Check if model is transformer-based (SiT, DiT, etc.)."""
    model_type = get_model_type(cfg)
    return model_type in ('sit', 'dit')
