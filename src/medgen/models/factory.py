"""
Model factory for diffusion models.

Provides unified creation of diffusion backbones (UNet or SiT) based on config.
"""

import logging
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

    # Validate spatial_dims
    if spatial_dims not in (2, 3):
        raise ValueError(
            f"spatial_dims must be 2 or 3, got {spatial_dims}. "
            f"Use 2 for 2D images, 3 for 3D volumes."
        )

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

    # Validate spatial_dims
    if spatial_dims not in (2, 3):
        raise ValueError(
            f"spatial_dims must be 2 or 3, got {spatial_dims}. "
            f"Use 2 for 2D images, 3 for 3D volumes."
        )

    variant = cfg.model.get('variant', 'B')
    patch_size = cfg.model.get('patch_size', 2)
    conditioning = cfg.model.get('conditioning', 'concat')
    mlp_ratio = cfg.model.get('mlp_ratio', 4.0)
    drop_rate = cfg.model.get('drop_rate', 0.0)
    drop_path_rate = cfg.model.get('drop_path_rate', 0.0)

    # Validate SiT-specific fields
    valid_variants = ('S', 'B', 'L', 'XL')
    if variant not in valid_variants:
        raise ValueError(
            f"SiT variant must be one of {valid_variants}, got '{variant}'"
        )

    valid_patch_sizes = (2, 4, 8, 16)
    if patch_size not in valid_patch_sizes:
        raise ValueError(
            f"SiT patch_size must be one of {valid_patch_sizes}, got {patch_size}"
        )

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

    # Get VAE scale factor (used for both 2D and 3D latent space)
    is_latent_space = cfg.get('diffusion', {}).get('space', 'pixel') == 'latent'
    vae_scale = cfg.get('vae', {}).get('spatial_scale', 8) if is_latent_space else 1

    # Get input size - prefer volume config for 3D, fallback to model config
    # This avoids redundant specification of dimensions
    if spatial_dims == 3 and 'volume' in cfg:
        # For 3D: derive from volume config (height/width should match)
        base_size = cfg.volume.get('height', cfg.model.get('image_size', 256))
        base_depth = cfg.volume.get('pad_depth_to', cfg.volume.get('depth', base_size))
    else:
        # For 2D or when volume config not available: use model config
        base_size = cfg.model.get('image_size', 256)
        base_depth = cfg.model.get('depth_size', base_size)

    input_size = base_size // vae_scale

    # 3D specific settings
    depth_size = None
    if spatial_dims == 3:
        depth_size = base_depth // vae_scale

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
        drop_path_rate=drop_path_rate,
    )

    variant_info = SIT_VARIANTS[variant]
    num_params = sum(p.numel() for p in model.parameters()) / 1e6

    drop_path_str = f", drop_path={drop_path_rate}" if drop_path_rate > 0 else ""
    logger.info(
        f"Created SiT-{variant}: spatial_dims={spatial_dims}, input_size={input_size}, "
        f"patch_size={patch_size}, hidden_size={variant_info['hidden_size']}, "
        f"depth={variant_info['depth']}, heads={variant_info['num_heads']}, "
        f"conditioning={conditioning}, params={num_params:.1f}M{drop_path_str}"
    )

    return model.to(device)


def get_model_type(cfg: DictConfig) -> str:
    """Get model type from config."""
    return cfg.model.get('type', 'unet')


def is_transformer_model(cfg: DictConfig) -> bool:
    """Check if model is transformer-based (SiT, DiT, etc.)."""
    model_type = get_model_type(cfg)
    return model_type in ('sit', 'dit')
