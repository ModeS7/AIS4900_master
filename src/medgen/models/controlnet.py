"""
ControlNet wrapper for conditional latent diffusion.

Provides pixel-resolution conditioning for latent diffusion models by:
1. Processing conditioning (seg masks) at full pixel resolution
2. Encoding to latent resolution via learned embedding
3. Injecting features into UNet via zero convolutions

Usage:
    # Create ControlNet matching UNet architecture
    controlnet = create_controlnet_for_unet(unet, cfg, device, spatial_dims=2)

    # Forward pass with conditioning
    down_residuals, mid_residual = controlnet(
        x=noisy_latent,           # [B, C, H_lat, W_lat]
        timesteps=t,
        controlnet_cond=seg_mask,  # [B, 1, H_pix, W_pix] - PIXEL SPACE
    )

    # UNet forward with ControlNet injection
    output = unet(
        x=noisy_latent,
        timesteps=t,
        down_block_additional_residuals=down_residuals,
        mid_block_additional_residual=mid_residual,
    )

Reference:
    Zhang & Agrawala, "Adding Conditional Control to Text-to-Image Diffusion Models"
    https://arxiv.org/abs/2302.05543
"""
import logging

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

logger = logging.getLogger(__name__)


def create_controlnet_for_unet(
    unet: nn.Module,
    cfg: DictConfig,
    device: torch.device,
    spatial_dims: int = 2,
    latent_channels: int = 4,
) -> nn.Module:
    """Create a ControlNet that matches the given UNet architecture.

    Args:
        unet: The DiffusionModelUNet to match.
        cfg: Configuration with controlnet settings.
        device: Device to create model on.
        spatial_dims: 2 for images, 3 for volumes.
        latent_channels: Number of latent channels (from VAE).

    Returns:
        ControlNet module ready for training.
    """
    from monai.networks.nets import ControlNet

    # Get UNet architecture from config or infer from model
    model_cfg = cfg.model
    controlnet_cfg = cfg.get('controlnet', {})

    # Use controlnet config or fall back to model config
    channels = controlnet_cfg.get('channels') or list(model_cfg.channels)
    attention_levels = controlnet_cfg.get('attention_levels') or list(model_cfg.attention_levels)
    num_res_blocks = controlnet_cfg.get('num_res_blocks') or model_cfg.num_res_blocks
    num_head_channels = controlnet_cfg.get('num_head_channels') or model_cfg.num_head_channels

    # Conditioning embedding config
    cond_in_channels = controlnet_cfg.get('conditioning_embedding_in_channels', 1)
    cond_num_channels = controlnet_cfg.get('conditioning_embedding_num_channels', [16, 32, 96, 256])

    # Ensure cond_num_channels is a tuple
    if isinstance(cond_num_channels, (list, tuple)):
        cond_num_channels = tuple(cond_num_channels)

    logger.info(f"Creating ControlNet (spatial_dims={spatial_dims}):")
    logger.info(f"  channels: {channels}")
    logger.info(f"  attention_levels: {attention_levels}")
    logger.info(f"  num_res_blocks: {num_res_blocks}")
    logger.info(f"  conditioning_embedding: {cond_in_channels} -> {cond_num_channels}")

    controlnet = ControlNet(
        spatial_dims=spatial_dims,
        in_channels=latent_channels,  # Matches UNet input (latent space)
        num_res_blocks=num_res_blocks,
        channels=tuple(channels),
        attention_levels=tuple(attention_levels),
        norm_num_groups=32,
        num_head_channels=num_head_channels,
        conditioning_embedding_in_channels=cond_in_channels,
        conditioning_embedding_num_channels=cond_num_channels,
    ).to(device)

    # Enable gradient checkpointing if requested
    if controlnet_cfg.get('gradient_checkpointing', False):
        if hasattr(controlnet, 'enable_gradient_checkpointing'):
            controlnet.enable_gradient_checkpointing()
            logger.info("ControlNet gradient checkpointing enabled")

    # Log parameter count
    num_params = sum(p.numel() for p in controlnet.parameters())
    logger.info(f"ControlNet parameters: {num_params:,}")

    return controlnet


def freeze_unet_for_controlnet(unet: nn.Module) -> None:
    """Freeze UNet parameters for Stage 2 ControlNet training.

    Args:
        unet: The UNet to freeze.
    """
    for param in unet.parameters():
        param.requires_grad = False
    logger.info("UNet parameters frozen for ControlNet training")


def unfreeze_unet(unet: nn.Module) -> None:
    """Unfreeze UNet parameters.

    Args:
        unet: The UNet to unfreeze.
    """
    for param in unet.parameters():
        param.requires_grad = True
    logger.info("UNet parameters unfrozen")


class ControlNetConditionedUNet(nn.Module):
    """Wrapper that combines UNet with ControlNet for conditional generation.

    This module handles:
    - ControlNet forward pass with pixel-space conditioning
    - Injection of ControlNet residuals into UNet
    - Conditioning scale control

    Args:
        unet: The base DiffusionModelUNet.
        controlnet: The ControlNet module.
        conditioning_scale: Scale factor for ControlNet output (default 1.0).
    """

    def __init__(
        self,
        unet: nn.Module,
        controlnet: nn.Module,
        conditioning_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.conditioning_scale = conditioning_scale

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        controlnet_cond: Tensor,
        context: Tensor | None = None,
        class_labels: Tensor | None = None,
        conditioning_scale: float | None = None,
    ) -> Tensor:
        """Forward pass with ControlNet conditioning.

        Args:
            x: Noisy latent input [B, C, H, W] or [B, C, D, H, W].
            timesteps: Diffusion timesteps [B].
            controlnet_cond: Conditioning at PIXEL space [B, 1, H_pix, W_pix].
            context: Optional cross-attention context.
            class_labels: Optional class labels.
            conditioning_scale: Override default conditioning scale.

        Returns:
            UNet output (noise/velocity prediction).
        """
        scale = conditioning_scale if conditioning_scale is not None else self.conditioning_scale

        # Get ControlNet residuals
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            x=x,
            timesteps=timesteps,
            controlnet_cond=controlnet_cond,
            conditioning_scale=scale,
            context=context,
            class_labels=class_labels,
        )

        # UNet forward with residual injection
        output = self.unet(
            x=x,
            timesteps=timesteps,
            context=context,
            class_labels=class_labels,
            down_block_additional_residuals=tuple(down_block_res_samples),
            mid_block_additional_residual=mid_block_res_sample,
        )

        return output

    def set_conditioning_scale(self, scale: float) -> None:
        """Update the conditioning scale.

        Args:
            scale: New conditioning scale (0.0 = no control, 1.0 = full control).
        """
        self.conditioning_scale = scale


def load_controlnet_checkpoint(
    controlnet: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
) -> None:
    """Load ControlNet weights from checkpoint.

    Args:
        controlnet: The ControlNet module.
        checkpoint_path: Path to checkpoint file.
        device: Device to load to.
        strict: Whether to require exact key matching.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'controlnet_state_dict' in checkpoint:
        state_dict = checkpoint['controlnet_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    controlnet.load_state_dict(state_dict, strict=strict)
    logger.info(f"Loaded ControlNet checkpoint from {checkpoint_path}")


def save_controlnet_checkpoint(
    controlnet: nn.Module,
    unet: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_path: str,
    cfg: DictConfig | None = None,
) -> None:
    """Save ControlNet checkpoint.

    Args:
        controlnet: The ControlNet module.
        unet: The UNet module (saved for reference).
        optimizer: The optimizer.
        epoch: Current epoch.
        save_path: Path to save checkpoint.
        cfg: Optional configuration to save.
    """
    checkpoint = {
        'epoch': epoch,
        'controlnet_state_dict': controlnet.state_dict(),
        'unet_state_dict': unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if cfg is not None:
        checkpoint['config'] = dict(cfg)

    torch.save(checkpoint, save_path)
    logger.info(f"Saved ControlNet checkpoint to {save_path}")


class ControlNetGenerationWrapper:
    """Wrapper for using ControlNet with strategy.generate().

    The strategy's generate method calls model(x, timesteps) but ControlNet
    needs controlnet_cond. This wrapper binds the conditioning so the strategy
    can use the model without modification.

    Usage:
        wrapper = ControlNetGenerationWrapper(model, conditioning)
        samples = strategy.generate(wrapper, noise, num_steps, device)

    Args:
        model: The ControlNetConditionedUNet or similar model.
        controlnet_cond: Conditioning tensor at pixel resolution.
    """

    def __init__(
        self,
        model: nn.Module,
        controlnet_cond: Tensor,
    ) -> None:
        self.model = model
        self.controlnet_cond = controlnet_cond

    def __call__(self, x: Tensor, timesteps: Tensor, **kwargs) -> Tensor:
        """Forward call that includes controlnet conditioning.

        The strategy calls model(x, timesteps, ...) and this routes
        to the ControlNet model with the bound conditioning.
        """
        return self.model(
            x=x,
            timesteps=timesteps,
            controlnet_cond=self.controlnet_cond,
            **kwargs,
        )

    def eval(self):
        """Set model to eval mode."""
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        """Set model to train mode."""
        self.model.train(mode)
        return self

    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()
