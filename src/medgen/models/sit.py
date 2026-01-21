"""
Scalable Interpolant Transformers (SiT) for diffusion models.

A transformer-based architecture designed for flow matching / interpolant-based
diffusion, supporting both 2D images and 3D volumes.

Reference: https://arxiv.org/abs/2401.08740 (SiT paper)
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Literal

from .embeddings import (
    PatchEmbed2D,
    PatchEmbed3D,
    TimestepEmbedder,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
)
from .sit_blocks import SiTBlock, FinalLayer


# Model variant configurations
SIT_VARIANTS = {
    'S': {'hidden_size': 384, 'depth': 12, 'num_heads': 6},
    'B': {'hidden_size': 768, 'depth': 12, 'num_heads': 12},
    'L': {'hidden_size': 1024, 'depth': 24, 'num_heads': 16},
    'XL': {'hidden_size': 1152, 'depth': 28, 'num_heads': 16},
}


class SiT(nn.Module):
    """Scalable Interpolant Transformer for diffusion.

    A vision transformer architecture with adaLN-Zero conditioning for
    diffusion / flow matching. Supports 2D and 3D inputs with either
    concatenation or cross-attention based conditioning.

    Args:
        spatial_dims: Number of spatial dimensions (2 or 3).
        input_size: Spatial size of input (assumes square/cubic for now).
        patch_size: Patch size for tokenization.
        in_channels: Number of input channels.
        hidden_size: Transformer hidden dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        conditioning: Conditioning mode ("concat" or "cross_attn").
        cond_channels: Number of conditioning channels (e.g., segmentation mask).
        learn_sigma: Whether to predict variance (doubles output channels).
        drop_rate: Dropout rate.
        drop_path_rate: Stochastic depth rate. Linearly increases from 0 to this
            value across transformer blocks. Default: 0.0 (disabled).
        qk_norm: Whether to use QK-normalization in attention. Improves training
            stability, especially for larger images/patches. Default: True.
        depth_size: Depth size for 3D (if different from input_size).

    Example:
        >>> model = SiT(spatial_dims=2, input_size=32, in_channels=4, hidden_size=768, depth=12, num_heads=12)
        >>> x = torch.randn(2, 4, 32, 32)  # Latent images
        >>> t = torch.rand(2)  # Timesteps
        >>> pred = model(x, t)  # Velocity/noise prediction
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        out_channels: Optional[int] = None,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        conditioning: Literal["concat", "cross_attn"] = "concat",
        cond_channels: int = 1,
        learn_sigma: bool = False,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        qk_norm: bool = True,
        depth_size: Optional[int] = None,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.input_size = input_size
        self.depth_size = depth_size or input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        # out_channels: explicit > in_channels (for unconditional), doubled if learn_sigma
        base_out = out_channels if out_channels is not None else in_channels
        self.out_channels = base_out * 2 if learn_sigma else base_out
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.conditioning = conditioning
        self.cond_channels = cond_channels

        # Validate input dimensions are divisible by patch_size
        if input_size % patch_size != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by patch_size ({patch_size})"
            )
        if spatial_dims == 3 and depth_size is not None and depth_size % patch_size != 0:
            raise ValueError(
                f"depth_size ({depth_size}) must be divisible by patch_size ({patch_size})"
            )

        # Calculate number of patches
        if spatial_dims == 2:
            self.num_patches = (input_size // patch_size) ** 2
            self.grid_size = input_size // patch_size
        else:
            self.num_patches = (self.depth_size // patch_size) * (input_size // patch_size) ** 2
            self.grid_size_d = self.depth_size // patch_size
            self.grid_size_h = input_size // patch_size
            self.grid_size_w = input_size // patch_size

        # Patch embedding
        # For concat mode, in_channels already includes conditioning (from trainer/factory)
        # For cross_attn mode, in_channels is just the target channels
        actual_in_channels = in_channels

        if spatial_dims == 2:
            self.x_embedder = PatchEmbed2D(patch_size, actual_in_channels, hidden_size)
        else:
            self.x_embedder = PatchEmbed3D(patch_size, actual_in_channels, hidden_size)

        # Conditioning embedder for cross-attention mode
        self.use_cross_attn = (conditioning == "cross_attn")
        if self.use_cross_attn:
            if spatial_dims == 2:
                self.cond_embedder = PatchEmbed2D(patch_size, cond_channels, hidden_size)
            else:
                self.cond_embedder = PatchEmbed3D(patch_size, cond_channels, hidden_size)

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Positional embedding (learnable, initialized from sincos)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        # Stochastic depth: linearly increasing drop rate from 0 to drop_path_rate
        # First block has 0 drop rate, last block has drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                use_cross_attn=self.use_cross_attn,
                qk_norm=qk_norm,
                drop=drop_rate,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, spatial_dims)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights following best practices."""
        # Initialize patch embeddings like linear
        def _init_conv(m):
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.x_embedder.apply(_init_conv)
        if hasattr(self, 'cond_embedder'):
            self.cond_embedder.apply(_init_conv)

        # Initialize positional embedding from sincos
        if self.spatial_dims == 2:
            pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.grid_size)
        else:
            pos_embed = get_3d_sincos_pos_embed(
                self.hidden_size,
                self.grid_size_d,
                self.grid_size_h,
                self.grid_size_w,
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize timestep embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize transformer blocks
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.qkv.weight)
            nn.init.xavier_uniform_(block.attn.proj.weight)
            nn.init.xavier_uniform_(block.mlp.fc1.weight)
            nn.init.xavier_uniform_(block.mlp.fc2.weight)

            # Zero-init adaLN modulation (key for training stability)
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

            if block.use_cross_attn:
                nn.init.xavier_uniform_(block.cross_attn.q.weight)
                nn.init.xavier_uniform_(block.cross_attn.kv.weight)
                nn.init.xavier_uniform_(block.cross_attn.proj.weight)

        # Zero-init final layer
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patch tokens back to 2D image.

        Args:
            x: [B, N, patch_size^2 * C] patch tokens

        Returns:
            [B, C, H, W] image
        """
        p = self.patch_size
        h = w = self.input_size // p
        c = self.out_channels

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, h, p, w, p]
        x = x.reshape(x.shape[0], c, h * p, w * p)
        return x

    def unpatchify_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patch tokens back to 3D volume.

        Args:
            x: [B, N, patch_size^3 * C] patch tokens

        Returns:
            [B, C, D, H, W] volume
        """
        p = self.patch_size
        d = self.grid_size_d
        h = self.grid_size_h
        w = self.grid_size_w
        c = self.out_channels

        x = x.reshape(x.shape[0], d, h, w, p, p, p, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # [B, C, d, p, h, p, w, p]
        x = x.reshape(x.shape[0], c, d * p, h * p, w * p)
        return x

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, C, H, W] or [B, C, D, H, W] noisy input
            timesteps: [B] diffusion timesteps (0-1 for flow matching)
            cond: [B, C_cond, ...] conditioning (e.g., segmentation mask)

        Returns:
            [B, C, H, W] or [B, C, D, H, W] velocity/noise prediction
        """
        # Handle conditioning
        context = None
        if self.conditioning == "concat" and cond is not None:
            x = torch.cat([x, cond], dim=1)
        elif self.conditioning == "cross_attn" and cond is not None:
            context = self.cond_embedder(cond)  # [B, N_cond, D]

        # Patchify + positional encoding
        x = self.x_embedder(x) + self.pos_embed  # [B, N, D]

        # Timestep conditioning
        t = self.t_embedder(timesteps)  # [B, D]

        # Transformer blocks
        for block in self.blocks:
            x = block(x, t, context)

        # Final projection
        x = self.final_layer(x, t)  # [B, N, p^d * C]

        # Unpatchify
        if self.spatial_dims == 2:
            x = self.unpatchify_2d(x)
        else:
            x = self.unpatchify_3d(x)

        return x


def create_sit(
    variant: str = 'B',
    spatial_dims: int = 2,
    input_size: int = 32,
    patch_size: int = 2,
    in_channels: int = 4,
    conditioning: str = "concat",
    cond_channels: int = 1,
    learn_sigma: bool = False,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    depth_size: Optional[int] = None,
    **kwargs,
) -> SiT:
    """Create a SiT model with predefined variant configuration.

    Args:
        variant: Model variant ('S', 'B', 'L', 'XL')
        spatial_dims: 2 or 3
        input_size: Spatial input size
        patch_size: Patch size
        in_channels: Input channels
        conditioning: "concat" or "cross_attn"
        cond_channels: Conditioning channels
        learn_sigma: Predict variance
        drop_rate: Dropout rate
        drop_path_rate: Stochastic depth rate (0.0 = disabled, 0.1-0.2 typical)
        depth_size: Depth for 3D (if different from input_size)
        **kwargs: Additional arguments passed to SiT

    Returns:
        SiT model
    """
    if variant not in SIT_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(SIT_VARIANTS.keys())}")

    config = SIT_VARIANTS[variant]

    return SiT(
        spatial_dims=spatial_dims,
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=config['hidden_size'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        conditioning=conditioning,
        cond_channels=cond_channels,
        learn_sigma=learn_sigma,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        depth_size=depth_size,
        **kwargs,
    )


# Convenience aliases
def SiT_S(**kwargs) -> SiT:
    """SiT-Small: 33M parameters"""
    return create_sit(variant='S', **kwargs)


def SiT_B(**kwargs) -> SiT:
    """SiT-Base: 130M parameters"""
    return create_sit(variant='B', **kwargs)


def SiT_L(**kwargs) -> SiT:
    """SiT-Large: 458M parameters"""
    return create_sit(variant='L', **kwargs)


def SiT_XL(**kwargs) -> SiT:
    """SiT-XLarge: 675M parameters"""
    return create_sit(variant='XL', **kwargs)
