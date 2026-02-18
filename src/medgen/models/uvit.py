"""
U-ViT: Vision Transformer with Skip Connections for Diffusion.

Faithful implementation of the U-ViT architecture from the paper:
"All are Worth Words: A ViT Backbone for Diffusion Models" (Bao et al., CVPR 2023)

Key differences from DiT:
- Token-based conditioning (timestep prepended as token, not adaLN modulation)
- Standard Pre-LN ViT blocks (no adaLN-Zero)
- Skip connections between encoder and decoder halves
- Depth must be odd (enc + 1 mid + dec)
- qkv_bias=False (paper default)
- trunc_normal_ init (no zero-init)
- Final conv to prevent patch-boundary artifacts

Reference: https://arxiv.org/abs/2209.12152
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .dit_blocks import Attention, Mlp
from .embeddings import (
    PatchEmbed2D,
    PatchEmbed3D,
    TimestepEmbedder,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
)

# U-ViT variant configurations (from paper Table 1)
UVIT_VARIANTS = {
    'S': {'hidden_size': 512, 'depth': 13, 'num_heads': 8},
    'S-Deep': {'hidden_size': 512, 'depth': 17, 'num_heads': 8},
    'M': {'hidden_size': 768, 'depth': 17, 'num_heads': 12},
    'L': {'hidden_size': 1024, 'depth': 21, 'num_heads': 16},
}


class UViTBlock(nn.Module):
    """Standard Pre-LN ViT block with optional skip connection.

    Unlike DiTBlock which uses adaLN-Zero for timestep conditioning,
    UViTBlock uses standard LayerNorm. Conditioning flows through
    self-attention via prepended timestep tokens.

    Args:
        hidden_size: Hidden dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        qkv_bias: Whether to use bias in QKV projection. Paper default: False.
        qk_norm: Whether to use QK-normalization.
        skip: Whether this block has a skip connection from encoder.
        drop: Dropout rate.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        skip: bool = False,
        drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(hidden_size, hidden_features=mlp_hidden, drop=drop)

        # Skip connection projection: cat([x, skip], dim=-1) -> Linear(2D, D)
        self.skip_linear = nn.Linear(2 * hidden_size, hidden_size) if skip else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input tokens (includes time token)
            skip: [B, N, D] skip connection from encoder (optional)

        Returns:
            [B, N, D] output tokens
        """
        if self.skip_linear is not None and skip is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class UViT(nn.Module):
    """U-ViT: Vision Transformer with Long Skip Connections for Diffusion.

    Token-based conditioning: timestep (and optional conditioning patches)
    are prepended as tokens to the sequence. All conditioning flows through
    self-attention — no per-block modulation.

    Architecture:
        Patchify -> [time_token, patch_tokens] + pos_embed
        -> depth//2 encoder blocks (save skip connections)
        -> 1 mid block
        -> depth//2 decoder blocks (consume skip connections)
        -> LayerNorm -> Linear -> strip tokens -> unpatchify -> Conv3x3

    Args:
        spatial_dims: Number of spatial dimensions (2 or 3).
        input_size: Spatial size of input.
        patch_size: Patch size for tokenization.
        in_channels: Number of input channels.
        out_channels: Output channels (None = same as in_channels).
        hidden_size: Transformer hidden dimension.
        depth: Number of transformer blocks (must be odd).
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        conditioning: Conditioning mode ("concat" or "cross_attn").
        cond_channels: Number of conditioning channels.
        qkv_bias: Whether to use bias in QKV. Paper default: False.
        qk_norm: Whether to use QK-normalization.
        drop_rate: Dropout rate.
        depth_size: Depth size for 3D (if different from input_size).
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        out_channels: int | None = None,
        hidden_size: int = 512,
        depth: int = 13,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        conditioning: Literal["concat", "cross_attn"] = "concat",
        cond_channels: int = 1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        drop_rate: float = 0.0,
        depth_size: int | None = None,
    ):
        super().__init__()

        # Validate depth is odd
        if depth % 2 == 0:
            raise ValueError(
                f"U-ViT depth must be odd (enc + 1 mid + dec), got {depth}. "
                f"Try {depth - 1} or {depth + 1}."
            )

        self.spatial_dims = spatial_dims
        self.input_size = input_size
        self.depth_size = depth_size or input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.conditioning = conditioning
        self.cond_channels = cond_channels
        self.depth = depth

        # Validate divisibility
        if input_size % patch_size != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by patch_size ({patch_size})"
            )
        if spatial_dims == 3 and self.depth_size % patch_size != 0:
            raise ValueError(
                f"depth_size ({self.depth_size}) must be divisible by patch_size ({patch_size})"
            )

        # Calculate grid and patch counts
        if spatial_dims == 2:
            self.num_patches = (input_size // patch_size) ** 2
            self.grid_size = input_size // patch_size
        else:
            self.grid_size_d = self.depth_size // patch_size
            self.grid_size_h = input_size // patch_size
            self.grid_size_w = input_size // patch_size
            self.num_patches = self.grid_size_d * self.grid_size_h * self.grid_size_w

        # Patch embedding
        # For concat mode, in_channels already includes conditioning
        if spatial_dims == 2:
            self.x_embedder = PatchEmbed2D(patch_size, in_channels, hidden_size)
        else:
            self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)

        # Cross-attention conditioning: embed cond patches as additional tokens
        self.use_cross_attn = (conditioning == "cross_attn")
        if self.use_cross_attn:
            if spatial_dims == 2:
                self.cond_embedder = PatchEmbed2D(patch_size, cond_channels, hidden_size)
            else:
                self.cond_embedder = PatchEmbed3D(patch_size, cond_channels, hidden_size)

        # Timestep embedding (output becomes a token, not modulation)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Learnable positional embedding: covers time token + all patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.num_patches, hidden_size)
        )

        # Separate positional embedding for conditioning tokens (cross_attn only)
        if self.use_cross_attn:
            self.cond_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, hidden_size)
            )

        # Encoder blocks (depth // 2)
        num_enc = depth // 2
        self.in_blocks = nn.ModuleList([
            UViTBlock(
                hidden_size, num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                skip=False,
                drop=drop_rate,
            )
            for _ in range(num_enc)
        ])

        # Mid block (1)
        self.mid_block = UViTBlock(
            hidden_size, num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            skip=False,
            drop=drop_rate,
        )

        # Decoder blocks (depth // 2, with skip connections)
        self.out_blocks = nn.ModuleList([
            UViTBlock(
                hidden_size, num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                skip=True,
                drop=drop_rate,
            )
            for _ in range(num_enc)
        ])

        # Final layers
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        patch_dim = patch_size ** spatial_dims * self.out_channels
        self.decoder_pred = nn.Linear(hidden_size, patch_dim)

        # Final conv to prevent patch-boundary artifacts (paper section 3.3)
        if spatial_dims == 2:
            self.final_conv = nn.Conv2d(
                self.out_channels, self.out_channels,
                kernel_size=3, padding=1,
            )
        else:
            self.final_conv = nn.Conv3d(
                self.out_channels, self.out_channels,
                kernel_size=3, padding=1,
            )

        # Gradient checkpointing flag
        self._use_gradient_checkpointing = False

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights following U-ViT paper conventions.

        Key differences from DiT:
        - trunc_normal_(std=0.02) for all Linear/Conv weights
        - bias = 0 everywhere
        - NO zero-init (no adaLN to zero-init)
        - Positional embedding initialized from sincos then made learnable
        """
        def _trunc_normal_init(module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_trunc_normal_init)

        # Initialize positional embedding from sincos (then learnable)
        if self.spatial_dims == 2:
            sincos = get_2d_sincos_pos_embed(self.hidden_size, self.grid_size)
        else:
            sincos = get_3d_sincos_pos_embed(
                self.hidden_size,
                self.grid_size_d,
                self.grid_size_h,
                self.grid_size_w,
            )
        # sincos is [num_patches, D], pos_embed is [1, 1+num_patches, D]
        # Time token position gets zero init (learned from scratch)
        sincos_tensor = torch.from_numpy(sincos).float()
        self.pos_embed.data[:, 1:, :] = sincos_tensor.unsqueeze(0)
        # pos_embed[:, 0, :] stays zero-initialized for time token

        # Initialize cond_pos_embed from same sincos (same spatial structure)
        if self.use_cross_attn:
            self.cond_pos_embed.data.copy_(sincos_tensor.unsqueeze(0))

        # Timestep embedder MLP
        nn.init.trunc_normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.trunc_normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for all transformer blocks."""
        self._use_gradient_checkpointing = True

    def _checkpoint_forward(self, block, *args):
        """Run block forward with optional gradient checkpointing."""
        if self._use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                block, *args, use_reentrant=False
            )
        return block(*args)

    def unpatchify_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patch tokens back to 2D image."""
        p = self.patch_size
        h = w = self.input_size // p
        c = self.out_channels
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, h, p, w, p]
        x = x.reshape(x.shape[0], c, h * p, w * p)
        return x

    def unpatchify_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patch tokens back to 3D volume."""
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
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, C, H, W] or [B, C, D, H, W] noisy input
            timesteps: [B] diffusion timesteps
            cond: [B, C_cond, ...] conditioning (e.g., segmentation mask)

        Returns:
            [B, C_out, H, W] or [B, C_out, D, H, W] prediction
        """
        # Handle conditioning
        if self.conditioning == "concat" and cond is not None:
            x = torch.cat([x, cond], dim=1)

        # Patchify
        patch_tokens = self.x_embedder(x)  # [B, N, D]

        # Time token
        t_token = self.t_embedder(timesteps).unsqueeze(1)  # [B, 1, D]

        # For cross_attn: prepend conditioning tokens between time and patches
        # Each token group gets its own positional embedding
        if self.use_cross_attn and cond is not None:
            cond_tokens = self.cond_embedder(cond) + self.cond_pos_embed
            # Sequence: [time_token + time_pos, cond_tokens + cond_pos, patches + patch_pos]
            x = torch.cat([
                t_token + self.pos_embed[:, :1, :],        # time pos
                cond_tokens,                                 # already has cond_pos_embed
                patch_tokens + self.pos_embed[:, 1:, :],   # patch pos
            ], dim=1)
        else:
            # Build sequence: [time_token, patch_tokens] + pos_embed
            x = torch.cat([t_token, patch_tokens], dim=1) + self.pos_embed

        # Encoder: save skip connections
        skips = []
        for block in self.in_blocks:
            x = self._checkpoint_forward(block, x)
            skips.append(x)

        # Mid block
        x = self._checkpoint_forward(self.mid_block, x)

        # Decoder: consume skip connections in reverse
        for block, skip in zip(self.out_blocks, reversed(skips)):
            x = self._checkpoint_forward(block, x, skip)

        # Final projection
        x = self.norm(x)
        x = self.decoder_pred(x)

        # Strip extra tokens to get only patch tokens
        # Sequence layout: [time, (cond if cross_attn), patches]
        num_prefix = 1 + (self.num_patches if self.use_cross_attn and cond is not None else 0)
        x = x[:, num_prefix:num_prefix + self.num_patches, :]

        # Unpatchify
        if self.spatial_dims == 2:
            x = self.unpatchify_2d(x)
        else:
            x = self.unpatchify_3d(x)

        # Final conv to smooth patch boundaries
        x = self.final_conv(x)

        return x


def create_uvit(
    variant: str = 'S',
    spatial_dims: int = 2,
    input_size: int = 32,
    patch_size: int = 2,
    in_channels: int = 4,
    out_channels: int | None = None,
    conditioning: str = "concat",
    cond_channels: int = 1,
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.0,
    depth_size: int | None = None,
    **kwargs,
) -> UViT:
    """Create a U-ViT model with predefined variant configuration.

    Args:
        variant: Model variant ('S', 'S-Deep', 'M', 'L')
        spatial_dims: 2 or 3
        input_size: Spatial input size
        patch_size: Patch size
        in_channels: Input channels
        conditioning: "concat" or "cross_attn"
        cond_channels: Conditioning channels
        drop_rate: Dropout rate
        depth_size: Depth for 3D (if different from input_size)
        **kwargs: Additional arguments passed to UViT

    Returns:
        UViT model
    """
    if variant not in UVIT_VARIANTS:
        raise ValueError(
            f"Unknown U-ViT variant: {variant}. "
            f"Choose from {list(UVIT_VARIANTS.keys())}"
        )

    config = UVIT_VARIANTS[variant]

    return UViT(
        spatial_dims=spatial_dims,
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_size=config['hidden_size'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=mlp_ratio,
        conditioning=conditioning,
        cond_channels=cond_channels,
        qkv_bias=False,  # Paper default
        qk_norm=False,  # Paper default (no QK-norm)
        drop_rate=drop_rate,
        depth_size=depth_size,
        **kwargs,
    )
