"""
LaMamba-Diff: Mamba + Local Attention U-Net for diffusion.

A U-Net architecture where each block combines:
1. SS2D (multi-directional Mamba) for global context with linear complexity
2. Windowed self-attention for local spatial detail
3. FFN for channel mixing

All conditioned via AdaLN-Zero (9 modulation params per block, same as DiT).

Supports both 2D images and 3D volumes. For 3D, SS2D uses 6-directional
scans (±D, ±H, ±W) instead of 4 (±row, ±col).

Reference: LaMamba-Diff (Fu et al., 2024) — https://arxiv.org/abs/2408.02615
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from .dit_blocks import Mlp
from .embeddings import (
    PatchEmbed2D,
    PatchEmbed3D,
    TimestepEmbedder,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
)
from .mamba_blocks import SS2D, WindowAttention

logger = logging.getLogger(__name__)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation for spatial tensors (2D/3D channels-last).

    Unlike dit_blocks.modulate which assumes [B, N, D], this works for
    [B, *spatial, C] where shift/scale already have matching spatial dims.
    """
    return x * (1 + scale) + shift


# Model size variants — embed_dim and num_heads (depths are configurable)
MAMBA_VARIANTS = {
    'S':  {'embed_dim': 128, 'num_heads': 4},
    'B':  {'embed_dim': 192, 'num_heads': 8},
    'L':  {'embed_dim': 256, 'num_heads': 16},
    'XL': {'embed_dim': 320, 'num_heads': 16},
}


# =============================================================================
# Core Block
# =============================================================================

class MambaDiffBlock(nn.Module):
    """LaMamba block: SS2D → WindowAttention → FFN, each with AdaLN-Zero.

    Args:
        dim: Feature dimension.
        num_heads: Attention heads for windowed attention.
        window_size: Window size for local attention.
        shift_size: Shift for shifted-window attention (0 or window_size//2).
        cond_dim: Conditioning dimension (timestep embedding size).
        ssm_d_state: SSM state dimension.
        ssm_ratio: SSM expansion ratio.
        mlp_ratio: FFN expansion ratio.
        spatial_dims: 2 or 3.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        cond_dim: int | None = None,
        ssm_d_state: int = 1,
        ssm_ratio: float = 2.0,
        mlp_ratio: float = 4.0,
        spatial_dims: int = 2,
    ):
        super().__init__()
        cond_dim = cond_dim or dim

        # Condition projection (if cond_dim != dim)
        self.cond_proj = nn.Linear(cond_dim, dim) if cond_dim != dim else nn.Identity()

        # AdaLN-Zero: 9 modulation params (shift, scale, gate for each of 3 sub-layers)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim),
        )
        # Zero-init gate parameters for stable training
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        # Sub-layer 1: SS2D (Mamba)
        self.norm_ssm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ssm = SS2D(
            d_model=dim, d_state=ssm_d_state, ssm_ratio=ssm_ratio,
            spatial_dims=spatial_dims,
        )

        # Sub-layer 2: Window Attention
        self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = WindowAttention(
            dim=dim, num_heads=num_heads, window_size=window_size,
            shift_size=shift_size, spatial_dims=spatial_dims,
        )

        # Sub-layer 3: FFN
        self.norm_ffn = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, *spatial, C] input features (channels-last).
            c: [B, cond_dim] conditioning (timestep embedding).

        Returns:
            [B, *spatial, C] output.
        """
        # Project condition and generate modulation params
        c = self.cond_proj(c)
        # Expand c to broadcast with spatial dims
        c_expanded = c
        for _ in range(len(x.shape) - 2):
            c_expanded = c_expanded.unsqueeze(1)
        mods = self.adaLN_modulation(c_expanded)
        (shift_s, scale_s, gate_s,
         shift_a, scale_a, gate_a,
         shift_f, scale_f, gate_f) = mods.chunk(9, dim=-1)

        # SS2D
        x = x + gate_s * self.ssm(_modulate(self.norm_ssm(x), shift_s, scale_s))

        # Window Attention
        x = x + gate_a * self.attn(_modulate(self.norm_attn(x), shift_a, scale_a))

        # FFN (operates per-token; flatten spatial dims for Mlp which expects [B, N, C])
        h = _modulate(self.norm_ffn(x), shift_f, scale_f)
        shape = h.shape  # [B, *spatial, C]
        h = h.reshape(shape[0], -1, shape[-1])  # [B, N, C]
        h = self.ffn(h)  # [B, N, C]
        h = h.reshape(shape)  # [B, *spatial, C]
        x = x + gate_f * h

        return x


# =============================================================================
# Downsampling / Upsampling
# =============================================================================

class Downsample(nn.Module):
    """Spatial downsampling via strided convolution (2× per spatial dim)."""

    def __init__(self, dim_in: int, dim_out: int, spatial_dims: int = 2):
        super().__init__()
        conv_cls = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.conv = conv_cls(dim_in, dim_out, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(dim_out)
        self.spatial_dims = spatial_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, *spatial, C] → [B, *spatial/2, C_out]."""
        if self.spatial_dims == 2:
            x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        else:
            x = x.permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
        x = self.conv(x)
        if self.spatial_dims == 2:
            x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        else:
            x = x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]
        return self.norm(x)


class Upsample(nn.Module):
    """Spatial upsampling via linear + pixel/voxel shuffle (2× per spatial dim)."""

    def __init__(self, dim_in: int, dim_out: int, spatial_dims: int = 2):
        super().__init__()
        self.spatial_dims = spatial_dims
        factor = 2 ** spatial_dims  # 4 for 2D, 8 for 3D
        self.expand = nn.Linear(dim_in, dim_out * factor)
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, *spatial, C] → [B, *spatial*2, C_out]."""
        x = self.expand(x)  # [B, *spatial, C_out * factor]
        if self.spatial_dims == 2:
            B, H, W, C = x.shape
            x = x.reshape(B, H, W, 2, 2, C // 4)
            x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H * 2, W * 2, C // 4)
        else:
            B, D, H, W, C = x.shape
            x = x.reshape(B, D, H, W, 2, 2, 2, C // 8)
            x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, D * 2, H * 2, W * 2, C // 8)
        return self.norm(x)


# =============================================================================
# U-Net Stages
# =============================================================================

class MambaDiffStage(nn.Module):
    """A stage of the U-Net: N blocks + optional down/upsample."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        cond_dim: int,
        ssm_d_state: int,
        ssm_ratio: float,
        mlp_ratio: float,
        spatial_dims: int,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            MambaDiffBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                cond_dim=cond_dim, ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio, mlp_ratio=mlp_ratio,
                spatial_dims=spatial_dims,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = ckpt.checkpoint(blk, x, c, use_reentrant=False)
            else:
                x = blk(x, c)
        return x


# =============================================================================
# Full Model
# =============================================================================

class MambaDiff(nn.Module):
    """LaMamba-Diff U-Net for diffusion.

    U-Net with LaMamba blocks (SS2D + WindowAttention + FFN).
    Supports both 2D and 3D pixel-space diffusion.

    Args:
        spatial_dims: 2 or 3.
        input_size: Spatial size (H/W for 2D, H for 3D).
        patch_size: Patch embedding size (2 recommended for pixel-space).
        in_channels: Input channels (e.g., 2 for bravo+seg).
        out_channels: Output channels (e.g., 1 for bravo).
        embed_dim: Base embedding dimension (expanded as [D, 2D, 4D, 4D]).
        depths: Number of blocks per encoder stage (e.g., [2, 2, 2, 2]).
        bottleneck_depth: Number of blocks in the bottleneck.
        num_heads: Number of attention heads.
        window_size: Window size for local attention.
        skip: Number of trailing encoder stages that don't downsample.
        ssm_d_state: SSM state dimension.
        ssm_ratio: SSM expansion ratio.
        mlp_ratio: FFN expansion ratio.
        depth_size: Depth dimension for 3D (None for 2D).
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        input_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 2,
        out_channels: int = 1,
        embed_dim: int = 192,
        depths: list[int] | None = None,
        bottleneck_depth: int = 2,
        num_heads: int = 8,
        window_size: int = 8,
        skip: int = 2,
        ssm_d_state: int = 1,
        ssm_ratio: float = 2.0,
        mlp_ratio: float = 4.0,
        depth_size: int | None = None,
    ):
        super().__init__()
        if depths is None:
            depths = [2, 2, 2, 2]

        self.spatial_dims = spatial_dims
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stages = len(depths)
        self.skip = skip
        self._use_checkpoint = False

        # Dimension schedule: [D, 2D, 4D, 4D] (last `skip` stages share max dim)
        dims = []
        for i in range(self.num_stages):
            scale = min(2 ** i, 2 ** (self.num_stages - skip))
            dims.append(embed_dim * scale)
        self.dims = dims

        # Conditioning dimension = largest dim
        cond_dim = dims[-1]
        self.hidden_size = cond_dim  # For torch.compile / ScoreAug detection

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size=cond_dim)

        # Patch embedding
        if spatial_dims == 2:
            self.patch_embed = PatchEmbed2D(
                patch_size=patch_size, in_channels=in_channels, embed_dim=dims[0],
            )
            grid_h = input_size // patch_size
            grid_w = grid_h
            self.grid_shape = (grid_h, grid_w)
            num_patches = grid_h * grid_w
            pos_embed_data = get_2d_sincos_pos_embed(dims[0], grid_h)
        else:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size, in_channels=in_channels, embed_dim=dims[0],
            )
            assert depth_size is not None, "depth_size required for 3D"
            grid_d = depth_size // patch_size
            grid_h = input_size // patch_size
            grid_w = grid_h
            self.grid_shape = (grid_d, grid_h, grid_w)
            num_patches = grid_d * grid_h * grid_w
            pos_embed_data = get_3d_sincos_pos_embed(dims[0], grid_d, grid_h, grid_w)

        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed_data).float().reshape(1, *self.grid_shape, dims[0]),
            requires_grad=True,
        )

        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(self.num_stages):
            self.encoder_stages.append(MambaDiffStage(
                dim=dims[i], depth=depths[i], num_heads=num_heads,
                window_size=window_size, cond_dim=cond_dim,
                ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio,
                mlp_ratio=mlp_ratio, spatial_dims=spatial_dims,
            ))
            # Downsample except for last `skip` stages
            if i < self.num_stages - skip:
                self.downsamples.append(Downsample(dims[i], dims[i + 1], spatial_dims))
            else:
                self.downsamples.append(nn.Identity())

        # Bottleneck
        self.bottleneck = MambaDiffStage(
            dim=dims[-1], depth=bottleneck_depth, num_heads=num_heads,
            window_size=window_size, cond_dim=cond_dim,
            ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio,
            mlp_ratio=mlp_ratio, spatial_dims=spatial_dims,
        )

        # Decoder stages (depth+1 blocks each, reversed order)
        # Per stage: upsample (if needed) → skip_add → blocks
        # Upsample mirrors encoder: encoder stage j downsampled dims[j]→dims[j+1],
        # so before decoder processes encoder stage j's skip, we upsample dims[j+1]→dims[j].
        self.decoder_stages = nn.ModuleList()
        self.pre_upsamples = nn.ModuleList()  # upsample BEFORE skip add
        for i in range(self.num_stages):
            enc_idx = self.num_stages - 1 - i  # reverse order
            dec_dim = dims[enc_idx]
            dec_depth = depths[enc_idx] + 1  # SD-UNet: decoder gets +1 block

            # Pre-upsample: reverse the encoder's downsample at this level.
            # Encoder stage enc_idx had a downsample if enc_idx < num_stages - skip.
            # That means x coming in is at dims[enc_idx+1], but the skip is at dims[enc_idx].
            # We need to upsample dims[enc_idx+1] → dims[enc_idx].
            if enc_idx < self.num_stages - skip and dims[enc_idx] != dims[enc_idx + 1]:
                self.pre_upsamples.append(Upsample(dims[enc_idx + 1], dims[enc_idx], spatial_dims))
            else:
                self.pre_upsamples.append(nn.Identity())

            self.decoder_stages.append(MambaDiffStage(
                dim=dec_dim, depth=dec_depth, num_heads=num_heads,
                window_size=window_size, cond_dim=cond_dim,
                ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio,
                mlp_ratio=mlp_ratio, spatial_dims=spatial_dims,
            ))

        # Final layer: AdaLN → Linear → unpatchify
        self.final_norm = nn.LayerNorm(dims[0], elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * dims[0]),
        )
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)

        self.final_linear = nn.Linear(
            dims[0], out_channels * patch_size ** spatial_dims,
        )
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following DiT conventions."""
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)
        # Re-zero gate params (may have been overwritten by _init)
        for stage in list(self.encoder_stages) + list(self.decoder_stages) + [self.bottleneck]:
            for blk in stage.blocks:
                nn.init.zeros_(blk.adaLN_modulation[1].weight)
                nn.init.zeros_(blk.adaLN_modulation[1].bias)
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for all stages."""
        self._use_checkpoint = True
        for stage in list(self.encoder_stages) + list(self.decoder_stages) + [self.bottleneck]:
            stage.use_checkpoint = True
        logger.info("Gradient checkpointing enabled for MambaDiff")

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patched representation back to spatial image/volume.

        Args:
            x: [B, *grid, C * patch_size^d] patched features.

        Returns:
            [B, out_channels, *spatial] output.
        """
        p = self.patch_size
        if self.spatial_dims == 2:
            B, gH, gW, C = x.shape
            c = self.out_channels
            x = x.reshape(B, gH, gW, c, p, p)
            x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, c, gH * p, gW * p)
        else:
            B, gD, gH, gW, C = x.shape
            c = self.out_channels
            x = x.reshape(B, gD, gH, gW, c, p, p, p)
            x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(B, c, gD * p, gH * p, gW * p)
        return x

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, C, H, W] or [B, C, D, H, W] input.
            timesteps: [B] timestep values.

        Returns:
            [B, out_C, H, W] or [B, out_C, D, H, W] prediction.
        """
        # Timestep conditioning
        c = self.t_embedder(timesteps)  # [B, cond_dim]

        # Patch embed: [B, C, *spatial] → [B, N, embed_dim] → [B, *grid, embed_dim]
        x = self.patch_embed(x)  # [B, N, D]
        x = x.reshape(x.shape[0], *self.grid_shape, -1)  # [B, *grid, D]
        x = x + self.pos_embed

        # Encoder
        skips = []
        for i in range(self.num_stages):
            x = self.encoder_stages[i](x, c)
            skips.append(x)
            x = self.downsamples[i](x)

        # Bottleneck (with residual)
        x = x + self.bottleneck(x, c)

        # Decoder: upsample → skip_add → blocks
        for i in range(self.num_stages):
            x = self.pre_upsamples[i](x)
            skip = skips[self.num_stages - 1 - i]
            x = x + skip
            x = self.decoder_stages[i](x, c)

        # Final layer: AdaLN → linear → unpatchify
        c_expanded = c
        for _ in range(len(x.shape) - 2):
            c_expanded = c_expanded.unsqueeze(1)
        shift, scale = self.final_adaLN(c_expanded).chunk(2, dim=-1)
        x = _modulate(self.final_norm(x), shift, scale)
        x = self.final_linear(x)

        return self._unpatchify(x)


# =============================================================================
# Factory
# =============================================================================

def create_mamba_diff(
    variant: str = 'B',
    spatial_dims: int = 2,
    input_size: int = 128,
    patch_size: int = 2,
    in_channels: int = 2,
    out_channels: int = 1,
    depth_size: int | None = None,
    depths: list[int] | None = None,
    bottleneck_depth: int = 2,
    window_size: int = 8,
    skip: int = 2,
    ssm_d_state: int = 1,
    ssm_ratio: float = 2.0,
    mlp_ratio: float = 4.0,
) -> MambaDiff:
    """Create a MambaDiff model from variant name.

    Args:
        variant: Model size variant ('S', 'B', 'L', 'XL').
        spatial_dims: 2 or 3.
        input_size: Spatial H/W size.
        patch_size: Patch embedding size.
        in_channels: Input channels.
        out_channels: Output channels.
        depth_size: Depth for 3D volumes (None for 2D).
        depths: Blocks per encoder stage.
        bottleneck_depth: Blocks in bottleneck.
        window_size: Window attention size.
        skip: Trailing encoder stages that don't downsample.
        ssm_d_state: SSM state dimension.
        ssm_ratio: SSM expansion ratio.
        mlp_ratio: FFN expansion ratio.

    Returns:
        Initialized MambaDiff model.
    """
    if variant not in MAMBA_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}', choose from {list(MAMBA_VARIANTS.keys())}")

    cfg = MAMBA_VARIANTS[variant]

    model = MambaDiff(
        spatial_dims=spatial_dims,
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=cfg['embed_dim'],
        depths=depths,
        bottleneck_depth=bottleneck_depth,
        num_heads=cfg['num_heads'],
        window_size=window_size,
        skip=skip,
        ssm_d_state=ssm_d_state,
        ssm_ratio=ssm_ratio,
        mlp_ratio=mlp_ratio,
        depth_size=depth_size,
    )

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(
        f"Created MambaDiff-{variant}: spatial_dims={spatial_dims}, "
        f"input_size={input_size}, patch_size={patch_size}, "
        f"embed_dim={cfg['embed_dim']}, num_heads={cfg['num_heads']}, "
        f"dims={model.dims}, params={num_params:.1f}M"
    )

    return model
