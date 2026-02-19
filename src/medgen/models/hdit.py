"""
HDiT: Hierarchical Diffusion Transformer.

A U-shaped transformer architecture that uses token merging/splitting
to process fine-resolution patches at manageable cost. Uses adaLN-Zero
DiTBlocks (same as standard DiT) with a hierarchical structure.

Design rationale:
- Enables patch_size=4 for 3D volumes (128x128x160 -> 40K tokens at level 0)
- Token merging reduces sequence length at each level (2x2x2 -> 8x reduction)
- Most compute happens at reduced token counts
- Skip connections between encoder and decoder levels
- Reuses existing DiTBlock unchanged

Inspired by U-DiT (Tian et al., NeurIPS 2024) concepts but adapted for
3D patchified sequences (U-DiT uses 2D convolutions at pixel resolution).
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .dit import DIT_VARIANTS
from .dit_blocks import DiTBlock, FinalLayer
from .embeddings import (
    PatchEmbed2D,
    PatchEmbed3D,
    TimestepEmbedder,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
)


class TokenMerge(nn.Module):
    """Merge adjacent tokens by grouping and projecting.

    Groups spatial neighbors (2x2 for 2D, 2x2x2 for 3D) and projects
    the concatenated features down to the original dimension.

    Args:
        hidden_size: Token dimension.
        spatial_dims: 2 or 3.
    """

    def __init__(self, hidden_size: int, spatial_dims: int = 2):
        super().__init__()
        self.spatial_dims = spatial_dims
        merge_factor = 4 if spatial_dims == 2 else 8  # 2^spatial_dims
        self.proj = nn.Linear(merge_factor * hidden_size, hidden_size)

    def forward(
        self, x: torch.Tensor, grid_dims: tuple[int, ...]
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        """
        Args:
            x: [B, N, D] flattened tokens
            grid_dims: (H, W) or (D, H, W) current grid dimensions

        Returns:
            merged: [B, N/merge_factor, D]
            new_grid_dims: halved grid dimensions
        """
        B, N, D = x.shape

        if self.spatial_dims == 2:
            H, W = grid_dims
            # Reshape to spatial grid
            x = x.reshape(B, H, W, D)
            # Group 2x2 neighbors
            x = x.reshape(B, H // 2, 2, W // 2, 2, D)
            x = x.permute(0, 1, 3, 2, 4, 5)  # [B, H/2, W/2, 2, 2, D]
            x = x.reshape(B, H // 2, W // 2, 4 * D)  # concat neighbors
            # Flatten spatial
            x = x.reshape(B, (H // 2) * (W // 2), 4 * D)
            new_grid = (H // 2, W // 2)
        else:
            Gd, Gh, Gw = grid_dims
            x = x.reshape(B, Gd, Gh, Gw, D)
            x = x.reshape(B, Gd // 2, 2, Gh // 2, 2, Gw // 2, 2, D)
            x = x.permute(0, 1, 3, 5, 2, 4, 6, 7)  # [B, Gd/2, Gh/2, Gw/2, 2, 2, 2, D]
            x = x.reshape(B, Gd // 2, Gh // 2, Gw // 2, 8 * D)
            x = x.reshape(B, (Gd // 2) * (Gh // 2) * (Gw // 2), 8 * D)
            new_grid = (Gd // 2, Gh // 2, Gw // 2)

        x = self.proj(x)
        return x, new_grid


class TokenSplit(nn.Module):
    """Split tokens back to higher resolution by projecting and scattering.

    Inverse of TokenMerge: projects each token to merge_factor * D features,
    then scatters them into a 2x2 (2D) or 2x2x2 (3D) neighborhood.

    Args:
        hidden_size: Token dimension.
        spatial_dims: 2 or 3.
    """

    def __init__(self, hidden_size: int, spatial_dims: int = 2):
        super().__init__()
        self.spatial_dims = spatial_dims
        merge_factor = 4 if spatial_dims == 2 else 8
        self.proj = nn.Linear(hidden_size, merge_factor * hidden_size)

    def forward(
        self, x: torch.Tensor, grid_dims: tuple[int, ...]
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        """
        Args:
            x: [B, N, D] tokens at coarse resolution
            grid_dims: (H, W) or (D, H, W) current (coarse) grid dimensions

        Returns:
            split: [B, N*merge_factor, D] tokens at finer resolution
            new_grid_dims: doubled grid dimensions
        """
        B, N, D = x.shape
        x = self.proj(x)  # [B, N, merge_factor * D]

        if self.spatial_dims == 2:
            H, W = grid_dims
            x = x.reshape(B, H, W, 4, D)
            # Scatter: 4 -> 2x2
            x = x.reshape(B, H, W, 2, 2, D)
            x = x.permute(0, 1, 3, 2, 4, 5)  # [B, H, 2, W, 2, D]
            x = x.reshape(B, H * 2, W * 2, D)
            x = x.reshape(B, (H * 2) * (W * 2), D)
            new_grid = (H * 2, W * 2)
        else:
            Gd, Gh, Gw = grid_dims
            x = x.reshape(B, Gd, Gh, Gw, 8, D)
            x = x.reshape(B, Gd, Gh, Gw, 2, 2, 2, D)
            x = x.permute(0, 1, 4, 2, 5, 3, 6, 7)  # [B, Gd, 2, Gh, 2, Gw, 2, D]
            x = x.reshape(B, Gd * 2, Gh * 2, Gw * 2, D)
            x = x.reshape(B, (Gd * 2) * (Gh * 2) * (Gw * 2), D)
            new_grid = (Gd * 2, Gh * 2, Gw * 2)

        return x, new_grid


class HDiT(nn.Module):
    """Hierarchical Diffusion Transformer.

    U-shaped transformer with token merging/splitting for multi-resolution
    processing. Enables fine patches (e.g., patch=4) at manageable cost.

    Architecture:
        Patchify(patch_size) -> [B, N, D]
        + pos_embed_level0

        Encoder level 0: blocks at full resolution -> skip[0]
        TokenMerge -> encoder level 1: blocks at 1/8 tokens -> skip[1]
        ...
        Bottleneck blocks at lowest resolution
        ...
        TokenSplit -> skip_proj(cat[x, skip[1]]) -> decoder level 1
        TokenSplit -> skip_proj(cat[x, skip[0]]) -> decoder level 0

        FinalLayer -> unpatchify

    Args:
        spatial_dims: Number of spatial dimensions (2 or 3).
        input_size: Spatial size of input.
        patch_size: Patch size for tokenization.
        in_channels: Number of input channels.
        out_channels: Output channels.
        hidden_size: Transformer hidden dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        conditioning: Conditioning mode ("concat" or "cross_attn").
        cond_channels: Number of conditioning channels.
        level_depths: Number of DiT blocks at each level.
            Must be odd length (symmetric encoder + bottleneck + decoder).
            Example: [2, 4, 6, 4, 2] = 2 enc levels, 6 bottleneck blocks, 2 dec levels.
        qk_norm: Whether to use QK-normalization.
        drop_rate: Dropout rate.
        drop_path_rate: Stochastic depth rate.
        depth_size: Depth size for 3D.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        input_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 4,
        out_channels: int | None = None,
        hidden_size: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        conditioning: Literal["concat", "cross_attn"] = "concat",
        cond_channels: int = 1,
        level_depths: list[int] | None = None,
        qk_norm: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        depth_size: int | None = None,
    ):
        super().__init__()

        if level_depths is None:
            level_depths = [2, 4, 6, 4, 2]

        if len(level_depths) % 2 == 0:
            raise ValueError(
                f"level_depths must have odd length (symmetric encoder + bottleneck + decoder), "
                f"got {len(level_depths)}"
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
        self.level_depths = level_depths

        # Parse level structure
        num_levels = len(level_depths)
        mid_idx = num_levels // 2
        self.num_downsample = mid_idx  # number of encoder/decoder levels (excluding bottleneck)
        enc_depths = level_depths[:mid_idx]
        mid_depth = level_depths[mid_idx]
        dec_depths = level_depths[mid_idx + 1:]

        # Validate input dimensions
        if input_size % patch_size != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by patch_size ({patch_size})"
            )
        if spatial_dims == 3 and self.depth_size % patch_size != 0:
            raise ValueError(
                f"depth_size ({self.depth_size}) must be divisible by patch_size ({patch_size})"
            )

        # Validate grid is divisible by 2^num_downsample at each spatial dim
        total_downsample = 2 ** self.num_downsample
        required_divisor = patch_size * total_downsample
        if input_size % required_divisor != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by "
                f"patch_size * 2^num_downsample = {required_divisor}"
            )
        if spatial_dims == 3 and self.depth_size % required_divisor != 0:
            raise ValueError(
                f"depth_size ({self.depth_size}) must be divisible by "
                f"patch_size * 2^num_downsample = {required_divisor}"
            )

        # Calculate grid dimensions at each level
        if spatial_dims == 2:
            base_h = input_size // patch_size
            base_w = input_size // patch_size
            self.grid_dims_per_level = []
            h, w = base_h, base_w
            for i in range(self.num_downsample + 1):  # +1 for bottleneck
                self.grid_dims_per_level.append((h, w))
                if i < self.num_downsample:
                    h, w = h // 2, w // 2
            self.num_patches = base_h * base_w
        else:
            base_d = self.depth_size // patch_size
            base_h = input_size // patch_size
            base_w = input_size // patch_size
            self.grid_dims_per_level = []
            d, h, w = base_d, base_h, base_w
            for i in range(self.num_downsample + 1):
                self.grid_dims_per_level.append((d, h, w))
                if i < self.num_downsample:
                    d, h, w = d // 2, h // 2, w // 2
            self.num_patches = base_d * base_h * base_w

        # Patch embedding
        if spatial_dims == 2:
            self.x_embedder = PatchEmbed2D(patch_size, in_channels, hidden_size)
        else:
            self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)

        # Cross-attention conditioning
        self.use_cross_attn = (conditioning == "cross_attn")
        if self.use_cross_attn:
            if spatial_dims == 2:
                self.cond_embedder = PatchEmbed2D(patch_size, cond_channels, hidden_size)
            else:
                self.cond_embedder = PatchEmbed3D(patch_size, cond_channels, hidden_size)

        # Timestep embedding (adaLN, same as DiT)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Positional embeddings per level (sincos initialized, learnable like DiT)
        self.pos_embeds = nn.ParameterList()
        for _i, grid in enumerate(self.grid_dims_per_level):
            if spatial_dims == 2:
                gh, gw = grid
                num_tokens = gh * gw
                sincos = get_2d_sincos_pos_embed(hidden_size, gh)
            else:
                gd, gh, gw = grid
                num_tokens = gd * gh * gw
                sincos = get_3d_sincos_pos_embed(hidden_size, gd, gh, gw)
            pe = nn.Parameter(torch.from_numpy(sincos).float().unsqueeze(0))
            self.pos_embeds.append(pe)

        # Compute total blocks for stochastic depth scheduling
        total_blocks = sum(level_depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        block_idx = 0

        # Encoder levels
        self.encoder_levels = nn.ModuleList()
        for _level_idx, num_blocks in enumerate(enc_depths):
            level_blocks = nn.ModuleList([
                DiTBlock(
                    hidden_size, num_heads,
                    mlp_ratio=mlp_ratio,
                    use_cross_attn=self.use_cross_attn,
                    qk_norm=qk_norm,
                    drop=drop_rate,
                    drop_path=dpr[block_idx + j],
                )
                for j in range(num_blocks)
            ])
            self.encoder_levels.append(level_blocks)
            block_idx += num_blocks

        # Token merge operators (one fewer than encoder levels)
        self.mergers = nn.ModuleList([
            TokenMerge(hidden_size, spatial_dims)
            for _ in range(self.num_downsample)
        ])

        # Bottleneck blocks
        self.mid_blocks = nn.ModuleList([
            DiTBlock(
                hidden_size, num_heads,
                mlp_ratio=mlp_ratio,
                use_cross_attn=self.use_cross_attn,
                qk_norm=qk_norm,
                drop=drop_rate,
                drop_path=dpr[block_idx + j],
            )
            for j in range(mid_depth)
        ])
        block_idx += mid_depth

        # Token split operators
        self.splitters = nn.ModuleList([
            TokenSplit(hidden_size, spatial_dims)
            for _ in range(self.num_downsample)
        ])

        # Skip projections: cat([x, skip], dim=-1) -> Linear(2D, D)
        self.skip_projs = nn.ModuleList([
            nn.Linear(2 * hidden_size, hidden_size)
            for _ in range(self.num_downsample)
        ])

        # Decoder levels
        self.decoder_levels = nn.ModuleList()
        for _level_idx, num_blocks in enumerate(dec_depths):
            level_blocks = nn.ModuleList([
                DiTBlock(
                    hidden_size, num_heads,
                    mlp_ratio=mlp_ratio,
                    use_cross_attn=self.use_cross_attn,
                    qk_norm=qk_norm,
                    drop=drop_rate,
                    drop_path=dpr[block_idx + j],
                )
                for j in range(num_blocks)
            ])
            self.decoder_levels.append(level_blocks)
            block_idx += num_blocks

        # Final layer (adaLN, same as DiT)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, spatial_dims)

        # Gradient checkpointing flag
        self._use_gradient_checkpointing = False

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights following DiT conventions."""
        # Patch embeddings
        def _init_conv(m):
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.x_embedder.apply(_init_conv)
        if hasattr(self, 'cond_embedder'):
            self.cond_embedder.apply(_init_conv)

        # Timestep embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # DiTBlocks: xavier for projections, zero-init for adaLN
        for blocks in [*self.encoder_levels, self.mid_blocks, *self.decoder_levels]:
            for block in blocks:
                nn.init.xavier_uniform_(block.attn.qkv.weight)
                nn.init.xavier_uniform_(block.attn.proj.weight)
                nn.init.xavier_uniform_(block.mlp.fc1.weight)
                nn.init.xavier_uniform_(block.mlp.fc2.weight)
                nn.init.zeros_(block.adaLN_modulation[-1].weight)
                nn.init.zeros_(block.adaLN_modulation[-1].bias)
                if block.use_cross_attn:
                    nn.init.xavier_uniform_(block.cross_attn.q.weight)
                    nn.init.xavier_uniform_(block.cross_attn.kv.weight)
                    nn.init.xavier_uniform_(block.cross_attn.proj.weight)

        # Token merge/split: xavier uniform
        for merger in self.mergers:
            nn.init.xavier_uniform_(merger.proj.weight)
            nn.init.zeros_(merger.proj.bias)
        for splitter in self.splitters:
            nn.init.xavier_uniform_(splitter.proj.weight)
            nn.init.zeros_(splitter.proj.bias)

        # Skip projections: xavier uniform
        for proj in self.skip_projs:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

        # Final layer: zero-init (same as DiT)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

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
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(x.shape[0], c, h * p, w * p)
        return x

    def unpatchify_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patch tokens back to 3D volume."""
        p = self.patch_size
        d = self.depth_size // p
        h = self.input_size // p
        w = self.input_size // p
        c = self.out_channels
        x = x.reshape(x.shape[0], d, h, w, p, p, p, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
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
            cond: [B, C_cond, ...] conditioning

        Returns:
            [B, C_out, H, W] or [B, C_out, D, H, W] prediction
        """
        # Handle conditioning
        context = None
        if self.conditioning == "concat" and cond is not None:
            x = torch.cat([x, cond], dim=1)
        elif self.conditioning == "cross_attn" and cond is not None:
            context = self.cond_embedder(cond)

        # Patchify + level-0 positional encoding
        x = self.x_embedder(x) + self.pos_embeds[0]  # [B, N, D]

        # Timestep conditioning (adaLN)
        t = self.t_embedder(timesteps)  # [B, D]

        # Current grid dimensions (needed for merge/split)
        grid_dims = self.grid_dims_per_level[0]

        # Encoder: process blocks at each level, save skips
        skips = []
        for level_idx, level_blocks in enumerate(self.encoder_levels):
            for block in level_blocks:
                x = self._checkpoint_forward(block, x, t, context)
            skips.append(x)

            # Merge tokens to next level
            x, grid_dims = self.mergers[level_idx](x, grid_dims)
            # Add positional embedding for the new level
            x = x + self.pos_embeds[level_idx + 1]

        # Bottleneck
        for block in self.mid_blocks:
            x = self._checkpoint_forward(block, x, t, context)

        # Decoder: split tokens and consume skips in reverse
        for level_idx in range(self.num_downsample):
            # Split tokens to finer resolution
            x, grid_dims = self.splitters[level_idx](x, grid_dims)

            # Skip connection
            skip = skips[self.num_downsample - 1 - level_idx]
            x = self.skip_projs[level_idx](torch.cat([x, skip], dim=-1))

            # Decoder blocks
            for block in self.decoder_levels[level_idx]:
                x = self._checkpoint_forward(block, x, t, context)

        # Final projection
        x = self.final_layer(x, t)  # [B, N, p^d * C]

        # Unpatchify
        if self.spatial_dims == 2:
            x = self.unpatchify_2d(x)
        else:
            x = self.unpatchify_3d(x)

        return x


def create_hdit(
    variant: str = 'S',
    spatial_dims: int = 2,
    input_size: int = 32,
    patch_size: int = 4,
    in_channels: int = 4,
    out_channels: int | None = None,
    conditioning: str = "concat",
    cond_channels: int = 1,
    mlp_ratio: float = 4.0,
    level_depths: list[int] | None = None,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    depth_size: int | None = None,
    **kwargs,
) -> HDiT:
    """Create an HDiT model with predefined variant configuration.

    Uses DiT variant sizes for hidden_size/num_heads. Level structure
    is specified via level_depths.

    Args:
        variant: DiT variant for size ('S', 'B', 'L', 'XL')
        spatial_dims: 2 or 3
        input_size: Spatial input size
        patch_size: Patch size (typically 4 for HDiT)
        in_channels: Input channels
        conditioning: "concat" or "cross_attn"
        cond_channels: Conditioning channels
        level_depths: Blocks per level [enc0, enc1, ..., mid, ..., dec1, dec0]
        drop_rate: Dropout rate
        drop_path_rate: Stochastic depth rate
        depth_size: Depth for 3D
        **kwargs: Additional arguments passed to HDiT

    Returns:
        HDiT model
    """
    if variant not in DIT_VARIANTS:
        raise ValueError(
            f"Unknown HDiT variant: {variant}. "
            f"Choose from {list(DIT_VARIANTS.keys())}"
        )

    config = DIT_VARIANTS[variant]

    return HDiT(
        spatial_dims=spatial_dims,
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_size=config['hidden_size'],
        num_heads=config['num_heads'],
        mlp_ratio=mlp_ratio,
        conditioning=conditioning,
        cond_channels=cond_channels,
        level_depths=level_depths,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        depth_size=depth_size,
        **kwargs,
    )
