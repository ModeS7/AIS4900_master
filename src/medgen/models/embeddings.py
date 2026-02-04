"""
Embedding layers for SiT (Scalable Interpolant Transformers).

Provides patch embeddings (2D/3D) and timestep embeddings for diffusion transformers.
"""

import math

import numpy as np
import torch
import torch.nn as nn


class PatchEmbed2D(nn.Module):
    """2D image to patch embedding.

    Converts images [B, C, H, W] to patch tokens [B, N, D] where N = (H/p) * (W/p).

    Args:
        patch_size: Size of each patch (assumes square patches).
        in_channels: Number of input channels.
        embed_dim: Embedding dimension.
        bias: Whether to use bias in convolution.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        embed_dim: int = 768,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input image

        Returns:
            [B, N, D] patch tokens where N = (H/p) * (W/p)
        """
        # [B, C, H, W] -> [B, D, H/p, W/p]
        x = self.proj(x)
        # [B, D, H/p, W/p] -> [B, D, N] -> [B, N, D]
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchEmbed3D(nn.Module):
    """3D volume to patch embedding.

    Converts volumes [B, C, D, H, W] to patch tokens [B, N, E] where N = (D/p) * (H/p) * (W/p).

    Args:
        patch_size: Size of each patch (assumes cubic patches).
        in_channels: Number of input channels.
        embed_dim: Embedding dimension.
        bias: Whether to use bias in convolution.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        embed_dim: int = 768,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W] input volume

        Returns:
            [B, N, E] patch tokens where N = (D/p) * (H/p) * (W/p)
        """
        # [B, C, D, H, W] -> [B, E, D/p, H/p, W/p]
        x = self.proj(x)
        # [B, E, D/p, H/p, W/p] -> [B, E, N] -> [B, N, E]
        x = x.flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embeddings with MLP projection.

    Uses sinusoidal positional encoding for timesteps, then projects through MLP.

    Args:
        hidden_size: Output embedding dimension.
        frequency_embedding_size: Dimension of sinusoidal embedding (before MLP).
    """

    def __init__(
        self,
        hidden_size: int = 768,
        frequency_embedding_size: int = 256,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.

        Args:
            t: [B] tensor of timesteps (can be float for flow matching)
            dim: Embedding dimension
            max_period: Maximum period for sinusoidal encoding

        Returns:
            [B, dim] sinusoidal embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] timesteps (0 to 1 for flow matching, or 0 to T for DDPM)

        Returns:
            [B, hidden_size] timestep embeddings
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
) -> np.ndarray:
    """Generate 2D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension (must be divisible by 2)
        grid_size: Height/width of the grid
        cls_token: If True, prepend a position for [CLS] token

    Returns:
        [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # [2, H, W]
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Generate 2D sinusoidal embeddings from grid coordinates.

    Args:
        embed_dim: Embedding dimension
        grid: [2, 1, H, W] grid coordinates

    Returns:
        [H*W, embed_dim] positional embeddings
    """
    assert embed_dim % 2 == 0

    # Use half of dimensions to encode grid_h, half for grid_w
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # [H*W, D/2]
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # [H*W, D/2]

    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # [H*W, D]
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D sinusoidal embeddings from position values.

    Args:
        embed_dim: Output dimension
        pos: Position values of any shape

    Returns:
        [len(pos.flatten()), embed_dim] positional embeddings
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)  # [D/2]

    pos = pos.flatten()  # [N]
    out = np.outer(pos, omega)  # [N, D/2]

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # [N, D]
    return emb


def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size_d: int,
    grid_size_h: int,
    grid_size_w: int,
) -> np.ndarray:
    """Generate 3D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension
        grid_size_d: Depth of the grid
        grid_size_h: Height of the grid
        grid_size_w: Width of the grid

    Returns:
        [D*H*W, embed_dim] positional embeddings
    """
    # Split dimensions into thirds, ensuring each is EVEN (required by sincos encoding)
    # Round down to nearest even number using & ~1
    dim_d = (embed_dim // 3) & ~1  # e.g., 341 -> 340
    dim_h = (embed_dim // 3) & ~1
    dim_w = embed_dim - dim_d - dim_h  # Remainder goes to W (will be even if embed_dim is even)

    # Create coordinate grids
    grid_d = np.arange(grid_size_d, dtype=np.float32)
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)

    # Meshgrid: [D, H, W]
    gd, gh, gw = np.meshgrid(grid_d, grid_h, grid_w, indexing='ij')

    # Generate embeddings for each dimension
    emb_d = get_1d_sincos_pos_embed_from_grid(dim_d, gd)  # [D*H*W, dim_d]
    emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, gh)  # [D*H*W, dim_h]
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, gw)  # [D*H*W, dim_w]

    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)  # [D*H*W, embed_dim]
    return pos_embed
