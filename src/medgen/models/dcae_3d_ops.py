"""3D operations for Deep Compression Autoencoder (DC-AE 3D).

This module provides the core building blocks for the 3D DC-AE architecture:
- PixelUnshuffle3D/PixelShuffle3D: Asymmetric space-to-channel operations
- RMSNorm3D: Channel-wise normalization for 3D tensors
- ResBlock3D: 3D residual block
- DCDownBlock3D/DCUpBlock3D: Down/upsampling blocks with shortcuts

The key innovation is asymmetric compression where spatial and depth factors
can differ (e.g., 2x2 spatial with 1x depth for spatial-only compression).
"""

import torch
import torch.nn as nn


class PixelUnshuffle3D(nn.Module):
    """Asymmetric 3D pixel unshuffle (space-to-channel).

    Rearranges elements from spatial dimensions to channel dimension.
    Supports different factors for spatial (H, W) and depth (D) dimensions.

    Args:
        spatial_factor: Downsampling factor for H and W dimensions (default: 2)
        depth_factor: Downsampling factor for D dimension (default: 2, use 1 for spatial-only)

    Example:
        >>> unshuffle = PixelUnshuffle3D(spatial_factor=2, depth_factor=2)
        >>> x = torch.randn(2, 4, 8, 16, 16)
        >>> y = unshuffle(x)  # [2, 32, 4, 8, 8]

        >>> unshuffle_spatial = PixelUnshuffle3D(spatial_factor=2, depth_factor=1)
        >>> y = unshuffle_spatial(x)  # [2, 16, 8, 8, 8] - depth unchanged
    """

    def __init__(self, spatial_factor: int = 2, depth_factor: int = 2) -> None:
        super().__init__()
        self.spatial_factor = spatial_factor
        self.depth_factor = depth_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply asymmetric 3D pixel unshuffle.

        Args:
            x: Input tensor [B, C, D, H, W]

        Returns:
            Output tensor with reduced spatial dims and increased channels
        """
        b, c, d, h, w = x.shape
        sf = self.spatial_factor
        df = self.depth_factor

        if df == 1:
            # Spatial-only compression (depth unchanged)
            # [B, C, D, H, W] -> [B, C*sf*sf, D, H/sf, W/sf]
            x = x.view(b, c, d, h // sf, sf, w // sf, sf)
            # Permute: [B, C, sf, sf, D, H/sf, W/sf]
            x = x.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
            return x.view(b, c * sf * sf, d, h // sf, w // sf)
        else:
            # Full 3D compression
            # [B, C, D, H, W] -> [B, C*df*sf*sf, D/df, H/sf, W/sf]
            x = x.view(b, c, d // df, df, h // sf, sf, w // sf, sf)
            # Permute: [B, C, df, sf, sf, D/df, H/sf, W/sf]
            x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
            return x.view(b, c * df * sf * sf, d // df, h // sf, w // sf)

    def extra_repr(self) -> str:
        return f'spatial_factor={self.spatial_factor}, depth_factor={self.depth_factor}'


class PixelShuffle3D(nn.Module):
    """Asymmetric 3D pixel shuffle (channel-to-space).

    Rearranges elements from channel dimension to spatial dimensions.
    Inverse of PixelUnshuffle3D.

    Args:
        spatial_factor: Upsampling factor for H and W dimensions (default: 2)
        depth_factor: Upsampling factor for D dimension (default: 2, use 1 for spatial-only)

    Example:
        >>> shuffle = PixelShuffle3D(spatial_factor=2, depth_factor=2)
        >>> x = torch.randn(2, 32, 4, 8, 8)
        >>> y = shuffle(x)  # [2, 4, 8, 16, 16]
    """

    def __init__(self, spatial_factor: int = 2, depth_factor: int = 2) -> None:
        super().__init__()
        self.spatial_factor = spatial_factor
        self.depth_factor = depth_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply asymmetric 3D pixel shuffle.

        Args:
            x: Input tensor [B, C*factor, D, H, W]

        Returns:
            Output tensor with increased spatial dims and reduced channels
        """
        b, c, d, h, w = x.shape
        sf = self.spatial_factor
        df = self.depth_factor

        if df == 1:
            # Spatial-only expansion (depth unchanged)
            # [B, C*sf*sf, D, H, W] -> [B, C, D, H*sf, W*sf]
            c_out = c // (sf * sf)
            # Reshape: [B, C_out, sf, sf, D, H, W]
            x = x.view(b, c_out, sf, sf, d, h, w)
            # Permute: [B, C_out, D, H, sf, W, sf]
            x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
            return x.view(b, c_out, d, h * sf, w * sf)
        else:
            # Full 3D expansion
            # [B, C*df*sf*sf, D, H, W] -> [B, C, D*df, H*sf, W*sf]
            c_out = c // (df * sf * sf)
            # Reshape: [B, C_out, df, sf, sf, D, H, W]
            x = x.view(b, c_out, df, sf, sf, d, h, w)
            # Permute: [B, C_out, D, df, H, sf, W, sf]
            x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
            return x.view(b, c_out, d * df, h * sf, w * sf)

    def extra_repr(self) -> str:
        return f'spatial_factor={self.spatial_factor}, depth_factor={self.depth_factor}'


class RMSNorm3D(nn.Module):
    """RMS Normalization for 3D tensors (channel-wise).

    Normalizes across the channel dimension using root mean square.
    More efficient than LayerNorm for certain architectures.

    Args:
        dim: Number of channels to normalize
        eps: Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor [B, C, D, H, W]

        Returns:
            Normalized tensor with same shape
        """
        # Move channel dim to last for normalization
        x = x.movedim(1, -1)  # [B, D, H, W, C]

        # Compute RMS and normalize
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x / rms * self.weight + self.bias

        # Move channel dim back
        return x.movedim(-1, 1)  # [B, C, D, H, W]


class ResBlock3D(nn.Module):
    """3D Residual block with RMSNorm.

    Architecture:
        x -> Conv3d -> SiLU -> Conv3d -> RMSNorm -> + x -> out

    Uses 3x3x3 convolutions with padding to preserve spatial dimensions.

    Args:
        channels: Number of input/output channels (must be equal for residual)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.nonlinearity = nn.SiLU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm = RMSNorm3D(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block.

        Args:
            x: Input tensor [B, C, D, H, W]

        Returns:
            Output tensor with same shape
        """
        residual = x
        x = self.conv1(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        x = self.norm(x)
        return x + residual


class DCDownBlock3D(nn.Module):
    """DC-AE 3D downsampling block with residual shortcut.

    Downsamples spatial dimensions using pixel unshuffle with optional
    residual shortcut for stable high-compression training.

    Main path: Conv3d -> PixelUnshuffle3D
    Shortcut:  PixelUnshuffle3D -> channel averaging

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        spatial_factor: Spatial downsampling factor (default: 2)
        depth_factor: Depth downsampling factor (default: 2, use 1 for spatial-only)
        shortcut: Whether to use residual shortcut (default: True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_factor: int = 2,
        depth_factor: int = 2,
        shortcut: bool = True,
    ) -> None:
        super().__init__()

        self.spatial_factor = spatial_factor
        self.depth_factor = depth_factor
        self.shortcut = shortcut

        # Total channel multiplication factor from pixel unshuffle
        total_factor = spatial_factor * spatial_factor * depth_factor

        # Conv outputs channels that will be multiplied by unshuffle
        conv_out_channels = out_channels // total_factor
        assert out_channels % total_factor == 0, \
            f"out_channels ({out_channels}) must be divisible by factor ({total_factor})"

        self.conv = nn.Conv3d(in_channels, conv_out_channels, kernel_size=3, padding=1)
        self.unshuffle = PixelUnshuffle3D(spatial_factor, depth_factor)

        if shortcut:
            # Shortcut uses same unshuffle, then averages channel groups
            self.shortcut_unshuffle = PixelUnshuffle3D(spatial_factor, depth_factor)
            # After unshuffle: in_channels * total_factor channels
            # Need to reduce to: out_channels
            shortcut_channels = in_channels * total_factor
            assert shortcut_channels % out_channels == 0, \
                f"Shortcut channels ({shortcut_channels}) must be divisible by out_channels ({out_channels})"
            self.group_size = shortcut_channels // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply downsampling block.

        Args:
            x: Input tensor [B, C_in, D, H, W]

        Returns:
            Output tensor [B, C_out, D/df, H/sf, W/sf]
        """
        # Main path
        y = self.conv(x)
        y = self.unshuffle(y)

        if self.shortcut:
            # Shortcut path: unshuffle then average groups
            s = self.shortcut_unshuffle(x)
            # Average channel groups: [B, C*factor, D', H', W'] -> [B, out_channels, D', H', W']
            b, c, d, h, w = s.shape
            s = s.view(b, -1, self.group_size, d, h, w)
            s = s.mean(dim=2)
            return y + s

        return y


class DCUpBlock3D(nn.Module):
    """DC-AE 3D upsampling block with residual shortcut.

    Upsamples spatial dimensions using pixel shuffle with optional
    residual shortcut for stable high-compression training.

    Main path: Conv3d -> PixelShuffle3D
    Shortcut:  channel repeat -> PixelShuffle3D

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        spatial_factor: Spatial upsampling factor (default: 2)
        depth_factor: Depth upsampling factor (default: 2, use 1 for spatial-only)
        shortcut: Whether to use residual shortcut (default: True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_factor: int = 2,
        depth_factor: int = 2,
        shortcut: bool = True,
    ) -> None:
        super().__init__()

        self.spatial_factor = spatial_factor
        self.depth_factor = depth_factor
        self.shortcut = shortcut

        # Total channel division factor from pixel shuffle
        total_factor = spatial_factor * spatial_factor * depth_factor

        # Conv outputs extra channels that will be divided by shuffle
        conv_out_channels = out_channels * total_factor

        self.conv = nn.Conv3d(in_channels, conv_out_channels, kernel_size=3, padding=1)
        self.shuffle = PixelShuffle3D(spatial_factor, depth_factor)

        if shortcut:
            # Shortcut repeats channels then shuffles
            self.shortcut_shuffle = PixelShuffle3D(spatial_factor, depth_factor)
            # Need to repeat in_channels to: out_channels * total_factor
            target_channels = out_channels * total_factor
            assert target_channels % in_channels == 0, \
                f"Target channels ({target_channels}) must be divisible by in_channels ({in_channels})"
            self.repeats = target_channels // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply upsampling block.

        Args:
            x: Input tensor [B, C_in, D, H, W]

        Returns:
            Output tensor [B, C_out, D*df, H*sf, W*sf]
        """
        # Main path
        y = self.conv(x)
        y = self.shuffle(y)

        if self.shortcut:
            # Shortcut path: repeat channels then shuffle
            s = x.repeat_interleave(self.repeats, dim=1)
            s = self.shortcut_shuffle(s)
            return y + s

        return y
