"""Adaptive convolution layers for DC-AE 1.5 structured latent space.

These layers enable dynamic channel slicing at the encoder output and decoder input,
implementing the official DC-Gen approach where gradients only flow through active channels.

Official DC-Gen implementation:
- AdaptiveOutputConvLayer: Slices weight[:out_channels] to only compute first N channels
- AdaptiveInputConvLayer: Slices weight[:, :in_channels] to handle variable input

Our implementation wraps diffusers AutoencoderDC with equivalent functionality.
"""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveOutputConv2d(nn.Module):
    """Adaptive encoder output convolution that supports dynamic channel slicing.

    This layer computes only the first N output channels by slicing the convolution
    weights, ensuring gradients only flow through active channels. This creates
    a structured hierarchy where early channels become more important.

    Unlike post-hoc masking (which zeros out channels after computation),
    weight slicing means:
    - Only first N channels are computed
    - No wasted computation on unused channels
    - Gradients only affect the active weight subset

    Args:
        in_channels: Number of input channels
        out_channels: Maximum number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding

    Example:
        >>> layer = AdaptiveOutputConv2d(512, 128)
        >>> x = torch.randn(2, 512, 8, 8)
        >>> y_full = layer(x, out_channels=128)  # [2, 128, 8, 8]
        >>> y_half = layer(x, out_channels=64)   # [2, 64, 8, 8]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Normalize kernel_size, stride, padding to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Store the underlying conv for weight/bias storage
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        out_channels: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass with optional channel slicing.

        Args:
            x: Input tensor [B, in_channels, H, W]
            out_channels: Number of output channels to compute. If None, uses all.

        Returns:
            Output tensor [B, out_channels, H', W']
        """
        if out_channels is None:
            out_channels = self.out_channels

        if out_channels > self.out_channels:
            raise ValueError(
                f"Requested out_channels ({out_channels}) exceeds "
                f"max out_channels ({self.out_channels})"
            )

        # Slice weights to only compute first N channels
        # Use contiguous() to ensure proper tensor (not just a view) before dtype conversion
        weight = self.conv.weight[:out_channels].contiguous()  # [out_channels, in_channels, kH, kW]
        bias = self.conv.bias[:out_channels].contiguous() if self.conv.bias is not None else None

        # F.conv2d requires all tensors to have matching dtypes
        # Cast weight and bias to input dtype for mixed precision compatibility
        # Use .to(device, dtype) to ensure both match the input tensor
        weight = weight.to(device=x.device, dtype=x.dtype)
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)

        return F.conv2d(x, weight, bias, self.stride, self.padding)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"
        )


class AdaptiveInputConv2d(nn.Module):
    """Adaptive decoder input convolution that handles variable input channels.

    This layer reads only the first N input channels by slicing the convolution
    weights. It automatically detects the input channel count and adjusts accordingly.

    This is the decoder counterpart to AdaptiveOutputConv2d - when the encoder
    outputs fewer channels, the decoder seamlessly handles the reduced input.

    Args:
        in_channels: Maximum number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding

    Example:
        >>> layer = AdaptiveInputConv2d(128, 512)
        >>> z_full = torch.randn(2, 128, 8, 8)
        >>> z_half = torch.randn(2, 64, 8, 8)
        >>> y_full = layer(z_full)  # Uses all 128 input channels
        >>> y_half = layer(z_half)  # Uses only 64 input channels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Normalize kernel_size, stride, padding to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Store the underlying conv for weight/bias storage
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic input channel detection.

        Args:
            x: Input tensor [B, C, H, W] where C <= in_channels

        Returns:
            Output tensor [B, out_channels, H', W']
        """
        actual_channels = x.shape[1]

        if actual_channels > self.in_channels:
            raise ValueError(
                f"Input channels ({actual_channels}) exceeds "
                f"max in_channels ({self.in_channels})"
            )

        # Slice weights to match actual input channels
        # Use contiguous() to ensure proper tensor (not just a view) before dtype conversion
        weight = self.conv.weight[:, :actual_channels].contiguous()  # [out_channels, actual_channels, kH, kW]
        bias = self.conv.bias  # Bias doesn't depend on input channels

        # F.conv2d requires all tensors to have matching dtypes
        # Cast weight and bias to input dtype for mixed precision compatibility
        # Use .to(device, dtype) to ensure both match the input tensor
        weight = weight.to(device=x.device, dtype=x.dtype)
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)

        return F.conv2d(x, weight, bias, self.stride, self.padding)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"
        )


def copy_conv_weights(
    source_conv: nn.Conv2d,
    target_adaptive: Union[AdaptiveOutputConv2d, AdaptiveInputConv2d],
) -> None:
    """Copy weights from a standard Conv2d to an adaptive conv layer.

    Args:
        source_conv: Source nn.Conv2d layer
        target_adaptive: Target adaptive conv layer
    """
    with torch.no_grad():
        target_adaptive.conv.weight.copy_(source_conv.weight)
        if source_conv.bias is not None and target_adaptive.conv.bias is not None:
            target_adaptive.conv.bias.copy_(source_conv.bias)
