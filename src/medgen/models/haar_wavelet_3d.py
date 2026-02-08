"""3D Haar wavelet decomposition (forward and inverse).

Lossless, orthogonal transform that decomposes a 3D volume into 8 frequency
subbands (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH) by applying the Haar
wavelet along each spatial axis sequentially.

Shape: [B, C, D, H, W] <-> [B, C*8, D/2, H/2, W/2]

Implementation uses reshape + arithmetic only (no convolutions), making it
fully differentiable and parameter-free.

The 1/sqrt(2) normalization factor ensures the transform is orthogonal,
meaning ||forward(x)||^2 == ||x||^2 (energy preservation).
"""

import torch
import torch.nn as nn
from torch import Tensor

_INV_SQRT2 = 1.0 / (2.0 ** 0.5)


def _haar_split(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    """Apply 1D Haar split along a given dimension.

    Splits even/odd indices and computes:
        low  = (even + odd) / sqrt(2)   (approximation / average)
        high = (even - odd) / sqrt(2)   (detail / difference)

    Args:
        x: Input tensor.
        dim: Dimension to split along (must have even size).

    Returns:
        Tuple of (low, high) tensors, each half the size along `dim`.

    Raises:
        ValueError: If the dimension size is not even.
    """
    size = x.shape[dim]
    if size % 2 != 0:
        raise ValueError(
            f"Haar wavelet requires even dimension size, "
            f"got {size} along dim {dim} (shape {x.shape})"
        )

    # Split into even and odd indices along dim
    even = x.narrow(dim, 0, size // 2 * 2).unfold(dim, 2, 2)
    # even has shape [..., size//2, 2] with the pair dimension at the end
    # Select even (index 0) and odd (index 1) from the last dim
    e = even.select(-1, 0)
    o = even.select(-1, 1)

    low = (e + o) * _INV_SQRT2
    high = (e - o) * _INV_SQRT2
    return low, high


def _haar_merge(low: Tensor, high: Tensor, dim: int) -> Tensor:
    """Inverse 1D Haar merge along a given dimension.

    Reconstructs even/odd from low/high:
        even = (low + high) / sqrt(2)
        odd  = (low - high) / sqrt(2)

    Then interleaves them back along `dim`.

    Args:
        low: Low-frequency (approximation) tensor.
        high: High-frequency (detail) tensor.
        dim: Dimension to merge along.

    Returns:
        Reconstructed tensor with twice the size along `dim`.
    """
    e = (low + high) * _INV_SQRT2
    o = (low - high) * _INV_SQRT2

    # Interleave even and odd along dim:
    # Stack along dim+1 so the pair dimension is adjacent to `dim`,
    # then reshape to merge the pair into `dim`.
    stacked = torch.stack([e, o], dim=dim + 1)
    # stacked shape: [..., half_size, 2, ...] with 2 at position dim+1
    target_shape = list(low.shape)
    target_shape[dim] = low.shape[dim] * 2
    return stacked.reshape(target_shape)


def haar_forward_3d(x: Tensor) -> Tensor:
    """Apply 3D Haar wavelet forward transform.

    Applies 1D Haar along W, then H, then D, producing 8 subbands
    packed into the channel dimension.

    Args:
        x: Input tensor [B, C, D, H, W]. All spatial dims must be even.

    Returns:
        Wavelet coefficients [B, C*8, D/2, H/2, W/2].
        Channel ordering: LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH
        where L/H refer to D, H, W axes respectively.
    """
    if x.dim() != 5:
        raise ValueError(
            f"haar_forward_3d requires 5D input [B, C, D, H, W], "
            f"got {x.dim()}D tensor with shape {x.shape}"
        )

    B, C, D, H, W = x.shape

    # Step 1: Split along W (dim=4) -> 2 subbands
    w_low, w_high = _haar_split(x, dim=4)
    # Each: [B, C, D, H, W/2]

    # Step 2: Split each along H (dim=3) -> 4 subbands
    hl_low, hl_high = _haar_split(w_low, dim=3)   # LL, LH (w_low split by H)
    hh_low, hh_high = _haar_split(w_high, dim=3)  # HL, HH (w_high split by H)
    # Each: [B, C, D, H/2, W/2]

    # Step 3: Split each along D (dim=2) -> 8 subbands
    lll, hll = _haar_split(hl_low, dim=2)   # LL split by D -> LLL, HLL
    llh, hlh = _haar_split(hl_high, dim=2)  # LH split by D -> LLH, HLH
    lhl, hhl = _haar_split(hh_low, dim=2)   # HL split by D -> LHL, HHL
    lhh, hhh = _haar_split(hh_high, dim=2)  # HH split by D -> LHH, HHH
    # Each: [B, C, D/2, H/2, W/2]

    # Pack all 8 subbands into channel dimension
    # Order: LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH
    return torch.cat([lll, llh, lhl, lhh, hll, hlh, hhl, hhh], dim=1)


def haar_inverse_3d(z: Tensor, in_channels: int = 1) -> Tensor:
    """Apply 3D Haar wavelet inverse transform.

    Splits channel dimension into 8 subbands and reconstructs along D, H, W.

    Args:
        z: Wavelet coefficients [B, C*8, D/2, H/2, W/2].
        in_channels: Original number of channels (default 1).
            Used to split the 8*C channels back into 8 groups.

    Returns:
        Reconstructed tensor [B, C, D, H, W].
    """
    if z.dim() != 5:
        raise ValueError(
            f"haar_inverse_3d requires 5D input [B, C*8, D/2, H/2, W/2], "
            f"got {z.dim()}D tensor with shape {z.shape}"
        )

    total_channels = z.shape[1]
    if total_channels % 8 != 0:
        raise ValueError(
            f"Channel count must be divisible by 8, got {total_channels}"
        )

    C = total_channels // 8

    # Unpack 8 subbands from channel dimension
    lll, llh, lhl, lhh, hll, hlh, hhl, hhh = z.chunk(8, dim=1)
    # Each: [B, C, D/2, H/2, W/2]

    # Step 1: Merge along D (dim=2) -> 4 subbands
    hl_low = _haar_merge(lll, hll, dim=2)   # LLL + HLL -> LL
    hl_high = _haar_merge(llh, hlh, dim=2)  # LLH + HLH -> LH
    hh_low = _haar_merge(lhl, hhl, dim=2)   # LHL + HHL -> HL
    hh_high = _haar_merge(lhh, hhh, dim=2)  # LHH + HHH -> HH
    # Each: [B, C, D, H/2, W/2]

    # Step 2: Merge along H (dim=3) -> 2 subbands
    w_low = _haar_merge(hl_low, hl_high, dim=3)   # LL + LH -> L
    w_high = _haar_merge(hh_low, hh_high, dim=3)  # HL + HH -> H
    # Each: [B, C, D, H, W/2]

    # Step 3: Merge along W (dim=4) -> original
    return _haar_merge(w_low, w_high, dim=4)
    # [B, C, D, H, W]


class HaarForward3D(nn.Module):
    """3D Haar wavelet forward transform (nn.Module wrapper).

    [B, C, D, H, W] -> [B, C*8, D/2, H/2, W/2]

    Parameter-free, fully differentiable, lossless.
    """

    def forward(self, x: Tensor) -> Tensor:
        return haar_forward_3d(x)


class HaarInverse3D(nn.Module):
    """3D Haar wavelet inverse transform (nn.Module wrapper).

    [B, C*8, D/2, H/2, W/2] -> [B, C, D, H, W]

    Parameter-free, fully differentiable, lossless.
    """

    def forward(self, z: Tensor) -> Tensor:
        return haar_inverse_3d(z)
