"""Lossless binary mask encoding/decoding for latent conditioning.

Packs 256x256 binary masks into latent-shaped tensors without information loss.
Supports multiple spatial resolutions matching DC-AE compression levels.

Key insight: 256x256 binary mask = 65,536 bits = 2,048 float32 values = DC-AE latent size

Supported formats:
    - f32:  8x8 spatial,  32 channels, 32x32 blocks (DC-AE f32 compatible)
    - f64:  4x4 spatial, 128 channels, 64x64 blocks (DC-AE f64 compatible)
    - f128: 2x2 spatial, 512 channels, 128x128 blocks (DC-AE f128 compatible)
    - k8x8: 32x32 spatial, 2 channels, 8x8 blocks (spatially coherent)

The k8x8 format is spatially coherent: each latent position (y, x) corresponds
directly to an 8x8 image patch at (y*8, x*8). This preserves local structure
better than the fXX formats which scatter bits across many channels.

Example usage:
    from medgen.data import encode_mask_lossless, decode_mask_lossless

    mask = torch.rand(256, 256) > 0.5
    latent = encode_mask_lossless(mask, 'k8x8')  # [2, 32, 32] - spatially coherent
    reconstructed = decode_mask_lossless(latent, 'k8x8')  # [256, 256]
    assert torch.equal(mask.float(), reconstructed)  # Always True (lossless)
"""

from typing import Literal

import torch
from torch import Tensor

# Format configs: (spatial_size, channels, block_size)
# block_size = 256 / spatial_size (size of image patch per latent position)
# channels = (block_size^2) / 32 (number of float32s needed to store block_size^2 bits)
FORMATS = {
    'f32': (8, 32, 32),     # 8x8 spatial, 32 channels, 32x32 blocks
    'f64': (4, 128, 64),    # 4x4 spatial, 128 channels, 64x64 blocks
    'f128': (2, 512, 128),  # 2x2 spatial, 512 channels, 128x128 blocks
    'k8x8': (32, 2, 8),     # 32x32 spatial, 2 channels, 8x8 blocks (spatially coherent)
}

FormatType = Literal['f32', 'f64', 'f128', 'k8x8']


def encode_mask_lossless(mask: Tensor, format: FormatType = 'f32') -> Tensor:
    """Encode 256x256 binary mask to latent shape losslessly.

    Args:
        mask: Binary mask [256, 256] or [B, 1, 256, 256].
        format: 'f32' (8x8x32), 'f64' (4x4x128), 'f128' (2x2x512).

    Returns:
        Latent tensor [C, S, S] or [B, C, S, S].
    """
    spatial, channels, block_size = FORMATS[format]

    # Handle batch dimension
    if mask.dim() == 4:
        B = mask.shape[0]
        latents = torch.stack([
            encode_mask_lossless(mask[b, 0], format) for b in range(B)
        ])
        return latents  # Already [B, C, S, S] from single-mask returns

    assert mask.shape == (256, 256), f"Expected 256x256, got {mask.shape}"

    # Binarize
    bits = (mask > 0.5).to(torch.int32)

    # Reshape to spatial grid of blocks
    # [256, 256] -> [S, block, S, block] -> [S, S, block, block] -> [S, S, block^2]
    blocks = (bits
        .view(spatial, block_size, spatial, block_size)
        .permute(0, 2, 1, 3)
        .reshape(spatial, spatial, -1))

    # Group into 32-bit chunks: [S, S, block^2] -> [S, S, C, 32]
    blocks = blocks.view(spatial, spatial, channels, 32)

    # Pack 32 bits into each int32 using bitwise operations (avoids overflow)
    packed = torch.zeros(spatial, spatial, channels, dtype=torch.int32, device=mask.device)
    for bit in range(32):
        packed = packed | (blocks[..., bit] << bit)

    # Reinterpret as float32 (preserves bit pattern exactly)
    latent = packed.view(torch.float32)

    # Channel-first: [S, S, C] -> [C, S, S]
    return latent.permute(2, 0, 1).contiguous()


def decode_mask_lossless(latent: Tensor, format: FormatType = 'f32') -> Tensor:
    """Decode latent back to 256x256 binary mask.

    Args:
        latent: Packed latent [C, S, S] or [B, C, S, S].
        format: 'f32', 'f64', or 'f128'.

    Returns:
        Binary mask [256, 256] or [B, 1, 256, 256].
    """
    spatial, channels, block_size = FORMATS[format]

    # Handle batch dimension
    if latent.dim() == 4:
        B = latent.shape[0]
        masks = torch.stack([
            decode_mask_lossless(latent[b], format) for b in range(B)
        ])
        return masks.unsqueeze(1)  # [B, 1, H, W]

    expected_shape = (channels, spatial, spatial)
    assert latent.shape == expected_shape, f"Expected {expected_shape}, got {latent.shape}"

    # Channel-last: [C, S, S] -> [S, S, C]
    latent = latent.permute(1, 2, 0)

    # Reinterpret as int32
    packed = latent.contiguous().view(torch.int32)  # [S, S, C]

    # Unpack bits: [S, S, C] -> [S, S, C, 32]
    bit_indices = torch.arange(32, device=latent.device, dtype=torch.int32)
    bits = (packed.unsqueeze(-1) >> bit_indices) & 1

    # Reshape: [S, S, C, 32] -> [S, S, block^2] -> [S, S, block, block]
    blocks = bits.view(spatial, spatial, -1).view(spatial, spatial, block_size, block_size)

    # Reconstruct: [S, S, block, block] -> [S, block, S, block] -> [256, 256]
    mask = blocks.permute(0, 2, 1, 3).reshape(256, 256)

    return mask.float()


def get_latent_shape(format: FormatType) -> tuple[int, int, int]:
    """Get latent shape (C, H, W) for a format."""
    spatial, channels, _ = FORMATS[format]
    return (channels, spatial, spatial)


# Convenience aliases
def encode_f32(mask: Tensor) -> Tensor:
    """Encode mask to f32 format (32x compression, 8x8x32 latent)."""
    return encode_mask_lossless(mask, 'f32')


def decode_f32(latent: Tensor) -> Tensor:
    """Decode f32 format latent to mask."""
    return decode_mask_lossless(latent, 'f32')


def encode_f64(mask: Tensor) -> Tensor:
    """Encode mask to f64 format (64x compression, 4x4x128 latent)."""
    return encode_mask_lossless(mask, 'f64')


def decode_f64(latent: Tensor) -> Tensor:
    """Decode f64 format latent to mask."""
    return decode_mask_lossless(latent, 'f64')


def encode_f128(mask: Tensor) -> Tensor:
    """Encode mask to f128 format (128x compression, 2x2x512 latent)."""
    return encode_mask_lossless(mask, 'f128')


def decode_f128(latent: Tensor) -> Tensor:
    """Decode f128 format latent to mask."""
    return decode_mask_lossless(latent, 'f128')


def encode_k8x8(mask: Tensor) -> Tensor:
    """Encode mask to k8x8 format (spatially coherent, 32x32x2 latent).

    Each 8x8 image patch maps to 2 float32 values at the corresponding
    latent position. Preserves spatial locality better than fXX formats.
    """
    return encode_mask_lossless(mask, 'k8x8')


def decode_k8x8(latent: Tensor) -> Tensor:
    """Decode k8x8 format latent to mask."""
    return decode_mask_lossless(latent, 'k8x8')
