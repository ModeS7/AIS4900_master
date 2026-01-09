"""3D Deep Compression Autoencoder (DC-AE 3D).

Custom implementation for volumetric medical image compression.
Adapts the 2D DC-AE architecture to 3D with asymmetric compression support.

Key features:
- 32× spatial compression (256×256 → 8×8)
- 4× depth compression (160 → 40)
- Residual autoencoding with encoder/decoder shortcuts
- No regularization (deterministic, unlike VAE)
- Gradient checkpointing support for memory efficiency

Based on:
- DC-AE: https://arxiv.org/abs/2410.10733
- DC-VideoGen: https://arxiv.org/abs/2509.25182
"""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .dcae_3d_ops import (
    DCDownBlock3D,
    DCUpBlock3D,
    PixelUnshuffle3D,
    PixelShuffle3D,
    ResBlock3D,
    RMSNorm3D,
)
from ..pipeline.checkpointing import BaseCheckpointedModel


class Encoder3D(nn.Module):
    """3D DC-AE Encoder.

    Compresses input volume through a series of downsampling stages.
    Each stage: [ResBlocks] -> DownBlock (except last stage)

    Args:
        in_channels: Input volume channels (e.g., 4 for multi-modality)
        latent_channels: Output latent channels (e.g., 32)
        block_out_channels: Channels at each stage (e.g., [64, 128, 256, 256, 512, 512])
        layers_per_block: ResBlocks per stage (e.g., [2, 2, 2, 2, 2, 2])
        depth_factors: Depth factor for each down block (e.g., [2, 2, 1, 1, 1])
        out_shortcut: Add shortcut at output conv
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        block_out_channels: Tuple[int, ...] = (64, 128, 256, 256, 512, 512),
        layers_per_block: Tuple[int, ...] = (2, 2, 2, 2, 2, 2),
        depth_factors: Tuple[int, ...] = (2, 2, 1, 1, 1),
        out_shortcut: bool = True,
    ) -> None:
        super().__init__()

        num_stages = len(block_out_channels)
        num_down_blocks = num_stages - 1

        assert len(layers_per_block) == num_stages, \
            f"layers_per_block ({len(layers_per_block)}) must match num_stages ({num_stages})"
        assert len(depth_factors) == num_down_blocks, \
            f"depth_factors ({len(depth_factors)}) must match num_down_blocks ({num_down_blocks})"

        self.out_shortcut = out_shortcut
        self.num_stages = num_stages

        # Initial conv
        self.conv_in = nn.Conv3d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # Build stages
        self.stages = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        for i in range(num_stages):
            # ResBlocks at this resolution
            stage_blocks = []
            channels = block_out_channels[i]
            for _ in range(layers_per_block[i]):
                stage_blocks.append(ResBlock3D(channels))
            self.stages.append(nn.Sequential(*stage_blocks))

            # Downsample (except last stage)
            if i < num_down_blocks:
                next_channels = block_out_channels[i + 1]
                df = depth_factors[i]
                self.down_blocks.append(
                    DCDownBlock3D(
                        in_channels=channels,
                        out_channels=next_channels,
                        spatial_factor=2,
                        depth_factor=df,
                        shortcut=True,
                    )
                )

        # Output conv with optional shortcut
        self.conv_out = nn.Conv3d(block_out_channels[-1], latent_channels, kernel_size=3, padding=1)

        if out_shortcut:
            # Compute group size for channel averaging in shortcut
            assert block_out_channels[-1] % latent_channels == 0, \
                f"Last stage channels ({block_out_channels[-1]}) must be divisible by latent_channels ({latent_channels})"
            self.out_shortcut_group_size = block_out_channels[-1] // latent_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input volume to latent space.

        Args:
            x: Input tensor [B, C_in, D, H, W]

        Returns:
            Latent tensor [B, latent_channels, D', H', W']
        """
        x = self.conv_in(x)

        for i in range(self.num_stages):
            x = self.stages[i](x)
            if i < len(self.down_blocks):
                x = self.down_blocks[i](x)

        if self.out_shortcut:
            # Average groups of channels for shortcut
            b, c, d, h, w = x.shape
            shortcut = x.view(b, -1, self.out_shortcut_group_size, d, h, w)
            shortcut = shortcut.mean(dim=2)
            x = self.conv_out(x) + shortcut
        else:
            x = self.conv_out(x)

        return x


class Decoder3D(nn.Module):
    """3D DC-AE Decoder.

    Expands latent through a series of upsampling stages.
    Each stage: UpBlock -> [ResBlocks] (except first stage)

    Architecture mirrors Encoder3D with reversed operations.

    Args:
        in_channels: Output volume channels (e.g., 4 for multi-modality)
        latent_channels: Input latent channels (e.g., 32)
        block_out_channels: Channels at each stage (same as encoder, e.g., [64, 128, 256, 256, 512, 512])
        layers_per_block: ResBlocks per stage (e.g., [2, 2, 2, 2, 2, 2])
        depth_factors: Depth factor for each up block (e.g., [2, 2, 1, 1, 1])
        in_shortcut: Add shortcut at input conv
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        block_out_channels: Tuple[int, ...] = (64, 128, 256, 256, 512, 512),
        layers_per_block: Tuple[int, ...] = (2, 2, 2, 2, 2, 2),
        depth_factors: Tuple[int, ...] = (2, 2, 1, 1, 1),
        in_shortcut: bool = True,
    ) -> None:
        super().__init__()

        num_stages = len(block_out_channels)
        num_up_blocks = num_stages - 1

        assert len(layers_per_block) == num_stages, \
            f"layers_per_block ({len(layers_per_block)}) must match num_stages ({num_stages})"
        assert len(depth_factors) == num_up_blocks, \
            f"depth_factors ({len(depth_factors)}) must match num_up_blocks ({num_up_blocks})"

        self.in_shortcut = in_shortcut
        self.num_stages = num_stages

        # Input conv with optional shortcut
        self.conv_in = nn.Conv3d(latent_channels, block_out_channels[-1], kernel_size=3, padding=1)

        if in_shortcut:
            assert block_out_channels[-1] % latent_channels == 0, \
                f"Last stage channels ({block_out_channels[-1]}) must be divisible by latent_channels ({latent_channels})"
            self.in_shortcut_repeats = block_out_channels[-1] // latent_channels

        # Build stages (in reverse order for decoder)
        self.stages = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # Process from deepest (highest channel count) to shallowest
        for i in range(num_stages - 1, -1, -1):
            # Upsample (except for deepest stage)
            if i < num_stages - 1:
                df = depth_factors[i]
                self.up_blocks.append(
                    DCUpBlock3D(
                        in_channels=block_out_channels[i + 1],
                        out_channels=block_out_channels[i],
                        spatial_factor=2,
                        depth_factor=df,
                        shortcut=True,
                    )
                )

            # ResBlocks at this resolution
            stage_blocks = []
            channels = block_out_channels[i]
            for _ in range(layers_per_block[i]):
                stage_blocks.append(ResBlock3D(channels))
            self.stages.append(nn.Sequential(*stage_blocks))

        # Output normalization and conv
        self.norm_out = RMSNorm3D(block_out_channels[0])
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv3d(block_out_channels[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent to output volume.

        Args:
            x: Latent tensor [B, latent_channels, D', H', W']

        Returns:
            Output tensor [B, C_out, D, H, W]
        """
        if self.in_shortcut:
            shortcut = x.repeat_interleave(self.in_shortcut_repeats, dim=1)
            x = self.conv_in(x) + shortcut
        else:
            x = self.conv_in(x)

        # Process stages (stages[0] is deepest, stages[-1] is shallowest)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.up_blocks):
                x = self.up_blocks[i](x)

        x = self.norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x


class AutoencoderDC3D(nn.Module):
    """3D Deep Compression Autoencoder.

    Deterministic autoencoder for volumetric data with high compression ratios.
    Uses pixel_unshuffle/pixel_shuffle for learned downsampling/upsampling.

    Key features:
    - Asymmetric compression (separate spatial and depth factors)
    - Residual autoencoding (encoder_out_shortcut, decoder_in_shortcut)
    - Pure ResBlock architecture (no attention for memory efficiency)

    Args:
        in_channels: Input channels (e.g., 4 for multi-modality MRI)
        latent_channels: Latent space channels (e.g., 32)
        encoder_block_out_channels: Encoder stage channels
        decoder_block_out_channels: Decoder stage channels
        encoder_layers_per_block: ResBlocks per encoder stage
        decoder_layers_per_block: ResBlocks per decoder stage
        depth_factors: Depth compression factor for each down/up block
        encoder_out_shortcut: Add shortcut at encoder output
        decoder_in_shortcut: Add shortcut at decoder input
        scaling_factor: Latent scaling for diffusion training

    Example:
        >>> model = AutoencoderDC3D(
        ...     in_channels=4,
        ...     latent_channels=32,
        ...     encoder_block_out_channels=(64, 128, 256, 256, 512, 512),
        ...     decoder_block_out_channels=(64, 128, 256, 256, 512, 512),
        ...     encoder_layers_per_block=(2, 2, 2, 2, 2, 2),
        ...     decoder_layers_per_block=(2, 2, 2, 2, 2, 2),
        ...     depth_factors=(2, 2, 1, 1, 1),
        ... )
        >>> x = torch.randn(1, 4, 160, 256, 256)
        >>> z = model.encode(x)  # [1, 32, 40, 8, 8]
        >>> y = model.decode(z)  # [1, 4, 160, 256, 256]
    """

    def __init__(
        self,
        in_channels: int = 4,
        latent_channels: int = 32,
        encoder_block_out_channels: Tuple[int, ...] = (64, 128, 256, 256, 512, 512),
        decoder_block_out_channels: Tuple[int, ...] = (64, 128, 256, 256, 512, 512),
        encoder_layers_per_block: Tuple[int, ...] = (2, 2, 2, 2, 2, 2),
        decoder_layers_per_block: Tuple[int, ...] = (2, 2, 2, 2, 2, 2),
        depth_factors: Tuple[int, ...] = (2, 2, 1, 1, 1),
        encoder_out_shortcut: bool = True,
        decoder_in_shortcut: bool = True,
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor

        # Compute compression ratios for logging
        n_down_blocks = len(encoder_block_out_channels) - 1
        self.spatial_compression = 2 ** n_down_blocks
        self.depth_compression = math.prod(depth_factors)

        self.encoder = Encoder3D(
            in_channels=in_channels,
            latent_channels=latent_channels,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=encoder_layers_per_block,
            depth_factors=depth_factors,
            out_shortcut=encoder_out_shortcut,
        )

        self.decoder = Decoder3D(
            in_channels=in_channels,
            latent_channels=latent_channels,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=decoder_layers_per_block,
            depth_factors=depth_factors,
            in_shortcut=decoder_in_shortcut,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input volume to latent space.

        Args:
            x: Input [B, C, D, H, W]

        Returns:
            Latent [B, latent_channels, D/depth_comp, H/spatial_comp, W/spatial_comp]
        """
        z = self.encoder(x)
        return z * self.scaling_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to output volume.

        Args:
            z: Latent tensor

        Returns:
            Reconstructed volume [B, C, D, H, W]
        """
        z = z / self.scaling_factor
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode-decode cycle.

        Args:
            x: Input volume [B, C, D, H, W]

        Returns:
            Reconstructed volume [B, C, D, H, W]
        """
        z = self.encode(x)
        return self.decode(z)


class CheckpointedAutoencoderDC3D(BaseCheckpointedModel):
    """Gradient-checkpointed wrapper for AutoencoderDC3D.

    Reduces activation memory by ~50% for 3D volumes by checkpointing
    encoder and decoder forward passes.

    Args:
        model: The underlying AutoencoderDC3D model.

    Example:
        >>> base_model = AutoencoderDC3D(...)
        >>> model = CheckpointedAutoencoderDC3D(base_model)
        >>> y = model(x)  # Same API, but uses gradient checkpointing
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with gradient checkpointing.

        Args:
            x: Input volume [B, C, D, H, W]

        Returns:
            Reconstructed volume [B, C, D, H, W]
        """
        def encode_fn(x: torch.Tensor) -> torch.Tensor:
            return self.model.encode(x)

        z = self.checkpoint(encode_fn, x)

        def decode_fn(z: torch.Tensor) -> torch.Tensor:
            return self.model.decode(z)

        return self.checkpoint(decode_fn, z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without checkpointing (for inference).

        Args:
            x: Input volume

        Returns:
            Latent tensor
        """
        return self.model.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode without checkpointing (for inference).

        Args:
            z: Latent tensor

        Returns:
            Reconstructed volume
        """
        return self.model.decode(z)

    @property
    def scaling_factor(self) -> float:
        """Get scaling factor from underlying model."""
        return self.model.scaling_factor

    @property
    def spatial_compression(self) -> int:
        """Get spatial compression ratio from underlying model."""
        return self.model.spatial_compression

    @property
    def depth_compression(self) -> int:
        """Get depth compression ratio from underlying model."""
        return self.model.depth_compression
