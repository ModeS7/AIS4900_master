"""Structured Latent Space wrapper for DC-AE (DC-AE 1.5).

This module provides StructuredAutoencoderDC, a wrapper around diffusers AutoencoderDC
that implements the official DC-Gen structured latent space approach using adaptive
convolution layers with dynamic weight slicing.

Key difference from naive channel masking:
- Naive: Compute all channels, then zero-mask some (gradients still flow through masked)
- Structured: Slice conv weights to only compute first N channels (true structured training)

Reference: DC-Gen (https://arxiv.org/abs/2412.09612)
"""
import logging
from typing import TYPE_CHECKING, Union

import torch
import torch.nn as nn

from .dcae_adaptive_layers import AdaptiveInputConv2d, AdaptiveOutputConv2d, copy_conv_weights

if TYPE_CHECKING:
    from diffusers.models.autoencoders.autoencoder_dc import DecoderOutput, EncoderOutput

logger = logging.getLogger(__name__)


class StructuredAutoencoderDC(nn.Module):
    """Wrapper for diffusers AutoencoderDC with structured latent space support.

    This wrapper replaces the encoder's output conv and decoder's input conv with
    adaptive versions that support dynamic channel slicing. This enables DC-AE 1.5
    structured latent space training where gradients only flow through active channels.

    The wrapper preserves the original model's behavior when full channels are used,
    but enables training with variable channel counts for structured hierarchy.

    Args:
        base_model: The underlying diffusers AutoencoderDC model
        channel_steps: List of channel counts to sample from during training

    Example:
        >>> from diffusers import AutoencoderDC
        >>> base = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0")
        >>> model = StructuredAutoencoderDC(base, channel_steps=[16, 32, 64, 128])
        >>> z = model.encode(images, latent_channels=64)  # Only 64 channels computed
        >>> recon = model.decode(z)  # Decoder auto-detects 64 channels
    """

    def __init__(
        self,
        base_model: nn.Module,
        channel_steps: list[int],
    ) -> None:
        super().__init__()

        self.base_model = base_model
        self.channel_steps = sorted(channel_steps)
        self.max_channels = self.channel_steps[-1]

        # Get the latent channels from the base model
        self.latent_channels = self._get_latent_channels()

        if self.max_channels > self.latent_channels:
            raise ValueError(
                f"Max channel step ({self.max_channels}) exceeds "
                f"model latent_channels ({self.latent_channels})"
            )

        # Create adaptive layers and replace in base model
        self._setup_adaptive_layers()

        logger.info(
            f"StructuredAutoencoderDC initialized: "
            f"latent_channels={self.latent_channels}, channel_steps={self.channel_steps}"
        )

    def _get_latent_channels(self) -> int:
        """Get the latent channels from the base model config."""
        if hasattr(self.base_model, 'config'):
            return self.base_model.config.latent_channels
        elif hasattr(self.base_model, 'latent_channels'):
            return self.base_model.latent_channels
        else:
            raise ValueError("Cannot determine latent_channels from base model")

    def _setup_adaptive_layers(self) -> None:
        """Create adaptive layers and replace them in the base model."""
        encoder = self.base_model.encoder
        decoder = self.base_model.decoder

        # Check if encoder has shortcut (needs special handling)
        self._encoder_has_shortcut = getattr(encoder, 'out_shortcut', False)
        if self._encoder_has_shortcut:
            self._shortcut_group_size = getattr(encoder, 'out_shortcut_average_group_size', 1)
            logger.debug(f"Encoder has shortcut with group_size={self._shortcut_group_size}")

        # Replace encoder.conv_out
        old_encoder_out = encoder.conv_out
        if isinstance(old_encoder_out, nn.Conv2d):
            self.encoder_out_adaptive = AdaptiveOutputConv2d(
                in_channels=old_encoder_out.in_channels,
                out_channels=old_encoder_out.out_channels,
                kernel_size=old_encoder_out.kernel_size,
                stride=old_encoder_out.stride,
                padding=old_encoder_out.padding,
                bias=old_encoder_out.bias is not None,
            )
            copy_conv_weights(old_encoder_out, self.encoder_out_adaptive)
            # Replace in base model
            encoder.conv_out = self.encoder_out_adaptive
            logger.debug("Replaced encoder.conv_out with AdaptiveOutputConv2d")
        else:
            raise ValueError(f"encoder.conv_out is not Conv2d: {type(old_encoder_out)}")

        # Check if decoder has shortcut
        self._decoder_has_shortcut = getattr(decoder, 'in_shortcut', False)
        if self._decoder_has_shortcut:
            self._decoder_shortcut_repeats = getattr(decoder, 'in_shortcut_repeats', 1)
            logger.debug(f"Decoder has shortcut with repeats={self._decoder_shortcut_repeats}")

        # Replace decoder.conv_in
        old_decoder_in = decoder.conv_in
        if isinstance(old_decoder_in, nn.Conv2d):
            self.decoder_in_adaptive = AdaptiveInputConv2d(
                in_channels=old_decoder_in.in_channels,
                out_channels=old_decoder_in.out_channels,
                kernel_size=old_decoder_in.kernel_size,
                stride=old_decoder_in.stride,
                padding=old_decoder_in.padding,
                bias=old_decoder_in.bias is not None,
            )
            copy_conv_weights(old_decoder_in, self.decoder_in_adaptive)
            # Replace in base model
            decoder.conv_in = self.decoder_in_adaptive
            logger.debug("Replaced decoder.conv_in with AdaptiveInputConv2d")
        else:
            raise ValueError(f"decoder.conv_in is not Conv2d: {type(old_decoder_in)}")

        # Track the current latent channels for encode
        self._current_latent_channels: int | None = None

        # Replace encoder forward to handle shortcut with variable channels
        if self._encoder_has_shortcut:
            self._original_encoder_forward = encoder.forward
            encoder.forward = self._encoder_forward_with_shortcut

        # Replace decoder forward to handle shortcut with variable input channels
        if self._decoder_has_shortcut:
            self._original_decoder_forward = decoder.forward
            decoder.forward = self._decoder_forward_with_shortcut

    def _encoder_forward_with_shortcut(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Custom encoder forward that handles shortcut with variable output channels.

        This replaces the original encoder.forward to properly slice the shortcut
        when using fewer than max latent channels.
        """
        encoder = self.base_model.encoder

        hidden_states = encoder.conv_in(hidden_states)
        for down_block in encoder.down_blocks:
            hidden_states = down_block(hidden_states)

        # Get the number of output channels we want
        out_channels = self._current_latent_channels or self.latent_channels

        # Compute shortcut: average groups of channels
        # Original: x = hidden_states.unflatten(1, (-1, group_size)).mean(dim=2)
        # This produces latent_channels output
        # We need to only take the first out_channels
        x = hidden_states.unflatten(1, (-1, self._shortcut_group_size))
        x = x.mean(dim=2)  # [B, latent_channels, H, W]
        x = x[:, :out_channels]  # Slice to match output

        # Apply adaptive conv_out with channel slicing
        hidden_states = self.encoder_out_adaptive(hidden_states, out_channels=out_channels) + x

        return hidden_states

    def _decoder_forward_with_shortcut(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Custom decoder forward that handles shortcut with variable input channels.

        This replaces the original decoder.forward to properly handle the shortcut
        when the input has fewer than max latent channels.
        """
        decoder = self.base_model.decoder
        input_channels = hidden_states.shape[1]

        # Compute shortcut: repeat the input to match conv_in output channels
        # Original: x = hidden_states.repeat_interleave(repeats, dim=1)
        # With full channels (128), repeats=8 gives 1024 channels
        #
        # For variable input channels, we need to handle cases where input_channels
        # doesn't evenly divide into conv_in_out_channels. Strategy:
        # 1. Repeat enough times to reach or exceed target channels
        # 2. Slice to exactly match target channels
        conv_in_out_channels = self.decoder_in_adaptive.out_channels  # 1024

        # Compute repeats: ceiling division to ensure we have enough channels
        needed_repeats = (conv_in_out_channels + input_channels - 1) // input_channels
        x = hidden_states.repeat_interleave(needed_repeats, dim=1)

        # Slice to exact target size if we repeated too much
        if x.shape[1] > conv_in_out_channels:
            x = x[:, :conv_in_out_channels]

        # Apply adaptive conv_in (automatically handles variable input channels)
        hidden_states = self.decoder_in_adaptive(hidden_states) + x

        for up_block in reversed(decoder.up_blocks):
            hidden_states = up_block(hidden_states)

        hidden_states = decoder.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        hidden_states = decoder.conv_act(hidden_states)
        hidden_states = decoder.conv_out(hidden_states)

        return hidden_states

    def encode(
        self,
        x: torch.Tensor,
        latent_channels: int | None = None,
        return_dict: bool = True,
    ) -> Union[tuple[torch.Tensor], "EncoderOutput"]:
        """Encode input to latent space with optional channel restriction.

        Args:
            x: Input tensor [B, C, H, W]
            latent_channels: Number of latent channels to output. If None, uses all.
            return_dict: Whether to return a dict-like object (for diffusers compatibility)

        Returns:
            If return_dict=True: EncoderOutput with .latent attribute
            If return_dict=False: Tuple of (latent,)
        """
        if latent_channels is None:
            latent_channels = self.latent_channels

        # Store for the encoder forward / adaptive layer to use
        self._current_latent_channels = latent_channels

        try:
            if self._encoder_has_shortcut:
                # Use our custom encoder forward that handles shortcut
                result = self.base_model.encode(x, return_dict=return_dict)
            else:
                # No shortcut - just need to patch the conv_out forward
                original_forward = self.encoder_out_adaptive.forward

                def patched_forward(input_tensor, out_channels=None):
                    return original_forward(input_tensor, out_channels=self._current_latent_channels)

                self.encoder_out_adaptive.forward = patched_forward
                try:
                    result = self.base_model.encode(x, return_dict=return_dict)
                finally:
                    self.encoder_out_adaptive.forward = original_forward
        finally:
            self._current_latent_channels = None

        return result

    def decode(
        self,
        z: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[tuple[torch.Tensor], "DecoderOutput"]:
        """Decode latent to output image.

        The decoder automatically handles variable input channels via the
        adaptive input conv.

        Args:
            z: Latent tensor [B, C, H, W] where C can be any valid channel count
            return_dict: Whether to return a dict-like object (for diffusers compatibility)

        Returns:
            If return_dict=True: DecoderOutput with .sample attribute
            If return_dict=False: Tuple of (sample,)
        """
        # AdaptiveInputConv2d automatically handles variable input channels
        return self.base_model.decode(z, return_dict=return_dict)

    def forward(
        self,
        x: torch.Tensor,
        latent_channels: int | None = None,
    ) -> torch.Tensor:
        """Full encode-decode cycle.

        Args:
            x: Input tensor [B, C, H, W]
            latent_channels: Number of latent channels to use. If None, uses all.

        Returns:
            Reconstructed tensor [B, C, H, W]
        """
        z = self.encode(x, latent_channels=latent_channels, return_dict=False)[0]
        return self.decode(z, return_dict=False)[0]

    # Delegate attribute access to base model for compatibility
    @property
    def encoder(self) -> nn.Module:
        """Access base model encoder (for compatibility)."""
        return self.base_model.encoder

    @property
    def decoder(self) -> nn.Module:
        """Access base model decoder (for compatibility)."""
        return self.base_model.decoder

    @property
    def config(self):
        """Access base model config (for compatibility)."""
        return getattr(self.base_model, 'config', None)

    def parameters(self, recurse: bool = True):
        """Return all parameters from base model (which now contains adaptive layers)."""
        return self.base_model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Return all named parameters from base model."""
        return self.base_model.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        """Return state dict from base model."""
        return self.base_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state dict into base model."""
        return self.base_model.load_state_dict(state_dict, strict=strict)

    def train(self, mode: bool = True):
        """Set training mode."""
        self.base_model.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        self.base_model.eval()
        return self

    def to(self, *args, **kwargs):
        """Move model to device/dtype."""
        self.base_model.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        """Move model to CUDA."""
        self.base_model.cuda(device)
        return self

    def cpu(self):
        """Move model to CPU."""
        self.base_model.cpu()
        return self


class _EncoderOutput:
    """Simple output container for encoder (diffusers compatibility)."""

    def __init__(self, latent: torch.Tensor):
        self.latent = latent

    def __getitem__(self, idx):
        if idx == 0:
            return self.latent
        raise IndexError(f"Index {idx} out of range")


class _DecoderOutput:
    """Simple output container for decoder (diffusers compatibility)."""

    def __init__(self, sample: torch.Tensor):
        self.sample = sample

    def __getitem__(self, idx):
        if idx == 0:
            return self.sample
        raise IndexError(f"Index {idx} out of range")
