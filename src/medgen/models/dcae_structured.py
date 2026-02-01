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
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .dcae_adaptive_layers import AdaptiveInputConv2d, AdaptiveOutputConv2d, copy_conv_weights

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
        channel_steps: List[int],
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

        # Find and replace projection layers
        self._replace_encoder_output()
        self._replace_decoder_input()

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

    def _replace_encoder_output(self) -> None:
        """Replace encoder's output conv with AdaptiveOutputConv2d."""
        encoder = self.base_model.encoder

        # Find the output conv - it's typically encoder.conv_out
        if hasattr(encoder, 'conv_out'):
            old_conv = encoder.conv_out
            if isinstance(old_conv, nn.Conv2d):
                # Direct Conv2d
                self.encoder_out_adaptive = AdaptiveOutputConv2d(
                    in_channels=old_conv.in_channels,
                    out_channels=old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None,
                )
                copy_conv_weights(old_conv, self.encoder_out_adaptive)
                self._encoder_out_is_direct = True
                logger.debug("Replaced encoder.conv_out (direct Conv2d)")
            else:
                # Might be a wrapper module with .conv attribute
                if hasattr(old_conv, 'conv') and isinstance(old_conv.conv, nn.Conv2d):
                    inner_conv = old_conv.conv
                    self.encoder_out_adaptive = AdaptiveOutputConv2d(
                        in_channels=inner_conv.in_channels,
                        out_channels=inner_conv.out_channels,
                        kernel_size=inner_conv.kernel_size,
                        stride=inner_conv.stride,
                        padding=inner_conv.padding,
                        bias=inner_conv.bias is not None,
                    )
                    copy_conv_weights(inner_conv, self.encoder_out_adaptive)
                    self._encoder_out_is_direct = False
                    self._encoder_out_wrapper = old_conv
                    logger.debug("Replaced encoder.conv_out.conv (wrapped Conv2d)")
                else:
                    raise ValueError(
                        f"encoder.conv_out is not a Conv2d or wrapper: {type(old_conv)}"
                    )
        else:
            raise ValueError("Cannot find encoder.conv_out in base model")

    def _replace_decoder_input(self) -> None:
        """Replace decoder's input conv with AdaptiveInputConv2d."""
        decoder = self.base_model.decoder

        # Find the input conv - it's typically decoder.conv_in
        if hasattr(decoder, 'conv_in'):
            old_conv = decoder.conv_in
            if isinstance(old_conv, nn.Conv2d):
                # Direct Conv2d
                self.decoder_in_adaptive = AdaptiveInputConv2d(
                    in_channels=old_conv.in_channels,
                    out_channels=old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None,
                )
                copy_conv_weights(old_conv, self.decoder_in_adaptive)
                self._decoder_in_is_direct = True
                logger.debug("Replaced decoder.conv_in (direct Conv2d)")
            else:
                # Might be a wrapper module with .conv attribute
                if hasattr(old_conv, 'conv') and isinstance(old_conv.conv, nn.Conv2d):
                    inner_conv = old_conv.conv
                    self.decoder_in_adaptive = AdaptiveInputConv2d(
                        in_channels=inner_conv.in_channels,
                        out_channels=inner_conv.out_channels,
                        kernel_size=inner_conv.kernel_size,
                        stride=inner_conv.stride,
                        padding=inner_conv.padding,
                        bias=inner_conv.bias is not None,
                    )
                    copy_conv_weights(inner_conv, self.decoder_in_adaptive)
                    self._decoder_in_is_direct = False
                    self._decoder_in_wrapper = old_conv
                    logger.debug("Replaced decoder.conv_in.conv (wrapped Conv2d)")
                else:
                    raise ValueError(
                        f"decoder.conv_in is not a Conv2d or wrapper: {type(old_conv)}"
                    )
        else:
            raise ValueError("Cannot find decoder.conv_in in base model")

    def _run_encoder_body(self, x: torch.Tensor) -> torch.Tensor:
        """Run encoder up to (but not including) the output conv.

        This runs all encoder layers except the final projection to latent space.
        """
        encoder = self.base_model.encoder

        # Initial conv
        x = encoder.conv_in(x)

        # Down blocks
        if hasattr(encoder, 'down_blocks'):
            for down_block in encoder.down_blocks:
                x = down_block(x)

        # Output normalization (if present)
        if hasattr(encoder, 'conv_norm_out'):
            x = encoder.conv_norm_out(x)

        # Activation before output conv (if present)
        if hasattr(encoder, 'conv_act'):
            x = encoder.conv_act(x)

        return x

    def _run_decoder_body(self, x: torch.Tensor) -> torch.Tensor:
        """Run decoder after the input conv to the output.

        This runs all decoder layers after the initial projection from latent space.
        """
        decoder = self.base_model.decoder

        # Up blocks
        if hasattr(decoder, 'up_blocks'):
            for up_block in decoder.up_blocks:
                x = up_block(x)

        # Output normalization and conv
        if hasattr(decoder, 'conv_norm_out'):
            x = decoder.conv_norm_out(x)

        if hasattr(decoder, 'conv_act'):
            x = decoder.conv_act(x)

        if hasattr(decoder, 'conv_out'):
            conv_out = decoder.conv_out
            if hasattr(conv_out, 'conv'):
                # Wrapper module
                x = conv_out(x)
            else:
                # Direct Conv2d
                x = conv_out(x)

        return x

    def encode(
        self,
        x: torch.Tensor,
        latent_channels: Optional[int] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], "EncoderOutput"]:
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

        # Run encoder body (everything except final conv)
        h = self._run_encoder_body(x)

        # Apply adaptive output conv with channel slicing
        z = self.encoder_out_adaptive(h, out_channels=latent_channels)

        # Apply scaling factor if present
        if hasattr(self.base_model, 'config') and hasattr(self.base_model.config, 'scaling_factor'):
            z = z * self.base_model.config.scaling_factor

        if return_dict:
            return _EncoderOutput(latent=z)
        return (z,)

    def decode(
        self,
        z: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], "DecoderOutput"]:
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
        # Undo scaling factor if present
        if hasattr(self.base_model, 'config') and hasattr(self.base_model.config, 'scaling_factor'):
            z = z / self.base_model.config.scaling_factor

        # Apply adaptive input conv (handles variable channels)
        h = self.decoder_in_adaptive(z)

        # Run decoder body (everything after input conv)
        x = self._run_decoder_body(h)

        if return_dict:
            return _DecoderOutput(sample=x)
        return (x,)

    def forward(
        self,
        x: torch.Tensor,
        latent_channels: Optional[int] = None,
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
        """Return all parameters including adaptive layers and base model."""
        # Yield adaptive layer parameters first
        yield from self.encoder_out_adaptive.parameters(recurse)
        yield from self.decoder_in_adaptive.parameters(recurse)

        # Yield base model parameters, excluding the replaced layers
        for name, param in self.base_model.named_parameters(recurse=recurse):
            # Skip the original conv layers that we replaced
            if 'encoder.conv_out' in name or 'decoder.conv_in' in name:
                continue
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Return all named parameters including adaptive layers."""
        # Yield adaptive layer parameters
        for name, param in self.encoder_out_adaptive.named_parameters(recurse=recurse):
            yield f'{prefix}encoder_out_adaptive.{name}' if prefix else f'encoder_out_adaptive.{name}', param
        for name, param in self.decoder_in_adaptive.named_parameters(recurse=recurse):
            yield f'{prefix}decoder_in_adaptive.{name}' if prefix else f'decoder_in_adaptive.{name}', param

        # Yield base model parameters, excluding replaced layers
        for name, param in self.base_model.named_parameters(prefix='base_model', recurse=recurse):
            if 'encoder.conv_out' in name or 'decoder.conv_in' in name:
                continue
            yield f'{prefix}{name}' if prefix else name, param

    def state_dict(self, *args, **kwargs):
        """Return state dict with adaptive layers."""
        state = {}

        # Add adaptive layer states
        for name, param in self.encoder_out_adaptive.state_dict().items():
            state[f'encoder_out_adaptive.{name}'] = param
        for name, param in self.decoder_in_adaptive.state_dict().items():
            state[f'decoder_in_adaptive.{name}'] = param

        # Add base model state (excluding replaced layers)
        for name, param in self.base_model.state_dict().items():
            if 'encoder.conv_out' in name or 'decoder.conv_in' in name:
                continue
            state[f'base_model.{name}'] = param

        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state dict with adaptive layers."""
        # Separate adaptive and base model states
        adaptive_encoder_state = {}
        adaptive_decoder_state = {}
        base_state = {}

        for name, param in state_dict.items():
            if name.startswith('encoder_out_adaptive.'):
                key = name.replace('encoder_out_adaptive.', '')
                adaptive_encoder_state[key] = param
            elif name.startswith('decoder_in_adaptive.'):
                key = name.replace('decoder_in_adaptive.', '')
                adaptive_decoder_state[key] = param
            elif name.startswith('base_model.'):
                key = name.replace('base_model.', '')
                base_state[key] = param
            else:
                # Handle legacy state dicts without prefixes
                base_state[name] = param

        # Load states
        if adaptive_encoder_state:
            self.encoder_out_adaptive.load_state_dict(adaptive_encoder_state, strict=strict)
        if adaptive_decoder_state:
            self.decoder_in_adaptive.load_state_dict(adaptive_decoder_state, strict=strict)

        # Load base model state (skip replaced layers)
        if base_state:
            # Filter out replaced layer keys from base_state
            filtered_base_state = {
                k: v for k, v in base_state.items()
                if 'encoder.conv_out' not in k and 'decoder.conv_in' not in k
            }
            # Get expected keys from base model
            base_keys = set(self.base_model.state_dict().keys())
            filtered_keys = set(filtered_base_state.keys())

            # Remove keys for replaced layers from expected
            expected_keys = {
                k for k in base_keys
                if 'encoder.conv_out' not in k and 'decoder.conv_in' not in k
            }

            missing = expected_keys - filtered_keys
            unexpected = filtered_keys - expected_keys

            if strict and (missing or unexpected):
                raise RuntimeError(
                    f"Error loading state_dict: missing keys: {missing}, unexpected keys: {unexpected}"
                )

            self.base_model.load_state_dict(filtered_base_state, strict=False)


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
