"""
Diffusion space abstractions for pixel-space and latent-space diffusion.

This module provides the DiffusionSpace abstraction that allows the same
diffusion training code to operate in either pixel space (identity) or
latent space (using a VAE encoder/decoder).
"""
from abc import ABC, abstractmethod
from typing import Dict, Union

import torch
from torch import Tensor


class DiffusionSpace(ABC):
    """Abstract base class for diffusion space operations.

    Defines the interface for encoding/decoding between pixel space
    and diffusion space. Subclasses implement either identity (pixel)
    or VAE-based (latent) transformations.
    """

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """Encode images from pixel space to diffusion space.

        Args:
            x: Images in pixel space [B, C, H, W].

        Returns:
            Encoded representation in diffusion space.
        """
        pass

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        """Decode from diffusion space back to pixel space.

        Args:
            z: Representation in diffusion space.

        Returns:
            Decoded images in pixel space [B, C, H, W].
        """
        pass

    @abstractmethod
    def get_latent_channels(self, input_channels: int) -> int:
        """Get the number of channels in diffusion space.

        Args:
            input_channels: Number of input channels in pixel space.

        Returns:
            Number of channels in diffusion space.
        """
        pass

    @property
    @abstractmethod
    def scale_factor(self) -> int:
        """Spatial downscale factor from pixel to diffusion space.

        Returns:
            Scale factor (1 for pixel space, typically 8 for latent).
        """
        pass

    def encode_batch(
        self, data: Union[Tensor, Dict[str, Tensor]]
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Encode a batch that may be a tensor or dict of tensors.

        Args:
            data: Either a single tensor or dict of tensors.

        Returns:
            Encoded data in same format as input.
        """
        if isinstance(data, dict):
            return {k: self.encode(v) for k, v in data.items()}
        return self.encode(data)

    def decode_batch(
        self, data: Union[Tensor, Dict[str, Tensor]]
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Decode a batch that may be a tensor or dict of tensors.

        Args:
            data: Either a single tensor or dict of tensors.

        Returns:
            Decoded data in same format as input.
        """
        if isinstance(data, dict):
            return {k: self.decode(v) for k, v in data.items()}
        return self.decode(data)


class PixelSpace(DiffusionSpace):
    """Identity space - diffusion operates directly on pixels.

    This is the default space that maintains backward compatibility
    with existing pixel-space diffusion training.
    """

    def encode(self, x: Tensor) -> Tensor:
        """Identity encoding - returns input unchanged."""
        return x

    def decode(self, z: Tensor) -> Tensor:
        """Identity decoding - returns input unchanged."""
        return z

    def get_latent_channels(self, input_channels: int) -> int:
        """Channels unchanged in pixel space."""
        return input_channels

    @property
    def scale_factor(self) -> int:
        """No spatial scaling in pixel space."""
        return 1


class LatentSpace(DiffusionSpace):
    """Latent space using AutoencoderKL for encoding/decoding.

    Wraps a trained VAE to provide encode/decode operations for
    latent diffusion model training.

    Args:
        vae: Trained AutoencoderKL model.
        device: Device for computations.
        deterministic: If True, use mean only (no sampling). Default False.
    """

    def __init__(
        self,
        vae: torch.nn.Module,
        device: torch.device,
        deterministic: bool = False,
    ) -> None:
        self.vae = vae.eval()
        self.device = device
        self.deterministic = deterministic

        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False

        # Get latent channels from VAE config
        self._latent_channels = vae.latent_channels

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Encode images to latent space.

        Uses reparameterization trick to sample from the latent distribution
        unless deterministic mode is enabled.

        Args:
            x: Images [B, C, H, W] in pixel space.

        Returns:
            Latent representation [B, latent_channels, H/8, W/8].
        """
        z_mu, z_logvar = self.vae.encode(x)

        if self.deterministic:
            return z_mu

        # Reparameterization trick: z = mu + sigma * epsilon
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mu + std * eps

        return z

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation to pixel space.

        Args:
            z: Latent representation [B, latent_channels, H/8, W/8].

        Returns:
            Decoded images [B, C, H, W] in pixel space.
        """
        return self.vae.decode(z)

    def get_latent_channels(self, input_channels: int) -> int:
        """Get latent channel count.

        For VAE, each input channel maps to latent_channels dimensions.

        Args:
            input_channels: Number of input channels.

        Returns:
            Number of latent channels (latent_channels * input_channels).
        """
        return self._latent_channels * input_channels

    @property
    def scale_factor(self) -> int:
        """Spatial downscale factor (8x for 3 downsampling stages)."""
        return 8

    @property
    def latent_channels(self) -> int:
        """Number of latent channels per input channel."""
        return self._latent_channels


def load_vae_for_latent_space(
    checkpoint_path: str,
    device: torch.device,
    vae_config: dict = None,
) -> LatentSpace:
    """Load a trained VAE and create a LatentSpace wrapper.

    Args:
        checkpoint_path: Path to VAE checkpoint (.pt file).
        device: Device to load model to.
        vae_config: Optional VAE configuration dict. If None, will try to
            load from checkpoint metadata.

    Returns:
        LatentSpace instance wrapping the loaded VAE.
    """
    from monai.networks.nets import AutoencoderKL

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Try to get config from checkpoint
    if vae_config is None:
        if 'config' in checkpoint:
            vae_config = checkpoint['config']
        else:
            raise ValueError(
                "VAE config not found in checkpoint and not provided. "
                "Please provide vae_config dict."
            )

    # Create VAE model
    vae = AutoencoderKL(
        spatial_dims=2,
        in_channels=vae_config.get('in_channels', 1),
        out_channels=vae_config.get('out_channels', 1),
        channels=tuple(vae_config['channels']),
        attention_levels=tuple(vae_config['attention_levels']),
        latent_channels=vae_config['latent_channels'],
        num_res_blocks=vae_config.get('num_res_blocks', 2),
        norm_num_groups=vae_config.get('norm_num_groups', 32),
        with_encoder_nonlocal_attn=vae_config.get('with_encoder_nonlocal_attn', True),
        with_decoder_nonlocal_attn=vae_config.get('with_decoder_nonlocal_attn', True),
    ).to(device)

    # Load weights
    if 'model_state_dict' in checkpoint:
        vae.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        vae.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume checkpoint is just the state dict
        vae.load_state_dict(checkpoint)

    return LatentSpace(vae, device)
