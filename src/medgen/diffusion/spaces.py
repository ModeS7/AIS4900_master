"""
Diffusion space abstractions for pixel-space and latent-space diffusion.

This module provides the DiffusionSpace abstraction that allows the same
diffusion training code to operate in either pixel space (identity) or
latent space (using a VAE encoder/decoder).
"""
from abc import ABC, abstractmethod

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
        self, data: Tensor | dict[str, Tensor]
    ) -> Tensor | dict[str, Tensor]:
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
        self, data: Tensor | dict[str, Tensor]
    ) -> Tensor | dict[str, Tensor]:
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

    @property
    def latent_channels(self) -> int:
        """Pixel space has 1 channel per input channel."""
        return 1


class LatentSpace(DiffusionSpace):
    """Latent space using a compression model for encoding/decoding.

    Wraps a trained compression model (VAE, VQ-VAE, or DC-AE) to provide
    encode/decode operations for latent diffusion model training.
    Supports both 2D and 3D.

    Args:
        compression_model: Trained compression model (VAE, VQ-VAE, DC-AE).
        device: Device for computations.
        deterministic: If True, use mean only (no sampling). Default False.
        spatial_dims: Spatial dimensions (2 for images, 3 for volumes).
        compression_type: Type of compression model ('vae', 'vqvae', 'dcae').
            Used for encoding strategy. Default 'vae'.
        scale_factor: Spatial downscale factor. If None, auto-detected from model.
            VAE/VQ-VAE typically use 8x, DC-AE uses 32x or 64x.
        depth_scale_factor: Depth downscale factor for 3D (may differ from spatial).
            If None, uses same as scale_factor.
        latent_channels: Number of channels in latent space. If None, auto-detected.
    """

    def __init__(
        self,
        compression_model: torch.nn.Module,
        device: torch.device,
        deterministic: bool = False,
        spatial_dims: int = 2,
        compression_type: str = 'vae',
        scale_factor: int | None = None,
        depth_scale_factor: int | None = None,
        latent_channels: int | None = None,
        slicewise_encoding: bool = False,
    ) -> None:
        # Store as 'vae' for backward compatibility with existing code
        self.vae = compression_model.eval()
        self.device = device
        self.deterministic = deterministic
        self.spatial_dims = spatial_dims
        self.compression_type = compression_type
        self.slicewise_encoding = slicewise_encoding

        # Freeze model parameters
        for param in self.vae.parameters():
            param.requires_grad = False

        # Auto-detect or use provided latent channels
        if latent_channels is not None:
            self._latent_channels = latent_channels
        else:
            self._latent_channels = self._detect_latent_channels(compression_model)

        # Auto-detect or use provided scale factor
        if scale_factor is not None:
            self._scale_factor = scale_factor
        else:
            self._scale_factor = self._detect_scale_factor(compression_model, compression_type)

        # Depth scale factor (3D only) - defaults to spatial scale factor
        # For slicewise encoding, depth is not compressed (scale factor = 1)
        if slicewise_encoding:
            self._depth_scale_factor = 1
        else:
            self._depth_scale_factor = depth_scale_factor if depth_scale_factor is not None else self._scale_factor

        # Expected tensor dimensions: batch + channels + spatial
        # For slicewise: model is 2D but we process 3D volumes
        if slicewise_encoding:
            self._expected_dims = 5  # 3D volumes [B, C, D, H, W]
        else:
            self._expected_dims = 2 + spatial_dims  # 4 for 2D, 5 for 3D

    def _detect_latent_channels(self, model: torch.nn.Module) -> int:
        """Detect latent channels from compression model."""
        # Try common attribute names
        if hasattr(model, 'latent_channels'):
            return int(model.latent_channels)  # type: ignore[arg-type]
        if hasattr(model, 'z_channels'):
            return int(model.z_channels)  # type: ignore[arg-type]
        if hasattr(model, 'embedding_dim'):
            return int(model.embedding_dim)  # type: ignore[arg-type]
        # DC-AE style
        if hasattr(model, 'config') and isinstance(model.config, dict):
            if 'latent_channels' in model.config:
                return int(model.config['latent_channels'])
        # Default
        return 4

    def _detect_scale_factor(self, model: torch.nn.Module, compression_type: str) -> int:
        """Detect spatial scale factor from compression model.

        Args:
            model: Compression model.
            compression_type: Type of model ('vae', 'vqvae', 'dcae').

        Returns:
            Spatial downscale factor.
        """
        import re

        # Check for explicit config
        if hasattr(model, 'config') and isinstance(model.config, dict):
            config = model.config
            # DC-AE uses spatial_compression_ratio
            if 'spatial_compression_ratio' in config:
                return int(config['spatial_compression_ratio'])
            # Some models use scale_factor directly
            if 'scale_factor' in config:
                return int(config['scale_factor'])
            # Check for f{N} notation (e.g., 'f32' -> 32)
            if 'name' in config:
                name_str = str(config['name'])
                # Guard against excessively long strings (ReDoS prevention)
                if len(name_str) <= 200:
                    match = re.search(r'f(\d+)', name_str)
                    if match:
                        return int(match.group(1))

        # Check for num_down_blocks (MONAI VAE pattern: 2^num_down_blocks)
        if hasattr(model, 'num_down_blocks'):
            return 2 ** int(model.num_down_blocks)  # type: ignore[arg-type, no-any-return]

        # Count encoder downsampling blocks by checking attribute names
        if hasattr(model, 'encoder'):
            encoder = model.encoder
            # Count 'down' blocks
            down_count = sum(1 for name in dir(encoder) if 'down' in name.lower() and not name.startswith('_'))
            if down_count > 0:
                return int(2 ** min(down_count, 6))  # Cap at 64x

        # Default based on compression type
        if compression_type == 'dcae':
            return 32  # DC-AE default
        else:
            return 8  # VAE/VQ-VAE default

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Encode images to latent space.

        Handles different compression model types:
        - VAE: Returns mean (deterministic) or samples from latent distribution
        - VQ-VAE: Returns quantized codes
        - DC-AE: Returns deterministic encoding

        For slicewise_encoding=True: applies 2D encoder slice-by-slice to 3D volume.

        Args:
            x: Images [B, C, H, W] (2D) or [B, C, D, H, W] (3D) in pixel space.

        Returns:
            Latent representation with shape based on scale_factor.

        Raises:
            ValueError: If input tensor doesn't have expected dimensions.
        """
        if x.dim() != self._expected_dims:
            shape_desc = "[B, C, D, H, W]" if self._expected_dims == 5 else "[B, C, H, W]"
            raise ValueError(
                f"LatentSpace.encode expects {self._expected_dims}D input {shape_desc}, "
                f"got {x.dim()}D tensor with shape {x.shape}"
            )

        # Slicewise encoding: apply 2D encoder to each depth slice
        if self.slicewise_encoding:
            return self._encode_slicewise(x)

        return self._encode_single(x)

    def _encode_single(self, x: Tensor) -> Tensor:
        """Encode a single tensor (2D or 3D) with the compression model."""
        if self.compression_type == 'vae':
            # VAE returns (mu, logvar)
            z_mu, z_logvar = self.vae.encode(x)  # type: ignore[operator]

            if self.deterministic:
                return z_mu  # type: ignore[no-any-return]

            # Reparameterization trick: z = mu + sigma * epsilon
            std = torch.exp(0.5 * z_logvar)
            eps = torch.randn_like(std)
            z: Tensor = z_mu + std * eps
            return z

        elif self.compression_type == 'vqvae':
            # VQ-VAE returns quantized directly
            return self.vae.encode(x)  # type: ignore[operator, no-any-return]

        elif self.compression_type == 'dcae':
            # DC-AE is deterministic - diffusers returns EncoderOutput object
            result = self.vae.encode(x)  # type: ignore[operator]
            if hasattr(result, 'latent'):
                return result.latent  # type: ignore[no-any-return]
            if isinstance(result, tuple):
                return result[0]  # type: ignore[no-any-return]
            return result  # type: ignore[no-any-return]

        else:
            # Generic fallback: try to handle tuple returns
            result = self.vae.encode(x)  # type: ignore[operator]
            if isinstance(result, tuple):
                return result[0]  # type: ignore[no-any-return]
            return result  # type: ignore[no-any-return]

    def _encode_slicewise(self, x: Tensor) -> Tensor:
        """Encode 3D volume slice-by-slice using 2D encoder.

        Args:
            x: 3D volume [B, C, D, H, W].

        Returns:
            Latent volume [B, C_latent, D, H_latent, W_latent].
        """
        B, C, D, H, W = x.shape
        latent_slices = []

        for d in range(D):
            # Extract slice [B, C, H, W]
            slice_2d = x[:, :, d, :, :]
            # Encode slice
            latent_slice = self._encode_single(slice_2d)
            latent_slices.append(latent_slice)

        # Stack along depth dimension: [B, C_lat, D, H_lat, W_lat]
        return torch.stack(latent_slices, dim=2)

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation to pixel space.

        For slicewise_encoding=True: applies 2D decoder slice-by-slice to 3D latent.

        Args:
            z: Latent representation [B, latent_channels, H/8, W/8] (2D)
               or [B, latent_channels, D, H/8, W/8] (3D).

        Returns:
            Decoded images [B, C, H, W] (2D) or [B, C, D, H, W] (3D) in pixel space.
        """
        # Slicewise decoding: apply 2D decoder to each depth slice
        if self.slicewise_encoding and z.dim() == 5:
            return self._decode_slicewise(z)

        result = self.vae.decode(z)  # type: ignore[operator]
        # Handle diffusers DecoderOutput (has .sample attribute)
        if hasattr(result, 'sample'):
            return result.sample  # type: ignore[no-any-return]
        return result  # type: ignore[no-any-return]

    def _decode_slicewise(self, z: Tensor) -> Tensor:
        """Decode 3D latent volume slice-by-slice using 2D decoder.

        Args:
            z: Latent volume [B, C_latent, D, H_latent, W_latent].

        Returns:
            Decoded volume [B, C, D, H, W].
        """
        B, C_lat, D, H_lat, W_lat = z.shape
        decoded_slices = []

        for d in range(D):
            # Extract latent slice [B, C_lat, H_lat, W_lat]
            latent_slice = z[:, :, d, :, :]
            # Decode slice
            result = self.vae.decode(latent_slice)  # type: ignore[operator]
            # Handle diffusers DecoderOutput
            if hasattr(result, 'sample'):
                decoded_slice = result.sample
            else:
                decoded_slice = result
            decoded_slices.append(decoded_slice)

        # Stack along depth dimension: [B, C, D, H, W]
        return torch.stack(decoded_slices, dim=2)

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
        """Spatial downscale factor (varies by compression model)."""
        return self._scale_factor

    @property
    def depth_scale_factor(self) -> int:
        """Depth downscale factor for 3D (may differ from spatial)."""
        return self._depth_scale_factor

    @property
    def latent_channels(self) -> int:
        """Number of latent channels per input channel."""
        return self._latent_channels


def load_vae_for_latent_space(
    checkpoint_path: str,
    device: torch.device,
    vae_config: dict | None = None,
    spatial_dims: int = 2,
) -> LatentSpace:
    """Load a trained VAE and create a LatentSpace wrapper.

    Note: This function is for backward compatibility. For new code, prefer
    using `load_compression_model` from `medgen.data.loaders.latent` which
    supports VAE, VQ-VAE, and DC-AE.

    Args:
        checkpoint_path: Path to VAE checkpoint (.pt file).
        device: Device to load model to.
        vae_config: Optional VAE configuration dict. If None, will try to
            load from checkpoint metadata.
        spatial_dims: Spatial dimensions (2 for images, 3 for volumes).

    Returns:
        LatentSpace instance wrapping the loaded VAE.
    """
    from monai.networks.nets import AutoencoderKL

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Try to get config from checkpoint
    if vae_config is None:
        if 'config' in checkpoint:
            vae_config = checkpoint['config']
        else:
            raise ValueError(
                "VAE config not found in checkpoint and not provided. "
                "Please provide vae_config dict."
            )

    # Auto-detect spatial_dims from config if available
    if 'spatial_dims' in vae_config:
        spatial_dims = vae_config['spatial_dims']

    # Create VAE model
    vae = AutoencoderKL(
        spatial_dims=spatial_dims,
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
        # Assume checkpoint is raw state dict - validate before loading
        try:
            vae.load_state_dict(checkpoint)
        except RuntimeError as e:
            raise ValueError(
                f"Could not load VAE checkpoint. Expected 'model_state_dict' or "
                f"'state_dict' key, or compatible raw state_dict. Error: {e}"
            ) from e

    return LatentSpace(
        vae, device,
        spatial_dims=spatial_dims,
        compression_type='vae',
        latent_channels=vae_config['latent_channels'],
    )
