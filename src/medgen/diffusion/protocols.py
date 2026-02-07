"""Protocol definitions for type-safe diffusion interfaces.

This module defines structural subtyping (duck typing) interfaces for
diffusion models and batch data. Using Protocols allows for flexible
type checking without requiring inheritance.

Usage:
    from medgen.diffusion.protocols import DiffusionModel, PreparedBatch

    def train_step(model: DiffusionModel, batch: PreparedBatch) -> Tensor:
        images = batch.images
        prediction = model(images, timesteps)
        ...
"""
from typing import Any, Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class DiffusionModel(Protocol):
    """Standard diffusion model interface.

    A model that predicts noise or velocity given noisy input and timesteps.
    This is the base interface for all diffusion models.
    """

    def __call__(self, x: Tensor, timesteps: Tensor, **kwargs: Any) -> Tensor:
        """Forward pass for noise/velocity prediction.

        Args:
            x: Noisy input tensor [B, C, H, W] or [B, C, D, H, W].
            timesteps: Timestep values [B] (continuous or discrete).
            **kwargs: Additional conditioning arguments (size_bins, omega, mode_id, etc.)

        Returns:
            Predicted noise or velocity tensor (same shape as x).
        """
        ...


@runtime_checkable
class ConditionalDiffusionModel(Protocol):
    """Diffusion model with optional conditioning.

    Extends DiffusionModel with additional conditioning inputs for
    classifier-free guidance (omega) and multi-modality (mode_id).
    """

    def __call__(
        self,
        x: Tensor,
        timesteps: Tensor,
        omega: Tensor | None = None,
        mode_id: Tensor | None = None,
    ) -> Tensor:
        """Forward pass with optional conditioning.

        Args:
            x: Noisy input tensor [B, C, H, W] or [B, C, D, H, W].
            timesteps: Timestep values [B].
            omega: Optional classifier-free guidance scale [B, 1].
            mode_id: Optional mode embedding index [B] for multi-modality.

        Returns:
            Predicted noise or velocity tensor.
        """
        ...


@runtime_checkable
class SizeBinModel(Protocol):
    """Model supporting size bin FiLM conditioning.

    Used for seg_conditioned mode where tumor size bins control
    the generated segmentation masks via FiLM (Feature-wise Linear Modulation).
    """

    # Marker attribute to identify models with size bin support
    size_bin_time_embed: Any

    def __call__(
        self,
        x: Tensor,
        timesteps: Tensor,
        size_bins: Tensor,
    ) -> Tensor:
        """Forward pass with size bin conditioning.

        Args:
            x: Noisy input tensor [B, C, H, W] or [B, C, D, H, W].
            timesteps: Timestep values [B].
            size_bins: Size bin indices [B] (0=small, 1=medium, 2=large, etc.).

        Returns:
            Predicted noise or velocity tensor.
        """
        ...


@runtime_checkable
class PreparedBatch(Protocol):
    """Standardized prepared batch format from TrainingMode.prepare_batch().

    This protocol defines the expected structure of batches after preprocessing.
    All training modes return data in this format for consistent handling.
    """

    @property
    def images(self) -> Tensor | dict[str, Tensor]:
        """Input images to diffuse.

        Returns:
            Single tensor [B, C, H, W] for most modes, or
            dict of tensors for dual mode {'t1_pre': [...], 't1_gd': [...]}.
        """
        ...

    @property
    def labels(self) -> Tensor | None:
        """Conditioning labels (segmentation masks).

        Returns:
            Tensor [B, 1, H, W] for conditional modes, None for unconditional.
        """
        ...

    @property
    def mode_id(self) -> Tensor | None:
        """Mode embedding index for multi-modality.

        Returns:
            Tensor [B] with mode indices, None if not multi-modality mode.
        """
        ...

    @property
    def is_latent(self) -> bool:
        """Whether data is already in latent space.

        Returns:
            True if data comes from latent cache (no encoding needed).
        """
        ...


class DiffusionSpaceProtocol(Protocol):
    """Protocol for pixel/latent space abstractions.

    Defines the interface for encoding/decoding between pixel and
    diffusion (possibly latent) spaces.
    """

    @property
    def scale_factor(self) -> int:
        """Spatial downsampling factor (1 for pixel space, >1 for latent)."""
        ...

    @property
    def latent_channels(self) -> int:
        """Number of channels in diffusion space."""
        ...

    def encode(self, x: Tensor) -> Tensor:
        """Encode tensor from pixel to diffusion space.

        Args:
            x: Pixel-space tensor [B, C, H, W].

        Returns:
            Diffusion-space tensor (possibly downsampled and different channels).
        """
        ...

    def decode(self, x: Tensor) -> Tensor:
        """Decode tensor from diffusion to pixel space.

        Args:
            x: Diffusion-space tensor.

        Returns:
            Pixel-space tensor [B, C, H, W].
        """
        ...

    def encode_batch(
        self, x: Tensor | dict[str, Tensor]
    ) -> Tensor | dict[str, Tensor]:
        """Encode batch (handles both single tensor and dict).

        Args:
            x: Single tensor or dict of tensors in pixel space.

        Returns:
            Encoded tensor(s) in diffusion space.
        """
        ...

    def decode_batch(
        self, x: Tensor | dict[str, Tensor]
    ) -> Tensor | dict[str, Tensor]:
        """Decode batch (handles both single tensor and dict).

        Args:
            x: Single tensor or dict of tensors in diffusion space.

        Returns:
            Decoded tensor(s) in pixel space.
        """
        ...


class TrainingModeProtocol(Protocol):
    """Protocol for training mode implementations.

    Training modes define how data is prepared and formatted for
    different conditioning setups (unconditional, conditional, dual, etc.).
    """

    @property
    def is_conditional(self) -> bool:
        """Whether this mode uses conditioning."""
        ...

    def prepare_batch(
        self, batch: Any, device: Any
    ) -> dict[str, Any]:
        """Prepare raw batch data for training.

        Args:
            batch: Raw batch from dataloader.
            device: Target device for tensors.

        Returns:
            Dict with 'images', 'labels', and other mode-specific keys.
        """
        ...

    def format_model_input(
        self, noisy_images: Tensor, labels_dict: dict[str, Tensor | None]
    ) -> Tensor:
        """Format noisy images and conditioning for model input.

        Args:
            noisy_images: Noised images [B, C, H, W].
            labels_dict: Dict with 'labels' key (may be None).

        Returns:
            Tensor ready for model forward pass.
        """
        ...

    def get_model_config(self) -> dict[str, Any]:
        """Get model architecture configuration.

        Returns:
            Dict with 'in_channels', 'out_channels', etc.
        """
        ...
