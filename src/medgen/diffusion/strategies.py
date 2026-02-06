"""
Diffusion strategy implementations for image generation.

This module provides the abstract base class for diffusion strategies
and re-exports concrete implementations (DDPM, RFlow) from submodules.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .protocols import DiffusionModel

if TYPE_CHECKING:
    from .conditioning import ConditioningContext

ImageTensor = torch.Tensor
ImageDict = dict[str, torch.Tensor]
ImageOrDict = ImageTensor | ImageDict


class ParsedModelInput:
    """Container for parsed model input components."""

    def __init__(
        self,
        noisy_images: torch.Tensor | None,
        noisy_pre: torch.Tensor | None,
        noisy_gd: torch.Tensor | None,
        conditioning: torch.Tensor | None,
        is_dual: bool,
    ):
        self.noisy_images = noisy_images
        self.noisy_pre = noisy_pre
        self.noisy_gd = noisy_gd
        self.conditioning = conditioning
        self.is_dual = is_dual


class DiffusionStrategy(ABC):
    """Abstract base class for diffusion algorithms.

    Defines the interface for diffusion strategies including DDPM and
    Rectified Flow. Subclasses must implement all abstract methods.

    Attributes:
        scheduler: The noise scheduler for the diffusion process.
        spatial_dims: Number of spatial dimensions (2 for images, 3 for volumes).
    """

    scheduler: Any
    spatial_dims: int = 2  # Default to 2D

    def _expand_to_broadcast(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Expand timestep tensor to broadcast with image tensor.

        Args:
            t: Timestep tensor [B] or scalar values.
            x: Image tensor [B, C, H, W] or [B, C, D, H, W].

        Returns:
            Expanded tensor that broadcasts with x.
        """
        # Expand to [B, 1, 1, 1] for 4D or [B, 1, 1, 1, 1] for 5D
        num_spatial = x.dim() - 2  # Subtract batch and channel dims
        shape = [-1] + [1] * (num_spatial + 1)  # +1 for channel dim
        return t.view(*shape)

    def _slice_channel(self, tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Slice tensor along channel dimension (dim 1), works for 4D or 5D.

        Args:
            tensor: Input tensor [B, C, ...] with any spatial dims.
            start: Start index for channel slice.
            end: End index for channel slice.

        Returns:
            Sliced tensor.
        """
        return tensor[:, start:end]

    def _prepare_cfg_context(
        self,
        cfg_scale: float,
        size_bins: torch.Tensor | None,
        bin_maps: torch.Tensor | None,
        conditioning: torch.Tensor | None,
        is_dual: bool,
    ) -> dict[str, Any]:
        """Prepare CFG context flags and unconditional tensors.

        Centralizes the common pattern of determining which CFG modes are active
        and creating the corresponding unconditional (zeros) tensors.

        Args:
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance).
            size_bins: Optional size bin conditioning [B, num_bins].
            bin_maps: Optional spatial bin maps [B, num_bins, ...].
            conditioning: Optional image conditioning (e.g., seg mask).
            is_dual: Whether this is dual-image mode (no image CFG for dual).

        Returns:
            Dict with CFG context:
                - use_cfg_size_bins: bool - CFG on size bins
                - use_cfg_bin_maps: bool - CFG on spatial bin maps
                - use_cfg_conditioning: bool - CFG on image conditioning
                - uncond_size_bins: Tensor | None - zeros tensor for uncond size bins
                - uncond_bin_maps: Tensor | None - zeros tensor for uncond bin maps
                - uncond_conditioning: Tensor | None - zeros tensor for uncond conditioning
        """
        ctx: dict[str, Any] = {
            'use_cfg_size_bins': cfg_scale > 1.0 and size_bins is not None,
            'use_cfg_bin_maps': cfg_scale > 1.0 and bin_maps is not None,
            'use_cfg_conditioning': cfg_scale > 1.0 and conditioning is not None and not is_dual,
            'uncond_size_bins': None,
            'uncond_bin_maps': None,
            'uncond_conditioning': None,
        }

        if ctx['use_cfg_size_bins']:
            ctx['uncond_size_bins'] = torch.zeros_like(size_bins)
        if ctx['use_cfg_bin_maps']:
            ctx['uncond_bin_maps'] = torch.zeros_like(bin_maps)
        if ctx['use_cfg_conditioning']:
            ctx['uncond_conditioning'] = torch.zeros_like(conditioning)

        return ctx

    def _apply_cfg_guidance(
        self,
        pred_uncond: torch.Tensor,
        pred_cond: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Apply classifier-free guidance formula.

        CFG: pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)

        Args:
            pred_uncond: Unconditional model prediction.
            pred_cond: Conditional model prediction.
            cfg_scale: Guidance scale (>1.0 for stronger conditioning).

        Returns:
            Guided prediction.
        """
        return pred_uncond + cfg_scale * (pred_cond - pred_uncond)

    def _compute_cfg_prediction(
        self,
        model: DiffusionModel,
        cfg_ctx: dict[str, Any],
        current_cfg: float,
        current_model_input: torch.Tensor,
        noisy_images: torch.Tensor,
        bin_maps: torch.Tensor | None,
        size_bins: torch.Tensor | None,
        timesteps_batch: torch.Tensor,
        omega: torch.Tensor | None,
        mode_id: torch.Tensor | None,
    ) -> torch.Tensor:
        """Unified CFG computation shared by DDPM and RFlow.

        Handles all CFG modes: bin_maps, size_bins, and image conditioning.
        Returns the prediction (noise for DDPM, velocity for RFlow) after
        applying appropriate CFG branching and model calls.

        Note: .clone() calls are required to prevent CUDA graph caching issues.

        Args:
            model: The diffusion model.
            cfg_ctx: CFG context from _prepare_cfg_context().
            current_cfg: Current CFG scale for this timestep.
            current_model_input: Model input (noisy + conditioning).
            noisy_images: Noisy images without conditioning.
            bin_maps: Optional spatial bin maps [B, num_bins, ...].
            size_bins: Optional size bin conditioning [B, num_bins].
            timesteps_batch: Timestep tensor.
            omega: Optional ScoreAug omega parameters.
            mode_id: Optional mode ID for multi-modality.

        Returns:
            Model prediction tensor.
        """
        use_cfg_bin_maps = cfg_ctx['use_cfg_bin_maps']
        use_cfg_size_bins = cfg_ctx['use_cfg_size_bins']
        use_cfg_conditioning = cfg_ctx['use_cfg_conditioning']
        uncond_size_bins = cfg_ctx['uncond_size_bins']
        uncond_bin_maps = cfg_ctx['uncond_bin_maps']
        uncond_conditioning = cfg_ctx['uncond_conditioning']

        if use_cfg_bin_maps:
            # CFG for bin_maps input conditioning (seg_conditioned_input mode)
            input_cond = torch.cat([noisy_images, bin_maps], dim=1)
            input_uncond = torch.cat([noisy_images, uncond_bin_maps], dim=1)
            pred_cond = self._call_model(
                model, input_cond, timesteps_batch, omega, mode_id, None
            ).clone()  # Clone to prevent CUDA graph issues
            pred_uncond = self._call_model(
                model, input_uncond, timesteps_batch, omega, mode_id, None
            )
            return self._apply_cfg_guidance(pred_uncond, pred_cond, current_cfg)
        elif bin_maps is not None:
            # bin_maps provided but no CFG - just concatenate and call model
            input_with_bin_maps = torch.cat([noisy_images, bin_maps], dim=1)
            return self._call_model(
                model, input_with_bin_maps, timesteps_batch, omega, mode_id, None
            )
        elif use_cfg_size_bins:
            # CFG for size_bins conditioning (FiLM mode)
            pred_cond = self._call_model(
                model, current_model_input, timesteps_batch, omega, mode_id, size_bins
            ).clone()  # Clone to prevent CUDA graph issues
            pred_uncond = self._call_model(
                model, current_model_input, timesteps_batch, omega, mode_id, uncond_size_bins
            )
            return self._apply_cfg_guidance(pred_uncond, pred_cond, current_cfg)
        elif use_cfg_conditioning:
            # CFG for image conditioning (seg mask)
            uncond_model_input = torch.cat([noisy_images, uncond_conditioning], dim=1)
            pred_cond = self._call_model(
                model, current_model_input, timesteps_batch, omega, mode_id, size_bins
            ).clone()  # Clone to prevent CUDA graph issues
            pred_uncond = self._call_model(
                model, uncond_model_input, timesteps_batch, omega, mode_id, size_bins
            )
            return self._apply_cfg_guidance(pred_uncond, pred_cond, current_cfg)
        else:
            return self._call_model(
                model, current_model_input, timesteps_batch, omega, mode_id, size_bins
            )

    def _prepare_dual_model_input(
        self,
        noisy_pre: torch.Tensor,
        noisy_gd: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate dual-image components for model input.

        Args:
            noisy_pre: Noisy pre-contrast image [B, 1, ...].
            noisy_gd: Noisy post-gadolinium image [B, 1, ...].
            conditioning: Conditioning tensor [B, 1, ...].

        Returns:
            Concatenated model input [B, 3, ...].
        """
        return torch.cat([noisy_pre, noisy_gd, conditioning], dim=1)

    def _split_dual_predictions(
        self,
        prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split dual predictions into pre and gd channels.

        Args:
            prediction: Model prediction [B, 2, ...].

        Returns:
            Tuple of (pred_pre, pred_gd), each [B, 1, ...].
        """
        return prediction[:, 0:1], prediction[:, 1:2]

    def _parse_model_input(
        self, model_input: torch.Tensor, latent_channels: int = 1
    ) -> ParsedModelInput:
        """Parse model input into components based on channel count.

        Works for both 4D (2D images) and 5D (3D volumes).
        Supports both pixel-space (1 channel noise) and latent-space (multi-channel noise).

        Args:
            model_input: Input tensor with noise and optional conditioning.
            latent_channels: Number of channels for noise in latent space (default 1 for pixel).
                For latent diffusion with 4-channel VAE, use latent_channels=4.

        Returns:
            ParsedModelInput with extracted components.

        Raises:
            ValueError: If channel count is unexpected.
        """
        num_channels = model_input.shape[1]

        # Latent space: multi-channel noise
        if latent_channels > 1:
            if num_channels == latent_channels:
                # Latent unconditional: just noise
                return ParsedModelInput(
                    noisy_images=model_input,
                    noisy_pre=None,
                    noisy_gd=None,
                    conditioning=None,
                    is_dual=False,
                )
            elif num_channels == latent_channels * 2:
                # Latent conditional (e.g., bravo_seg_cond): [noise, conditioning]
                return ParsedModelInput(
                    noisy_images=self._slice_channel(model_input, 0, latent_channels),
                    noisy_pre=None,
                    noisy_gd=None,
                    conditioning=self._slice_channel(model_input, latent_channels, num_channels),
                    is_dual=False,
                )
            else:
                raise ValueError(
                    f"Unexpected latent channels: {num_channels} (latent_channels={latent_channels})"
                )

        # Pixel space: 1 channel noise
        if num_channels == 1:
            # Unconditional: just noise
            return ParsedModelInput(
                noisy_images=model_input,
                noisy_pre=None,
                noisy_gd=None,
                conditioning=None,
                is_dual=False,
            )
        elif num_channels == 2:
            # Conditional single: [noise, conditioning]
            return ParsedModelInput(
                noisy_images=self._slice_channel(model_input, 0, 1),
                noisy_pre=None,
                noisy_gd=None,
                conditioning=self._slice_channel(model_input, 1, 2),
                is_dual=False,
            )
        elif num_channels == 3:
            # Conditional dual: [noise_pre, noise_gd, conditioning]
            return ParsedModelInput(
                noisy_images=None,
                noisy_pre=self._slice_channel(model_input, 0, 1),
                noisy_gd=self._slice_channel(model_input, 1, 2),
                conditioning=self._slice_channel(model_input, 2, 3),
                is_dual=True,
            )
        elif num_channels > 3:
            # Multi-channel conditioning: [noise (1 ch), conditioning (remaining)]
            # Handles seg_conditioned_input with variable num_bins
            return ParsedModelInput(
                noisy_images=self._slice_channel(model_input, 0, 1),
                noisy_pre=None,
                noisy_gd=None,
                conditioning=self._slice_channel(model_input, 1, num_channels),
                is_dual=False,
            )
        else:
            raise ValueError(f"Unexpected number of channels: {num_channels}")

    def _call_model(
        self,
        model: DiffusionModel,
        model_input: torch.Tensor,
        timesteps: torch.Tensor,
        omega: torch.Tensor | None,
        mode_id: torch.Tensor | None,
        size_bins: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Call model with appropriate arguments based on conditioning.

        Args:
            model: Diffusion model conforming to DiffusionModel protocol.
                May be wrapped with ScoreAug, ModeEmbed, or SizeBin conditioning.
            model_input: Formatted input tensor.
            timesteps: Current timesteps.
            omega: Optional ScoreAug omega conditioning.
            mode_id: Optional mode ID for multi-modality.
            size_bins: Optional size bin conditioning [B, num_bins] for seg_conditioned mode.

        Returns:
            Model prediction (noise or velocity).
        """
        with torch.no_grad():
            # Check if model is SizeBinModelWrapper (has size_bins parameter)
            if size_bins is not None and hasattr(model, 'size_bin_time_embed'):
                return model(model_input, timesteps=timesteps, size_bins=size_bins)
            elif omega is not None or mode_id is not None:
                return model(model_input, timesteps=timesteps, omega=omega, mode_id=mode_id)
            else:
                return model(x=model_input, timesteps=timesteps)

    @abstractmethod
    def setup_scheduler(self, num_timesteps: int, image_size: int) -> Any:
        """Setup the noise scheduler.

        Args:
            num_timesteps: Number of diffusion timesteps.
            image_size: Size of input images (assumes square).

        Returns:
            Configured scheduler instance.
        """
        pass

    def add_noise(
        self, clean_images: ImageOrDict, noise: ImageOrDict, timesteps: torch.Tensor
    ) -> ImageOrDict:
        """Add noise to clean images at specified timesteps.

        Uses the scheduler's add_noise method. Handles both single-image
        (tensor) and dual-image (dict) formats.

        Args:
            clean_images: Clean images (tensor or dict for dual-image mode).
            noise: Noise to add (same format as clean_images).
            timesteps: Diffusion timesteps tensor.

        Returns:
            Noisy images in same format as input.

        Raises:
            TypeError: If noise is not a dict when clean_images is a dict.
            ValueError: If dict keys don't match between clean_images and noise.
        """
        if isinstance(clean_images, dict):
            if not isinstance(noise, dict):
                raise TypeError(f"noise must be dict when clean_images is dict, got {type(noise).__name__}")
            if set(clean_images.keys()) != set(noise.keys()):
                raise ValueError(f"Key mismatch: images={set(clean_images.keys())}, noise={set(noise.keys())}")
            return {
                key: self.scheduler.add_noise(clean_images[key], noise[key], timesteps)
                for key in clean_images.keys()
            }
        else:
            return self.scheduler.add_noise(clean_images, noise, timesteps)

    @abstractmethod
    def predict_noise_or_velocity(
        self, model: nn.Module, model_input: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Get model prediction (noise for DDPM, velocity for RFlow).

        Args:
            model: Diffusion model.
            model_input: Formatted model input tensor.
            timesteps: Current timesteps.

        Returns:
            Model prediction tensor.
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        prediction: torch.Tensor,
        target_images: ImageOrDict,
        noise: ImageOrDict,
        noisy_images: ImageOrDict,
        timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, ImageOrDict]:
        """Compute loss and predicted clean images.

        Args:
            prediction: Model output (can be multi-channel).
            target_images: Clean target images (tensor or dict for dual-image).
            noise: Noise that was added (same format as target_images).
            noisy_images: Noisy images at timestep t (same format as target_images).
            timesteps: Diffusion timesteps.

        Returns:
            Tuple of (mse_loss, predicted_clean_images) where
            predicted_clean_images matches the format of target_images.
        """
        pass

    @abstractmethod
    def sample_timesteps(
        self,
        images: ImageOrDict,
        curriculum_range: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Sample timesteps for training.

        Args:
            images: Image tensor or dict of tensors (for batch size/device).
            curriculum_range: Optional (min_t, max_t) tuple for curriculum learning.
                If provided, samples from restricted timestep range.

        Returns:
            Sampled timesteps tensor.
        """
        pass

    @abstractmethod
    def compute_target(
        self,
        clean_images: ImageOrDict,
        noise: ImageOrDict,
    ) -> ImageOrDict:
        """Compute training target from clean images and noise.

        Args:
            clean_images: Clean images (x_0).
            noise: Gaussian noise (x_1 for RFlow, epsilon for DDPM).

        Returns:
            Target for loss computation (velocity for RFlow, noise for DDPM).
        """
        pass

    @abstractmethod
    def compute_predicted_clean(
        self,
        noisy_images: ImageOrDict,
        prediction: ImageOrDict,
        timesteps: torch.Tensor,
    ) -> ImageOrDict:
        """Compute predicted clean image from model output.

        Args:
            noisy_images: Noisy images at timestep t (x_t).
            prediction: Model output (velocity or noise prediction).
            timesteps: Current timesteps.

        Returns:
            Predicted clean images (x_0).
        """
        pass

    @abstractmethod
    def generate(
        self,
        model: nn.Module,
        noise: torch.Tensor,
        num_steps: int,
        device: torch.device,
        use_progress_bars: bool = False,
        # NEW: Unified conditioning context
        conditioning: ConditioningContext | None = None,
        # DEPRECATED: Individual params (kept for backward compat)
        omega: torch.Tensor | None = None,
        mode_id: torch.Tensor | None = None,
        size_bins: torch.Tensor | None = None,
        bin_maps: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        cfg_scale_end: float | None = None,
        latent_channels: int = 1,
    ) -> torch.Tensor:
        """Generate samples using the diffusion process.

        Args:
            model: Trained diffusion model.
            noise: Initial noise tensor.
            num_steps: Number of sampling steps.
            device: Computation device.
            use_progress_bars: Whether to show progress bars.
            conditioning: Unified ConditioningContext (preferred). If provided,
                individual params below are ignored.
            omega: [DEPRECATED] ScoreAug omega conditioning tensor.
            mode_id: [DEPRECATED] Mode ID tensor for multi-modality conditioning.
            size_bins: [DEPRECATED] Size bin conditioning [B, num_bins] for FiLM.
            bin_maps: [DEPRECATED] Spatial bin maps for input conditioning.
            cfg_scale: [DEPRECATED] CFG scale (1.0 = no guidance).
            cfg_scale_end: [DEPRECATED] End CFG scale for dynamic CFG.
            latent_channels: [DEPRECATED] Noise channels (1=pixel, 4=latent).

        Returns:
            Generated image tensor.

        Note:
            The individual conditioning parameters (omega, mode_id, etc.) are
            deprecated. Use conditioning=ConditioningContext(...) instead.
        """
        pass


# =============================================================================
# Re-exports from submodules for backward compatibility
# =============================================================================

from .strategy_ddpm import DDPMStrategy  # noqa: E402, F401
from .strategy_rflow import RFlowStrategy  # noqa: E402, F401
