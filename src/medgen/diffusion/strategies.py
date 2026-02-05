"""
Diffusion strategy implementations for image generation.

This module provides abstract and concrete implementations of diffusion
strategies including DDPM and Rectified Flow algorithms.
"""
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
from monai.networks.schedulers import DDPMScheduler, RFlowScheduler
from torch import nn
from tqdm import tqdm

from .protocols import DiffusionModel

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
        omega: torch.Tensor | None = None,
        mode_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate samples using the diffusion process.

        Args:
            model: Trained diffusion model.
            noise: Initial noise tensor.
            num_steps: Number of sampling steps.
            device: Computation device.
            use_progress_bars: Whether to show progress bars.
            omega: Optional ScoreAug omega conditioning tensor.
            mode_id: Optional mode ID tensor for multi-modality conditioning.

        Returns:
            Generated image tensor.
        """
        pass


class DDPMStrategy(DiffusionStrategy):
    """Denoising Diffusion Probabilistic Model strategy.

    Implements the DDPM algorithm for noise prediction-based diffusion.
    """

    def setup_scheduler(
        self, num_timesteps: int = 1000, image_size: int = 128, **kwargs
    ) -> DDPMScheduler:
        """Setup DDPM scheduler with cosine schedule.

        Args:
            num_timesteps: Number of diffusion timesteps.
            image_size: Size of input images (unused but kept for interface).
            **kwargs: Ignored (for interface compatibility with RFlowStrategy).

        Returns:
            Configured DDPMScheduler instance.
        """
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps, schedule='cosine'
        )
        return self.scheduler

    def predict_noise_or_velocity(
        self, model: nn.Module, model_input: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """DDPM predicts noise"""
        return model(x=model_input, timesteps=timesteps)

    def compute_target(
        self,
        clean_images: ImageOrDict,
        noise: ImageOrDict,
    ) -> ImageOrDict:
        """DDPM predicts noise, so target is the noise itself."""
        return noise

    def compute_predicted_clean(
        self,
        noisy_images: ImageOrDict,
        prediction: ImageOrDict,
        timesteps: torch.Tensor,
    ) -> ImageOrDict:
        """Reconstruct clean from noise prediction: x_0 = (x_t - sqrt(1-a) * eps) / sqrt(a).

        Args:
            noisy_images: Noisy images at timestep t (x_t).
            prediction: Model noise prediction (epsilon).
            timesteps: Current timesteps (integer indices).

        Returns:
            Predicted clean images (x_0), clamped to [0, 1].
        """
        # Handle dual-image case
        if isinstance(noisy_images, dict):
            keys = list(noisy_images.keys())
            # For dual mode, prediction has 2 channels
            noise_pred_0 = self._slice_channel(prediction, 0, 1)
            noise_pred_1 = self._slice_channel(prediction, 1, 2)

            # Get alpha values
            alphas_cumprod = self.scheduler.alphas_cumprod.to(noisy_images[keys[0]].device)
            alpha_t = self._expand_to_broadcast(alphas_cumprod[timesteps], prediction)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

            return {
                keys[0]: torch.clamp(
                    (noisy_images[keys[0]] - sqrt_one_minus_alpha_t * noise_pred_0) / sqrt_alpha_t,
                    0, 1
                ),
                keys[1]: torch.clamp(
                    (noisy_images[keys[1]] - sqrt_one_minus_alpha_t * noise_pred_1) / sqrt_alpha_t,
                    0, 1
                )
            }
        else:
            # Single-image case
            alphas_cumprod = self.scheduler.alphas_cumprod.to(noisy_images.device)
            alpha_t = self._expand_to_broadcast(alphas_cumprod[timesteps], prediction)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

            return torch.clamp(
                (noisy_images - sqrt_one_minus_alpha_t * prediction) / sqrt_alpha_t,
                0, 1
            )

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target_images: ImageOrDict,
        noise: ImageOrDict,
        noisy_images: ImageOrDict,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, ImageOrDict]:
        """
        Compute DDPM loss

        Works for both 2D (4D tensors) and 3D (5D tensors).

        Args:
            prediction: Model output [B, C, H, W] or [B, C, D, H, W] where C=1 (single) or C=2 (dual)
            target_images: Clean images (tensor or dict)
            noise: Noise added (tensor or dict)
            noisy_images: Noisy images at timestep t (tensor or dict)
            timesteps: Timestep tensor

        Returns:
            (mse_loss, predicted_clean_images)
        """
        # Get target (noise for DDPM)
        target = self.compute_target(target_images, noise)

        # Compute loss
        if isinstance(target, dict):
            keys = list(target.keys())
            noise_pred_0 = self._slice_channel(prediction, 0, 1)
            noise_pred_1 = self._slice_channel(prediction, 1, 2)
            mse_loss_0 = F.mse_loss(noise_pred_0.float(), target[keys[0]].float())
            mse_loss_1 = F.mse_loss(noise_pred_1.float(), target[keys[1]].float())
            mse_loss = (mse_loss_0 + mse_loss_1) / 2
        else:
            mse_loss = F.mse_loss(prediction.float(), target.float())

        # Compute predicted clean images
        predicted_clean = self.compute_predicted_clean(noisy_images, prediction, timesteps)

        return mse_loss, predicted_clean

    def sample_timesteps(
        self,
        images: ImageOrDict,
        curriculum_range: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Sample random timesteps for training.

        Args:
            images: Input images tensor [B, C, H, W] or dict of tensors.
                Used to determine batch size and device.
            curriculum_range: Optional (min_t, max_t) tuple to restrict timestep
                range for curriculum learning. Values in [0, 1] are scaled to
                [0, num_train_timesteps].

        Returns:
            Integer tensor [B] of sampled timesteps in [0, num_train_timesteps).
        """
        # Extract batch size from images
        if isinstance(images, dict):
            batch_size = list(images.values())[0].shape[0]
            device = list(images.values())[0].device
        else:
            batch_size = images.shape[0]
            device = images.device

        if curriculum_range is not None:
            # Sample from restricted timestep range
            min_t, max_t = curriculum_range
            num_steps = self.scheduler.num_train_timesteps
            min_step = int(min_t * num_steps)
            max_step = int(max_t * num_steps)
            return torch.randint(min_step, max_step, (batch_size,), device=device).long()

        return torch.randint(
            0, self.scheduler.num_train_timesteps,
            (batch_size,), device=device
        ).long()

    def generate(
        self,
        model: nn.Module,
        model_input: torch.Tensor,
        num_steps: int,
        device: torch.device,
        use_progress_bars: bool = False,
        omega: torch.Tensor | None = None,
        mode_id: torch.Tensor | None = None,
        size_bins: torch.Tensor | None = None,
        bin_maps: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        cfg_scale_end: float | None = None,
        latent_channels: int = 1,
    ) -> torch.Tensor:
        """
        Generate using DDPM sampling

        Handles both unconditional and conditional generation:
        - Unconditional: model_input is [B, 1, H, W] (just noise)
        - Conditional single: model_input is [B, 2, H, W] (noise + conditioning)
        - Conditional dual: model_input is [B, 3, H, W] (noise_pre + noise_gd + conditioning)

        Args:
            model: Trained diffusion model (may be wrapped with ScoreAug/ModeEmbed).
            model_input: Input tensor with noise and optional conditioning.
            num_steps: Number of sampling steps.
            device: Computation device.
            use_progress_bars: Whether to show progress bars.
            omega: Optional ScoreAug omega conditioning tensor [B, 5].
            mode_id: Optional mode ID tensor [B] for multi-modality conditioning.
            size_bins: Optional size bin conditioning [B, num_bins] for seg_conditioned mode (FiLM).
            bin_maps: Optional spatial bin maps [B, num_bins, ...] for seg_conditioned_input mode.
                      These are concatenated with noise for input channel conditioning.
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance, >1.0 = stronger conditioning).
                       For dynamic CFG, this is the starting scale (at t=T, high noise).
            cfg_scale_end: Optional ending CFG scale (at t=0, low noise). If None, uses constant cfg_scale.
            latent_channels: Number of noise channels (1 for pixel space, 4 for latent space).
        """
        batch_size = model_input.shape[0]

        # Dynamic CFG: interpolate from cfg_scale (at t=T) to cfg_scale_end (at t=0)
        use_dynamic_cfg = cfg_scale_end is not None and cfg_scale_end != cfg_scale

        # Set timesteps for inference
        self.scheduler.set_timesteps(num_inference_steps=num_steps)

        # Parse model input into components
        parsed = self._parse_model_input(model_input, latent_channels=latent_channels)
        noisy_images = parsed.noisy_images
        noisy_pre = parsed.noisy_pre
        noisy_gd = parsed.noisy_gd
        conditioning = parsed.conditioning
        is_dual = parsed.is_dual

        # Prepare CFG context (flags and unconditional tensors)
        cfg_ctx = self._prepare_cfg_context(cfg_scale, size_bins, bin_maps, conditioning, is_dual)

        # Sampling loop
        timesteps = list(self.scheduler.timesteps)
        total_steps = len(timesteps)
        if use_progress_bars:
            timesteps_iter = tqdm(enumerate(timesteps), total=total_steps, desc="DDPM sampling")
        else:
            timesteps_iter = enumerate(timesteps)

        for step_idx, t in timesteps_iter:
            # Compute current CFG scale (dynamic or constant)
            # Note: For single-step (num_steps=1), progress=0 so cfg_scale_end is ignored
            # and only cfg_scale (start value) is used. This is intentional.
            if use_dynamic_cfg:
                progress = step_idx / max(total_steps - 1, 1)
                current_cfg = cfg_scale + progress * (cfg_scale_end - cfg_scale)
            else:
                current_cfg = cfg_scale
            timesteps_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            if is_dual:
                # Dual-image: process each channel through model together
                current_model_input = self._prepare_dual_model_input(noisy_pre, noisy_gd, conditioning)
                noise_pred = self._call_model(model, current_model_input, timesteps_batch, omega, mode_id, size_bins)

                # Split predictions for each channel
                noise_pred_pre, noise_pred_gd = self._split_dual_predictions(noise_pred)

                # Denoise each channel SEPARATELY using scheduler
                noisy_pre, _ = self.scheduler.step(noise_pred_pre, t, noisy_pre)
                noisy_gd, _ = self.scheduler.step(noise_pred_gd, t, noisy_gd)

            else:
                # Single image or unconditional
                # Build base model input (without bin_maps - those are handled separately)
                if conditioning is not None:
                    current_model_input = torch.cat([noisy_images, conditioning], dim=1)
                else:
                    current_model_input = noisy_images

                noise_pred = self._compute_cfg_prediction(
                    model, cfg_ctx, current_cfg, current_model_input, noisy_images,
                    bin_maps, size_bins, timesteps_batch, omega, mode_id
                )

                noisy_images, _ = self.scheduler.step(noise_pred, t, noisy_images)

        # Return final denoised images
        if is_dual:
            return torch.cat([noisy_pre, noisy_gd], dim=1)  # [B, 2, H, W]
        else:
            return noisy_images


class RFlowStrategy(DiffusionStrategy):
    """Rectified Flow algorithm.

    Supports both 2D (images) and 3D (volumes).
    """

    def setup_scheduler(
        self,
        num_timesteps: int = 1000,
        image_size: int = 128,
        depth_size: int | None = None,
        spatial_dims: int = 2,
        use_discrete_timesteps: bool = True,
        sample_method: str = 'logit-normal',
        use_timestep_transform: bool = True,
    ):
        """Setup RFlow scheduler.

        Args:
            num_timesteps: Number of diffusion timesteps (default 1000).
            image_size: Height/width of input.
            depth_size: Depth for 3D volumes (required if spatial_dims=3).
            spatial_dims: Number of spatial dimensions (2 or 3).
            use_discrete_timesteps: Use discrete integer timesteps (default True).
            sample_method: Timestep sampling - 'uniform' or 'logit-normal' (default).
            use_timestep_transform: Apply resolution-based timestep transform (default True).
        """
        self.spatial_dims = spatial_dims

        if spatial_dims == 3:
            if depth_size is None:
                raise ValueError("depth_size required for 3D RFlowStrategy")
            base_numel = image_size * image_size * depth_size
        else:
            base_numel = image_size * image_size

        self.scheduler = RFlowScheduler(
            num_train_timesteps=num_timesteps,
            use_discrete_timesteps=use_discrete_timesteps,
            sample_method=sample_method,
            use_timestep_transform=use_timestep_transform,
            base_img_size_numel=base_numel,
            spatial_dim=spatial_dims
        )
        return self.scheduler

    def predict_noise_or_velocity(
        self, model: nn.Module, model_input: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """RFlow predicts velocity"""
        return model(x=model_input, timesteps=timesteps)

    def compute_target(
        self,
        clean_images: ImageOrDict,
        noise: ImageOrDict,
    ) -> ImageOrDict:
        """RFlow predicts velocity: v = x_0 - x_1 (clean - noise)."""
        if isinstance(clean_images, dict):
            return {k: clean_images[k] - noise[k] for k in clean_images.keys()}
        return clean_images - noise

    def compute_predicted_clean(
        self,
        noisy_images: ImageOrDict,
        prediction: ImageOrDict,
        timesteps: torch.Tensor,
    ) -> ImageOrDict:
        """Reconstruct clean from velocity: x_0 = x_t + t * v.

        Args:
            noisy_images: Noisy images at timestep t (x_t).
            prediction: Model velocity prediction.
            timesteps: Current timesteps (can be continuous or discrete).

        Returns:
            Predicted clean images (x_0), clamped to [0, 1].
        """
        # Get normalized timestep t in [0, 1]
        t = timesteps.float() / self.scheduler.num_train_timesteps

        # Handle dual-image case
        if isinstance(noisy_images, dict):
            keys = list(noisy_images.keys())
            velocity_pred_0 = self._slice_channel(prediction, 0, 1)
            velocity_pred_1 = self._slice_channel(prediction, 1, 2)
            t_expanded = self._expand_to_broadcast(t, prediction)
            return {
                keys[0]: torch.clamp(noisy_images[keys[0]] + t_expanded * velocity_pred_0, 0, 1),
                keys[1]: torch.clamp(noisy_images[keys[1]] + t_expanded * velocity_pred_1, 0, 1)
            }
        else:
            t_expanded = self._expand_to_broadcast(t, prediction)
            return torch.clamp(noisy_images + t_expanded * prediction, 0, 1)

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target_images: ImageOrDict,
        noise: ImageOrDict,
        noisy_images: ImageOrDict,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, ImageOrDict]:
        """
        Compute RFlow loss (velocity prediction)

        Works for both 2D (4D tensors) and 3D (5D tensors).

        Args:
            prediction: Model velocity prediction
            target_images: Clean images (x_0)
            noise: Pure Gaussian noise (x_1)
            noisy_images: Interpolated images at timestep t (x_t)
            timesteps: Timestep tensor

        Returns:
            (mse_loss, predicted_clean_images)
        """
        # Get target (velocity for RFlow)
        target = self.compute_target(target_images, noise)

        # Compute loss
        if isinstance(target, dict):
            keys = list(target.keys())
            velocity_pred_0 = self._slice_channel(prediction, 0, 1)
            velocity_pred_1 = self._slice_channel(prediction, 1, 2)
            mse_loss_0 = F.mse_loss(velocity_pred_0.float(), target[keys[0]].float())
            mse_loss_1 = F.mse_loss(velocity_pred_1.float(), target[keys[1]].float())
            mse_loss = (mse_loss_0 + mse_loss_1) / 2
        else:
            mse_loss = F.mse_loss(prediction.float(), target.float())

        # Compute predicted clean images
        predicted_clean = self.compute_predicted_clean(noisy_images, prediction, timesteps)

        return mse_loss, predicted_clean

    def sample_timesteps(
        self,
        images: ImageOrDict,
        curriculum_range: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Sample timesteps for RFlow training.

        For RFlow with continuous timesteps (use_discrete_timesteps=False),
        samples float timesteps in [0, num_train_timesteps]. For discrete mode,
        samples integers like DDPM.

        Args:
            images: Input images tensor [B, C, H, W] or dict of tensors.
                Used to determine batch size and device.
            curriculum_range: Optional (min_t, max_t) tuple to restrict timestep
                range for curriculum learning. Values in [0, 1] are scaled to
                [0, num_train_timesteps].

        Returns:
            Tensor [B] of sampled timesteps. Float for continuous mode,
            int for discrete mode.
        """
        # Extract batch size and device from images
        if isinstance(images, dict):
            sample_tensor = list(images.values())[0]
        else:
            sample_tensor = images

        if curriculum_range is not None:
            # Sample from restricted timestep range (uniform sampling)
            min_t, max_t = curriculum_range
            batch_size = sample_tensor.shape[0]
            device = sample_tensor.device
            # RFlow uses continuous timesteps in [0, 1], then scales to [0, num_train_timesteps]
            t_float = torch.rand(batch_size, device=device) * (max_t - min_t) + min_t
            t_scaled = t_float * self.scheduler.num_train_timesteps
            # Respect use_discrete_timesteps config
            if self.scheduler.use_discrete_timesteps:
                return t_scaled.long()
            else:
                return t_scaled

        # Default: logit-normal sampling
        return self.scheduler.sample_timesteps(sample_tensor)

    def generate(
        self,
        model: nn.Module,
        model_input: torch.Tensor,
        num_steps: int,
        device: torch.device,
        use_progress_bars: bool = False,
        omega: torch.Tensor | None = None,
        mode_id: torch.Tensor | None = None,
        size_bins: torch.Tensor | None = None,
        bin_maps: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        cfg_scale_end: float | None = None,
        latent_channels: int = 1,
    ) -> torch.Tensor:
        """
        Generate using RFlow sampling

        Works for both 2D and 3D:
        - 2D: model_input is [B, C, H, W]
        - 3D: model_input is [B, C, D, H, W]

        Handles both unconditional and conditional generation:
        - Unconditional: C=1 (just noise)
        - Conditional single: C=2 (noise + conditioning)
        - Conditional dual: C=3 (noise_pre + noise_gd + conditioning)

        Args:
            model: Trained diffusion model (may be wrapped with ScoreAug/ModeEmbed).
            model_input: Input tensor with noise and optional conditioning.
            num_steps: Number of sampling steps.
            device: Computation device.
            use_progress_bars: Whether to show progress bars.
            omega: Optional ScoreAug omega conditioning tensor [B, 5].
            mode_id: Optional mode ID tensor [B] for multi-modality conditioning.
            size_bins: Optional size bin conditioning [B, num_bins] for seg_conditioned mode (FiLM).
            bin_maps: Optional spatial bin maps [B, num_bins, ...] for seg_conditioned_input mode.
                      These are concatenated with noise for input channel conditioning.
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance, >1.0 = stronger conditioning).
                       Works with size_bins, bin_maps, and image conditioning (seg mask).
                       For dynamic CFG, this is the starting scale (at t=T, high noise).
            cfg_scale_end: Optional ending CFG scale (at t=0, low noise). If None, uses constant cfg_scale.
                          Set to 1.0 for "high CFG early, no CFG late" schedule.
            latent_channels: Number of noise channels (1 for pixel space, 4 for latent space).
        """
        # Dynamic CFG: interpolate from cfg_scale (at t=T) to cfg_scale_end (at t=0)
        use_dynamic_cfg = cfg_scale_end is not None and cfg_scale_end != cfg_scale
        batch_size = model_input.shape[0]

        # Calculate numel based on spatial dimensions
        if model_input.dim() == 5:
            # 3D: [B, C, D, H, W]
            input_img_size_numel = model_input.shape[2] * model_input.shape[3] * model_input.shape[4]
        else:
            # 2D: [B, C, H, W]
            input_img_size_numel = model_input.shape[2] * model_input.shape[3]

        # Parse model input into components
        parsed = self._parse_model_input(model_input, latent_channels=latent_channels)
        noisy_images = parsed.noisy_images
        noisy_pre = parsed.noisy_pre
        noisy_gd = parsed.noisy_gd
        conditioning = parsed.conditioning
        is_dual = parsed.is_dual

        # Prepare CFG context (flags and unconditional tensors)
        cfg_ctx = self._prepare_cfg_context(cfg_scale, size_bins, bin_maps, conditioning, is_dual)

        # Setup scheduler
        self.scheduler.set_timesteps(
            num_inference_steps=num_steps,
            device=device,
            input_img_size_numel=input_img_size_numel
        )

        all_next_timesteps = torch.cat((
            self.scheduler.timesteps[1:],
            torch.tensor([0], dtype=self.scheduler.timesteps.dtype, device=device)
        ))

        # Sampling loop
        timestep_pairs = list(zip(self.scheduler.timesteps, all_next_timesteps))
        total_steps = len(timestep_pairs)
        if use_progress_bars:
            timestep_pairs = tqdm(timestep_pairs, desc="RFlow sampling")

        for step_idx, (t, next_t) in enumerate(timestep_pairs):
            # Compute current CFG scale (dynamic or constant)
            # Note: For single-step (num_steps=1), progress=0 so cfg_scale_end is ignored
            # and only cfg_scale (start value) is used. This is intentional.
            if use_dynamic_cfg:
                # Linear interpolation: cfg_scale at step 0, cfg_scale_end at last step
                progress = step_idx / max(total_steps - 1, 1)
                current_cfg = cfg_scale + progress * (cfg_scale_end - cfg_scale)
            else:
                current_cfg = cfg_scale
            timesteps_batch = t.unsqueeze(0).repeat(batch_size).to(device)
            next_timestep = next_t.to(device) if isinstance(next_t, torch.Tensor) else torch.tensor(next_t,
                                                                                                    device=device)

            if is_dual:
                # Dual-image: process each channel through model together
                current_model_input = self._prepare_dual_model_input(noisy_pre, noisy_gd, conditioning)
                velocity_pred = self._call_model(model, current_model_input, timesteps_batch, omega, mode_id, size_bins)

                # Split predictions for each channel (works for both 2D and 3D)
                velocity_pred_pre, velocity_pred_gd = self._split_dual_predictions(velocity_pred)

                # Update each channel SEPARATELY using scheduler
                noisy_pre, _ = self.scheduler.step(velocity_pred_pre, t, noisy_pre, next_timestep)
                noisy_gd, _ = self.scheduler.step(velocity_pred_gd, t, noisy_gd, next_timestep)

            else:
                # Single image or unconditional
                # Build base model input (without bin_maps - those are handled separately)
                if conditioning is not None:
                    current_model_input = torch.cat([noisy_images, conditioning], dim=1)
                else:
                    current_model_input = noisy_images

                velocity_pred = self._compute_cfg_prediction(
                    model, cfg_ctx, current_cfg, current_model_input, noisy_images,
                    bin_maps, size_bins, timesteps_batch, omega, mode_id
                )

                noisy_images, _ = self.scheduler.step(velocity_pred, t, noisy_images, next_timestep)

        # Return final denoised images
        if is_dual:
            return torch.cat([noisy_pre, noisy_gd], dim=1)  # [B, 2, H, W]
        else:
            return noisy_images
