"""DDPM (Denoising Diffusion Probabilistic Models) strategy implementation.

Moved from strategies.py during file split.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from monai.networks.schedulers import DDPMScheduler
from torch import nn
from tqdm import tqdm

from .strategies import DiffusionStrategy, ImageOrDict

if TYPE_CHECKING:
    from .conditioning import ConditioningContext


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
        return self.scheduler  # type: ignore[no-any-return]

    def predict_noise_or_velocity(
        self, model: nn.Module, model_input: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """DDPM predicts noise"""
        return model(x=model_input, timesteps=timesteps)  # type: ignore[no-any-return]

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
            assert isinstance(prediction, torch.Tensor)
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
            assert isinstance(prediction, torch.Tensor)
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
            conditioning: Unified ConditioningContext (preferred). If provided,
                individual params below are ignored.
            omega: [DEPRECATED] ScoreAug omega conditioning tensor [B, 5].
            mode_id: [DEPRECATED] Mode ID tensor [B] for multi-modality conditioning.
            size_bins: [DEPRECATED] Size bin conditioning [B, num_bins] for FiLM.
            bin_maps: [DEPRECATED] Spatial bin maps [B, num_bins, ...] for input conditioning.
            cfg_scale: [DEPRECATED] CFG scale (1.0 = no guidance).
            cfg_scale_end: [DEPRECATED] End CFG scale for dynamic CFG.
            latent_channels: [DEPRECATED] Noise channels (1 for pixel, 4 for latent).
        """
        # Build context from individual params if not provided
        if conditioning is None:
            # Emit deprecation warning if using old API with conditioning params
            has_old_conditioning = any([
                omega is not None,
                mode_id is not None,
                size_bins is not None,
                bin_maps is not None,
            ])
            if has_old_conditioning:
                warnings.warn(
                    "Passing conditioning params individually is deprecated. "
                    "Use conditioning=ConditioningContext(...) instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            from .conditioning import ConditioningContext
            conditioning = ConditioningContext(
                omega=omega,
                mode_id=mode_id,
                size_bins=size_bins,
                bin_maps=bin_maps,
                cfg_scale=cfg_scale,
                cfg_scale_end=cfg_scale_end,
                latent_channels=latent_channels,
            )

        # Extract from conditioning context
        omega = conditioning.omega
        mode_id = conditioning.mode_id
        size_bins = conditioning.size_bins
        bin_maps = conditioning.bin_maps
        cfg_scale = conditioning.cfg_scale
        cfg_scale_end = conditioning.cfg_scale_end
        latent_channels = conditioning.latent_channels

        batch_size = model_input.shape[0]

        # Dynamic CFG: interpolate from cfg_scale (at t=T) to cfg_scale_end (at t=0)
        use_dynamic_cfg = conditioning.use_dynamic_cfg

        # Set timesteps for inference
        self.scheduler.set_timesteps(num_inference_steps=num_steps)

        # Parse model input into components
        parsed = self._parse_model_input(model_input, latent_channels=latent_channels)
        noisy_images = parsed.noisy_images
        noisy_pre = parsed.noisy_pre
        noisy_gd = parsed.noisy_gd
        image_conditioning = parsed.conditioning
        is_dual = parsed.is_dual

        # Prepare CFG context (flags and unconditional tensors)
        cfg_ctx = self._prepare_cfg_context(cfg_scale, size_bins, bin_maps, image_conditioning, is_dual)

        # Sampling loop
        timesteps = list(self.scheduler.timesteps)
        total_steps = len(timesteps)
        if use_progress_bars:
            timesteps_iter = tqdm(enumerate(timesteps), total=total_steps, desc="DDPM sampling")
        else:
            timesteps_iter = enumerate(timesteps)  # type: ignore[assignment]

        for step_idx, t in timesteps_iter:
            # Compute current CFG scale (dynamic or constant)
            # Note: For single-step (num_steps=1), progress=0 so cfg_scale_end is ignored
            # and only cfg_scale (start value) is used. This is intentional.
            if use_dynamic_cfg:
                assert cfg_scale_end is not None
                progress = step_idx / max(total_steps - 1, 1)
                current_cfg = cfg_scale + progress * (cfg_scale_end - cfg_scale)
            else:
                current_cfg = cfg_scale
            timesteps_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            if is_dual:
                # Dual-image: process each channel through model together
                assert noisy_pre is not None
                assert noisy_gd is not None
                assert image_conditioning is not None
                current_model_input = self._prepare_dual_model_input(noisy_pre, noisy_gd, image_conditioning)
                noise_pred = self._call_model(model, current_model_input, timesteps_batch, omega, mode_id, size_bins)

                # Split predictions for each channel
                noise_pred_pre, noise_pred_gd = self._split_dual_predictions(noise_pred)

                # Denoise each channel SEPARATELY using scheduler
                noisy_pre, _ = self.scheduler.step(noise_pred_pre, t, noisy_pre)
                noisy_gd, _ = self.scheduler.step(noise_pred_gd, t, noisy_gd)

            else:
                # Single image or unconditional
                assert noisy_images is not None
                # Build base model input (without bin_maps - those are handled separately)
                if image_conditioning is not None:
                    current_model_input = torch.cat([noisy_images, image_conditioning], dim=1)
                else:
                    current_model_input = noisy_images

                noise_pred = self._compute_cfg_prediction(
                    model, cfg_ctx, current_cfg, current_model_input, noisy_images,
                    bin_maps, size_bins, timesteps_batch, omega, mode_id
                )

                noisy_images, _ = self.scheduler.step(noise_pred, t, noisy_images)

        # Return final denoised images
        if is_dual:
            assert noisy_pre is not None
            assert noisy_gd is not None
            return torch.cat([noisy_pre, noisy_gd], dim=1)  # [B, 2, H, W]
        else:
            assert noisy_images is not None
            return noisy_images
