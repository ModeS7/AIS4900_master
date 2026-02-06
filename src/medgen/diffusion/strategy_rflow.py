"""Rectified Flow (RFlow) strategy implementation.

Moved from strategies.py during file split.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from monai.networks.schedulers import RFlowScheduler
from torch import nn
from tqdm import tqdm

from .strategies import DiffusionStrategy, ImageOrDict

if TYPE_CHECKING:
    from .conditioning import ConditioningContext


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

        # Dynamic CFG: interpolate from cfg_scale (at t=T) to cfg_scale_end (at t=0)
        use_dynamic_cfg = conditioning.use_dynamic_cfg
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
