"""Rectified Flow (RFlow) strategy implementation.

Moved from strategies.py during file split.
"""
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from monai.networks.schedulers import RFlowScheduler
from torch import nn
from tqdm import tqdm

from .strategies import DiffusionStrategy, ImageOrDict

if TYPE_CHECKING:
    from .conditioning import ConditioningContext

logger = logging.getLogger(__name__)


class RFlowStrategy(DiffusionStrategy):
    """Rectified Flow algorithm.

    Supports both 2D (images) and 3D (volumes).

    Attributes:
        ode_solver: ODE solver for generation. 'euler' uses built-in loop,
            others use torchdiffeq. Fixed-step: midpoint, heun2, heun3, rk4.
            Adaptive: fehlberg2, bosh3, dopri5, dopri8.
        ode_atol: Absolute tolerance for adaptive solvers.
        ode_rtol: Relative tolerance for adaptive solvers.
    """

    ADAPTIVE_SOLVERS = frozenset({'fehlberg2', 'bosh3', 'dopri5', 'dopri8', 'adaptive_heun'})

    ode_solver: str = 'euler'
    ode_atol: float = 1e-5
    ode_rtol: float = 1e-5
    prediction_type: str = 'velocity'  # 'velocity' or 'sample' (x₀)

    def setup_scheduler(
        self,
        num_timesteps: int = 1000,
        image_size: int = 128,
        depth_size: int | None = None,
        spatial_dims: int = 2,
        use_discrete_timesteps: bool = True,
        sample_method: str = 'logit-normal',
        use_timestep_transform: bool = True,
        prediction_type: str = 'velocity',
        **kwargs,
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
            prediction_type: 'velocity' (default) or 'sample' (predict x₀ directly).
            **kwargs: Ignored (for interface compatibility with DDPMStrategy).
        """
        self.prediction_type = prediction_type
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

    def _to_velocity(
        self, model_output: torch.Tensor, noisy: torch.Tensor, timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Convert model output to velocity if prediction_type is 'sample'.

        For velocity prediction: no-op.
        For sample (x₀) prediction: v = (x₀ - x_t) / t_norm.
        """
        if self.prediction_type != 'sample':
            return model_output
        t_norm = timesteps.float() / self.scheduler.num_train_timesteps
        t_expanded = self._expand_to_broadcast(t_norm, model_output).clamp(min=1e-5)
        return (model_output - noisy) / t_expanded

    def predict_noise_or_velocity(
        self, model: nn.Module, model_input: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """RFlow model forward pass (predicts velocity or x₀ depending on prediction_type)."""
        return model(x=model_input, timesteps=timesteps)  # type: ignore[no-any-return]

    def compute_target(
        self,
        clean_images: ImageOrDict,
        noise: ImageOrDict,
    ) -> ImageOrDict:
        """RFlow target: velocity v = x_0 - x_1, or x₀ directly."""
        if self.prediction_type == 'sample':
            return clean_images
        if isinstance(clean_images, dict):
            assert isinstance(noise, dict)
            return {k: clean_images[k] - noise[k] for k in clean_images.keys()}
        assert isinstance(noise, torch.Tensor)
        return clean_images - noise

    def compute_predicted_clean(
        self,
        noisy_images: ImageOrDict,
        prediction: ImageOrDict,
        timesteps: torch.Tensor,
    ) -> ImageOrDict:
        """Reconstruct clean images from model prediction.

        For velocity prediction: x_0 = x_t + t * v.
        For sample prediction: x_0 = prediction (model output IS x₀).

        Args:
            noisy_images: Noisy images at timestep t (x_t).
            prediction: Model prediction (velocity or x₀).
            timesteps: Current timesteps (can be continuous or discrete).

        Returns:
            Predicted clean images (x_0). Not clamped — values may be outside
            [0, 1] for latent/wavelet space or due to prediction error.
        """
        if self.prediction_type == 'sample':
            # Model directly predicts x₀
            if isinstance(noisy_images, dict):
                keys = list(noisy_images.keys())
                assert isinstance(prediction, torch.Tensor)
                return {
                    keys[0]: self._slice_channel(prediction, 0, 1),
                    keys[1]: self._slice_channel(prediction, 1, 2),
                }
            return prediction

        # Velocity: x₀ = x_t + t * v
        t = timesteps.float() / self.scheduler.num_train_timesteps

        # Handle dual-image case
        if isinstance(noisy_images, dict):
            keys = list(noisy_images.keys())
            assert isinstance(prediction, torch.Tensor)
            velocity_pred_0 = self._slice_channel(prediction, 0, 1)
            velocity_pred_1 = self._slice_channel(prediction, 1, 2)
            t_expanded = self._expand_to_broadcast(t, prediction)
            return {
                keys[0]: noisy_images[keys[0]] + t_expanded * velocity_pred_0,
                keys[1]: noisy_images[keys[1]] + t_expanded * velocity_pred_1,
            }
        else:
            assert isinstance(prediction, torch.Tensor)
            t_expanded = self._expand_to_broadcast(t, prediction)
            return noisy_images + t_expanded * prediction

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
                return t_scaled.long()  # type: ignore[no-any-return]
            else:
                return t_scaled  # type: ignore[no-any-return]

        # Default: logit-normal sampling
        return self.scheduler.sample_timesteps(sample_tensor)  # type: ignore[no-any-return]

    def _generate_torchdiffeq(
        self,
        model: nn.Module,
        parsed: Any,  # ParsedModelInput
        cfg_ctx: dict[str, Any],
        conditioning: ConditioningContext,
        num_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate samples using torchdiffeq ODE solvers.

        Reparameterizes the RFlow ODE as dx/ds = v(x, T*(1-s)) where s in [0,1],
        s=0 is noise, s=1 is clean data. T = num_train_timesteps.

        Args:
            model: Trained diffusion model.
            parsed: ParsedModelInput from _parse_model_input().
            cfg_ctx: CFG context from _prepare_cfg_context().
            conditioning: ConditioningContext with all generation params.
            num_steps: Number of sampling steps (for fixed-step methods).
            device: Computation device.

        Returns:
            Generated image tensor.
        """
        try:
            from torchdiffeq import odeint
        except ImportError as err:
            raise ImportError(
                f"torchdiffeq is required for ode_solver='{self.ode_solver}'. "
                "Install it with: pip install torchdiffeq"
            ) from err

        T = self.scheduler.num_train_timesteps
        omega = conditioning.omega
        mode_id = conditioning.mode_id
        size_bins = conditioning.size_bins
        bin_maps = conditioning.bin_maps
        cfg_scale = conditioning.cfg_scale
        cfg_scale_end = conditioning.cfg_scale_end
        use_dynamic_cfg = conditioning.use_dynamic_cfg
        batch_size = parsed.noisy_images.shape[0] if parsed.noisy_images is not None else parsed.noisy_pre.shape[0]

        is_dual = parsed.is_dual
        image_conditioning = parsed.conditioning

        if is_dual:
            assert parsed.noisy_pre is not None and parsed.noisy_gd is not None
            assert image_conditioning is not None
            # State vector: concatenate [pre, gd] along channel dim
            y0 = torch.cat([parsed.noisy_pre, parsed.noisy_gd], dim=1)  # [B, 2, ...]
        else:
            assert parsed.noisy_images is not None
            y0 = parsed.noisy_images  # [B, C, ...]

        def ode_fn(s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            """ODE function: dx/ds = v(x, t) where t = T*(1-s).

            The RFlow ODE in model timestep space: x_{next} = x + v * (t - t_next) / T.
            Reparameterized with s = 1 - t/T (s in [0,1], noise->clean):
                dx/ds = v(x, T*(1-s))

            Args:
                s: Current ODE time in [0, 1] (scalar tensor).
                x: Current state.

            Returns:
                dx/ds = v(x, t).
            """
            # Convert s -> t (model timestep space)
            t_val = T * (1.0 - s.item())
            timesteps_batch = torch.full((batch_size,), t_val, device=device)

            # Compute current CFG scale
            if use_dynamic_cfg:
                assert cfg_scale_end is not None
                progress = s.item()  # s=0 -> start, s=1 -> end
                current_cfg = cfg_scale + progress * (cfg_scale_end - cfg_scale)
            else:
                current_cfg = cfg_scale

            if is_dual:
                # Split state vector back into pre and gd
                noisy_pre = x[:, 0:1]
                noisy_gd = x[:, 1:2]
                current_model_input = self._prepare_dual_model_input(
                    noisy_pre, noisy_gd, image_conditioning
                )
                model_pred = self._call_model(
                    model, current_model_input, timesteps_batch, omega, mode_id, size_bins
                )
                return self._to_velocity(model_pred, x, timesteps_batch)
            else:
                # Build model input
                if image_conditioning is not None:
                    current_model_input = torch.cat([x, image_conditioning], dim=1)
                else:
                    current_model_input = x

                model_pred = self._compute_cfg_prediction(
                    model, cfg_ctx, current_cfg, current_model_input, x,
                    bin_maps, size_bins, timesteps_batch, omega, mode_id
                )
                return self._to_velocity(model_pred, x, timesteps_batch)

        # Build time span and solver options
        t_span = torch.tensor([0.0, 1.0], device=device)
        solver_kwargs: dict[str, Any] = {'method': self.ode_solver}

        if self.ode_solver in self.ADAPTIVE_SOLVERS:
            solver_kwargs['atol'] = self.ode_atol
            solver_kwargs['rtol'] = self.ode_rtol
        else:
            # Fixed-step methods: step_size = 1/num_steps
            solver_kwargs['options'] = {'step_size': 1.0 / num_steps}

        logger.debug(
            "ODE generation: solver=%s, steps=%s",
            self.ode_solver,
            'adaptive' if self.ode_solver in self.ADAPTIVE_SOLVERS else num_steps,
        )

        # Solve ODE: y0 at s=0 (noise) -> y1 at s=1 (clean)
        solution = odeint(ode_fn, y0, t_span, **solver_kwargs)
        result = solution[-1]  # Final state at s=1

        # Return result
        if is_dual:
            return result  # [B, 2, ...]
        else:
            return result

    def _generate_diffrs(
        self,
        model: nn.Module,
        parsed: Any,  # ParsedModelInput
        cfg_ctx: dict[str, Any],
        conditioning: ConditioningContext,
        num_steps: int,
        device: torch.device,
        diffrs_discriminator: Any,  # DiffRSDiscriminator
        diffrs_config: dict[str, Any],
    ) -> torch.Tensor:
        """Generate samples using DiffRS (Diffusion Rejection Sampling).

        Wraps the normal Euler sampling loop with a discriminator that
        rejects bad intermediate samples and retries with new noise.

        Args:
            model: Trained diffusion model.
            parsed: ParsedModelInput from _parse_model_input().
            cfg_ctx: CFG context from _prepare_cfg_context().
            conditioning: ConditioningContext with all generation params.
            num_steps: Number of sampling steps.
            device: Computation device.
            diffrs_discriminator: DiffRSDiscriminator instance.
            diffrs_config: Dict with DiffRS hyperparameters:
                rej_percentile, backsteps, max_iter, iter_warmup.

        Returns:
            Generated image tensor.
        """
        from .diffrs import (
            diffrs_sampling_loop,
            estimate_adaptive_thresholds,
        )

        if parsed.is_dual:
            raise NotImplementedError("DiffRS does not support dual-image mode yet")

        if cfg_ctx.get('cfg_scale', 1.0) > 1.0:
            raise NotImplementedError(
                "DiffRS does not support CFG (cfg_scale > 1.0) yet. "
                "The sampling loop calls model() directly, bypassing "
                "_compute_cfg_prediction(). Set cfg_scale=1.0 or implement "
                "CFG support in diffrs.diffrs_sampling_loop()."
            )

        assert parsed.noisy_images is not None
        noise = parsed.noisy_images
        conditioning_input = parsed.conditioning  # May be None for unconditional

        rej_percentile = diffrs_config.get('rej_percentile', 0.75)
        backsteps = diffrs_config.get('backsteps', 1)
        max_iter = diffrs_config.get('max_iter', 999999)
        iter_warmup = diffrs_config.get('iter_warmup', 10)

        # Estimate adaptive thresholds (cached per session)
        if not hasattr(self, '_diffrs_adaptive_cache'):
            self._diffrs_adaptive_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        cache_key = f"{num_steps}_{rej_percentile}_{iter_warmup}"
        if cache_key not in self._diffrs_adaptive_cache:
            logger.info(
                "DiffRS: estimating adaptive thresholds "
                "(warmup=%d, percentile=%.2f)...",
                iter_warmup, rej_percentile,
            )
            adaptive, adaptive2 = estimate_adaptive_thresholds(
                strategy=self,
                model=model,
                discriminator=diffrs_discriminator,
                sample_shape=noise.shape,
                num_steps=num_steps,
                device=device,
                iter_warmup=iter_warmup,
                rej_percentile=rej_percentile,
                conditioning=conditioning,
                conditioning_input=conditioning_input,
            )
            self._diffrs_adaptive_cache[cache_key] = (adaptive, adaptive2)
        else:
            logger.info("DiffRS: using cached adaptive thresholds")

        adaptive, adaptive2 = self._diffrs_adaptive_cache[cache_key]

        logger.info(
            "DiffRS sampling: steps=%d, backsteps=%d, max_iter=%d",
            num_steps, backsteps, max_iter,
        )

        return diffrs_sampling_loop(
            strategy=self,
            model=model,
            discriminator=diffrs_discriminator,
            noise=noise,
            num_steps=num_steps,
            device=device,
            adaptive=adaptive,
            adaptive2=adaptive2,
            conditioning=conditioning,
            conditioning_input=conditioning_input,
            backsteps=backsteps,
            max_iter=max_iter,
        )

    def _generate_restart(
        self,
        model: nn.Module,
        parsed: Any,  # ParsedModelInput
        cfg_ctx: dict[str, Any],
        conditioning: ConditioningContext,
        num_steps: int,
        device: torch.device,
        restart_config: dict[str, Any],
    ) -> torch.Tensor:
        """Generate samples using Restart Sampling.

        Alternates between forward noise injection and backward ODE within a
        restart interval [tmin, tmax] to contract accumulated discretization
        errors. No auxiliary model needed — purely algorithmic improvement.

        Paper: "Restart Sampling for Improving Generative Processes" (NeurIPS 2023)

        Args:
            model: Trained diffusion model.
            parsed: ParsedModelInput from _parse_model_input().
            cfg_ctx: CFG context from _prepare_cfg_context().
            conditioning: ConditioningContext with all generation params.
            num_steps: Number of main sampling steps.
            device: Computation device.
            restart_config: Dict with restart hyperparameters:
                tmin: float — restart interval start (fraction of T, e.g. 0.1)
                tmax: float — restart interval end (fraction of T, e.g. 0.33)
                K: int — number of restart iterations
                n_restart: int — Euler steps per restart backward pass

        Returns:
            Generated image tensor.
        """
        if parsed.is_dual:
            raise NotImplementedError("Restart sampling does not support dual-image mode")

        assert parsed.noisy_images is not None
        noisy_images = parsed.noisy_images
        image_conditioning = parsed.conditioning

        # Extract restart hyperparameters
        T = float(self.scheduler.num_train_timesteps)
        tmin_frac = restart_config.get('tmin', 0.1)
        tmax_frac = restart_config.get('tmax', 0.33)
        K = restart_config.get('K', 2)
        n_restart = restart_config.get('n_restart', 5)

        tmin = tmin_frac * T  # e.g. 100.0
        tmax = tmax_frac * T  # e.g. 330.0

        # Extract conditioning params
        omega = conditioning.omega
        mode_id = conditioning.mode_id
        size_bins = conditioning.size_bins
        bin_maps = conditioning.bin_maps
        cfg_scale = conditioning.cfg_scale
        batch_size = noisy_images.shape[0]

        # Compute numel for scheduler
        if noisy_images.dim() == 5:
            input_img_size_numel = noisy_images.shape[2] * noisy_images.shape[3] * noisy_images.shape[4]
        else:
            input_img_size_numel = noisy_images.shape[2] * noisy_images.shape[3]

        # Setup main scheduler timesteps
        self.scheduler.set_timesteps(
            num_inference_steps=num_steps, device=device,
            input_img_size_numel=input_img_size_numel,
        )

        all_next_timesteps = torch.cat((
            self.scheduler.timesteps[1:],
            torch.tensor([0], dtype=self.scheduler.timesteps.dtype, device=device),
        ))

        # Snap tmin to nearest main timestep
        main_timesteps = self.scheduler.timesteps.float()
        # Snap tmin to nearest main timestep that is BELOW tmax
        # (With coarse grids, the nearest timestep could exceed tmax, which is invalid)
        valid_mask = main_timesteps < tmax
        if not valid_mask.any():
            logger.warning(
                "No main timestep below tmax=%.1f — skipping restart, running plain Euler",
                tmax,
            )
            # Fall through to plain Euler (restart_trigger_idx = len-1 means no restart)
            tmin_idx = len(main_timesteps) - 1
            tmin_snapped = main_timesteps[tmin_idx].item()
            K = 0  # Disable restart iterations
        else:
            candidates = main_timesteps.clone()
            candidates[~valid_mask] = float('inf')
            tmin_idx = (candidates - tmin).abs().argmin().item()
            tmin_snapped = main_timesteps[tmin_idx].item()

        logger.info(
            "Restart sampling: tmin=%.1f (snapped from %.1f), tmax=%.1f, K=%d, n_restart=%d",
            tmin_snapped, tmin, tmax, K, n_restart,
        )

        def _euler_step(x: torch.Tensor, t_val: torch.Tensor, next_t_val: torch.Tensor) -> torch.Tensor:
            """Single Euler step with CFG support."""
            timesteps_batch = t_val.unsqueeze(0).repeat(batch_size).to(device)
            next_timestep = next_t_val.to(device)

            if image_conditioning is not None:
                current_model_input = torch.cat([x, image_conditioning], dim=1)
            else:
                current_model_input = x

            model_pred = self._compute_cfg_prediction(
                model, cfg_ctx, cfg_scale, current_model_input, x,
                bin_maps, size_bins, timesteps_batch, omega, mode_id,
            )

            velocity_pred = self._to_velocity(model_pred, x, timesteps_batch)
            x, _ = self.scheduler.step(velocity_pred, t_val, x, next_timestep)
            return x

        # ── Main backward: T → tmin ──
        timestep_pairs = list(zip(self.scheduler.timesteps, all_next_timesteps))
        restart_trigger_idx = tmin_idx

        # Phase 1: Steps from T down to tmin (inclusive — we stop AFTER reaching tmin)
        for step_idx, (t, next_t) in enumerate(timestep_pairs):
            noisy_images = _euler_step(noisy_images, t, next_t)

            # Check if we've reached the restart trigger point
            if step_idx == restart_trigger_idx:
                break

        # Current timestep after Phase 1 is the next_t of the trigger step
        current_t = all_next_timesteps[restart_trigger_idx].item()

        # ── Restart iterations ──
        for _k in range(K):
            # Forward noise: current_t → tmax (add noise back up)
            alpha = (1.0 - tmax / T) / (1.0 - current_t / T)
            sigma_cond_sq = (tmax / T) ** 2 - alpha ** 2 * (current_t / T) ** 2
            sigma_cond = sigma_cond_sq ** 0.5

            z_new = torch.randn_like(noisy_images)
            noisy_images = alpha * noisy_images + sigma_cond * z_new

            # Backward ODE: tmax → current_t using n_restart evenly-spaced steps
            restart_timesteps = torch.linspace(tmax, current_t, n_restart + 1, device=device)
            for j in range(n_restart):
                noisy_images = _euler_step(
                    noisy_images,
                    restart_timesteps[j],
                    restart_timesteps[j + 1],
                )

        # ── Phase 2: Continue main backward tmin → 0 ──
        remaining_pairs = timestep_pairs[restart_trigger_idx + 1:]
        for t, next_t in remaining_pairs:
            noisy_images = _euler_step(noisy_images, t, next_t)

        return noisy_images

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
        # DiffRS (opt-in)
        diffrs_discriminator: Any | None = None,
        diffrs_config: dict[str, Any] | None = None,
        # Restart Sampling (opt-in)
        restart_config: dict[str, Any] | None = None,
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
        image_conditioning = parsed.conditioning
        is_dual = parsed.is_dual

        # Prepare CFG context (flags and unconditional tensors)
        cfg_ctx = self._prepare_cfg_context(cfg_scale, size_bins, bin_maps, image_conditioning, is_dual)

        # Branch: Restart Sampling (opt-in)
        if restart_config is not None:
            return self._generate_restart(
                model, parsed, cfg_ctx, conditioning, num_steps, device,
                restart_config,
            )

        # Branch: DiffRS rejection sampling (opt-in)
        if diffrs_discriminator is not None:
            return self._generate_diffrs(
                model, parsed, cfg_ctx, conditioning, num_steps, device,
                diffrs_discriminator, diffrs_config or {},
            )

        # Branch: use torchdiffeq for non-euler solvers
        if self.ode_solver != 'euler':
            return self._generate_torchdiffeq(
                model, parsed, cfg_ctx, conditioning, num_steps, device,
            )

        # Setup scheduler (Euler path)
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
            timestep_pairs = tqdm(timestep_pairs, desc="RFlow sampling")  # type: ignore[assignment]

        for step_idx, (t, next_t) in enumerate(timestep_pairs):
            # Compute current CFG scale (dynamic or constant)
            # Note: For single-step (num_steps=1), progress=0 so cfg_scale_end is ignored
            # and only cfg_scale (start value) is used. This is intentional.
            if use_dynamic_cfg:
                # Linear interpolation: cfg_scale at step 0, cfg_scale_end at last step
                assert cfg_scale_end is not None
                progress = step_idx / max(total_steps - 1, 1)
                current_cfg = cfg_scale + progress * (cfg_scale_end - cfg_scale)
            else:
                current_cfg = cfg_scale
            timesteps_batch = t.unsqueeze(0).repeat(batch_size).to(device)
            next_timestep = next_t.to(device) if isinstance(next_t, torch.Tensor) else torch.tensor(next_t,
                                                                                                    device=device)

            if is_dual:
                # Dual-image: process each channel through model together
                assert noisy_pre is not None
                assert noisy_gd is not None
                assert image_conditioning is not None
                current_model_input = self._prepare_dual_model_input(noisy_pre, noisy_gd, image_conditioning)
                model_pred = self._call_model(model, current_model_input, timesteps_batch, omega, mode_id, size_bins)

                # Split predictions for each channel (works for both 2D and 3D)
                pred_pre, pred_gd = self._split_dual_predictions(model_pred)

                # Convert x₀ → velocity if prediction_type='sample'
                velocity_pred_pre = self._to_velocity(pred_pre, noisy_pre, timesteps_batch)
                velocity_pred_gd = self._to_velocity(pred_gd, noisy_gd, timesteps_batch)

                # Update each channel SEPARATELY using scheduler
                noisy_pre, _ = self.scheduler.step(velocity_pred_pre, t, noisy_pre, next_timestep)
                noisy_gd, _ = self.scheduler.step(velocity_pred_gd, t, noisy_gd, next_timestep)

            else:
                # Single image or unconditional
                assert noisy_images is not None
                # Build base model input (without bin_maps - those are handled separately)
                if image_conditioning is not None:
                    current_model_input = torch.cat([noisy_images, image_conditioning], dim=1)
                else:
                    current_model_input = noisy_images

                model_pred = self._compute_cfg_prediction(
                    model, cfg_ctx, current_cfg, current_model_input, noisy_images,
                    bin_maps, size_bins, timesteps_batch, omega, mode_id
                )

                # Convert x₀ → velocity if prediction_type='sample'
                velocity_pred = self._to_velocity(model_pred, noisy_images, timesteps_batch)
                noisy_images, _ = self.scheduler.step(velocity_pred, t, noisy_images, next_timestep)

        # Return final denoised images
        if is_dual:
            assert noisy_pre is not None
            assert noisy_gd is not None
            return torch.cat([noisy_pre, noisy_gd], dim=1)  # [B, 2, H, W]
        else:
            assert noisy_images is not None
            return noisy_images
