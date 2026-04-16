"""Diffusion Bridge Strategy for Image Restoration.

Implements the Diffusion Bridge Model from Zhang et al. (2025), adapted for
3D brain MRI restoration. The key idea: interpolate between clean (x_0) and
degraded (x_1) with a time-dependent noise bubble that prevents memorization.

Forward process:
    x_t = (1-t)*x_0 + t*x_1 + γ_max * sqrt(4*t*(1-t)) * ε

where:
    - x_0 = clean volume, x_1 = degraded volume
    - γ_max controls noise intensity (0.125 validated on 3D brain MRI)
    - The noise bubble peaks at t=0.5 and vanishes at t=0 and t=1

Training: Model predicts x̂_0 (denoiser). Loss = MSE(x̂_0_pred, x_0).
Sampling: ODE or SDE from t=1 (degraded) to t=0 (clean), ~40 steps.

Reference:
    Zhang et al., "Diffusion Bridge Models for 3D Medical Image Translation",
    arXiv:2504.15267, April 2025.
"""
import logging
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast

from .strategies import DiffusionStrategy, ImageOrDict

logger = logging.getLogger(__name__)


class BridgeStrategy(DiffusionStrategy):
    """Diffusion Bridge Model for paired image restoration.

    Adds a time-dependent noise bubble to the linear interpolation between
    clean and degraded images. This prevents the memorization problem that
    occurs with deterministic bridges (exp33_2) by forcing the model to
    denoise at intermediate timesteps.

    The model predicts x̂_0 (the clean image) given (x_t, x_1, t).
    """

    def __init__(self) -> None:
        super().__init__()
        self.gamma_max: float = 0.125  # Paper default for 3D brain MRI
        self.num_steps: int = 40       # Paper default sampling steps
        self.spatial_dims: int = 3
        # Continuous timesteps in [0, 1], discretized for sampling
        self._num_train_timesteps: int = 1000  # For UNet timestep encoding

    @property
    def num_timesteps(self) -> int:
        return self._num_train_timesteps

    def setup_scheduler(
        self,
        num_timesteps: int = 1000,
        image_size: int = 256,
        depth_size: int | None = None,
        spatial_dims: int = 3,
        **kwargs: Any,
    ) -> None:
        """Setup bridge parameters. No scheduler needed — we handle noise directly."""
        self._num_train_timesteps = num_timesteps
        self.spatial_dims = spatial_dims
        self.scheduler = None  # No MONAI scheduler

        logger.info(
            f"Bridge setup: γ_max={self.gamma_max}, "
            f"num_train_timesteps={num_timesteps}, "
            f"sampling_steps={self.num_steps}"
        )

    def _expand(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Expand [B] to broadcast with [B, C, ...] target."""
        while x.dim() < target.dim():
            x = x.unsqueeze(-1)
        return x

    def _gamma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise standard deviation: γ_max * sqrt(4*t*(1-t)).

        This is the Brownian bridge variance — peaks at t=0.5, zero at endpoints.
        """
        return self.gamma_max * torch.sqrt(4.0 * t * (1.0 - t))

    # ── DiffusionStrategy interface ────────────────────────────────────

    def add_noise(
        self, clean_images: ImageOrDict, noise: ImageOrDict, timesteps: torch.Tensor,
    ) -> ImageOrDict:
        """Bridge forward: x_t = (1-t)*x_0 + t*x_1 + γ(t)*ε.

        Args:
            clean_images: x_0 (clean volume).
            noise: x_1 (degraded volume, passed via noise arg in restoration mode).
            timesteps: Integer timesteps [B] in [0, num_train_timesteps].
        """
        assert not isinstance(clean_images, dict)
        x0 = clean_images
        x1 = noise  # degraded

        # Convert integer timesteps to continuous t in [0, 1]
        t = timesteps.float() / self._num_train_timesteps
        t = self._expand(t, x0)

        # Bridge interpolation with noise bubble
        gamma_t = self.gamma_max * torch.sqrt(4.0 * t * (1.0 - t))
        epsilon = torch.randn_like(x0)
        x_t = (1.0 - t) * x0 + t * x1 + gamma_t * epsilon

        return x_t

    def sample_timesteps(
        self, images: ImageOrDict, curriculum_range: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Sample uniform integer timesteps in [1, num_train_timesteps-1].

        Avoids t=0 (pure clean, trivial) and t=T (pure degraded, no noise bubble).
        """
        if isinstance(images, dict):
            batch_size = next(iter(images.values())).shape[0]
            device = next(iter(images.values())).device
        else:
            batch_size = images.shape[0]
            device = images.device
        # Sample in [1, T-1] to avoid degenerate endpoints
        return torch.randint(
            1, self._num_train_timesteps, (batch_size,), device=device
        ).long()

    def predict_noise_or_velocity(
        self, model: nn.Module, model_input: torch.Tensor, timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Model predicts x̂_0 (clean image). Timesteps passed as-is."""
        return model(x=model_input, timesteps=timesteps)

    def compute_target(
        self, clean_images: ImageOrDict, noise: ImageOrDict,
    ) -> ImageOrDict:
        """Target is the clean image x_0 (x̂_0 prediction)."""
        return clean_images

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target_images: ImageOrDict,
        noise: ImageOrDict,
        noisy_images: ImageOrDict,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, ImageOrDict]:
        """MSE loss on x̂_0 prediction (unweighted, per Ho et al. 2020).

        Loss = MSE(model(x_t, x_1, t), x_0)

        The paper found that omitting the time-dependent weight w_t
        gives better results, consistent with DDPM findings.
        """
        assert not isinstance(target_images, dict)
        x0 = target_images.float()
        x0_pred = prediction.float()

        loss = F.mse_loss(x0_pred, x0)

        # Predicted clean is directly the model output
        predicted_clean = x0_pred.clamp(0, 1)

        return loss, predicted_clean

    def compute_predicted_clean(
        self, noisy_images: ImageOrDict, prediction: ImageOrDict, timesteps: torch.Tensor,
    ) -> ImageOrDict:
        """Model directly predicts x̂_0, so prediction IS the clean estimate."""
        assert not isinstance(prediction, dict)
        return prediction.clamp(0, 1)

    @torch.no_grad()
    def generate(
        self,
        model: nn.Module,
        model_input: torch.Tensor,
        num_steps: int,
        device: torch.device,
        use_progress_bars: bool = False,
        conditioning: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """ODE sampling from t=1 (degraded) to t=0 (clean).

        Algorithm (from Zhang et al. 2025, Algorithm 1):
        1. Start at x_N = x_1 (degraded)
        2. For each step i = N down to 1:
            a. Predict x̂_0 = model(x_i, x_1, t_i)
            b. Compute normalized residual ẑ_i = (x_i - α_i*x̂_0 - β_i*x_1) / γ_i
            c. If last step (i=1): x_0 = α_0*x̂_0 + β_0*x_1 + γ_0*ẑ (deterministic)
               Else: use Euler step or analytic update

        For ODE (η=0): deterministic, same output every time.
        We use the analytic posterior (Eq. 12) which is more stable.
        """
        # Parse input: [x_t, degraded] or just degraded
        if model_input.shape[1] == 2:
            x1 = model_input[:, 1:2]  # degraded
        else:
            x1 = model_input
        batch_size = x1.shape[0]

        N = num_steps if num_steps > 0 else self.num_steps

        # Time discretization: evenly spaced from t=1 to t=0
        # t_steps[0] = 1.0 (start), t_steps[N] = 0.0 (end)
        t_steps = torch.linspace(1.0, 0.0, N + 1, device=device)

        # Start from x_1 (degraded)
        x = x1.clone()

        steps = range(N)
        if use_progress_bars:
            from tqdm import tqdm
            steps = tqdm(steps, desc="Bridge ODE")

        for i in steps:
            t_i = t_steps[i]
            t_next = t_steps[i + 1]

            # Integer timestep for UNet encoding
            t_int = (t_i * self._num_train_timesteps).long().clamp(1, self._num_train_timesteps - 1)
            t_batch = t_int.expand(batch_size)

            # Model predicts x̂_0
            current_input = torch.cat([x, x1], dim=1)
            with autocast('cuda', dtype=torch.bfloat16):
                x0_pred = model(x=current_input, timesteps=t_batch)
            x0_pred = x0_pred.float()

            # Bridge coefficients at current and next time
            alpha_i = 1.0 - t_i.item()
            beta_i = t_i.item()
            gamma_i = self.gamma_max * math.sqrt(max(4.0 * t_i.item() * (1.0 - t_i.item()), 1e-10))

            alpha_next = 1.0 - t_next.item()
            beta_next = t_next.item()
            gamma_next = self.gamma_max * math.sqrt(max(4.0 * t_next.item() * (1.0 - t_next.item()), 1e-10))

            if i == N - 1:
                # Final step: go directly to t=0 where γ=0
                # x_0 = x̂_0 (the prediction is the answer)
                x = x0_pred
            else:
                # Analytic posterior update (Eq. 12 from Zhang et al.):
                # x_{t-Δt} = α_{t-Δt} * x̂_0 + β_{t-Δt} * x_1 + γ_{t-Δt} * ẑ_t
                # where ẑ_t = (x_t - α_t * x̂_0 - β_t * x_1) / γ_t
                z_hat = (x - alpha_i * x0_pred - beta_i * x1) / max(gamma_i, 1e-10)
                x = alpha_next * x0_pred + beta_next * x1 + gamma_next * z_hat

        return x.clamp(0, 1)
