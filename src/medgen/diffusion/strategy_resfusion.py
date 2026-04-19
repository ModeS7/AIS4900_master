"""Resfusion Strategy: Residual Noise Diffusion for Image Restoration.

Faithful reimplementation of Shi et al. (NeurIPS 2024), matching the official
GitHub repository: https://github.com/nkicsl/Resfusion

Key details matched to official code:
- Short linear schedule (T=12 for restoration, NOT T=1000)
- Resnoise coefficient uses α_t (single-step), NOT ᾱ_t (cumulative)
- Data scaled to [-1, 1] internally (official convention)
- T_acc = argmin|√ᾱ - 0.5| + 1 (off-by-one matters)
- Reverse loop includes t=0 (with zero noise)
- Training range: [0, T_acc-1]

Reference:
    Shi et al., "Resfusion: Denoising Diffusion Probabilistic Models for
    Image Restoration Based on Prior Residual Noise", NeurIPS 2024.
    https://github.com/nkicsl/Resfusion
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


class ResfusionStrategy(DiffusionStrategy):
    """Resfusion: DDPM with residual term for image restoration.

    Uses a short linear schedule (T=12) with truncated inference starting
    from T_acc where √ᾱ ≈ 0.5. Only ~5-12 reverse steps needed.

    The model predicts resnoise = ε + coeff * R, where R = degraded - clean.
    """

    def __init__(self) -> None:
        super().__init__()
        self.T: int = 12           # Short schedule for restoration (official default)
        self.T_acc: int | None = None  # Acceleration point (computed in setup)
        self.spatial_dims: int = 3

        # Schedule arrays (0-indexed, length T)
        self._betas: torch.Tensor | None = None
        self._alphas: torch.Tensor | None = None            # α_t = 1 - β_t (single step)
        self._alphas_hat: torch.Tensor | None = None         # ᾱ_t = cumulative product
        self._sqrt_alphas: torch.Tensor | None = None        # √α_t
        self._sqrt_alphas_hat: torch.Tensor | None = None    # √ᾱ_t
        self._sqrt_1m_alphas_hat: torch.Tensor | None = None  # √(1-ᾱ_t)

    @property
    def num_timesteps(self) -> int:
        return self.T

    def setup_scheduler(
        self,
        num_timesteps: int = 12,
        image_size: int = 256,
        depth_size: int | None = None,
        spatial_dims: int = 3,
        **kwargs: Any,
    ) -> None:
        """Setup linear schedule with Pro-style truncation.

        Official uses scaled linear schedule: beta_start = (1000/T)*0.0001,
        beta_end = (1000/T)*0.02 (from improved diffusion).
        """
        self.T = num_timesteps
        self.spatial_dims = spatial_dims

        # Scaled linear schedule (matching official LinearProScheduler)
        scale = 1000.0 / num_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        betas = betas.clamp(max=0.999)

        alphas = 1.0 - betas
        alphas_hat = torch.cumprod(alphas, dim=0)

        self._betas = betas
        self._alphas = alphas
        self._alphas_hat = alphas_hat
        self._sqrt_alphas = torch.sqrt(alphas)
        self._sqrt_alphas_hat = torch.sqrt(alphas_hat)
        self._sqrt_1m_alphas_hat = torch.sqrt(1.0 - alphas_hat)

        # Find acceleration point T_acc = argmin|√ᾱ - 0.5| + 1
        diffs = torch.abs(self._sqrt_alphas_hat - 0.5)
        self.T_acc = int(diffs.argmin().item()) + 1  # Official adds +1

        logger.info(
            f"Resfusion setup: T={num_timesteps}, T_acc={self.T_acc}, "
            f"√ᾱ_T_acc={self._sqrt_alphas_hat[self.T_acc - 1]:.4f}, "
            f"betas=[{betas[0]:.4f}, {betas[-1]:.4f}]"
        )

    def _expand(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Expand [B] to broadcast with [B, C, ...] target."""
        while x.dim() < target.dim():
            x = x.unsqueeze(-1)
        return x

    def _to_m1p1(self, x: torch.Tensor) -> torch.Tensor:
        """Scale [0, 1] → [-1, 1]."""
        return x * 2.0 - 1.0

    def _to_01(self, x: torch.Tensor) -> torch.Tensor:
        """Scale [-1, 1] → [0, 1]."""
        return (x + 1.0) / 2.0

    def add_noise(
        self, clean_images: ImageOrDict, noise: ImageOrDict, timesteps: torch.Tensor,
    ) -> ImageOrDict:
        """Resfusion forward: x_t = √ᾱ_t * x₀ + (1-√ᾱ_t) * R + √(1-ᾱ_t) * ε.

        Operates in [-1, 1] space internally.

        Args:
            clean_images: x₀ in [0, 1].
            noise: Degraded volume in [0, 1] (passed via noise arg).
            timesteps: 0-based timesteps [B].
        """
        assert not isinstance(clean_images, dict)
        # Convert to [-1, 1]
        x0 = self._to_m1p1(clean_images)
        degraded = self._to_m1p1(noise)
        R = degraded - x0

        assert self._sqrt_alphas_hat is not None
        device = x0.device

        sqrt_ah = self._sqrt_alphas_hat.to(device)[timesteps]
        sqrt_ah = self._expand(sqrt_ah, x0)
        sqrt_1m_ah = self._sqrt_1m_alphas_hat.to(device)[timesteps]
        sqrt_1m_ah = self._expand(sqrt_1m_ah, x0)

        epsilon = torch.randn_like(x0)
        x_t = sqrt_ah * x0 + (1.0 - sqrt_ah) * R + sqrt_1m_ah * epsilon

        return self._to_01(x_t)  # Back to [0, 1] for trainer interface

    def sample_timesteps(
        self, images: ImageOrDict, curriculum_range: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Sample from [0, T_acc-1] (matching official)."""
        assert self.T_acc is not None
        if isinstance(images, dict):
            batch_size = next(iter(images.values())).shape[0]
            device = next(iter(images.values())).device
        else:
            batch_size = images.shape[0]
            device = images.device
        return torch.randint(0, self.T_acc, (batch_size,), device=device).long()

    def predict_noise_or_velocity(
        self, model: nn.Module, model_input: torch.Tensor, timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Model predicts resnoise."""
        return model(x=model_input, timesteps=timesteps)

    def compute_target(
        self, clean_images: ImageOrDict, noise: ImageOrDict,
    ) -> ImageOrDict:
        """Placeholder — resnoise target computed in compute_loss."""
        return clean_images

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target_images: ImageOrDict,
        noise: ImageOrDict,
        noisy_images: ImageOrDict,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, ImageOrDict]:
        """Resnoise prediction loss (eq 6).

        resnoise = ε + (1 - √α_t) * √(1-ᾱ_t) / β_t * R

        NOTE: Uses α_t (single-step), NOT ᾱ_t (cumulative). This matches
        the official code which uses self.alphas[t], not self.alphas_hat[t].
        """
        assert not isinstance(target_images, dict)
        assert not isinstance(noise, dict)
        assert not isinstance(noisy_images, dict)
        assert self._alphas is not None
        assert self._betas is not None

        device = prediction.device
        assert self._sqrt_alphas_hat is not None
        assert self._sqrt_1m_alphas_hat is not None

        # Deterministically recover epsilon + R from the inputs rather than relying on
        # stored instance state from add_noise — avoids race/reentrancy hazards under
        # gradient accumulation.
        x0_m1p1 = self._to_m1p1(target_images)
        degraded_m1p1 = self._to_m1p1(noise)
        R = degraded_m1p1 - x0_m1p1
        x_t_m1p1 = self._to_m1p1(noisy_images)
        sqrt_ah = self._sqrt_alphas_hat.to(device)[timesteps]
        sqrt_ah = self._expand(sqrt_ah, prediction)
        sqrt_1m_ah = self._sqrt_1m_alphas_hat.to(device)[timesteps]
        sqrt_1m_ah = self._expand(sqrt_1m_ah, prediction)
        epsilon = (x_t_m1p1 - sqrt_ah * x0_m1p1 - (1.0 - sqrt_ah) * R) / sqrt_1m_ah.clamp(min=1e-8)

        # α_t (SINGLE STEP, not cumulative) — this is the critical difference
        alpha_t = self._alphas.to(device)[timesteps]
        alpha_t = self._expand(alpha_t, prediction)
        sqrt_alpha_t = torch.sqrt(alpha_t)

        # β_t
        beta_t = self._betas.to(device)[timesteps]
        beta_t = self._expand(beta_t, prediction)

        # Resnoise target (eq 6): ε + (1 - √α_t) * √(1-ᾱ_t) / β_t * R
        resnoise_coeff = (1.0 - sqrt_alpha_t) * sqrt_1m_ah / beta_t.clamp(min=1e-8)
        resnoise_target = epsilon + resnoise_coeff * R

        loss = F.mse_loss(prediction.float(), resnoise_target.float())

        # Predicted clean: reverse the DDPM mean formula with resnoise
        # μ_θ = (1/√α_t) * (x_t - β_t/√(1-ᾱ_t) * resnoise)
        assert not isinstance(noisy_images, dict)
        x_t_m1p1 = self._to_m1p1(noisy_images)
        pred_mean = (1.0 / sqrt_alpha_t) * (
            x_t_m1p1 - (beta_t / sqrt_1m_ah.clamp(min=1e-8)) * prediction
        )
        predicted_clean = self._to_01(pred_mean).clamp(0, 1)

        return loss, predicted_clean

    def compute_predicted_clean(
        self, noisy_images: ImageOrDict, prediction: ImageOrDict, timesteps: torch.Tensor,
    ) -> ImageOrDict:
        """Approximate."""
        assert not isinstance(noisy_images, dict)
        return noisy_images

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
        """Truncated DDPM reverse from T_acc (Algorithm 2).

        Starting point: x_{T_acc} = √ᾱ_{T_acc} * degraded + √(1-ᾱ_{T_acc}) * ε
        Reverse: T_acc steps from T_acc-1 down to 0 (inclusive).
        At t=0: no noise added.
        """
        assert self.T_acc is not None
        assert self._alphas_hat is not None

        # Parse input
        if model_input.shape[1] == 2:
            degraded_01 = model_input[:, 1:2]
        else:
            degraded_01 = model_input
        batch_size = degraded_01.shape[0]

        # Convert to [-1, 1]
        degraded = self._to_m1p1(degraded_01)

        alphas = self._alphas.to(device)
        alphas_hat = self._alphas_hat.to(device)
        betas = self._betas.to(device)
        sqrt_1m_ah = self._sqrt_1m_alphas_hat.to(device)

        # Starting point: x_{T_acc} (using standard DDPM noising of degraded)
        t_start = self.T_acc - 1  # 0-indexed
        sqrt_ah_start = self._sqrt_alphas_hat[t_start].item()
        sqrt_1m_ah_start = self._sqrt_1m_alphas_hat[t_start].item()
        epsilon = torch.randn_like(degraded)
        x = sqrt_ah_start * degraded + sqrt_1m_ah_start * epsilon

        # Reverse from T_acc-1 down to 0 (inclusive, T_acc steps total)
        steps = list(range(self.T_acc - 1, -1, -1))
        if use_progress_bars:
            from tqdm import tqdm
            steps = tqdm(steps, desc="Resfusion reverse")

        for t in steps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Model input: [x_t, degraded] concatenated
            x_01 = self._to_01(x)  # Model expects [0, 1]
            current_input = torch.cat([x_01, degraded_01], dim=1)

            with autocast('cuda', dtype=torch.bfloat16):
                resnoise_pred = model(x=current_input, timesteps=t_batch)
            resnoise_pred = resnoise_pred.float()

            # DDPM reverse step with resnoise (eq 13)
            alpha_t = alphas[t].item()
            sqrt_alpha_t = math.sqrt(alpha_t)
            beta_t = betas[t].item()
            sqrt_1m_ah_t = sqrt_1m_ah[t].item()

            # Mean: (1/√α_t) * (x_t - β_t/√(1-ᾱ_t) * resnoise)
            mean = (1.0 / sqrt_alpha_t) * (
                x - (beta_t / max(sqrt_1m_ah_t, 1e-8)) * resnoise_pred
            )

            if t > 0:
                # Posterior variance: β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
                ah_prev = alphas_hat[t - 1].item()
                ah_t = alphas_hat[t].item()
                beta_tilde = beta_t * (1.0 - ah_prev) / max(1.0 - ah_t, 1e-10)
                z = torch.randn_like(x)
                x = mean + math.sqrt(max(beta_tilde, 1e-10)) * z
            else:
                # t=0: no noise (final step to clean)
                x = mean

        # Convert back to [0, 1]
        return self._to_01(x).clamp(0, 1)
