"""IR-SDE Strategy: Image Restoration using Mean-Reverting SDE.

Faithful reimplementation of Luo et al. (ICML 2023), matching the official
GitHub repository: https://github.com/Algolzw/image-restoration-sde

Key details matched to official code:
- max_sigma = 10/255 (standard deviation), variance = max_sigma² = (10/255)²
- Theta schedule: DDPM-style cosine with T+2 offset, cumsum subtracts thetas[0]
- Loss: L1 (ML objective, eq 15)
- Sampling: Posterior (estimate x₀ → optimal mean → calibrated noise), NOT Euler-Maruyama
- Timesteps: 1-based [1, T]

Reference:
    Luo et al., "Image Restoration with Mean-Reverting Stochastic Differential
    Equations", ICML 2023. https://github.com/Algolzw/image-restoration-sde
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


class IRSDEStrategy(DiffusionStrategy):
    """Mean-reverting SDE for image restoration (official implementation).

    Forward: dx = θ_t(μ - x)dt + σ_t dw
    where σ_t² = 2 * max_sigma² * θ_t (stationary variance condition)

    Training: ML objective (eq 15) with L1 loss on predicted vs optimal reverse step.
    Sampling: Posterior sampling (more accurate than Euler-Maruyama).
    """

    def __init__(self) -> None:
        super().__init__()
        # Paper defaults (matching official repo)
        self.max_sigma: float = 10.0 / 255.0   # λ (std dev, NOT variance)
        self.eps: float = 0.005                  # δ: exp(-θ̄_T * dt) ≈ eps
        self.T: int = 100
        self.spatial_dims: int = 3

        # Pre-computed schedules (1-indexed: index 0 unused, indices 1..T)
        self._thetas: torch.Tensor | None = None          # [T+1], θ values
        self._thetas_cumsum: torch.Tensor | None = None    # [T+1], cumulative θ (cumsum[0]=0)
        self._dt: float = 0.0

    @property
    def num_timesteps(self) -> int:
        return self.T

    def setup_scheduler(
        self,
        num_timesteps: int = 100,
        image_size: int = 256,
        depth_size: int | None = None,
        spatial_dims: int = 3,
        **kwargs: Any,
    ) -> None:
        """Setup cosine theta schedule (matching official cosine_theta_schedule)."""
        self.T = num_timesteps
        self.spatial_dims = spatial_dims

        # Official: cosine_theta_schedule with T+2 offset
        T_sched = num_timesteps + 2
        steps = T_sched + 1
        s = 0.008
        x = torch.linspace(0, T_sched, steps)
        alphas_cumprod = torch.cos(((x / T_sched) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - alphas_cumprod[1:-1]  # Length T+1 (indices 0..T)
        betas = betas.clamp(max=0.999)

        # thetas = betas, thetas_cumsum starts at 0 (subtract first element)
        thetas = betas
        thetas_cumsum = torch.cumsum(thetas, dim=0) - thetas[0]
        # Now thetas_cumsum[0] = 0, thetas_cumsum[T] = sum(thetas) - thetas[0]

        # dt calibrated so exp(-thetas_cumsum[-1] * dt) = eps
        self._dt = -math.log(self.eps) / thetas_cumsum[-1].item()

        self._thetas = thetas        # [T+1], indices 0..T
        self._thetas_cumsum = thetas_cumsum  # [T+1], cumsum[0]=0

        # Verify
        exp_T = math.exp(-self._thetas_cumsum[-1].item() * self._dt)
        var_T = self.max_sigma ** 2 * (1.0 - math.exp(-2.0 * self._thetas_cumsum[-1].item() * self._dt))
        logger.info(
            f"IR-SDE setup: T={num_timesteps}, max_σ={self.max_sigma:.4f}, "
            f"exp(-θ̄_T·dt)={exp_T:.4f}, var_T={var_T:.6f}, dt={self._dt:.4f}"
        )

    # ── Schedule helper functions (matching official sde_utils.py) ──────

    def _mu_bar(self, x0: torch.Tensor, mu: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward mean: μ + (x₀ - μ) * exp(-θ̄_t * dt)."""
        theta_cumsum_t = self._thetas_cumsum.to(x0.device)[t]
        exp_neg = torch.exp(-theta_cumsum_t * self._dt)
        exp_neg = self._expand(exp_neg, x0)
        return mu + (x0 - mu) * exp_neg

    def _sigma_bar(self, t: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Forward std: sqrt(max_σ² * (1 - exp(-2θ̄_t * dt)))."""
        theta_cumsum_t = self._thetas_cumsum.to(device)[t]
        return torch.sqrt(
            self.max_sigma ** 2 * (1.0 - torch.exp(-2.0 * theta_cumsum_t * self._dt))
        )

    def _expand(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Expand [B] to broadcast with [B, C, ...] target."""
        while x.dim() < target.dim():
            x = x.unsqueeze(-1)
        return x

    # ── DiffusionStrategy interface ────────────────────────────────────

    def add_noise(
        self, clean_images: ImageOrDict, noise: ImageOrDict, timesteps: torch.Tensor,
    ) -> ImageOrDict:
        """Forward kernel: x_t = mu_bar(x₀, t) + sigma_bar(t) * ε.

        Args:
            clean_images: x₀ (clean).
            noise: μ (degraded, passed via noise arg in restoration mode).
            timesteps: 1-based timesteps [B] in [1, T].
        """
        assert not isinstance(clean_images, dict)
        mu = noise  # degraded
        device = clean_images.device

        mean = self._mu_bar(clean_images, mu, timesteps)
        sigma = self._sigma_bar(timesteps, device)
        sigma = self._expand(sigma, clean_images)

        epsilon = torch.randn_like(clean_images)
        return mean + sigma * epsilon

    def sample_timesteps(
        self, images: ImageOrDict, curriculum_range: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Sample 1-based timesteps in [1, T] (matching official)."""
        if isinstance(images, dict):
            batch_size = next(iter(images.values())).shape[0]
            device = next(iter(images.values())).device
        else:
            batch_size = images.shape[0]
            device = images.device
        return torch.randint(1, self.T + 1, (batch_size,), device=device).long()

    def predict_noise_or_velocity(
        self, model: nn.Module, model_input: torch.Tensor, timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Model predicts noise ε. Timesteps passed as-is (1-based)."""
        # Scale to match UNet's expected range. UNet trained with [0, 1000],
        # IR-SDE uses [1, 100]. Map linearly.
        scale = 1000.0 / self.T
        scaled_t = (timesteps.float() * scale).long()
        return model(x=model_input, timesteps=scaled_t)

    def compute_target(
        self, clean_images: ImageOrDict, noise: ImageOrDict,
    ) -> ImageOrDict:
        """Placeholder — ML target computed in compute_loss."""
        return clean_images

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target_images: ImageOrDict,
        noise: ImageOrDict,
        noisy_images: ImageOrDict,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, ImageOrDict]:
        """Maximum likelihood objective (eq 15) with L1 loss.

        1. Model predicts noise ε
        2. Compute score from ε: score = -ε / sigma_bar(t)
        3. Compute predicted x_{t-1} via reverse drift
        4. Compute optimal x*_{t-1} from Proposition 3.2
        5. Loss = L1(predicted, optimal)
        """
        assert not isinstance(target_images, dict)
        assert not isinstance(noise, dict)
        assert not isinstance(noisy_images, dict)
        assert self._thetas is not None
        assert self._thetas_cumsum is not None

        x0 = target_images.float()
        mu = noise.float()  # degraded
        xt = noisy_images.float()
        eps_pred = prediction.float()
        device = xt.device
        dt = self._dt

        thetas = self._thetas.to(device)
        thetas_cumsum = self._thetas_cumsum.to(device)

        # ── Score from noise prediction ──
        sigma_bar_t = self._sigma_bar(timesteps, device)
        sigma_bar_t = self._expand(sigma_bar_t, xt)
        score = -eps_pred / sigma_bar_t.clamp(min=1e-10)

        # ── Predicted x_{t-1} via reverse SDE drift ──
        theta_t = thetas[timesteps]
        theta_t = self._expand(theta_t, xt)
        sigma_t_sq = self.max_sigma ** 2 * 2.0 * theta_t  # σ² = max_σ² * 2θ

        drift_dt = (theta_t * (mu - xt) - sigma_t_sq * score) * dt
        x_pred_prev = xt - drift_dt  # reverse time

        # ── Optimal x*_{t-1} (Proposition 3.2 / official reverse_optimum_step) ──
        tc_t = thetas_cumsum[timesteps]        # θ̄_t
        tc_prev = thetas_cumsum[timesteps - 1]  # θ̄_{t-1} (safe: t≥1, cumsum[0]=0)
        th_t = thetas[timesteps]               # θ_t (single step)

        tc_t = self._expand(tc_t, xt)
        tc_prev = self._expand(tc_prev, xt)
        th_t_exp = self._expand(th_t, xt)

        A = torch.exp(-th_t_exp * dt)
        B = torch.exp(-tc_t * dt)
        C = torch.exp(-tc_prev * dt)

        denom = (1.0 - B ** 2).clamp(min=1e-10)
        coeff1 = A * (1.0 - C ** 2) / denom
        coeff2 = C * (1.0 - A ** 2) / denom

        x_star_prev = coeff1 * (xt - mu) + coeff2 * (x0 - mu) + mu

        # ── L1 loss (matching official) ──
        loss = F.l1_loss(x_pred_prev, x_star_prev)

        # ── Predicted clean via Tweedie ──
        # x₀ = (x_t - μ - σ̄(t)*ε) * exp(θ̄_t*dt) + μ
        exp_pos = torch.exp(tc_t * dt)
        predicted_clean = (xt - mu - sigma_bar_t * eps_pred) * exp_pos + mu
        predicted_clean = predicted_clean.clamp(0, 1)

        return loss, predicted_clean

    def compute_predicted_clean(
        self, noisy_images: ImageOrDict, prediction: ImageOrDict, timesteps: torch.Tensor,
    ) -> ImageOrDict:
        """Approximate clean estimate (full Tweedie in compute_loss)."""
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
        """Posterior sampling (official default, more accurate than Euler-Maruyama).

        For each reverse step t → t-1:
        1. Predict noise ε from model
        2. Estimate x₀ via Tweedie: x₀ = (x_t - μ - σ̄(t)ε) * exp(θ̄_t·dt) + μ
        3. Compute optimal mean x*_{t-1} from Proposition 3.2
        4. Add calibrated posterior noise

        Args:
            model: Trained restoration model.
            model_input: [B, 2, D, H, W] = [degraded, degraded].
            num_steps: Number of reverse steps.
            device: CUDA device.
        """
        assert self._thetas is not None
        assert self._thetas_cumsum is not None

        # Parse input
        if model_input.shape[1] == 2:
            mu = model_input[:, 1:2]
        else:
            mu = model_input
        batch_size = mu.shape[0]

        # Terminal state: x_T = μ + randn * max_sigma (matching official noise_state)
        x = mu + torch.randn_like(mu) * self.max_sigma

        thetas = self._thetas.to(device)
        thetas_cumsum = self._thetas_cumsum.to(device)
        dt = self._dt
        T = min(num_steps, self.T)

        # Reverse from T down to 1
        steps = list(range(T, 0, -1))
        if use_progress_bars:
            from tqdm import tqdm
            steps = tqdm(steps, desc="IR-SDE posterior")

        for t_val in steps:
            t_batch = torch.full((batch_size,), t_val, device=device, dtype=torch.long)

            # Model prediction (noise ε)
            current_input = torch.cat([x, mu], dim=1)
            scale = 1000.0 / self.T
            scaled_t = (t_batch.float() * scale).long()

            with autocast('cuda', dtype=torch.bfloat16):
                noise_pred = model(x=current_input, timesteps=scaled_t)
            noise_pred = noise_pred.float()

            # ── Posterior sampling step ──

            # 1. Estimate x₀ from noise (Tweedie)
            tc_t = thetas_cumsum[t_val]
            sigma_bar_t = math.sqrt(self.max_sigma ** 2 * (1.0 - math.exp(-2.0 * tc_t.item() * dt)))
            exp_pos_t = math.exp(tc_t.item() * dt)
            x0_est = (x - mu - sigma_bar_t * noise_pred) * exp_pos_t + mu

            # 2. Compute optimal mean x*_{t-1}
            tc_prev = thetas_cumsum[t_val - 1]  # t≥1, so t-1≥0, cumsum[0]=0
            th_t = thetas[t_val]

            A = math.exp(-th_t.item() * dt)
            B = math.exp(-tc_t.item() * dt)
            C = math.exp(-tc_prev.item() * dt)

            denom = max(1.0 - B ** 2, 1e-10)
            coeff1 = A * (1.0 - C ** 2) / denom
            coeff2 = C * (1.0 - A ** 2) / denom

            mean = coeff1 * (x - mu) + coeff2 * (x0_est - mu) + mu

            # 3. Add posterior noise (skip at t=1 → x₀)
            if t_val > 1:
                A2 = math.exp(-2.0 * th_t.item() * dt)
                B2 = math.exp(-2.0 * tc_t.item() * dt)
                C2 = math.exp(-2.0 * tc_prev.item() * dt)
                posterior_var = (1.0 - A2) * (1.0 - C2) / max(1.0 - B2, 1e-10)
                log_pvar = math.log(max(posterior_var, 1e-20 * dt))
                std = math.exp(0.5 * log_pvar) * self.max_sigma
                x = mean + std * torch.randn_like(x)
            else:
                x = mean

        return x.clamp(0, 1)
