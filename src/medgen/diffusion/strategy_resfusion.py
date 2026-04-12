"""Resfusion Strategy: Residual Noise Diffusion for Image Restoration.

Implements Resfusion (Shi et al., NeurIPS 2024) which incorporates the
residual term R = degraded - clean into the DDPM forward process:

    x_t = √ᾱ_t * x_0 + (1-√ᾱ_t) * R + √(1-ᾱ_t) * ε

The model predicts "resnoise" — a combined residual+noise target:

    resnoise = ε + ((1-√ᾱ_t) * √(1-ᾱ_t) / β_t) * R

Inference starts from T' (where √ᾱ_{T'} ≈ 0.5) instead of T,
requiring only ~5 sampling steps.

Reference:
    Shi et al., "Resfusion: Denoising Diffusion Probabilistic Models for
    Image Restoration Based on Prior Residual Noise", NeurIPS 2024.
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

    The key insight is that starting from x_T' (a mix of degraded + noise)
    instead of pure noise reduces inference to ~5 steps. The forward process
    encodes the residual (degraded - clean) into the diffusion chain.

    At inference, T' is chosen where √ᾱ_{T'} ≈ 0.5, which means:
        x_{T'} ≈ 0.5*x̂_0 + noise  ≈  0.5*degraded + noise
    Since x_0 is unknown, we approximate x_{T'} from the degraded image.
    """

    def __init__(self) -> None:
        super().__init__()
        self.num_timesteps: int = 1000
        self.T_prime: int | None = None  # Auto-computed truncation point
        self.spatial_dims: int = 3

        # DDPM schedule parameters
        self._alphas_cumprod: torch.Tensor | None = None
        self._betas: torch.Tensor | None = None
        self._sqrt_alphas_cumprod: torch.Tensor | None = None
        self._sqrt_one_minus_alphas_cumprod: torch.Tensor | None = None

    def setup_scheduler(
        self,
        num_timesteps: int = 1000,
        image_size: int = 256,
        depth_size: int | None = None,
        spatial_dims: int = 3,
        **kwargs: Any,
    ) -> None:
        """Setup cosine beta schedule and find truncation point T'.

        T' is the timestep where √ᾱ_{T'} is closest to 0.5.
        """
        self.num_timesteps = num_timesteps
        self.spatial_dims = spatial_dims

        # Cosine schedule (Nichol & Dhariwal 2021)
        s = 0.008
        t = torch.linspace(0, num_timesteps, num_timesteps + 1)
        f = torch.cos(((t / num_timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
        alphas_cumprod = f / f[0]
        alphas_cumprod = alphas_cumprod[1:]  # Remove t=0 entry

        # Clip for numerical stability
        alphas_cumprod = alphas_cumprod.clamp(min=1e-5, max=1.0 - 1e-5)

        # Derive betas
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        betas = 1.0 - alphas_cumprod / alphas_cumprod_prev
        betas = betas.clamp(min=1e-5, max=0.999)

        self._alphas_cumprod = alphas_cumprod
        self._betas = betas
        self._sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self._sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Find T': argmin |√ᾱ_t - 0.5|
        diffs = torch.abs(self._sqrt_alphas_cumprod - 0.5)
        self.T_prime = int(diffs.argmin().item())

        logger.info(
            f"Resfusion setup: T={num_timesteps}, T'={self.T_prime}, "
            f"√ᾱ_T'={self._sqrt_alphas_cumprod[self.T_prime]:.4f}"
        )

    def _expand_dims(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Expand 1D tensor to broadcast with target."""
        while x.dim() < target.dim():
            x = x.unsqueeze(-1)
        return x

    def add_noise(
        self,
        clean_images: ImageOrDict,
        noise: ImageOrDict,
        timesteps: torch.Tensor,
    ) -> ImageOrDict:
        """Resfusion forward process.

        x_t = √ᾱ_t * x_0 + (1-√ᾱ_t) * R + √(1-ᾱ_t) * ε

        where R = degraded - clean (residual).

        Args:
            clean_images: x_0 (clean volume).
            noise: Degraded volume μ (passed via noise argument in restoration mode).
            timesteps: Integer timesteps [B].

        Returns:
            x_t at the given timesteps.
        """
        assert not isinstance(clean_images, dict)
        degraded = noise  # In restoration mode
        R = degraded - clean_images  # Residual

        assert self._sqrt_alphas_cumprod is not None
        sqrt_alpha = self._sqrt_alphas_cumprod.to(clean_images.device)[timesteps]
        sqrt_alpha = self._expand_dims(sqrt_alpha, clean_images)

        sqrt_one_minus_alpha = self._sqrt_one_minus_alphas_cumprod.to(clean_images.device)[timesteps]
        sqrt_one_minus_alpha = self._expand_dims(sqrt_one_minus_alpha, clean_images)

        epsilon = torch.randn_like(clean_images)
        x_t = sqrt_alpha * clean_images + (1.0 - sqrt_alpha) * R + sqrt_one_minus_alpha * epsilon

        # Store epsilon for loss computation (accessed via closure in compute_loss)
        self._last_epsilon = epsilon

        return x_t

    def sample_timesteps(
        self,
        images: ImageOrDict,
        curriculum_range: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Sample timesteps uniformly in [0, T'-1] (truncated range).

        Only train on timesteps up to T' since inference starts there.
        """
        assert self.T_prime is not None, "Call setup_scheduler first"
        if isinstance(images, dict):
            batch_size = next(iter(images.values())).shape[0]
            device = next(iter(images.values())).device
        else:
            batch_size = images.shape[0]
            device = images.device
        return torch.randint(0, self.T_prime + 1, (batch_size,), device=device).long()

    def predict_noise_or_velocity(
        self,
        model: nn.Module,
        model_input: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass — model predicts resnoise."""
        return model(x=model_input, timesteps=timesteps)

    def compute_target(
        self,
        clean_images: ImageOrDict,
        noise: ImageOrDict,
    ) -> ImageOrDict:
        """Compute resnoise target. Placeholder — real computation in compute_loss."""
        return clean_images

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target_images: ImageOrDict,
        noise: ImageOrDict,
        noisy_images: ImageOrDict,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, ImageOrDict]:
        """Resnoise prediction loss.

        Target: resnoise = ε + ((1-√ᾱ_t) * √(1-ᾱ_t) / β_t) * R

        Args:
            prediction: Model's resnoise prediction.
            target_images: Clean images x_0.
            noise: Degraded images μ.
            noisy_images: x_t.
            timesteps: Current timesteps.

        Returns:
            (loss, predicted_clean) tuple.
        """
        assert not isinstance(target_images, dict)
        assert not isinstance(noise, dict)

        clean = target_images
        degraded = noise
        R = degraded - clean

        assert self._sqrt_alphas_cumprod is not None
        assert self._betas is not None

        sqrt_alpha = self._sqrt_alphas_cumprod.to(clean.device)[timesteps]
        sqrt_alpha = self._expand_dims(sqrt_alpha, clean)
        sqrt_one_minus_alpha = self._sqrt_one_minus_alphas_cumprod.to(clean.device)[timesteps]
        sqrt_one_minus_alpha = self._expand_dims(sqrt_one_minus_alpha, clean)
        beta = self._betas.to(clean.device)[timesteps]
        beta = self._expand_dims(beta, clean)

        # True resnoise target (eq 6 from paper)
        # resnoise = ε + ((1-√ᾱ_t) * √(1-ᾱ_t) / β_t) * R
        epsilon = self._last_epsilon.to(clean.device)
        resnoise_coeff = (1.0 - sqrt_alpha) * sqrt_one_minus_alpha / beta.clamp(min=1e-8)
        resnoise_target = epsilon + resnoise_coeff * R

        loss = F.mse_loss(prediction.float(), resnoise_target.float())

        # Predicted clean from resnoise (reverse of eq 13)
        # μ_θ(x_t, t) = (1/√α_t) * (x_t - β_t/√(1-ᾱ_t) * resnoise)
        alpha_t = (1.0 - beta)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        assert not isinstance(noisy_images, dict)
        predicted_mean = (1.0 / sqrt_alpha_t) * (
            noisy_images - (beta / sqrt_one_minus_alpha.clamp(min=1e-8)) * prediction
        )
        predicted_clean = predicted_mean.clamp(0, 1)

        return loss, predicted_clean

    def compute_predicted_clean(
        self,
        noisy_images: ImageOrDict,
        prediction: ImageOrDict,
        timesteps: torch.Tensor,
    ) -> ImageOrDict:
        """Estimate clean image from resnoise prediction."""
        return prediction  # Approximate

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
        """Truncated DDPM reverse from T' (only ~5 steps).

        Starting from x_{T'} ≈ √ᾱ_{T'} * degraded + √(1-ᾱ_{T'}) * ε
        (since √ᾱ_{T'} ≈ 0.5, this is a noisy version of the degraded image).

        Args:
            model: Trained restoration model.
            model_input: [B, 2, D, H, W] = [degraded, degraded].
            num_steps: Ignored — always uses T' steps.
            device: CUDA device.

        Returns:
            Restored volume [B, 1, D, H, W].
        """
        assert self.T_prime is not None
        assert self._alphas_cumprod is not None
        assert self._betas is not None

        # Parse input
        if model_input.shape[1] == 2:
            degraded = model_input[:, 1:2]
        else:
            degraded = model_input

        batch_size = degraded.shape[0]

        # Start from x_{T'} (eq 9 from paper)
        sqrt_alpha_Tp = self._sqrt_alphas_cumprod[self.T_prime].item()
        sqrt_1m_alpha_Tp = self._sqrt_one_minus_alphas_cumprod[self.T_prime].item()
        epsilon = torch.randn_like(degraded)
        x = sqrt_alpha_Tp * degraded + sqrt_1m_alpha_Tp * epsilon

        alphas_cumprod = self._alphas_cumprod.to(device)
        betas = self._betas.to(device)

        # Reverse from T' down to 0
        steps = list(range(self.T_prime, -1, -1))
        if use_progress_bars:
            from tqdm import tqdm
            steps = tqdm(steps, desc="Resfusion reverse")

        for t in steps:
            timesteps_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            current_input = torch.cat([x, degraded], dim=1)

            with autocast('cuda', dtype=torch.bfloat16):
                resnoise_pred = model(x=current_input, timesteps=timesteps_batch)

            resnoise_pred = resnoise_pred.float()

            # DDPM reverse step with resnoise (eq 13 from paper)
            alpha_t = 1.0 - betas[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_1m_alpha_cumprod = self._sqrt_one_minus_alphas_cumprod.to(device)[t]
            beta_t = betas[t]

            # Predicted mean: (1/√α_t) * (x_t - β_t/√(1-ᾱ_t) * resnoise)
            mean = (1.0 / sqrt_alpha_t) * (
                x - (beta_t / sqrt_1m_alpha_cumprod.clamp(min=1e-8)) * resnoise_pred
            )

            if t > 0:
                # Add noise with posterior variance
                alpha_cumprod_prev = alphas_cumprod[t - 1]
                alpha_cumprod_t = alphas_cumprod[t]
                beta_tilde = beta_t * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)
                z = torch.randn_like(x)
                x = mean + torch.sqrt(beta_tilde.clamp(min=1e-10)) * z
            else:
                x = mean

        return x.clamp(0, 1)
