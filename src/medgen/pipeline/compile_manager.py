"""Compiled forward management for diffusion training.

This module provides:
- CompiledForwardManager: Manages torch.compile optimization for diffusion training

Consolidates the compiled forward logic from DiffusionTrainer into a reusable class.
"""
import logging
from collections.abc import Callable

import torch
from torch import nn

logger = logging.getLogger(__name__)


class CompiledForwardManager:
    """Manages torch.compile optimization for diffusion training.

    Handles compilation of fused forward+loss computation for:
    - Single-channel modes (seg, bravo)
    - Dual-channel mode (dual)

    The compiled functions fuse model forward pass with loss computation
    to enable CUDA graph capture for maximum performance.
    """

    def __init__(
        self,
        use_fp32_loss: bool = True,
        strategy_name: str = 'rflow',
        num_timesteps: int = 1000,
        is_main_process: bool = True,
    ) -> None:
        """Initialize compiled forward manager.

        Args:
            use_fp32_loss: Whether to compute loss in FP32.
            strategy_name: Diffusion strategy name ('rflow' or 'ddpm').
            num_timesteps: Number of training timesteps.
            is_main_process: Whether this is the main process (for logging).
        """
        self.use_fp32_loss = use_fp32_loss
        self.strategy_name = strategy_name
        self.num_timesteps = num_timesteps
        self.is_main_process = is_main_process
        self._compiled_single: Callable | None = None
        self._compiled_dual: Callable | None = None
        self._enabled = False

    @property
    def is_compiled(self) -> bool:
        """Check if compiled forward is enabled."""
        return self._enabled

    def setup(self, enabled: bool) -> None:
        """Compile forward functions.

        Args:
            enabled: Whether to enable compilation.
        """
        self._enabled = enabled

        if not enabled:
            self._compiled_single = None
            self._compiled_dual = None
            return

        # Capture settings for closures
        use_fp32 = self.use_fp32_loss
        strategy_name = self.strategy_name
        num_train_timesteps = self.num_timesteps

        def _forward_single(
            model: nn.Module,
            perceptual_fn: nn.Module,
            model_input: torch.Tensor,
            timesteps: torch.Tensor,
            images: torch.Tensor,
            noise: torch.Tensor,
            noisy_images: torch.Tensor,
            perceptual_weight: float,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Single-channel forward with loss computation.

            Fuses model prediction and loss computation for CUDA graph capture.

            Args:
                model: Diffusion model.
                perceptual_fn: Perceptual loss function.
                model_input: Input to model (noisy images + conditioning).
                timesteps: Timestep tensor.
                images: Clean images.
                noise: Noise tensor.
                noisy_images: Noisy images.
                perceptual_weight: Weight for perceptual loss.

            Returns:
                Tuple of (total_loss, mse_loss, perceptual_loss, predicted_clean).
            """
            prediction = model(model_input, timesteps)

            if strategy_name == 'rflow':
                t_normalized = timesteps.float() / float(num_train_timesteps)
                # Auto-detect spatial dims from input shape for 2D/3D support
                spatial_dims = noisy_images.ndim - 2  # 2 for [B,C,H,W], 3 for [B,C,D,H,W]
                expand_shape = (-1,) + (1,) * (spatial_dims + 1)
                t_expanded = t_normalized.view(*expand_shape)
                predicted_clean = torch.clamp(noisy_images + t_expanded * prediction, 0, 1)
            else:
                predicted_clean = torch.clamp(noisy_images - prediction, 0, 1)

            if strategy_name == 'rflow':
                target = images - noise
                if use_fp32:
                    mse_loss = ((prediction.float() - target.float()) ** 2).mean()
                else:
                    mse_loss = ((prediction - target) ** 2).mean()
            else:
                if use_fp32:
                    mse_loss = ((prediction.float() - noise.float()) ** 2).mean()
                else:
                    mse_loss = ((prediction - noise) ** 2).mean()

            # Perceptual loss always uses FP32 (pretrained networks need it)
            if perceptual_weight > 0:
                p_loss = perceptual_fn(predicted_clean.float(), images.float())
            else:
                p_loss = torch.tensor(0.0, device=images.device)

            total_loss = mse_loss + perceptual_weight * p_loss
            return total_loss, mse_loss, p_loss, predicted_clean

        def _forward_dual(
            model: nn.Module,
            perceptual_fn: nn.Module,
            model_input: torch.Tensor,
            timesteps: torch.Tensor,
            images_0: torch.Tensor,
            images_1: torch.Tensor,
            noise_0: torch.Tensor,
            noise_1: torch.Tensor,
            noisy_0: torch.Tensor,
            noisy_1: torch.Tensor,
            perceptual_weight: float,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Dual-channel forward with loss computation.

            Fuses model prediction and loss computation for CUDA graph capture.

            Args:
                model: Diffusion model.
                perceptual_fn: Perceptual loss function.
                model_input: Input to model (noisy images + conditioning).
                timesteps: Timestep tensor.
                images_0: Clean images for channel 0.
                images_1: Clean images for channel 1.
                noise_0: Noise for channel 0.
                noise_1: Noise for channel 1.
                noisy_0: Noisy images for channel 0.
                noisy_1: Noisy images for channel 1.
                perceptual_weight: Weight for perceptual loss.

            Returns:
                Tuple of (total_loss, mse_loss, perceptual_loss, clean_0, clean_1).
            """
            prediction = model(model_input, timesteps)
            pred_0 = prediction[:, 0:1, :, :]
            pred_1 = prediction[:, 1:2, :, :]

            if strategy_name == 'rflow':
                t_normalized = timesteps.float() / float(num_train_timesteps)
                # Auto-detect spatial dims from input shape for 2D/3D support
                spatial_dims = noisy_0.ndim - 2  # 2 for [B,C,H,W], 3 for [B,C,D,H,W]
                expand_shape = (-1,) + (1,) * (spatial_dims + 1)
                t_expanded = t_normalized.view(*expand_shape)
                clean_0 = torch.clamp(noisy_0 + t_expanded * pred_0, 0, 1)
                clean_1 = torch.clamp(noisy_1 + t_expanded * pred_1, 0, 1)
                target_0 = images_0 - noise_0
                target_1 = images_1 - noise_1
                if use_fp32:
                    mse_loss = (((pred_0.float() - target_0.float()) ** 2).mean() +
                               ((pred_1.float() - target_1.float()) ** 2).mean()) / 2
                else:
                    mse_loss = (((pred_0 - target_0) ** 2).mean() +
                               ((pred_1 - target_1) ** 2).mean()) / 2
            else:
                clean_0 = torch.clamp(noisy_0 - pred_0, 0, 1)
                clean_1 = torch.clamp(noisy_1 - pred_1, 0, 1)
                if use_fp32:
                    mse_loss = (((pred_0.float() - noise_0.float()) ** 2).mean() +
                               ((pred_1.float() - noise_1.float()) ** 2).mean()) / 2
                else:
                    mse_loss = (((pred_0 - noise_0) ** 2).mean() +
                               ((pred_1 - noise_1) ** 2).mean()) / 2

            if perceptual_weight > 0:
                p_loss = (perceptual_fn(clean_0.float(), images_0.float()) +
                         perceptual_fn(clean_1.float(), images_1.float())) / 2
            else:
                p_loss = torch.tensor(0.0, device=images_0.device)

            total_loss = mse_loss + perceptual_weight * p_loss
            return total_loss, mse_loss, p_loss, clean_0, clean_1

        # Compile functions
        self._compiled_single = torch.compile(
            _forward_single, mode="reduce-overhead", fullgraph=True
        )
        self._compiled_dual = torch.compile(
            _forward_dual, mode="reduce-overhead", fullgraph=True
        )

        if self.is_main_process:
            precision = "FP32" if use_fp32 else "BF16 (legacy)"
            logger.info(f"Compiled fused forward passes (CUDA graphs enabled, MSE precision: {precision})")

    def forward_single(
        self,
        model: nn.Module,
        perceptual_fn: nn.Module,
        model_input: torch.Tensor,
        timesteps: torch.Tensor,
        images: torch.Tensor,
        noise: torch.Tensor,
        noisy_images: torch.Tensor,
        perceptual_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run compiled single-channel forward.

        Args:
            model: Diffusion model.
            perceptual_fn: Perceptual loss function.
            model_input: Input to model.
            timesteps: Timestep tensor.
            images: Clean images.
            noise: Noise tensor.
            noisy_images: Noisy images.
            perceptual_weight: Weight for perceptual loss.

        Returns:
            Tuple of (total_loss, mse_loss, perceptual_loss, predicted_clean).

        Raises:
            RuntimeError: If compiled forward not setup.
        """
        if self._compiled_single is None:
            raise RuntimeError("Compiled forward not setup. Call setup(enabled=True) first.")
        return self._compiled_single(
            model, perceptual_fn, model_input, timesteps,
            images, noise, noisy_images, perceptual_weight
        )

    def forward_dual(
        self,
        model: nn.Module,
        perceptual_fn: nn.Module,
        model_input: torch.Tensor,
        timesteps: torch.Tensor,
        images_0: torch.Tensor,
        images_1: torch.Tensor,
        noise_0: torch.Tensor,
        noise_1: torch.Tensor,
        noisy_0: torch.Tensor,
        noisy_1: torch.Tensor,
        perceptual_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run compiled dual-channel forward.

        Args:
            model: Diffusion model.
            perceptual_fn: Perceptual loss function.
            model_input: Input to model.
            timesteps: Timestep tensor.
            images_0: Clean images for channel 0.
            images_1: Clean images for channel 1.
            noise_0: Noise for channel 0.
            noise_1: Noise for channel 1.
            noisy_0: Noisy images for channel 0.
            noisy_1: Noisy images for channel 1.
            perceptual_weight: Weight for perceptual loss.

        Returns:
            Tuple of (total_loss, mse_loss, perceptual_loss, clean_0, clean_1).

        Raises:
            RuntimeError: If compiled forward not setup.
        """
        if self._compiled_dual is None:
            raise RuntimeError("Compiled forward not setup. Call setup(enabled=True) first.")
        return self._compiled_dual(
            model, perceptual_fn, model_input, timesteps,
            images_0, images_1, noise_0, noise_1, noisy_0, noisy_1,
            perceptual_weight
        )


def create_compile_manager(
    use_fp32_loss: bool = True,
    strategy_name: str = 'rflow',
    num_timesteps: int = 1000,
    is_main_process: bool = True,
) -> CompiledForwardManager:
    """Factory function to create a CompiledForwardManager.

    Args:
        use_fp32_loss: Whether to compute loss in FP32.
        strategy_name: Diffusion strategy name.
        num_timesteps: Number of training timesteps.
        is_main_process: Whether this is the main process.

    Returns:
        Configured CompiledForwardManager.
    """
    return CompiledForwardManager(
        use_fp32_loss=use_fp32_loss,
        strategy_name=strategy_name,
        num_timesteps=num_timesteps,
        is_main_process=is_main_process,
    )
