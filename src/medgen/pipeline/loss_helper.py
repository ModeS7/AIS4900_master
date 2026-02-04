"""Loss computation helpers for diffusion training.

This module provides:
- DiffusionLossHelper: Helper class for advanced loss computation techniques

Consolidates loss-related wrapper methods from DiffusionTrainer that delegate
to training_tricks.py and losses.py functions.
"""

import torch


class DiffusionLossHelper:
    """Helper for advanced loss computation techniques in diffusion training.

    Provides a clean interface for:
    - Conditioning dropout (CFG)
    - Noise augmentation
    - Timestep jitter
    - Curriculum learning
    - Min-SNR weighting
    - Regional weighting
    - Gradient noise injection
    - Feature perturbation
    - Self-conditioning
    - Augmented diffusion channel masking
    """

    def __init__(self, trainer) -> None:
        """Initialize with trainer reference for config access.

        Args:
            trainer: DiffusionTrainer instance.
        """
        self.trainer = trainer

    def apply_conditioning_dropout(
        self,
        conditioning: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor | None:
        """Apply per-sample CFG dropout to conditioning tensor.

        Args:
            conditioning: Conditioning tensor to apply dropout to.
            batch_size: Batch size for sampling dropout mask.

        Returns:
            Conditioning tensor with some samples zeroed out.
        """
        from .training_tricks import apply_conditioning_dropout
        return apply_conditioning_dropout(self.trainer, conditioning, batch_size)

    def apply_noise_augmentation(
        self,
        noise: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Apply noise offset/scaling augmentation.

        Args:
            noise: Noise tensor or dict of noise tensors.

        Returns:
            Augmented noise.
        """
        from .training_tricks import apply_noise_augmentation
        return apply_noise_augmentation(self.trainer, noise)

    def apply_timestep_jitter(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to timesteps for regularization.

        Args:
            timesteps: Timestep tensor.

        Returns:
            Jittered timesteps.
        """
        from .training_tricks import apply_timestep_jitter
        return apply_timestep_jitter(self.trainer, timesteps)

    def get_curriculum_range(self, epoch: int) -> tuple[float, float] | None:
        """Get timestep range for curriculum learning.

        Args:
            epoch: Current epoch number.

        Returns:
            Tuple of (min_t, max_t) or None if curriculum disabled.
        """
        from .training_tricks import get_curriculum_range
        return get_curriculum_range(self.trainer, epoch)

    def compute_min_snr_weighted_mse(
        self,
        prediction: torch.Tensor,
        images: torch.Tensor | dict[str, torch.Tensor],
        noise: torch.Tensor | dict[str, torch.Tensor],
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss with Min-SNR weighting.

        Args:
            prediction: Model prediction.
            images: Clean images.
            noise: Noise tensor.
            timesteps: Timestep tensor.

        Returns:
            Weighted MSE loss.
        """
        from .losses import compute_min_snr_weighted_mse
        return compute_min_snr_weighted_mse(self.trainer, prediction, images, noise, timesteps)

    def compute_region_weighted_mse(
        self,
        prediction: torch.Tensor,
        images: torch.Tensor | dict[str, torch.Tensor],
        noise: torch.Tensor | dict[str, torch.Tensor],
        seg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss with per-pixel regional weighting.

        Args:
            prediction: Model prediction.
            images: Clean images.
            noise: Noise tensor.
            seg_mask: Segmentation mask for weighting.

        Returns:
            Regionally weighted MSE loss.
        """
        from .losses import compute_region_weighted_mse
        return compute_region_weighted_mse(self.trainer, prediction, images, noise, seg_mask)

    def add_gradient_noise(self, step: int) -> None:
        """Add Gaussian noise to gradients for regularization.

        Args:
            step: Current training step.
        """
        from .training_tricks import add_gradient_noise
        add_gradient_noise(self.trainer, step)

    def setup_feature_perturbation(self) -> None:
        """Setup forward hooks for feature perturbation."""
        from .training_tricks import setup_feature_perturbation
        setup_feature_perturbation(self.trainer)

    def remove_feature_perturbation_hooks(self) -> None:
        """Remove feature perturbation hooks."""
        from .training_tricks import remove_feature_perturbation_hooks
        remove_feature_perturbation_hooks(self.trainer)

    def compute_self_conditioning_loss(
        self,
        model_input: torch.Tensor,
        timesteps: torch.Tensor,
        prediction: torch.Tensor,
        mode_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute self-conditioning consistency loss.

        Args:
            model_input: Model input tensor.
            timesteps: Timestep tensor.
            prediction: Previous prediction for self-conditioning.
            mode_id: Optional mode ID for multi-modality.

        Returns:
            Self-conditioning loss.
        """
        from .losses import compute_self_conditioning_loss
        return compute_self_conditioning_loss(self.trainer, model_input, timesteps, prediction, mode_id)

    def create_aug_diff_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create channel mask for augmented diffusion training.

        Implements DC-AE 1.5 channel masking for latent diffusion.

        Args:
            tensor: Tensor to create mask for (used to get shape/device).

        Returns:
            Binary channel mask [1, C, 1, 1, ...].
        """
        from .training_tricks import create_aug_diff_mask
        return create_aug_diff_mask(self.trainer, tensor)

    def get_aug_diff_channel_steps(self, num_channels: int) -> list[int]:
        """Get list of channel counts for augmented diffusion masking.

        Args:
            num_channels: Total number of channels.

        Returns:
            List of valid channel counts.
        """
        from .training_tricks import get_aug_diff_channel_steps
        return get_aug_diff_channel_steps(self.trainer, num_channels)


def create_loss_helper(trainer) -> DiffusionLossHelper:
    """Factory function to create a DiffusionLossHelper.

    Args:
        trainer: DiffusionTrainer instance.

    Returns:
        Configured DiffusionLossHelper.
    """
    return DiffusionLossHelper(trainer)
