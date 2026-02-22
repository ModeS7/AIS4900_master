"""Training step result dataclass for unified trainer return types.

Provides a standardized return type for train_step() across all trainers,
enabling generic training loop handling and consistent logging.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

# Type alias for batch inputs accepted by train_step() and prepare_batch().
# Covers all formats: dict batches, tuple/list batches, and raw tensors.
BatchType = dict[str, Tensor] | tuple[Tensor, ...] | Tensor


@dataclass
class TrainingStepResult:
    """Unified return type for train_step() across all trainers.

    All trainers return this type, enabling generic handling in train_epoch().
    Optional fields default to 0.0 to avoid None checks.

    Attributes:
        total_loss: Combined loss used for optimization (generator loss for GANs).
        reconstruction_loss: L1/MSE reconstruction error.
        perceptual_loss: LPIPS or similar perceptual loss.
        regularization_loss: KL (VAE), VQ (VQVAE), or 0 (DCAE/Diffusion).
        adversarial_loss: Generator adversarial loss (0 if GAN disabled).
        discriminator_loss: Discriminator loss (0 if GAN disabled).
        mse_loss: MSE for diffusion noise prediction (0 for compression models).
        aux_bin_loss: Auxiliary bin prediction loss (0 if disabled).
    """

    total_loss: float
    reconstruction_loss: float
    perceptual_loss: float
    regularization_loss: float = 0.0
    adversarial_loss: float = 0.0
    discriminator_loss: float = 0.0
    mse_loss: float = 0.0
    aux_bin_loss: float = 0.0

    def to_legacy_dict(self, reg_key: str | None = 'kl') -> dict[str, float]:
        """Convert to legacy dict format for backward compatibility.

        Uses short key names matching existing TensorBoard logs.

        Args:
            reg_key: Key for regularization loss ('kl' for VAE, 'vq' for VQVAE,
                     None to exclude regularization key).

        Returns:
            Dict with keys: 'gen', 'disc', 'recon', 'perc', '{reg_key}', 'adv'.
            If reg_key is None, regularization is excluded.
        """
        result = {
            'gen': self.total_loss,
            'disc': self.discriminator_loss,
            'recon': self.reconstruction_loss,
            'perc': self.perceptual_loss,
            'adv': self.adversarial_loss,
        }
        if reg_key is not None:
            result[reg_key] = self.regularization_loss
        return result
