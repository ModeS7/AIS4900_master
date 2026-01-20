"""
DEPRECATED: 3D VQ-VAE trainer module.

This module is deprecated. Use VQVAETrainer with spatial_dims=3 instead:

    from medgen.pipeline import VQVAETrainer
    trainer = VQVAETrainer(cfg, spatial_dims=3)
    # or
    trainer = VQVAETrainer.create_3d(cfg)

This file is maintained for backward compatibility only.
"""
import warnings

from .vqvae_trainer import VQVAETrainer, CheckpointedVQVAE

__all__ = ['VQVAE3DTrainer', 'CheckpointedVQVAE']


class VQVAE3DTrainer(VQVAETrainer):
    """DEPRECATED: Use VQVAETrainer with spatial_dims=3 instead.

    This class is maintained for backward compatibility only.
    """

    def __init__(self, cfg, **kwargs):
        """Initialize 3D VQ-VAE trainer.

        Args:
            cfg: Hydra configuration object.
            **kwargs: Additional arguments.
        """
        warnings.warn(
            "VQVAE3DTrainer is deprecated. Use VQVAETrainer with spatial_dims=3 instead: "
            "VQVAETrainer(cfg, spatial_dims=3) or VQVAETrainer.create_3d(cfg)",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(cfg, spatial_dims=3, **kwargs)
