"""
DEPRECATED: 3D VAE trainer module.

This module is deprecated. Use VAETrainer with spatial_dims=3 instead:

    from medgen.pipeline import VAETrainer
    trainer = VAETrainer(cfg, spatial_dims=3)
    # or
    trainer = VAETrainer.create_3d(cfg)

This file is maintained for backward compatibility only.
"""
import warnings

from .vae_trainer import VAETrainer, CheckpointedAutoencoder

__all__ = ['VAE3DTrainer', 'CheckpointedAutoencoder']


class VAE3DTrainer(VAETrainer):
    """DEPRECATED: Use VAETrainer with spatial_dims=3 instead.

    This class is maintained for backward compatibility only.
    """

    def __init__(self, cfg, **kwargs):
        """Initialize 3D VAE trainer.

        Args:
            cfg: Hydra configuration object.
            **kwargs: Additional arguments.
        """
        warnings.warn(
            "VAE3DTrainer is deprecated. Use VAETrainer with spatial_dims=3 instead: "
            "VAETrainer(cfg, spatial_dims=3) or VAETrainer.create_3d(cfg)",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(cfg, spatial_dims=3, **kwargs)
