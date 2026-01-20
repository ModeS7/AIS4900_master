"""
DEPRECATED: 3D DC-AE trainer module.

This module is deprecated. Use DCAETrainer with spatial_dims=3 instead:

    from medgen.pipeline import DCAETrainer
    trainer = DCAETrainer(cfg, spatial_dims=3)
    # or
    trainer = DCAETrainer.create_3d(cfg)

This file is maintained for backward compatibility only.
"""
import warnings

from .dcae_trainer import DCAETrainer, CheckpointedAutoencoderDC3D

__all__ = ['DCAE3DTrainer', 'CheckpointedAutoencoderDC3D']


class DCAE3DTrainer(DCAETrainer):
    """DEPRECATED: Use DCAETrainer with spatial_dims=3 instead.

    This class is maintained for backward compatibility only.
    """

    def __init__(self, cfg, **kwargs):
        """Initialize 3D DC-AE trainer.

        Args:
            cfg: Hydra configuration object.
            **kwargs: Additional arguments.
        """
        warnings.warn(
            "DCAE3DTrainer is deprecated. Use DCAETrainer with spatial_dims=3 instead: "
            "DCAETrainer(cfg, spatial_dims=3) or DCAETrainer.create_3d(cfg)",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(cfg, spatial_dims=3, **kwargs)
