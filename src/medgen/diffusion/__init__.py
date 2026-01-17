"""
Diffusion model training components.

This package provides:
- Strategies: DDPM and Rectified Flow diffusion algorithms
- Modes: Training modes for different conditioning setups
- Spaces: Pixel and latent space abstractions for diffusion

Usage:
    from medgen.diffusion import (
        # Strategies
        DiffusionStrategy, DDPMStrategy, RFlowStrategy,
        # Modes
        TrainingMode, SegmentationMode, ConditionalSingleMode,
        ConditionalDualMode, MultiModalityMode, SegmentationConditionedMode,
        # Spaces
        DiffusionSpace, PixelSpace, LatentSpace, load_vae_for_latent_space,
    )
"""

# Strategies
from .strategies import (
    DiffusionStrategy,
    DDPMStrategy,
    RFlowStrategy,
    ParsedModelInput,
)

# Modes
from .modes import (
    TrainingMode,
    SegmentationMode,
    ConditionalSingleMode,
    ConditionalDualMode,
    MultiModalityMode,
    SegmentationConditionedMode,
)

# Spaces
from .spaces import (
    DiffusionSpace,
    PixelSpace,
    LatentSpace,
    load_vae_for_latent_space,
)

__all__ = [
    # Strategies
    'DiffusionStrategy',
    'DDPMStrategy',
    'RFlowStrategy',
    'ParsedModelInput',
    # Modes
    'TrainingMode',
    'SegmentationMode',
    'ConditionalSingleMode',
    'ConditionalDualMode',
    'MultiModalityMode',
    'SegmentationConditionedMode',
    # Spaces
    'DiffusionSpace',
    'PixelSpace',
    'LatentSpace',
    'load_vae_for_latent_space',
]
