"""
Diffusion model training components.

This package provides:
- Strategies: DDPM and Rectified Flow diffusion algorithms
- Modes: Training modes for different conditioning setups
- Spaces: Pixel and latent space abstractions for diffusion
- Loading: Utilities for loading trained models from checkpoints

Usage:
    from medgen.diffusion import (
        # Strategies
        DiffusionStrategy, DDPMStrategy, RFlowStrategy,
        # Modes
        TrainingMode, SegmentationMode, ConditionalSingleMode,
        ConditionalDualMode, MultiModalityMode, SegmentationConditionedMode,
        # Spaces
        DiffusionSpace, PixelSpace, LatentSpace, load_vae_for_latent_space,
        # Loading
        load_diffusion_model, load_diffusion_model_with_metadata,
        detect_wrapper_type, LoadedModel,
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
    SegmentationConditionedInputMode,
)

# Spaces
from .spaces import (
    DiffusionSpace,
    PixelSpace,
    LatentSpace,
    load_vae_for_latent_space,
)

# Loading utilities
from .loading import (
    load_diffusion_model,
    load_diffusion_model_with_metadata,
    detect_wrapper_type,
    LoadedModel,
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
    'SegmentationConditionedInputMode',
    # Spaces
    'DiffusionSpace',
    'PixelSpace',
    'LatentSpace',
    'load_vae_for_latent_space',
    # Loading
    'load_diffusion_model',
    'load_diffusion_model_with_metadata',
    'detect_wrapper_type',
    'LoadedModel',
]
