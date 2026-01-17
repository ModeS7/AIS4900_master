"""Model wrappers for conditioning embeddings.

This package provides model wrappers that add conditioning signals to
diffusion models (MONAI DiffusionModelUNet):

- ModeEmbedModelWrapper: Per-sample modality conditioning
- SizeBinModelWrapper: Tumor size distribution conditioning
- CombinedModelWrapper: Combined omega + mode conditioning (ScoreAug + multi-modality)

Usage:
    from medgen.models.wrappers import (
        create_conditioning_wrapper,  # Factory for auto-selecting wrapper
        ModeEmbedModelWrapper,
        SizeBinModelWrapper,
        CombinedModelWrapper,
    )
"""

from .base_embed import create_zero_init_mlp, create_film_mlp

from .mode_embed import (
    ModeEmbedModelWrapper,
    ModeEmbedDropoutModelWrapper,
    NoModeModelWrapper,
    LateModeModelWrapper,
    FiLMModeModelWrapper,
    ModeTimeEmbed,
    MODE_ID_MAP,
    MODE_ENCODING_DIM,
    encode_mode_id,
)

from .combined_embed import (
    CombinedModelWrapper,
    CombinedTimeEmbed,
    CombinedFiLMModelWrapper,
    CombinedFiLMTimeEmbed,
    create_conditioning_wrapper,
)

from .size_bin_embed import (
    SizeBinModelWrapper,
    SizeBinTimeEmbed,
    encode_size_bins,
    DEFAULT_BIN_EDGES,
    DEFAULT_NUM_BINS,
    format_size_bins,
)

__all__ = [
    # Base utilities
    'create_zero_init_mlp',
    'create_film_mlp',
    # Mode embedding
    'ModeEmbedModelWrapper',
    'ModeEmbedDropoutModelWrapper',
    'NoModeModelWrapper',
    'LateModeModelWrapper',
    'FiLMModeModelWrapper',
    'ModeTimeEmbed',
    'MODE_ID_MAP',
    'MODE_ENCODING_DIM',
    'encode_mode_id',
    # Combined embedding
    'CombinedModelWrapper',
    'CombinedTimeEmbed',
    'CombinedFiLMModelWrapper',
    'CombinedFiLMTimeEmbed',
    'create_conditioning_wrapper',
    # Size bin embedding
    'SizeBinModelWrapper',
    'SizeBinTimeEmbed',
    'encode_size_bins',
    'DEFAULT_BIN_EDGES',
    'DEFAULT_NUM_BINS',
    'format_size_bins',
]
