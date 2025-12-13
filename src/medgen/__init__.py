"""MedGen - Synthetic Medical Image Generation using Diffusion Models."""

__version__ = "0.1.0"

from medgen.diffusion import (
    DiffusionTrainer,
    DDPMStrategy,
    RFlowStrategy,
    DiffusionStrategy,
    SegmentationMode,
    ConditionalSingleMode,
    ConditionalDualMode,
    TrainingMode,
)
from medgen.data import (
    NiFTIDataset,
    create_dataloader,
    create_dual_image_dataloader,
    extract_slices_single,
    extract_slices_dual,
    merge_sequences,
    make_binary,
)

__all__ = [
    # Trainer
    'DiffusionTrainer',
    # Strategies
    'DDPMStrategy',
    'RFlowStrategy',
    'DiffusionStrategy',
    # Modes
    'SegmentationMode',
    'ConditionalSingleMode',
    'ConditionalDualMode',
    'TrainingMode',
    # Data
    'NiFTIDataset',
    'create_dataloader',
    'create_dual_image_dataloader',
    'extract_slices_single',
    'extract_slices_dual',
    'merge_sequences',
    'make_binary',
]
