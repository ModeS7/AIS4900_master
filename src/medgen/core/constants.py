"""Named constants extracted from codebase.

This module centralizes magic numbers and threshold values used throughout
the training and generation pipelines.
"""
from enum import Enum


class ModeType(str, Enum):
    """Valid training/generation modes.

    Using str inheritance allows direct comparison with config strings:
        mode_name == ModeType.DUAL  # works even if mode_name is "dual"

    Note on multi-modality modes (easy to confuse):
    - MULTI: Multi-modality DIFFUSION training with mode embedding conditioning.
      The model learns to generate specific modalities (t1_pre, t1_gd, bravo, seg)
      based on a mode ID embedding. Use this when training conditional diffusion.
    - MULTI_MODALITY: Multi-modality VAE/compression training (pools all modalities).
      The autoencoder treats all modality slices identically without knowing which
      modality they are. Use this for VAE, VQ-VAE, and DC-AE training.
    """
    SEG = "seg"               # Segmentation mask only
    SEG_CONDITIONED = "seg_conditioned"  # Seg mask generation conditioned on size bins
    SEG_CONDITIONED_INPUT = "seg_conditioned_input"  # Seg conditioned via input channel concat
    BRAVO = "bravo"           # Single MRI modality (BRAVO/FLAIR)
    DUAL = "dual"             # Two MRI modalities (t1_pre + t1_gd)
    MULTI = "multi"           # Multi-modality diffusion with mode embedding
    MULTI_MODALITY = "multi_modality"  # Multi-modality VAE (no mode embedding)


# Data processing thresholds
# Ground truth masks: low threshold to preserve all positive pixels
BINARY_THRESHOLD_GT = 0.01
# Generated masks: higher threshold to filter out noise
BINARY_THRESHOLD_GEN = 0.1
# Max tumor size for valid generated masks (used in generate.py)
MAX_WHITE_PERCENTAGE = 0.04

# Model defaults (used in generate.py for inference)
DEFAULT_CHANNELS = (128, 256, 256)
DEFAULT_ATTENTION_LEVELS = (False, True, True)
DEFAULT_NUM_RES_BLOCKS = 1
DEFAULT_NUM_HEAD_CHANNELS = 256

# Data loading
DEFAULT_NUM_WORKERS = 4  # Parallel data loading workers

# Dual mode default image keys
DEFAULT_DUAL_IMAGE_KEYS = ['t1_pre', 't1_gd']
