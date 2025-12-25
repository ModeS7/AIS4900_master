"""Named constants extracted from codebase.

This module centralizes magic numbers and threshold values used throughout
the training and generation pipelines.
"""
from enum import Enum


class ModeType(str, Enum):
    """Valid training/generation modes.

    Using str inheritance allows direct comparison with config strings:
        mode_name == ModeType.DUAL  # works even if mode_name is "dual"
    """
    SEG = "seg"
    BRAVO = "bravo"
    DUAL = "dual"
    MULTI = "multi"  # Multi-modality diffusion with mode embedding
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
