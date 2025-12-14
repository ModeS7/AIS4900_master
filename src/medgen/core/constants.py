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
    T1_PRE = "t1_pre"
    T1_GD = "t1_gd"
    MULTI_MODALITY = "multi_modality"


# Data processing thresholds
# Ground truth masks: low threshold to preserve all positive pixels
BINARY_THRESHOLD_GT = 0.01
# Generated masks: higher threshold to filter out noise
BINARY_THRESHOLD_GEN = 0.1
# Legacy alias (prefer explicit GT/GEN versions)
BINARY_THRESHOLD = BINARY_THRESHOLD_GT
MAX_WHITE_PERCENTAGE = 0.04

# Training defaults (also in configs, kept as fallbacks)
DEFAULT_GRADIENT_CLIP_NORM = 1.0
DEFAULT_EMA_UPDATE_AFTER_STEP = 100
DEFAULT_EMA_UPDATE_EVERY = 10
DEFAULT_MIN_SNR_GAMMA = 5.0
DEFAULT_PERCEPTUAL_WEIGHT = 0.001

# Model defaults
DEFAULT_CHANNELS = (128, 256, 256)
DEFAULT_ATTENTION_LEVELS = (False, True, True)
DEFAULT_NUM_RES_BLOCKS = 1
DEFAULT_NUM_HEAD_CHANNELS = 256

# Data loading
DEFAULT_NUM_WORKERS = 4

# Dual mode default image keys
DEFAULT_DUAL_IMAGE_KEYS = ['t1_pre', 't1_gd']
