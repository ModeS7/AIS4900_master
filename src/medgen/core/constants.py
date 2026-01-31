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
    BRAVO_SEG_COND = "bravo_seg_cond"  # Latent BRAVO conditioned on latent seg
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


def get_modality_for_mode(mode: str) -> str:
    """Map diffusion mode to data modality for file loading.

    Diffusion modes (e.g., 'bravo_seg_cond') don't always map 1:1 to
    modality file names (e.g., 'bravo.nii.gz'). This function handles
    the translation.

    Args:
        mode: Diffusion training mode (e.g., 'bravo', 'bravo_seg_cond', 'seg')

    Returns:
        Modality name for file loading (e.g., 'bravo', 'seg', 'dual')

    Raises:
        ValueError: If mode requires special handling not supported here
    """
    # Direct modalities: mode == file modality
    if mode in ('bravo', 'seg', 'flair', 't1_pre', 't1_gd', 'dual'):
        return mode

    # Segmentation modes load seg.nii.gz
    if mode in ('seg_conditioned', 'seg_conditioned_input'):
        return 'seg'

    # bravo_seg_cond loads bravo.nii.gz (latent seg comes from cache)
    if mode == 'bravo_seg_cond':
        return 'bravo'

    # Multi-modality modes need special dataloaders
    if mode in ('multi', 'multi_modality'):
        raise ValueError(
            f"Mode '{mode}' requires a multi-modality dataloader, "
            "not create_vae_3d_dataloader()"
        )

    raise ValueError(f"Unknown mode for modality mapping: {mode}")
