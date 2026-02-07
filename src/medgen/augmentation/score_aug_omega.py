"""Omega encoding and mode intensity scaling for ScoreAug.

Contains standalone functions for encoding transforms to tensor format
and applying/inverting modality-specific intensity scaling.

Moved from score_aug.py during file split.
"""

from typing import Any

import torch

from medgen.core.spatial_utils import broadcast_to_spatial

# =============================================================================
# Mode-Specific Intensity Scaling
# =============================================================================
# Each modality gets a different intensity scale applied AFTER noise addition.
# This makes mode conditioning NECESSARY - model cannot predict correct output
# without knowing the scale factor (similar to how rotation requires omega).
#
# Scales are intentionally asymmetric around 1.0 to force the model to learn
# modality-specific features rather than just inverting a simple transform.

MODE_INTENSITY_SCALE = {
    0: 0.85,   # bravo  - darker
    1: 1.15,   # flair  - brighter
    2: 0.92,   # t1_pre - slightly darker
    3: 1.08,   # t1_gd  - slightly brighter
}

# Reverse mapping for inference
MODE_INTENSITY_SCALE_INV = {k: 1.0 / v for k, v in MODE_INTENSITY_SCALE.items()}

# 3D aliases
MODE_INTENSITY_SCALE_3D = MODE_INTENSITY_SCALE
MODE_INTENSITY_SCALE_INV_3D = MODE_INTENSITY_SCALE_INV


def apply_mode_intensity_scale(
    x: torch.Tensor,
    mode_id: torch.Tensor | None,
    spatial_dims: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply modality-specific intensity scaling to input (per-sample).

    This makes mode conditioning NECESSARY for correct predictions.
    The model sees scaled input but must predict unscaled target.

    Supports mixed modalities within a batch - each sample gets its own
    scale factor based on its mode_id.

    Args:
        x: Input tensor [B, C, H, W] or [B, C, D, H, W]
        mode_id: Mode ID tensor [B] or None (0=bravo, 1=flair, 2=t1_pre, 3=t1_gd)
        spatial_dims: Number of spatial dimensions (2 or 3)

    Returns:
        Tuple of (scaled_input, scale_factors)
        - scaled_input: x * scale_factors (per-sample scaling)
        - scale_factors: Tensor [B, 1, 1, 1] or [B, 1, 1, 1, 1] with per-sample scales
    """
    if mode_id is None:
        return x, torch.ones(1, device=x.device, dtype=x.dtype)

    if mode_id.dim() == 0:
        mode_id = mode_id.unsqueeze(0)

    if mode_id.numel() == 0:
        return x, torch.ones(1, device=x.device, dtype=x.dtype)

    # Per-sample scales [B]
    scales = torch.tensor(
        [MODE_INTENSITY_SCALE.get(int(m.item()), 1.0) for m in mode_id],
        device=x.device,
        dtype=x.dtype,
    )
    scales = broadcast_to_spatial(scales, spatial_dims)

    return x * scales, scales


def inverse_mode_intensity_scale(
    x: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Inverse the mode intensity scaling (per-sample).

    Args:
        x: Scaled tensor [B, C, H, W] or [B, C, D, H, W]
        scale: The scale factors that were applied [B, 1, ...] or [1]

    Returns:
        Unscaled tensor: x / scale
    """
    if scale.numel() == 1 and scale.item() == 1.0:
        return x
    return x / scale


# =============================================================================
# Omega Conditioning (compile-compatible implementation)
# =============================================================================

# Omega encoding format (36 dims total):
#
# Layout:
#   Dims 0-3: Active mask (spatial, translation, cutout, pattern)
#   Dims 4-9: Spatial type (rot90, hflip, vflip, rot90_hflip/flip_d, flip_h, flip_w)
#   Dim 10: rot_k normalized (0-1)
#   Dims 11-13: translation params (dx/dd, dy/dh, dw)
#   Dims 14-15: reserved
#   Dims 16-31: Pattern ID one-hot (16 patterns)
#   Dims 32-35: Mode one-hot (4 modalities) - for mode intensity scaling
#
OMEGA_ENCODING_DIM = 36


def encode_omega(
    omega: dict[str, Any] | None,
    device: torch.device,
    mode_id: torch.Tensor | None = None,
    spatial_dims: int = 2,
) -> torch.Tensor:
    """Encode omega dict into tensor format for MLP.

    All samples in a batch get the same transform, so we return shape (1, 36)
    which broadcasts to (B, 36) in the MLP. This keeps the buffer shape constant
    for torch.compile compatibility.

    Supports single-transform mode, compose mode, v2 mode, and mode intensity scaling.

    Args:
        omega: Transform parameters dict or None
        device: Target device
        mode_id: Optional mode ID tensor for intensity scaling (0=bravo, 1=flair, etc.)
        spatial_dims: Number of spatial dimensions (2 or 3)

    Returns:
        Tensor [1, OMEGA_ENCODING_DIM] encoding the transform + mode
    """
    enc = torch.zeros(1, OMEGA_ENCODING_DIM, device=device)

    # Encode mode intensity scaling in dims 32-35 (always, if provided)
    # NOTE: Uses first sample's mode_id since omega transforms are per-batch.
    # Actual per-sample mode conditioning is handled by ModeTimeEmbed.
    if mode_id is not None and mode_id.numel() > 0:
        if mode_id.dim() == 0:
            idx = mode_id.item()
        else:
            idx = mode_id[0].item()
        if 0 <= idx < 4:
            enc[0, 32 + int(idx)] = 1.0

    if omega is None:
        # Identity: type_onehot[0] = 1, rest = 0
        enc[0, 0] = 1.0
        return enc

    # Check for v2 mode
    if omega.get('v2', False):
        transforms = omega['transforms']
        for t, p in transforms:
            _encode_single_transform(enc, t, p, spatial_dims, is_v2=True)
        return enc

    # Check for compose mode (legacy)
    if omega.get('compose', False):
        transforms = omega['transforms']
        for t, p in transforms:
            _encode_single_transform(enc, t, p, spatial_dims, is_v2=True)
        return enc

    # Single transform mode (legacy)
    _encode_single_transform_legacy(enc, omega['type'], omega['params'], spatial_dims)
    return enc


def _encode_single_transform(
    enc: torch.Tensor,
    transform_type: str,
    params: dict[str, Any],
    spatial_dims: int,
    is_v2: bool = True,
) -> None:
    """Encode a single transform for v2/compose mode (in-place).

    Layout:
        Dims 0-3: Active mask (spatial, translation, cutout, pattern)
        Dims 4-9: Spatial type (rot90/rot90_d, hflip/rot90_h, vflip/rot90_w, rot90_hflip/flip_d, flip_h, flip_w)
        Dim 10: rot_k normalized
        Dims 11-13: dx/dd, dy/dh, (dw for 3D) (translation)
        Dims 14-15: reserved
        Dims 16-31: Pattern ID one-hot
    """
    # 2D spatial transforms
    if transform_type == 'rot90':
        enc[0, 0] = 1.0  # spatial active
        enc[0, 4] = 1.0  # rot90 type
        enc[0, 10] = params['k'] / 3.0  # normalized k
    elif transform_type == 'hflip':
        enc[0, 0] = 1.0  # spatial active
        enc[0, 5] = 1.0  # hflip type
    elif transform_type == 'vflip':
        enc[0, 0] = 1.0  # spatial active
        enc[0, 6] = 1.0  # vflip type
    elif transform_type == 'rot90_hflip':
        enc[0, 0] = 1.0  # spatial active
        enc[0, 7] = 1.0  # rot90_hflip type
        enc[0, 10] = params['k'] / 3.0

    # 3D spatial transforms
    elif transform_type == 'rot90_3d':
        enc[0, 0] = 1.0  # spatial active
        axis = params['axis']
        k = params['k']
        if axis == 'd':
            enc[0, 4] = 1.0  # rot90_d
        elif axis == 'h':
            enc[0, 5] = 1.0  # rot90_h
        elif axis == 'w':
            enc[0, 6] = 1.0  # rot90_w
        enc[0, 10] = k / 3.0  # normalized k
    elif transform_type == 'flip_d':
        enc[0, 0] = 1.0  # spatial active
        enc[0, 7] = 1.0  # flip_d type
    elif transform_type == 'flip_h':
        enc[0, 0] = 1.0  # spatial active
        enc[0, 8] = 1.0  # flip_h type
    elif transform_type == 'flip_w':
        enc[0, 0] = 1.0  # spatial active
        enc[0, 9] = 1.0  # flip_w type

    # Translation
    elif transform_type == 'translate':
        enc[0, 1] = 1.0  # translation active
        if 'dx' in params:  # 2D
            enc[0, 11] = params['dx']
            enc[0, 12] = params['dy']
        else:  # 3D
            enc[0, 11] = params['dd']
            enc[0, 12] = params['dh']
            enc[0, 13] = params['dw']

    # Cutout
    elif transform_type == 'cutout':
        enc[0, 2] = 1.0  # cutout active

    # Fixed pattern
    elif transform_type == 'pattern':
        enc[0, 3] = 1.0  # pattern active
        pattern_id = params['pattern_id']
        enc[0, 16 + pattern_id] = 1.0  # pattern one-hot


def _encode_single_transform_legacy(
    enc: torch.Tensor,
    transform_type: str,
    params: dict[str, Any],
    spatial_dims: int,
) -> None:
    """Encode a single transform for legacy single-transform mode (in-place)."""
    # Use same layout as v2 for consistency
    _encode_single_transform(enc, transform_type, params, spatial_dims, is_v2=False)


# =============================================================================
# 3D Backward Compatibility Aliases
# =============================================================================

OMEGA_ENCODING_DIM_3D = OMEGA_ENCODING_DIM


def apply_mode_intensity_scale_3d(
    x: torch.Tensor,
    mode_id: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply modality-specific intensity scaling to 3D input (per-sample).

    Backwards compatibility alias for apply_mode_intensity_scale with spatial_dims=3.
    """
    return apply_mode_intensity_scale(x, mode_id, spatial_dims=3)


def inverse_mode_intensity_scale_3d(
    x: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Inverse the mode intensity scaling for 3D (per-sample).

    Backwards compatibility alias for inverse_mode_intensity_scale.
    """
    return inverse_mode_intensity_scale(x, scale)


def encode_omega_3d(
    omega: dict[str, Any] | None,
    device: torch.device,
    mode_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encode 3D omega dict into tensor format for MLP.

    Backwards compatibility alias for encode_omega with spatial_dims=3.
    """
    return encode_omega(omega, device, mode_id=mode_id, spatial_dims=3)
