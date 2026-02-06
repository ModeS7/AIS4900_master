"""Mode-specific intensity scaling for diffusion training.

Each modality gets a different intensity scale applied AFTER noise addition.
This makes mode conditioning NECESSARY - model cannot predict correct output
without knowing the scale factor (similar to how rotation requires omega).

Scales are intentionally asymmetric around 1.0 to force the model to learn
modality-specific features rather than just inverting a simple transform.
"""


import torch
from torch import Tensor

from medgen.core.spatial_utils import broadcast_to_spatial

MODE_INTENSITY_SCALE = {
    0: 0.85,   # bravo  - darker
    1: 1.15,   # flair  - brighter
    2: 0.92,   # t1_pre - slightly darker
    3: 1.08,   # t1_gd  - slightly brighter
}

# Reverse mapping for inference
MODE_INTENSITY_SCALE_INV = {k: 1.0 / v for k, v in MODE_INTENSITY_SCALE.items()}


def apply_mode_intensity_scale(
    x: Tensor,
    mode_id: Tensor | None,
    spatial_dims: int = 2,
) -> tuple[Tensor, Tensor]:
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
    x: Tensor,
    scale: Tensor,
) -> Tensor:
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


# Backwards compatibility aliases
MODE_INTENSITY_SCALE_3D = MODE_INTENSITY_SCALE
MODE_INTENSITY_SCALE_INV_3D = MODE_INTENSITY_SCALE_INV


def apply_mode_intensity_scale_3d(
    x: Tensor,
    mode_id: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Apply modality-specific intensity scaling to 3D input (per-sample).

    Backwards compatibility alias for apply_mode_intensity_scale with spatial_dims=3.
    """
    return apply_mode_intensity_scale(x, mode_id, spatial_dims=3)


def inverse_mode_intensity_scale_3d(
    x: Tensor,
    scale: Tensor,
) -> Tensor:
    """Inverse the mode intensity scaling for 3D (per-sample).

    Backwards compatibility alias for inverse_mode_intensity_scale.
    """
    return inverse_mode_intensity_scale(x, scale)
