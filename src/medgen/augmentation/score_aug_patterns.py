"""Fixed pattern mask generation for ScoreAug v2 mode.

Contains deterministic mask patterns (checkerboard, grid dropout,
coarse dropout, patch dropout) and their caching infrastructure.

Moved from score_aug.py during file split.
"""

from functools import lru_cache

import numpy as np
import torch


def _checkerboard_mask(H: int, W: int, grid_size: int, offset: bool) -> torch.Tensor:
    """Generate checkerboard mask (alternating grid cells).

    Args:
        H, W: Image dimensions
        grid_size: Number of cells per dimension
        offset: If True, shift pattern by 1 cell

    Returns:
        Mask tensor [H, W] with 0=keep, 1=drop
    """
    cell_h = H // grid_size
    cell_w = W // grid_size
    mask = torch.zeros(H, W)

    for i in range(grid_size):
        for j in range(grid_size):
            # Checkerboard: drop if (i+j) is even (or odd if offset)
            drop = ((i + j) % 2 == 0) if not offset else ((i + j) % 2 == 1)
            if drop:
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                mask[y1:y2, x1:x2] = 1

    return mask


def _grid_dropout_mask(H: int, W: int, grid_size: int, drop_ratio: float, seed: int) -> torch.Tensor:
    """Generate grid dropout mask (random cells dropped).

    Args:
        H, W: Image dimensions
        grid_size: Number of cells per dimension
        drop_ratio: Fraction of cells to drop
        seed: Random seed for reproducibility

    Returns:
        Mask tensor [H, W] with 0=keep, 1=drop
    """
    rng = np.random.RandomState(seed)
    cell_h = H // grid_size
    cell_w = W // grid_size
    mask = torch.zeros(H, W)

    n_cells = grid_size * grid_size
    n_drop = int(n_cells * drop_ratio)

    # Randomly select cells to drop
    drop_indices = rng.choice(n_cells, n_drop, replace=False)

    for idx in drop_indices:
        i = idx // grid_size
        j = idx % grid_size
        y1, y2 = i * cell_h, (i + 1) * cell_h
        x1, x2 = j * cell_w, (j + 1) * cell_w
        mask[y1:y2, x1:x2] = 1

    return mask


def _coarse_dropout_mask(H: int, W: int, pattern_id: int) -> torch.Tensor:
    """Generate coarse dropout mask (predefined large holes).

    Pattern definitions:
        0: 2 holes - top-left, bottom-right corners
        1: 2 holes - top-right, bottom-left corners
        2: 3 holes - top-center, bottom-left, bottom-right
        3: 4 holes - all corners

    Args:
        H, W: Image dimensions
        pattern_id: Which pattern (0-3)

    Returns:
        Mask tensor [H, W] with 0=keep, 1=drop
    """
    mask = torch.zeros(H, W)
    hole_h = H // 4  # 25% of height
    hole_w = W // 4  # 25% of width

    if pattern_id == 0:
        # Top-left and bottom-right
        mask[:hole_h, :hole_w] = 1
        mask[-hole_h:, -hole_w:] = 1
    elif pattern_id == 1:
        # Top-right and bottom-left
        mask[:hole_h, -hole_w:] = 1
        mask[-hole_h:, :hole_w] = 1
    elif pattern_id == 2:
        # Top-center, bottom-left, bottom-right
        mask[:hole_h, W//2 - hole_w//2:W//2 + hole_w//2] = 1
        mask[-hole_h:, :hole_w] = 1
        mask[-hole_h:, -hole_w:] = 1
    elif pattern_id == 3:
        # All four corners
        mask[:hole_h, :hole_w] = 1
        mask[:hole_h, -hole_w:] = 1
        mask[-hole_h:, :hole_w] = 1
        mask[-hole_h:, -hole_w:] = 1

    return mask


def _patch_dropout_mask(H: int, W: int, patch_size: int, drop_ratio: float, seed: int) -> torch.Tensor:
    """Generate patch dropout mask (MAE-style random patches).

    Args:
        H, W: Image dimensions
        patch_size: Size of each patch in pixels
        drop_ratio: Fraction of patches to drop
        seed: Random seed for reproducibility

    Returns:
        Mask tensor [H, W] with 0=keep, 1=drop
    """
    rng = np.random.RandomState(seed)

    n_patches_h = H // patch_size
    n_patches_w = W // patch_size
    n_patches = n_patches_h * n_patches_w
    n_drop = int(n_patches * drop_ratio)

    mask = torch.zeros(H, W)

    # Randomly select patches to drop
    drop_indices = rng.choice(n_patches, n_drop, replace=False)

    for idx in drop_indices:
        i = idx // n_patches_w
        j = idx % n_patches_w
        y1, y2 = i * patch_size, (i + 1) * patch_size
        x1, x2 = j * patch_size, (j + 1) * patch_size
        mask[y1:y2, x1:x2] = 1

    return mask


def generate_pattern_mask(
    pattern_id: int,
    H: int,
    W: int,
    D: int | None = None,
    spatial_dims: int = 2,
) -> torch.Tensor:
    """Generate mask for a fixed pattern ID.

    Pattern IDs (16 total):
        0-3:   Checkerboard (4x4 std, 4x4 offset, 8x8 std, 8x8 offset)
        4-7:   Grid dropout (4x4 25% seed0, 4x4 25% seed1, 4x4 50% seed0, 4x4 50% seed1)
        8-11:  Coarse dropout (patterns 0-3)
        12-15: Patch dropout (8x8 25% seedA, 8x8 25% seedB, 8x8 50% seedA, 8x8 50% seedB)

    Args:
        pattern_id: Pattern index (0-15)
        H, W: Spatial dimensions (height, width)
        D: Depth dimension for 3D (required if spatial_dims=3)
        spatial_dims: Number of spatial dimensions (2 or 3)

    Returns:
        Mask tensor [H, W] for 2D or [D, H, W] for 3D, with 0=keep, 1=drop
    """
    # Generate 2D base mask
    if pattern_id < 4:
        # Checkerboard patterns
        if pattern_id == 0:
            mask_2d = _checkerboard_mask(H, W, grid_size=4, offset=False)
        elif pattern_id == 1:
            mask_2d = _checkerboard_mask(H, W, grid_size=4, offset=True)
        elif pattern_id == 2:
            mask_2d = _checkerboard_mask(H, W, grid_size=8, offset=False)
        elif pattern_id == 3:
            mask_2d = _checkerboard_mask(H, W, grid_size=8, offset=True)

    elif pattern_id < 8:
        # Grid dropout patterns
        if pattern_id == 4:
            mask_2d = _grid_dropout_mask(H, W, grid_size=4, drop_ratio=0.25, seed=42)
        elif pattern_id == 5:
            mask_2d = _grid_dropout_mask(H, W, grid_size=4, drop_ratio=0.25, seed=123)
        elif pattern_id == 6:
            mask_2d = _grid_dropout_mask(H, W, grid_size=4, drop_ratio=0.50, seed=42)
        elif pattern_id == 7:
            mask_2d = _grid_dropout_mask(H, W, grid_size=4, drop_ratio=0.50, seed=123)

    elif pattern_id < 12:
        # Coarse dropout patterns
        mask_2d = _coarse_dropout_mask(H, W, pattern_id=pattern_id - 8)

    else:
        # Patch dropout patterns (8x8 patches for 128px image = 16 patches)
        patch_size = max(H // 8, 1)
        if pattern_id == 12:
            mask_2d = _patch_dropout_mask(H, W, patch_size=patch_size, drop_ratio=0.25, seed=42)
        elif pattern_id == 13:
            mask_2d = _patch_dropout_mask(H, W, patch_size=patch_size, drop_ratio=0.25, seed=123)
        elif pattern_id == 14:
            mask_2d = _patch_dropout_mask(H, W, patch_size=patch_size, drop_ratio=0.50, seed=42)
        elif pattern_id == 15:
            mask_2d = _patch_dropout_mask(H, W, patch_size=patch_size, drop_ratio=0.50, seed=123)
        else:
            mask_2d = torch.zeros(H, W)

    if spatial_dims == 2:
        return mask_2d
    else:
        # Expand to 3D: uniform across depth
        if D is None:
            raise ValueError("D required for spatial_dims=3")
        return mask_2d.unsqueeze(0).expand(D, -1, -1).contiguous()


# Pattern category names for debugging/logging
PATTERN_NAMES = [
    "checker_4x4",      # 0
    "checker_4x4_off",  # 1
    "checker_8x8",      # 2
    "checker_8x8_off",  # 3
    "grid_4x4_25_s0",   # 4
    "grid_4x4_25_s1",   # 5
    "grid_4x4_50_s0",   # 6
    "grid_4x4_50_s1",   # 7
    "coarse_diag1",     # 8
    "coarse_diag2",     # 9
    "coarse_tri",       # 10
    "coarse_corners",   # 11
    "patch_25_s0",      # 12
    "patch_25_s1",      # 13
    "patch_50_s0",      # 14
    "patch_50_s1",      # 15
]

NUM_PATTERNS = 16

# 3D aliases
PATTERN_NAMES_3D = PATTERN_NAMES
NUM_PATTERNS_3D = NUM_PATTERNS


@lru_cache(maxsize=256)  # 16 patterns x 16 size combinations
def _cached_generate_pattern_mask(
    pattern_id: int,
    H: int,
    W: int,
    D: int | None,
    spatial_dims: int,
) -> torch.Tensor:
    """Thread-safe cached pattern mask generation.

    LRU cache provides:
    - Automatic size bounding (maxsize=256)
    - Thread-safe operations (GIL + atomic dict)
    - LRU eviction when full
    """
    return generate_pattern_mask(pattern_id, H, W, D=D, spatial_dims=spatial_dims)


def clear_pattern_cache() -> None:
    """Clear the pattern mask cache. Call between training runs if needed."""
    _cached_generate_pattern_mask.cache_clear()


# 3D alias
generate_pattern_mask_3d = generate_pattern_mask
