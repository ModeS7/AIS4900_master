"""Score Augmentation for 3D Diffusion Models.

Reference: https://arxiv.org/abs/2508.07926

Adapts ScoreAug transforms for 3D volumetric data [B, C, D, H, W].

Key differences from 2D:
- Rotations can be around any of 3 axes (D, H, W)
- Flips along all 3 axes
- 3D translations (±40% X, ±20% Y, 0% Z - brain is centered in Z)
- 3D cutout regions
- Fixed patterns: 2D patterns uniform across depth dimension

Conditioning requirements (per paper):
- Rotation: REQUIRES omega conditioning (noise is rotation-invariant, model can cheat)
- Translation/Cutout: Work without conditioning but risk data leakage

v2 mode adds structured masking patterns:
- Non-destructive transforms (can stack): rotation, flip, translation
- Destructive transforms (pick one): cutout OR fixed patterns
- Fixed patterns: checkerboard, grid dropout, coarse dropout, patch dropout
  (applied uniformly across depth dimension - "2.5D patterns")
"""

import random
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from medgen.models.wrappers import create_zero_init_mlp


# =============================================================================
# Fixed Pattern Definitions for 3D (2D patterns uniform across depth)
# =============================================================================
# These are deterministic masks that the network learns via one-hot embedding.
# Patterns are generated as 2D masks and then extended uniformly across D.

def _checkerboard_mask_3d(D: int, H: int, W: int, grid_size: int, offset: bool) -> torch.Tensor:
    """Generate 3D checkerboard mask (2D pattern uniform across depth).

    Args:
        D, H, W: Volume dimensions
        grid_size: Number of cells per spatial dimension
        offset: If True, shift pattern by 1 cell

    Returns:
        Mask tensor [D, H, W] with 0=keep, 1=drop
    """
    cell_h = H // grid_size
    cell_w = W // grid_size
    mask_2d = torch.zeros(H, W)

    for i in range(grid_size):
        for j in range(grid_size):
            # Checkerboard: drop if (i+j) is even (or odd if offset)
            drop = ((i + j) % 2 == 0) if not offset else ((i + j) % 2 == 1)
            if drop:
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                mask_2d[y1:y2, x1:x2] = 1

    # Expand to 3D: uniform across depth
    return mask_2d.unsqueeze(0).expand(D, -1, -1).contiguous()


def _grid_dropout_mask_3d(D: int, H: int, W: int, grid_size: int, drop_ratio: float, seed: int) -> torch.Tensor:
    """Generate 3D grid dropout mask (2D pattern uniform across depth).

    Args:
        D, H, W: Volume dimensions
        grid_size: Number of cells per spatial dimension
        drop_ratio: Fraction of cells to drop
        seed: Random seed for reproducibility

    Returns:
        Mask tensor [D, H, W] with 0=keep, 1=drop
    """
    rng = np.random.RandomState(seed)
    cell_h = H // grid_size
    cell_w = W // grid_size
    mask_2d = torch.zeros(H, W)

    n_cells = grid_size * grid_size
    n_drop = int(n_cells * drop_ratio)

    # Randomly select cells to drop
    drop_indices = rng.choice(n_cells, n_drop, replace=False)

    for idx in drop_indices:
        i = idx // grid_size
        j = idx % grid_size
        y1, y2 = i * cell_h, (i + 1) * cell_h
        x1, x2 = j * cell_w, (j + 1) * cell_w
        mask_2d[y1:y2, x1:x2] = 1

    # Expand to 3D: uniform across depth
    return mask_2d.unsqueeze(0).expand(D, -1, -1).contiguous()


def _coarse_dropout_mask_3d(D: int, H: int, W: int, pattern_id: int) -> torch.Tensor:
    """Generate 3D coarse dropout mask (2D pattern uniform across depth).

    Pattern definitions:
        0: 2 holes - top-left, bottom-right corners
        1: 2 holes - top-right, bottom-left corners
        2: 3 holes - top-center, bottom-left, bottom-right
        3: 4 holes - all corners

    Args:
        D, H, W: Volume dimensions
        pattern_id: Which pattern (0-3)

    Returns:
        Mask tensor [D, H, W] with 0=keep, 1=drop
    """
    mask_2d = torch.zeros(H, W)
    hole_h = H // 4  # 25% of height
    hole_w = W // 4  # 25% of width

    if pattern_id == 0:
        # Top-left and bottom-right
        mask_2d[:hole_h, :hole_w] = 1
        mask_2d[-hole_h:, -hole_w:] = 1
    elif pattern_id == 1:
        # Top-right and bottom-left
        mask_2d[:hole_h, -hole_w:] = 1
        mask_2d[-hole_h:, :hole_w] = 1
    elif pattern_id == 2:
        # Top-center, bottom-left, bottom-right
        mask_2d[:hole_h, W//2 - hole_w//2:W//2 + hole_w//2] = 1
        mask_2d[-hole_h:, :hole_w] = 1
        mask_2d[-hole_h:, -hole_w:] = 1
    elif pattern_id == 3:
        # All four corners
        mask_2d[:hole_h, :hole_w] = 1
        mask_2d[:hole_h, -hole_w:] = 1
        mask_2d[-hole_h:, :hole_w] = 1
        mask_2d[-hole_h:, -hole_w:] = 1

    # Expand to 3D: uniform across depth
    return mask_2d.unsqueeze(0).expand(D, -1, -1).contiguous()


def _patch_dropout_mask_3d(D: int, H: int, W: int, patch_size: int, drop_ratio: float, seed: int) -> torch.Tensor:
    """Generate 3D patch dropout mask (2D pattern uniform across depth).

    Args:
        D, H, W: Volume dimensions
        patch_size: Size of each patch in pixels
        drop_ratio: Fraction of patches to drop
        seed: Random seed for reproducibility

    Returns:
        Mask tensor [D, H, W] with 0=keep, 1=drop
    """
    rng = np.random.RandomState(seed)

    n_patches_h = H // patch_size
    n_patches_w = W // patch_size
    n_patches = n_patches_h * n_patches_w
    n_drop = int(n_patches * drop_ratio)

    mask_2d = torch.zeros(H, W)

    # Randomly select patches to drop
    drop_indices = rng.choice(n_patches, n_drop, replace=False)

    for idx in drop_indices:
        i = idx // n_patches_w
        j = idx % n_patches_w
        y1, y2 = i * patch_size, (i + 1) * patch_size
        x1, x2 = j * patch_size, (j + 1) * patch_size
        mask_2d[y1:y2, x1:x2] = 1

    # Expand to 3D: uniform across depth
    return mask_2d.unsqueeze(0).expand(D, -1, -1).contiguous()


def generate_pattern_mask_3d(pattern_id: int, D: int, H: int, W: int) -> torch.Tensor:
    """Generate 3D mask for a fixed pattern ID (2D pattern uniform across depth).

    Pattern IDs (16 total):
        0-3:   Checkerboard (4x4 std, 4x4 offset, 8x8 std, 8x8 offset)
        4-7:   Grid dropout (4x4 25% seed0, 4x4 25% seed1, 4x4 50% seed0, 4x4 50% seed1)
        8-11:  Coarse dropout (patterns 0-3)
        12-15: Patch dropout (8x8 25% seedA, 8x8 25% seedB, 8x8 50% seedA, 8x8 50% seedB)

    Args:
        pattern_id: Pattern index (0-15)
        D, H, W: Volume dimensions

    Returns:
        Mask tensor [D, H, W] with 0=keep, 1=drop
    """
    if pattern_id < 4:
        # Checkerboard patterns
        if pattern_id == 0:
            return _checkerboard_mask_3d(D, H, W, grid_size=4, offset=False)
        elif pattern_id == 1:
            return _checkerboard_mask_3d(D, H, W, grid_size=4, offset=True)
        elif pattern_id == 2:
            return _checkerboard_mask_3d(D, H, W, grid_size=8, offset=False)
        elif pattern_id == 3:
            return _checkerboard_mask_3d(D, H, W, grid_size=8, offset=True)

    elif pattern_id < 8:
        # Grid dropout patterns
        if pattern_id == 4:
            return _grid_dropout_mask_3d(D, H, W, grid_size=4, drop_ratio=0.25, seed=42)
        elif pattern_id == 5:
            return _grid_dropout_mask_3d(D, H, W, grid_size=4, drop_ratio=0.25, seed=123)
        elif pattern_id == 6:
            return _grid_dropout_mask_3d(D, H, W, grid_size=4, drop_ratio=0.50, seed=42)
        elif pattern_id == 7:
            return _grid_dropout_mask_3d(D, H, W, grid_size=4, drop_ratio=0.50, seed=123)

    elif pattern_id < 12:
        # Coarse dropout patterns
        return _coarse_dropout_mask_3d(D, H, W, pattern_id=pattern_id - 8)

    else:
        # Patch dropout patterns (8x8 patches for 128px image = 16 patches)
        patch_size = max(H // 8, 1)
        if pattern_id == 12:
            return _patch_dropout_mask_3d(D, H, W, patch_size=patch_size, drop_ratio=0.25, seed=42)
        elif pattern_id == 13:
            return _patch_dropout_mask_3d(D, H, W, patch_size=patch_size, drop_ratio=0.25, seed=123)
        elif pattern_id == 14:
            return _patch_dropout_mask_3d(D, H, W, patch_size=patch_size, drop_ratio=0.50, seed=42)
        elif pattern_id == 15:
            return _patch_dropout_mask_3d(D, H, W, patch_size=patch_size, drop_ratio=0.50, seed=123)

    # Fallback: no masking
    return torch.zeros(D, H, W)


# Pattern category names for debugging/logging
PATTERN_NAMES_3D = [
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

NUM_PATTERNS_3D = 16


# =============================================================================
# Mode-Specific Intensity Scaling (same as 2D)
# =============================================================================
# Each modality gets a different intensity scale applied AFTER noise addition.

MODE_INTENSITY_SCALE_3D = {
    0: 0.85,   # bravo  - darker
    1: 1.15,   # flair  - brighter
    2: 0.92,   # t1_pre - slightly darker
    3: 1.08,   # t1_gd  - slightly brighter
}

# Reverse mapping for inference
MODE_INTENSITY_SCALE_INV_3D = {k: 1.0 / v for k, v in MODE_INTENSITY_SCALE_3D.items()}


def apply_mode_intensity_scale_3d(
    x: torch.Tensor,
    mode_id: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply modality-specific intensity scaling to 3D input (per-sample).

    This makes mode conditioning NECESSARY for correct predictions.
    The model sees scaled input but must predict unscaled target.

    Args:
        x: Input tensor [B, C, D, H, W]
        mode_id: Mode ID tensor [B] or None (0=bravo, 1=flair, 2=t1_pre, 3=t1_gd)

    Returns:
        Tuple of (scaled_input, scale_factors)
        - scaled_input: x * scale_factors (per-sample scaling)
        - scale_factors: Tensor [B, 1, 1, 1, 1] with per-sample scales
    """
    if mode_id is None:
        return x, torch.ones(1, device=x.device, dtype=x.dtype)

    if mode_id.dim() == 0:
        mode_id = mode_id.unsqueeze(0)

    if mode_id.numel() == 0:
        return x, torch.ones(1, device=x.device, dtype=x.dtype)

    # Per-sample scales [B]
    scales = torch.tensor(
        [MODE_INTENSITY_SCALE_3D.get(int(m.item()), 1.0) for m in mode_id],
        device=x.device,
        dtype=x.dtype,
    )
    # Reshape for 5D broadcasting [B, 1, 1, 1, 1]
    scales = scales.view(-1, 1, 1, 1, 1)
    return x * scales, scales


def inverse_mode_intensity_scale_3d(
    x: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Inverse the mode intensity scaling for 3D (per-sample).

    Args:
        x: Scaled tensor [B, C, D, H, W]
        scale: The scale factors that were applied [B, 1, 1, 1, 1] or [1]

    Returns:
        Unscaled tensor: x / scale
    """
    if scale.numel() == 1 and scale.item() == 1.0:
        return x
    return x / scale


class ScoreAugTransform3D:
    """Applies transforms to 3D noisy input and target per ScoreAug paper.

    Transforms:
    - Rotation: 90, 180, 270 degrees around D/H/W axes (REQUIRES omega conditioning)
    - Flip: Along D/H/W axes (REQUIRES omega conditioning)
    - Translation: ±40% X, ±20% Y, 0% Z (brain centered in Z)
    - Cutout: Random 3D rectangular region zeroed
    - Fixed patterns (v2): 2D patterns applied uniformly across depth

    v2 mode adds structured masking patterns:
    - Non-destructive transforms (can stack): rotation, flip, translation
    - Destructive transforms (pick one): cutout OR fixed patterns
    """

    def __init__(
        self,
        rotation: bool = True,
        flip: bool = True,
        translation: bool = False,
        cutout: bool = False,
        compose: bool = False,
        compose_prob: float = 0.5,
        # v2 mode parameters
        v2_mode: bool = False,
        nondestructive_prob: float = 0.5,
        destructive_prob: float = 0.5,
        cutout_vs_pattern: float = 0.5,
        patterns_checkerboard: bool = True,
        patterns_grid_dropout: bool = True,
        patterns_coarse_dropout: bool = True,
        patterns_patch_dropout: bool = True,
    ):
        """Initialize 3D ScoreAug transform.

        Three modes:
        - compose=False, v2_mode=False (default): One transform sampled with equal probability
        - compose=True, v2_mode=False: Each transform applied independently with compose_prob
        - v2_mode=True: Structured augmentation (non-destructive stack + one destructive)

        v2 mode separates transforms into:
        - Non-destructive (can stack): rotation, flip, translation
        - Destructive (pick one): cutout OR fixed pattern (2D patterns across depth)

        Args:
            rotation: Enable 90, 180, 270 degree rotations around axes
            flip: Enable flips along D/H/W axes
            translation: Enable ±40% X, ±20% Y, 0% Z translation with zero-padding
            cutout: Enable random 3D cutout (10-30% each dimension)
            compose: If True, apply transforms independently (legacy mode)
            compose_prob: Probability for each transform when compose=True
            v2_mode: Enable structured non-destructive/destructive augmentation
            nondestructive_prob: Probability for each non-destructive transform in v2
            destructive_prob: Probability of applying any destructive transform in v2
            cutout_vs_pattern: In v2, probability of cutout vs fixed patterns (0.5 = 50/50)
            patterns_checkerboard: Enable checkerboard patterns (IDs 0-3)
            patterns_grid_dropout: Enable grid dropout patterns (IDs 4-7)
            patterns_coarse_dropout: Enable coarse dropout patterns (IDs 8-11)
            patterns_patch_dropout: Enable patch dropout patterns (IDs 12-15)
        """
        self.rotation = rotation
        self.flip = flip
        self.translation = translation
        self.cutout = cutout
        self.compose = compose
        self.compose_prob = compose_prob

        # v2 mode parameters
        self.v2_mode = v2_mode
        self.nondestructive_prob = nondestructive_prob
        self.destructive_prob = destructive_prob
        self.cutout_vs_pattern = cutout_vs_pattern
        self.patterns_checkerboard = patterns_checkerboard
        self.patterns_grid_dropout = patterns_grid_dropout
        self.patterns_coarse_dropout = patterns_coarse_dropout
        self.patterns_patch_dropout = patterns_patch_dropout

        # Build list of enabled pattern IDs for v2 mode
        self._enabled_patterns = []
        if patterns_checkerboard:
            self._enabled_patterns.extend([0, 1, 2, 3])
        if patterns_grid_dropout:
            self._enabled_patterns.extend([4, 5, 6, 7])
        if patterns_coarse_dropout:
            self._enabled_patterns.extend([8, 9, 10, 11])
        if patterns_patch_dropout:
            self._enabled_patterns.extend([12, 13, 14, 15])

        # Cache for pattern masks (generated lazily per volume size)
        self._pattern_cache: Dict[Tuple[int, int, int, int], torch.Tensor] = {}

    def sample_transform(self) -> Tuple[str, Dict[str, Any]]:
        """Sample a random transform with equal probability.

        Returns:
            Tuple of (transform_type, params_dict)
        """
        transforms = ['identity']
        if self.rotation or self.flip:
            transforms.append('spatial')
        if self.translation:
            transforms.append('translate')
        if self.cutout:
            transforms.append('cutout')

        transform_type = random.choice(transforms)

        if transform_type == 'identity':
            return 'identity', {}
        elif transform_type == 'spatial':
            # Build list of 3D spatial transforms
            spatial_options = []
            if self.rotation:
                # Rotations around each axis: 90, 180, 270 degrees
                for axis in ['d', 'h', 'w']:  # depth, height, width
                    for k in [1, 2, 3]:  # 90, 180, 270
                        spatial_options.append(('rot90_3d', {'axis': axis, 'k': k}))
            if self.flip:
                # Flips along each axis
                spatial_options.extend([
                    ('flip_d', {}),
                    ('flip_h', {}),
                    ('flip_w', {}),
                ])
            if not spatial_options:
                return 'identity', {}
            return random.choice(spatial_options)
        elif transform_type == 'translate':
            # 3D translation: ±40% X (W), ±20% Y (H), 0% Z (D)
            # Brain is centered in Z dimension, asymmetric in X/Y
            dd = 0.0  # No translation in depth (Z)
            dh = random.uniform(-0.2, 0.2)  # ±20% Y
            dw = random.uniform(-0.4, 0.4)  # ±40% X
            return 'translate', {'dd': dd, 'dh': dh, 'dw': dw}
        elif transform_type == 'cutout':
            # Random 3D rectangular region
            size_d = random.uniform(0.1, 0.3)
            size_h = random.uniform(0.1, 0.3)
            size_w = random.uniform(0.1, 0.3)
            cd = random.uniform(size_d / 2, 1 - size_d / 2)
            ch = random.uniform(size_h / 2, 1 - size_h / 2)
            cw = random.uniform(size_w / 2, 1 - size_w / 2)
            return 'cutout', {'cd': cd, 'ch': ch, 'cw': cw,
                             'size_d': size_d, 'size_h': size_h, 'size_w': size_w}

        return 'identity', {}

    def sample_compose_transforms(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Sample multiple transforms independently for compose mode.

        Returns:
            List of (transform_type, params) tuples
        """
        transforms = []

        # Spatial (rotation/flip)
        if (self.rotation or self.flip) and random.random() < self.compose_prob:
            spatial_options = []
            if self.rotation:
                for axis in ['d', 'h', 'w']:
                    for k in [1, 2, 3]:
                        spatial_options.append(('rot90_3d', {'axis': axis, 'k': k}))
            if self.flip:
                spatial_options.extend([
                    ('flip_d', {}),
                    ('flip_h', {}),
                    ('flip_w', {}),
                ])
            if spatial_options:
                transforms.append(random.choice(spatial_options))

        # Translation (±40% X, ±20% Y, 0% Z)
        if self.translation and random.random() < self.compose_prob:
            dd = 0.0  # No translation in depth (Z)
            dh = random.uniform(-0.2, 0.2)  # ±20% Y
            dw = random.uniform(-0.4, 0.4)  # ±40% X
            transforms.append(('translate', {'dd': dd, 'dh': dh, 'dw': dw}))

        # Cutout
        if self.cutout and random.random() < self.compose_prob:
            size_d = random.uniform(0.1, 0.3)
            size_h = random.uniform(0.1, 0.3)
            size_w = random.uniform(0.1, 0.3)
            cd = random.uniform(size_d / 2, 1 - size_d / 2)
            ch = random.uniform(size_h / 2, 1 - size_h / 2)
            cw = random.uniform(size_w / 2, 1 - size_w / 2)
            transforms.append(('cutout', {'cd': cd, 'ch': ch, 'cw': cw,
                                         'size_d': size_d, 'size_h': size_h, 'size_w': size_w}))

        return transforms

    def sample_v2_transforms(self) -> Tuple[List[Tuple[str, Dict[str, Any]]], Optional[Tuple[str, Dict[str, Any]]]]:
        """Sample transforms for v2 mode: non-destructive stack + one destructive.

        Non-destructive transforms (can stack): rotation, flip, translation
        Destructive transforms (pick one): cutout OR fixed pattern (2D across depth)

        Returns:
            Tuple of:
                - List of non-destructive (transform_type, params) tuples
                - Optional destructive (transform_type, params) tuple, or None
        """
        nondestructive = []

        # Sample non-destructive transforms (each with nondestructive_prob)
        # Spatial (rotation/flip)
        if (self.rotation or self.flip) and random.random() < self.nondestructive_prob:
            spatial_options = []
            if self.rotation:
                for axis in ['d', 'h', 'w']:
                    for k in [1, 2, 3]:
                        spatial_options.append(('rot90_3d', {'axis': axis, 'k': k}))
            if self.flip:
                spatial_options.extend([
                    ('flip_d', {}),
                    ('flip_h', {}),
                    ('flip_w', {}),
                ])
            if spatial_options:
                nondestructive.append(random.choice(spatial_options))

        # Translation (non-destructive since brain is centered)
        if self.translation and random.random() < self.nondestructive_prob:
            dd = 0.0  # No translation in depth (Z)
            dh = random.uniform(-0.2, 0.2)  # ±20% Y
            dw = random.uniform(-0.4, 0.4)  # ±40% X
            nondestructive.append(('translate', {'dd': dd, 'dh': dh, 'dw': dw}))

        # Sample destructive transform (with destructive_prob)
        destructive = None
        if random.random() < self.destructive_prob:
            # Choose between cutout and fixed patterns
            if random.random() < self.cutout_vs_pattern:
                # Random 3D cutout
                size_d = random.uniform(0.1, 0.3)
                size_h = random.uniform(0.1, 0.3)
                size_w = random.uniform(0.1, 0.3)
                cd = random.uniform(size_d / 2, 1 - size_d / 2)
                ch = random.uniform(size_h / 2, 1 - size_h / 2)
                cw = random.uniform(size_w / 2, 1 - size_w / 2)
                destructive = ('cutout', {'cd': cd, 'ch': ch, 'cw': cw,
                                         'size_d': size_d, 'size_h': size_h, 'size_w': size_w})
            elif self._enabled_patterns:
                # Fixed pattern (uniform over enabled patterns)
                pattern_id = random.choice(self._enabled_patterns)
                destructive = ('pattern', {'pattern_id': pattern_id})

        return nondestructive, destructive

    def _get_pattern_mask(self, pattern_id: int, D: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Get cached pattern mask, generating if needed.

        Args:
            pattern_id: Pattern index (0-15)
            D, H, W: Volume dimensions
            device: Target device

        Returns:
            Mask tensor [D, H, W] with 0=keep, 1=drop
        """
        cache_key = (pattern_id, D, H, W)
        if cache_key not in self._pattern_cache:
            mask = generate_pattern_mask_3d(pattern_id, D, H, W)
            self._pattern_cache[cache_key] = mask
        return self._pattern_cache[cache_key].to(device)

    def _apply_pattern(self, x: torch.Tensor, pattern_id: int) -> torch.Tensor:
        """Apply fixed pattern mask to 3D tensor.

        Args:
            x: Input tensor [B, C, D, H, W]
            pattern_id: Pattern index (0-15)

        Returns:
            Tensor with pattern regions zeroed
        """
        B, C, D, H, W = x.shape
        mask = self._get_pattern_mask(pattern_id, D, H, W, x.device)

        # Expand mask to [1, 1, D, H, W] for broadcasting
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Zero out masked regions
        return x * (1 - mask)

    def apply(
        self,
        x: torch.Tensor,
        transform_type: str,
        params: Dict[str, Any],
    ) -> torch.Tensor:
        """Apply transform to 3D tensor [B, C, D, H, W].

        Args:
            x: Input tensor
            transform_type: Type of transform
            params: Transform parameters

        Returns:
            Transformed tensor
        """
        if transform_type == 'identity':
            return x
        elif transform_type == 'rot90_3d':
            return self._rotate_3d(x, params['axis'], params['k'])
        elif transform_type == 'flip_d':
            return torch.flip(x, dims=[2])  # D dimension
        elif transform_type == 'flip_h':
            return torch.flip(x, dims=[3])  # H dimension
        elif transform_type == 'flip_w':
            return torch.flip(x, dims=[4])  # W dimension
        elif transform_type == 'translate':
            return self._translate_3d(x, params['dd'], params['dh'], params['dw'])
        elif transform_type == 'cutout':
            return self._cutout_3d(x, params['cd'], params['ch'], params['cw'],
                                   params['size_d'], params['size_h'], params['size_w'])
        elif transform_type == 'pattern':
            return self._apply_pattern(x, params['pattern_id'])
        return x

    def _rotate_3d(self, x: torch.Tensor, axis: str, k: int) -> torch.Tensor:
        """Rotate 3D tensor by k*90 degrees around axis.

        Args:
            x: Input tensor [B, C, D, H, W]
            axis: Rotation axis ('d', 'h', or 'w')
            k: Number of 90-degree rotations (1, 2, or 3)

        Returns:
            Rotated tensor
        """
        # Determine which dimensions to rotate
        if axis == 'd':
            # Rotate around D axis = rotate H-W plane
            dims = (3, 4)  # H, W
        elif axis == 'h':
            # Rotate around H axis = rotate D-W plane
            dims = (2, 4)  # D, W
        elif axis == 'w':
            # Rotate around W axis = rotate D-H plane
            dims = (2, 3)  # D, H
        else:
            return x

        return torch.rot90(x, k=k, dims=dims)

    def _translate_3d(
        self, x: torch.Tensor, dd: float, dh: float, dw: float
    ) -> torch.Tensor:
        """Translate 3D tensor, zero-pad edges.

        Args:
            x: Input tensor [B, C, D, H, W]
            dd, dh, dw: Shifts as fractions of dimension sizes

        Returns:
            Translated tensor with zero-padded edges
        """
        B, C, D, H, W = x.shape
        shift_d = int(dd * D)
        shift_h = int(dh * H)
        shift_w = int(dw * W)

        result = torch.roll(x, shifts=(shift_d, shift_h, shift_w), dims=(2, 3, 4))

        # Zero out wrapped regions
        if shift_d > 0:
            result[:, :, :shift_d, :, :] = 0
        elif shift_d < 0:
            result[:, :, shift_d:, :, :] = 0

        if shift_h > 0:
            result[:, :, :, :shift_h, :] = 0
        elif shift_h < 0:
            result[:, :, :, shift_h:, :] = 0

        if shift_w > 0:
            result[:, :, :, :, :shift_w] = 0
        elif shift_w < 0:
            result[:, :, :, :, shift_w:] = 0

        return result

    def _cutout_3d(
        self, x: torch.Tensor,
        cd: float, ch: float, cw: float,
        size_d: float, size_h: float, size_w: float,
    ) -> torch.Tensor:
        """Apply 3D rectangular cutout.

        Args:
            x: Input tensor [B, C, D, H, W]
            cd, ch, cw: Center coordinates as fractions
            size_d, size_h, size_w: Size fractions for each dimension

        Returns:
            Tensor with 3D rectangular region zeroed
        """
        B, C, D, H, W = x.shape

        half_d = int(size_d * D / 2)
        half_h = int(size_h * H / 2)
        half_w = int(size_w * W / 2)
        center_d = int(cd * D)
        center_h = int(ch * H)
        center_w = int(cw * W)

        d1 = max(0, center_d - half_d)
        d2 = min(D, center_d + half_d)
        h1 = max(0, center_h - half_h)
        h2 = min(H, center_h + half_h)
        w1 = max(0, center_w - half_w)
        w2 = min(W, center_w + half_w)

        result = x.clone()
        result[:, :, d1:d2, h1:h2, w1:w2] = 0
        return result

    def requires_omega(self, omega: Optional[Dict[str, Any]]) -> bool:
        """Check if transform requires omega conditioning.

        All spatial transforms (rotations, flips) require omega conditioning.

        Args:
            omega: Transform parameters

        Returns:
            True if omega conditioning is required
        """
        if omega is None:
            return False
        transform_type = omega.get('type')
        return transform_type in ('rot90_3d', 'flip_d', 'flip_h', 'flip_w')

    def __call__(
        self,
        noisy_input: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        """Apply ScoreAug transform to 3D noisy input and target.

        Args:
            noisy_input: [B, C, D, H, W] - noisy volume (optionally concatenated with seg)
            target: [B, C_out, D, H, W] - velocity target

        Returns:
            aug_input: Transformed noisy input
            aug_target: Transformed target
            omega: Transform parameters for conditioning, None if identity
        """
        if self.v2_mode:
            # v2 mode: non-destructive stack + one destructive
            nondestructive, destructive = self.sample_v2_transforms()

            if not nondestructive and destructive is None:
                return noisy_input, target, None

            # Apply non-destructive transforms first
            aug_input = noisy_input
            aug_target = target
            all_transforms = []

            for transform_type, params in nondestructive:
                aug_input = self.apply(aug_input, transform_type, params)
                aug_target = self.apply(aug_target, transform_type, params)
                all_transforms.append((transform_type, params))

            # Apply destructive transform if sampled
            if destructive is not None:
                transform_type, params = destructive
                aug_input = self.apply(aug_input, transform_type, params)
                aug_target = self.apply(aug_target, transform_type, params)
                all_transforms.append((transform_type, params))

            if not all_transforms:
                return aug_input, aug_target, None

            omega = {
                'v2': True,
                'transforms': all_transforms,
            }
            return aug_input, aug_target, omega

        elif self.compose:
            transforms = self.sample_compose_transforms()

            if not transforms:
                return noisy_input, target, None

            aug_input = noisy_input
            aug_target = target
            for transform_type, params in transforms:
                aug_input = self.apply(aug_input, transform_type, params)
                aug_target = self.apply(aug_target, transform_type, params)

            omega = {
                'compose': True,
                'transforms': transforms,
            }
            return aug_input, aug_target, omega

        else:
            transform_type, params = self.sample_transform()

            aug_input = self.apply(noisy_input, transform_type, params)
            aug_target = self.apply(target, transform_type, params)

            if transform_type == 'identity':
                return aug_input, aug_target, None

            omega = {
                'type': transform_type,
                'params': params,
            }
            return aug_input, aug_target, omega


# =============================================================================
# Omega Conditioning for 3D (extended for v2 mode with patterns)
# =============================================================================
#
# Omega encoding format (36 dims total - matching 2D for consistency):
#
# Legacy mode (compose=False, v2=False) - uses dims 0-15:
#   [type_onehot(9), axis_onehot(3), rot_k_norm, dd, dh, dw]
#   Types: 0=identity, 1=rot90_d, 2=rot90_h, 3=rot90_w,
#          4=flip_d, 5=flip_h, 6=flip_w, 7=translate, 8=cutout
#
# Compose/v2 mode - uses dims 0-31:
#   Dims 0-3: Active mask (spatial, translation, cutout, pattern)
#   Dims 4-8: Spatial type (rot90_d, rot90_h, rot90_w, flip_d, flip_h)
#   Dim 9: flip_w type
#   Dims 10-12: axis_onehot for 3D rotations
#   Dim 13: rot_k normalized
#   Dims 14-16: dd, dh, dw (translation)
#   Dims 17-22: cd, ch, cw, size_d, size_h, size_w (cutout)
#   Dims 23-38 would be pattern_onehot(16) but we use 16-31 for compatibility
#
# Actually, let's use a cleaner layout matching 2D:
#   Dims 0-3: Active mask (spatial, translation, cutout, pattern)
#   Dims 4-9: Spatial type (rot90_d, rot90_h, rot90_w, flip_d, flip_h, flip_w)
#   Dim 10: rot_k normalized
#   Dims 11-13: dd, dh, dw (translation)
#   Dims 14-15: reserved
#   Dims 16-31: Pattern ID one-hot (16 patterns)
#   Dims 32-35: Mode one-hot (4 modalities) for intensity scaling

OMEGA_ENCODING_DIM_3D = 36


def encode_omega_3d(
    omega: Optional[Dict[str, Any]],
    device: torch.device,
    mode_id: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Encode 3D omega dict into tensor format for MLP.

    Supports single-transform mode, compose mode, v2 mode, and mode intensity scaling.

    Args:
        omega: Transform parameters dict or None
        device: Target device
        mode_id: Optional mode ID tensor for intensity scaling (0=bravo, 1=flair, etc.)

    Returns:
        Tensor [1, OMEGA_ENCODING_DIM_3D] encoding the transform + mode
    """
    enc = torch.zeros(1, OMEGA_ENCODING_DIM_3D, device=device)

    # Encode mode intensity scaling in dims 32-35 (always, if provided)
    if mode_id is not None and mode_id.numel() > 0:
        if mode_id.dim() == 0:
            idx = mode_id.item()
        else:
            idx = mode_id[0].item()
        if 0 <= idx < 4:
            enc[0, 32 + int(idx)] = 1.0

    if omega is None:
        # Identity: no active transforms
        return enc

    # Check for v2 mode
    if omega.get('v2', False):
        transforms = omega['transforms']
        for t, p in transforms:
            _encode_single_transform_3d_v2(enc, t, p)
        return enc

    # Check for compose mode
    if omega.get('compose', False):
        transforms = omega['transforms']
        for t, p in transforms:
            _encode_single_transform_3d_v2(enc, t, p)
        return enc

    # Single transform mode (legacy)
    _encode_single_transform_3d_legacy(enc, omega['type'], omega['params'])
    return enc


def _encode_single_transform_3d_v2(
    enc: torch.Tensor,
    transform_type: str,
    params: Dict[str, Any],
) -> None:
    """Encode a single transform for v2/compose mode (in-place).

    Layout:
        Dims 0-3: Active mask (spatial, translation, cutout, pattern)
        Dims 4-9: Spatial type (rot90_d, rot90_h, rot90_w, flip_d, flip_h, flip_w)
        Dim 10: rot_k normalized
        Dims 11-13: dd, dh, dw (translation)
        Dims 16-31: Pattern ID one-hot
    """
    if transform_type == 'rot90_3d':
        enc[0, 0] = 1.0  # spatial active
        axis = params['axis']
        k = params['k']
        if axis == 'd':
            enc[0, 4] = 1.0  # rot90_d type
        elif axis == 'h':
            enc[0, 5] = 1.0  # rot90_h type
        elif axis == 'w':
            enc[0, 6] = 1.0  # rot90_w type
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
    elif transform_type == 'translate':
        enc[0, 1] = 1.0  # translation active
        enc[0, 11] = params['dd']
        enc[0, 12] = params['dh']
        enc[0, 13] = params['dw']
    elif transform_type == 'cutout':
        enc[0, 2] = 1.0  # cutout active
        # Store cutout params (can use dims 14-15 and beyond if needed)
        # For simplicity, just mark as active - detailed params not critical
    elif transform_type == 'pattern':
        enc[0, 3] = 1.0  # pattern active
        pattern_id = params['pattern_id']
        enc[0, 16 + pattern_id] = 1.0  # pattern one-hot


def _encode_single_transform_3d_legacy(
    enc: torch.Tensor,
    transform_type: str,
    params: Dict[str, Any],
) -> None:
    """Encode a single transform for legacy single-transform mode (in-place).

    Uses same layout as v2 for compatibility.
    """
    _encode_single_transform_3d_v2(enc, transform_type, params)


class OmegaTimeEmbed3D(nn.Module):
    """Wrapper around time_embed that adds 3D omega conditioning.

    This is compile-compatible because:
    - No hooks, just module replacement
    - No data-dependent control flow in forward
    - Always adds omega embedding (zero for identity)
    """

    def __init__(self, original_time_embed: nn.Module, embed_dim: int):
        """Initialize omega-aware time embedding.

        Args:
            original_time_embed: The original time_embed module
            embed_dim: Output dimension (should match original)
        """
        super().__init__()
        self.original = original_time_embed
        self.embed_dim = embed_dim

        # MLP that maps omega encoding to embedding (zero-init for neutral start)
        self.omega_mlp = create_zero_init_mlp(OMEGA_ENCODING_DIM_3D, embed_dim)

        # Buffer to store current omega encoding
        self.register_buffer('_omega_encoding', torch.zeros(1, OMEGA_ENCODING_DIM_3D))

    def set_omega_encoding(self, omega_encoding: torch.Tensor):
        """Set omega encoding for next forward pass.

        Uses in-place copy to maintain buffer identity for torch.compile.

        Args:
            omega_encoding: Tensor [1, OMEGA_ENCODING_DIM_3D]
        """
        self._omega_encoding.copy_(omega_encoding)

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass: original time_embed + omega embedding.

        Args:
            t_emb: Timestep embedding input [B, C]

        Returns:
            Time embedding with omega conditioning [B, embed_dim]
        """
        # Original time embedding
        out = self.original(t_emb)

        # Add omega embedding (always computed, zero for identity due to init)
        omega_emb = self.omega_mlp(self._omega_encoding)

        return out + omega_emb


class ScoreAugModelWrapper3D(nn.Module):
    """Wrapper to inject omega conditioning into 3D MONAI UNet.

    This implementation is compile-compatible:
    - Replaces time_embed with OmegaTimeEmbed3D (no hooks)
    - Sets omega encoding before forward (outside traced graph)
    - Forward has fixed control flow (no data-dependent branches)
    """

    def __init__(self, model: nn.Module, embed_dim: int = 256):
        """Initialize wrapper.

        Args:
            model: MONAI DiffusionModelUNet to wrap
            embed_dim: Embedding dimension (should match model's time_embed output)
        """
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim

        if not hasattr(model, 'time_embed'):
            raise ValueError("Model does not have 'time_embed' attribute")

        original_time_embed = model.time_embed
        self.omega_time_embed = OmegaTimeEmbed3D(original_time_embed, embed_dim)

        # Replace the model's time_embed
        model.time_embed = self.omega_time_embed

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        omega: Optional[Dict[str, Any]] = None,
        mode_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with omega conditioning.

        Args:
            x: Noisy input tensor [B, C, D, H, W]
            timesteps: Timestep tensor [B]
            omega: Optional augmentation parameters for conditioning
            mode_id: Optional mode ID for intensity scaling (0=bravo, 1=flair, etc.)

        Returns:
            Model prediction [B, C_out, D, H, W]
        """
        # Encode omega + mode_id as (1, 36) - broadcasts to batch in MLP
        omega_encoding = encode_omega_3d(omega, x.device, mode_id=mode_id)
        self.omega_time_embed.set_omega_encoding(omega_encoding)

        # Call model normally
        return self.model(x, timesteps)

    def parameters(self, recurse: bool = True):
        """Get all parameters including omega MLP."""
        return self.model.parameters(recurse=recurse)

    @property
    def parameters_without_omega(self):
        """Get model parameters excluding omega embedding.

        Useful if you want to use different learning rates.
        """
        # Get all params except omega_mlp
        omega_param_ids = {id(p) for p in self.omega_time_embed.omega_mlp.parameters()}
        return (p for p in self.model.parameters() if id(p) not in omega_param_ids)
