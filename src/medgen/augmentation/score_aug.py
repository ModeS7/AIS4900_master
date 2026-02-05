"""Score Augmentation for Diffusion Models.

Reference: https://arxiv.org/abs/2508.07926

Applies transforms to noisy data (after noise addition) to improve
generalization without changing the output distribution.

Traditional augmentation: x -> T(x) -> add noise -> denoise -> T(x)
ScoreAug:                 x -> add noise -> T(x + noise) -> denoise -> T(x)

This aligns with diffusion's denoising mechanism and provides stronger
regularization without changing the output distribution.

Supports both 2D images [B, C, H, W] and 3D volumes [B, C, D, H, W].

Conditioning requirements (per paper):
- Rotation/Flip: REQUIRES omega conditioning (noise is rotation-invariant, model can cheat)
- Translation/Cutout: Work without conditioning but risk data leakage

v2 mode adds structured masking patterns:
- Non-destructive transforms (can stack): rotation, flip, translation
- Destructive transforms (pick one): cutout OR fixed patterns
- Fixed patterns: checkerboard, grid dropout, coarse dropout, patch dropout
"""

import random
from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from medgen.models.wrappers import create_zero_init_mlp

# =============================================================================
# Fixed Pattern Definitions (16 total)
# =============================================================================
# These are deterministic masks that the network learns via one-hot embedding.
# Patterns are generated as functions of spatial dimensions.

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


class ScoreAugTransform:
    """Applies transforms to noisy input and target per ScoreAug paper.

    Supports both 2D images [B, C, H, W] and 3D volumes [B, C, D, H, W].

    Transforms:
    - Rotation: 90, 180, 270 degrees (REQUIRES omega conditioning)
      - 2D: around single axis
      - 3D: around D, H, or W axes
    - Flip: horizontal/vertical (2D) or D/H/W axes (3D)
    - Translation: with zero-padding
      - 2D: +/-40% X, +/-20% Y
      - 3D: +/-40% W, +/-20% H, 0% D (brain centered in Z)
    - Cutout: Random rectangular region zeroed

    Note: Rotation/flip requires omega conditioning because Gaussian noise is
    rotation-invariant, allowing the model to detect rotation from the
    noise pattern and "cheat" by inverting it before denoising.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
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
        """Initialize ScoreAug transform.

        Three modes:
        - compose=False, v2_mode=False (default): One transform sampled with equal probability (per paper)
        - compose=True, v2_mode=False: Each transform applied independently with compose_prob
        - v2_mode=True: Structured augmentation (non-destructive stack + one destructive)

        v2 mode separates transforms into:
        - Non-destructive (can stack): rotation, flip, translation
        - Destructive (pick one): cutout OR fixed pattern (checkerboard/grid/coarse/patch)

        Args:
            spatial_dims: Number of spatial dimensions (2 or 3).
            rotation: Enable rotations (requires omega conditioning)
            flip: Enable flips (requires omega conditioning for consistency)
            translation: Enable translation with zero-padding
            cutout: Enable random rectangular cutout
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
        self.spatial_dims = spatial_dims
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

    def sample_transform(self) -> tuple[str, dict[str, Any]]:
        """Sample a random transform with equal probability (per paper).

        Identity is always included as one option. All enabled transforms
        have equal probability of being selected.

        Returns:
            Tuple of (transform_type, params_dict)
        """
        # Build list of all enabled transforms + identity
        transforms = ['identity']
        if self.rotation or self.flip:
            transforms.append('spatial')  # Unified D4/D8 symmetries
        if self.translation:
            transforms.append('translate')
        if self.cutout:
            transforms.append('cutout')

        # Sample uniformly across all options (per paper)
        transform_type = random.choice(transforms)

        if transform_type == 'identity':
            return 'identity', {}
        elif transform_type == 'spatial':
            return self._sample_spatial_transform()
        elif transform_type == 'translate':
            return self._sample_translation()
        elif transform_type == 'cutout':
            return self._sample_cutout()

        return 'identity', {}

    def _sample_spatial_transform(self) -> tuple[str, dict[str, Any]]:
        """Sample a spatial transform (rotation or flip)."""
        spatial_options = []

        if self.spatial_dims == 2:
            if self.rotation:
                spatial_options.extend([
                    ('rot90', {'k': 1}),
                    ('rot90', {'k': 2}),  # 180°
                    ('rot90', {'k': 3}),  # 270°
                ])
            if self.flip:
                spatial_options.extend([
                    ('hflip', {}),
                    ('vflip', {}),
                ])
            if self.rotation and self.flip:
                # Combined transforms (diagonal/anti-diagonal reflections)
                spatial_options.extend([
                    ('rot90_hflip', {'k': 1}),  # rot90 then hflip
                    ('rot90_hflip', {'k': 3}),  # rot270 then hflip
                ])
        else:  # 3D
            if self.rotation:
                # Rotations around each axis: 90, 180, 270 degrees
                for axis in ['d', 'h', 'w']:
                    for k in [1, 2, 3]:
                        spatial_options.append(('rot90_3d', {'axis': axis, 'k': k}))
            if self.flip:
                spatial_options.extend([
                    ('flip_d', {}),
                    ('flip_h', {}),
                    ('flip_w', {}),
                ])

        if not spatial_options:
            return 'identity', {}

        return random.choice(spatial_options)

    def _sample_translation(self) -> tuple[str, dict[str, Any]]:
        """Sample a translation transform."""
        if self.spatial_dims == 2:
            # Asymmetric: ±40% X, ±20% Y (brain is oval, more vertical space taken)
            dx = random.uniform(-0.4, 0.4)
            dy = random.uniform(-0.2, 0.2)
            return 'translate', {'dx': dx, 'dy': dy}
        else:  # 3D
            # ±40% W, ±20% H, 0% D (brain centered in Z)
            dd = 0.0
            dh = random.uniform(-0.2, 0.2)
            dw = random.uniform(-0.4, 0.4)
            return 'translate', {'dd': dd, 'dh': dh, 'dw': dw}

    def _sample_cutout(self) -> tuple[str, dict[str, Any]]:
        """Sample a cutout transform."""
        if self.spatial_dims == 2:
            # Random rectangle (not square) - sample width/height independently
            size_x = random.uniform(0.1, 0.3)  # 10-30% of width
            size_y = random.uniform(0.1, 0.3)  # 10-30% of height
            cx = random.uniform(size_x / 2, 1 - size_x / 2)
            cy = random.uniform(size_y / 2, 1 - size_y / 2)
            return 'cutout', {'cx': cx, 'cy': cy, 'size_x': size_x, 'size_y': size_y}
        else:  # 3D
            size_d = random.uniform(0.1, 0.3)
            size_h = random.uniform(0.1, 0.3)
            size_w = random.uniform(0.1, 0.3)
            cd = random.uniform(size_d / 2, 1 - size_d / 2)
            ch = random.uniform(size_h / 2, 1 - size_h / 2)
            cw = random.uniform(size_w / 2, 1 - size_w / 2)
            return 'cutout', {'cd': cd, 'ch': ch, 'cw': cw,
                             'size_d': size_d, 'size_h': size_h, 'size_w': size_w}

    def sample_compose_transforms(self) -> list[tuple[str, dict[str, Any]]]:
        """Sample multiple transforms independently for compose mode.

        Each enabled transform has compose_prob chance of being applied.
        Order: spatial → translate → cutout

        Returns:
            List of (transform_type, params) tuples to apply in sequence
        """
        transforms = []

        # Spatial (rotation/flip) - sample one from D4/D8 if triggered
        if (self.rotation or self.flip) and random.random() < self.compose_prob:
            t, p = self._sample_spatial_transform()
            if t != 'identity':
                transforms.append((t, p))

        # Translation
        if self.translation and random.random() < self.compose_prob:
            t, p = self._sample_translation()
            transforms.append((t, p))

        # Cutout
        if self.cutout and random.random() < self.compose_prob:
            t, p = self._sample_cutout()
            transforms.append((t, p))

        return transforms

    def sample_v2_transforms(self) -> tuple[list[tuple[str, dict[str, Any]]], tuple[str, dict[str, Any]] | None]:
        """Sample transforms for v2 mode: non-destructive stack + one destructive.

        Non-destructive transforms (can stack): rotation, flip, translation
        Destructive transforms (pick one): cutout OR fixed pattern

        Returns:
            Tuple of:
                - List of non-destructive (transform_type, params) tuples
                - Optional destructive (transform_type, params) tuple, or None
        """
        nondestructive = []

        # Sample non-destructive transforms (each with nondestructive_prob)
        # Spatial (rotation/flip)
        if (self.rotation or self.flip) and random.random() < self.nondestructive_prob:
            t, p = self._sample_spatial_transform()
            if t != 'identity':
                nondestructive.append((t, p))

        # Translation (non-destructive since brain is centered)
        if self.translation and random.random() < self.nondestructive_prob:
            t, p = self._sample_translation()
            nondestructive.append((t, p))

        # Sample destructive transform (with destructive_prob)
        destructive = None
        if random.random() < self.destructive_prob:
            # Choose between cutout and fixed patterns
            if random.random() < self.cutout_vs_pattern:
                # Random cutout
                _, p = self._sample_cutout()
                destructive = ('cutout', p)
            elif self._enabled_patterns:
                # Fixed pattern (uniform over enabled patterns)
                pattern_id = random.choice(self._enabled_patterns)
                destructive = ('pattern', {'pattern_id': pattern_id})

        return nondestructive, destructive

    def _get_pattern_mask(self, pattern_id: int, x: torch.Tensor) -> torch.Tensor:
        """Get cached pattern mask, generating if needed.

        Thread-safe via module-level LRU cache.

        Args:
            pattern_id: Pattern index (0-15)
            x: Input tensor to get dimensions and device from

        Returns:
            Mask tensor [H, W] or [D, H, W] with 0=keep, 1=drop
        """
        if self.spatial_dims == 2:
            H, W = x.shape[-2], x.shape[-1]
            mask = _cached_generate_pattern_mask(pattern_id, H, W, None, 2)
        else:
            D, H, W = x.shape[-3], x.shape[-2], x.shape[-1]
            mask = _cached_generate_pattern_mask(pattern_id, H, W, D, 3)
        return mask.to(x.device)

    def _apply_pattern(self, x: torch.Tensor, pattern_id: int) -> torch.Tensor:
        """Apply fixed pattern mask to tensor.

        Args:
            x: Input tensor [B, C, H, W] or [B, C, D, H, W]
            pattern_id: Pattern index (0-15)

        Returns:
            Tensor with pattern regions zeroed
        """
        mask = self._get_pattern_mask(pattern_id, x)

        if self.spatial_dims == 2:
            # Expand mask to [1, 1, H, W] for broadcasting
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            # Expand mask to [1, 1, D, H, W] for broadcasting
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Zero out masked regions
        return x * (1 - mask)

    def apply(
        self,
        x: torch.Tensor,
        transform_type: str,
        params: dict[str, Any],
    ) -> torch.Tensor:
        """Apply transform to tensor.

        Args:
            x: Input tensor [B, C, H, W] or [B, C, D, H, W]
            transform_type: Type of transform to apply
            params: Transform parameters

        Returns:
            Transformed tensor
        """
        if transform_type == 'identity':
            return x

        # 2D transforms
        elif transform_type == 'rot90':
            return torch.rot90(x, k=params['k'], dims=(-2, -1))
        elif transform_type == 'hflip':
            return torch.flip(x, dims=[-1])
        elif transform_type == 'vflip':
            return torch.flip(x, dims=[-2])
        elif transform_type == 'rot90_hflip':
            # Rotate then flip horizontally (diagonal/anti-diagonal reflections)
            rotated = torch.rot90(x, k=params['k'], dims=(-2, -1))
            return torch.flip(rotated, dims=[-1])

        # 3D transforms
        elif transform_type == 'rot90_3d':
            return self._rotate_3d(x, params['axis'], params['k'])
        elif transform_type == 'flip_d':
            return torch.flip(x, dims=[2])  # D dimension
        elif transform_type == 'flip_h':
            return torch.flip(x, dims=[3])  # H dimension
        elif transform_type == 'flip_w':
            return torch.flip(x, dims=[4])  # W dimension

        # Common transforms
        elif transform_type == 'translate':
            if self.spatial_dims == 2:
                return self._translate_2d(x, params['dx'], params['dy'])
            else:
                return self._translate_3d(x, params['dd'], params['dh'], params['dw'])
        elif transform_type == 'cutout':
            if self.spatial_dims == 2:
                return self._cutout_2d(x, params['cx'], params['cy'], params['size_x'], params['size_y'])
            else:
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

    def _translate_2d(self, x: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
        """Translate 2D tensor by (dx, dy) fraction, zero-pad edges.

        Args:
            x: Input tensor [B, C, H, W]
            dx: Horizontal shift as fraction of width (-1 to 1)
            dy: Vertical shift as fraction of height (-1 to 1)

        Returns:
            Translated tensor with zero-padded edges
        """
        B, C, H, W = x.shape
        shift_x = int(dx * W)
        shift_y = int(dy * H)

        result = torch.roll(x, shifts=(shift_y, shift_x), dims=(-2, -1))

        # Zero out wrapped regions
        if shift_y > 0:
            result[:, :, :shift_y, :] = 0
        elif shift_y < 0:
            result[:, :, shift_y:, :] = 0

        if shift_x > 0:
            result[:, :, :, :shift_x] = 0
        elif shift_x < 0:
            result[:, :, :, shift_x:] = 0

        return result

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

    def _cutout_2d(self, x: torch.Tensor, cx: float, cy: float, size_x: float, size_y: float) -> torch.Tensor:
        """Apply rectangular cutout centered at (cx, cy) with given size fractions.

        Args:
            x: Input tensor [B, C, H, W]
            cx: Center x as fraction of width (0 to 1)
            cy: Center y as fraction of height (0 to 1)
            size_x: Width of cutout as fraction of image width (0 to 1)
            size_y: Height of cutout as fraction of image height (0 to 1)

        Returns:
            Tensor with rectangular region zeroed
        """
        B, C, H, W = x.shape

        half_h = int(size_y * H / 2)
        half_w = int(size_x * W / 2)
        center_y = int(cy * H)
        center_x = int(cx * W)

        y1 = max(0, center_y - half_h)
        y2 = min(H, center_y + half_h)
        x1 = max(0, center_x - half_w)
        x2 = min(W, center_x + half_w)

        result = x.clone()
        result[:, :, y1:y2, x1:x2] = 0
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

    def inverse_apply(
        self,
        x: torch.Tensor,
        omega: dict[str, Any] | None,
    ) -> torch.Tensor | None:
        """Apply inverse transform to tensor.

        For invertible transforms (rotation, flip), applies the inverse to
        recover original values. For non-invertible transforms (translation,
        cutout), returns None since information is lost.

        Args:
            x: Transformed tensor [B, C, H, W] or [B, C, D, H, W]
            omega: Transform parameters from __call__

        Returns:
            Inverse-transformed tensor, or None if not invertible
        """
        if omega is None:
            return x

        transform_type = omega['type']
        params = omega['params']

        # 2D inverse transforms
        if transform_type == 'rot90':
            # Inverse of rot90(k) is rot90(4-k)
            inv_k = 4 - params['k']
            return torch.rot90(x, k=inv_k, dims=(-2, -1))
        elif transform_type == 'hflip':
            # Flip is self-inverse: flip(flip(x)) = x
            return torch.flip(x, dims=[-1])
        elif transform_type == 'vflip':
            # Flip is self-inverse
            return torch.flip(x, dims=[-2])
        elif transform_type == 'rot90_hflip':
            # Inverse: flip first, then inverse rotate
            flipped = torch.flip(x, dims=[-1])
            inv_k = 4 - params['k']
            return torch.rot90(flipped, k=inv_k, dims=(-2, -1))

        # 3D inverse transforms
        elif transform_type == 'rot90_3d':
            axis = params['axis']
            inv_k = 4 - params['k']
            return self._rotate_3d(x, axis, inv_k)
        elif transform_type == 'flip_d':
            return torch.flip(x, dims=[2])
        elif transform_type == 'flip_h':
            return torch.flip(x, dims=[3])
        elif transform_type == 'flip_w':
            return torch.flip(x, dims=[4])

        # Non-invertible transforms
        elif transform_type in ('translate', 'cutout', 'pattern'):
            return None

        return x

    def requires_omega(self, omega: dict[str, Any] | None) -> bool:
        """Check if transform requires omega conditioning.

        Per paper: rotation requires conditioning because noise is rotation-invariant.
        Flip uses omega conditioning for consistency (same reasoning applies).

        Args:
            omega: Transform parameters

        Returns:
            True if omega conditioning is required for this transform
        """
        if omega is None:
            return False

        # Check for v2 or compose mode
        if omega.get('v2', False) or omega.get('compose', False):
            transforms = omega.get('transforms', [])
            for t, _ in transforms:
                if t in ('rot90', 'hflip', 'vflip', 'rot90_hflip',
                         'rot90_3d', 'flip_d', 'flip_h', 'flip_w'):
                    return True
            return False

        # Single transform mode
        return omega.get('type') in ('rot90', 'hflip', 'vflip', 'rot90_hflip',
                                     'rot90_3d', 'flip_d', 'flip_h', 'flip_w')

    def apply_omega(
        self,
        x: torch.Tensor,
        omega: dict[str, Any] | None,
    ) -> torch.Tensor:
        """Apply transform(s) from omega dict to tensor.

        Handles both single-transform and compose mode omega formats.

        Args:
            x: Input tensor [B, C, H, W] or [B, C, D, H, W]
            omega: Transform parameters (single or compose format), or None for identity

        Returns:
            Transformed tensor
        """
        if omega is None:
            return x

        if omega.get('compose', False) or omega.get('v2', False):
            # Compose/v2 mode: apply all transforms in sequence
            result = x
            for transform_type, params in omega['transforms']:
                result = self.apply(result, transform_type, params)
            return result
        else:
            # Single transform mode
            return self.apply(x, omega['type'], omega['params'])

    def inverse_apply_omega(
        self,
        x: torch.Tensor,
        omega: dict[str, Any] | None,
    ) -> torch.Tensor | None:
        """Apply inverse transform(s) from omega dict to tensor.

        Handles both single-transform and compose mode omega formats.
        For compose mode, applies inverses in reverse order.

        Args:
            x: Transformed tensor [B, C, H, W] or [B, C, D, H, W]
            omega: Transform parameters (single or compose format), or None for identity

        Returns:
            Inverse-transformed tensor, or None if any transform is non-invertible
        """
        if omega is None:
            return x

        if omega.get('compose', False) or omega.get('v2', False):
            # Compose/v2 mode: apply inverse transforms in REVERSE order
            result = x
            for transform_type, params in reversed(omega['transforms']):
                single_omega = {'type': transform_type, 'params': params}
                result = self.inverse_apply(result, single_omega)
                if result is None:
                    return None  # Non-invertible transform encountered
            return result
        else:
            # Single transform mode
            return self.inverse_apply(x, omega)

    def __call__(
        self,
        noisy_input: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
        """Apply ScoreAug transform to noisy input and target.

        Args:
            noisy_input: [B, C, H, W] or [B, C, D, H, W] - concatenated [noisy_images, seg_mask]
            target: [B, C_out, H, W] or [B, C_out, D, H, W] - velocity target for RFlow

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
            # Compose mode: apply multiple transforms independently
            transforms = self.sample_compose_transforms()

            if not transforms:
                return noisy_input, target, None

            # Apply all transforms in sequence
            aug_input = noisy_input
            aug_target = target
            for transform_type, params in transforms:
                aug_input = self.apply(aug_input, transform_type, params)
                aug_target = self.apply(aug_target, transform_type, params)

            # Return list of transforms for omega encoding
            omega = {
                'compose': True,
                'transforms': transforms,  # List of (type, params) tuples
            }
            return aug_input, aug_target, omega

        else:
            # Single transform mode (per paper)
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
    # Reshape for broadcasting
    if spatial_dims == 2:
        scales = scales.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
    else:
        scales = scales.view(-1, 1, 1, 1, 1)  # [B, 1, 1, 1, 1]

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


class OmegaTimeEmbed(nn.Module):
    """Wrapper around time_embed that adds omega conditioning.

    Supports both 2D and 3D models.

    This is compile-compatible because:
    - No hooks, just module replacement
    - No data-dependent control flow in forward
    - Always adds omega embedding (zero for identity)
    """

    def __init__(self, original_time_embed: nn.Module, embed_dim: int):
        """Initialize omega-aware time embedding.

        Args:
            original_time_embed: The original time_embed module from DiffusionModelUNet
            embed_dim: Output dimension (should match original time_embed output)
        """
        super().__init__()
        self.original = original_time_embed
        self.embed_dim = embed_dim

        # MLP that maps omega encoding to embedding (zero-init for neutral start)
        self.omega_mlp = create_zero_init_mlp(OMEGA_ENCODING_DIM, embed_dim)

        # Buffer to store current omega encoding
        # This is set by ScoreAugModelWrapper before each forward
        self.register_buffer('_omega_encoding', torch.zeros(1, OMEGA_ENCODING_DIM))

    def set_omega_encoding(self, omega_encoding: torch.Tensor):
        """Set omega encoding for next forward pass.

        Uses in-place copy which is atomic for single CUDA kernel.
        Buffer identity preserved for torch.compile compatibility.

        Args:
            omega_encoding: Tensor [1, OMEGA_ENCODING_DIM]
        """
        # Ensure source is on same device and contiguous for atomic copy
        omega_encoding = omega_encoding.to(self._omega_encoding.device).contiguous()
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


class ScoreAugModelWrapper(nn.Module):
    """Wrapper to inject omega conditioning into MONAI UNet.

    Supports both 2D and 3D models.

    This implementation is compile-compatible:
    - Replaces time_embed with OmegaTimeEmbed (no hooks)
    - Sets omega encoding before forward (outside traced graph)
    - Forward has fixed control flow (no data-dependent branches)
    """

    def __init__(self, model: nn.Module, embed_dim: int = 256, spatial_dims: int = 2):
        """Initialize wrapper.

        Args:
            model: MONAI DiffusionModelUNet to wrap
            embed_dim: Embedding dimension (should match model's time_embed output)
            spatial_dims: Number of spatial dimensions (2 or 3)
        """
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim
        self.spatial_dims = spatial_dims

        # Replace time_embed with omega-aware version
        if not hasattr(model, 'time_embed'):
            raise ValueError("Model does not have 'time_embed' attribute")

        original_time_embed = model.time_embed
        self.omega_time_embed = OmegaTimeEmbed(original_time_embed, embed_dim)

        # Replace the model's time_embed
        model.time_embed = self.omega_time_embed

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        conditioning: dict[str, Any] | None = None,
        # Deprecated individual params (kept for backward compat)
        omega: dict[str, Any] | None = None,
        mode_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with omega conditioning.

        Args:
            x: Noisy input tensor [B, C, H, W] or [B, C, D, H, W]
            timesteps: Timestep tensor [B]
            conditioning: Optional dict with conditioning params. Supported keys:
                - 'omega': ScoreAug parameters dict
                - 'mode_id': Mode ID tensor for intensity scaling
            omega: (Deprecated) Use conditioning={'omega': ...} instead.
            mode_id: (Deprecated) Use conditioning={'mode_id': ...} instead.

        Returns:
            Model prediction [B, C_out, H, W] or [B, C_out, D, H, W]
        """
        # Build conditioning from dict or individual params
        if conditioning is not None:
            omega = conditioning.get('omega', omega)
            mode_id = conditioning.get('mode_id', mode_id)

        # Encode omega + mode_id as (1, 36) - broadcasts to batch in MLP
        # Using fixed shape keeps torch.compile happy
        omega_encoding = encode_omega(omega, x.device, mode_id=mode_id, spatial_dims=self.spatial_dims)
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


# =============================================================================
# Backwards Compatibility Aliases
# =============================================================================

# OmegaTimeEmbed doesn't depend on spatial_dims, so simple alias is fine
OmegaTimeEmbed3D = OmegaTimeEmbed


class ScoreAugTransform3D(ScoreAugTransform):
    """3D ScoreAug transform (backwards compatibility wrapper).

    Equivalent to ScoreAugTransform(spatial_dims=3, ...).
    """

    def __init__(self, **kwargs):
        # Force spatial_dims=3 for 3D compatibility
        kwargs['spatial_dims'] = 3
        super().__init__(**kwargs)


class ScoreAugModelWrapper3D(ScoreAugModelWrapper):
    """3D ScoreAug model wrapper (backwards compatibility wrapper).

    Equivalent to ScoreAugModelWrapper(..., spatial_dims=3).
    """

    def __init__(self, model: nn.Module, embed_dim: int = 256, **kwargs):
        # Force spatial_dims=3 for 3D compatibility
        super().__init__(model, embed_dim, spatial_dims=3, **kwargs)

# 3D pattern names alias
PATTERN_NAMES_3D = PATTERN_NAMES
NUM_PATTERNS_3D = NUM_PATTERNS

# 3D mode intensity scale aliases
MODE_INTENSITY_SCALE_3D = MODE_INTENSITY_SCALE
MODE_INTENSITY_SCALE_INV_3D = MODE_INTENSITY_SCALE_INV


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


# Alias for dimension constant
OMEGA_ENCODING_DIM_3D = OMEGA_ENCODING_DIM

# 2D-specific pattern generators (backwards compat)
generate_pattern_mask_3d = generate_pattern_mask
