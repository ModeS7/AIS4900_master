"""Score Augmentation for Diffusion Models.

Reference: https://arxiv.org/abs/2508.07926

Applies transforms to noisy data (after noise addition) to improve
generalization without changing the output distribution.

Traditional augmentation: x -> T(x) -> add noise -> denoise -> T(x)
ScoreAug:                 x -> add noise -> T(x + noise) -> denoise -> T(x)

This aligns with diffusion's denoising mechanism and provides stronger
regularization without changing the output distribution.

Conditioning requirements (per paper):
- Rotation: REQUIRES omega conditioning (noise is rotation-invariant, model can cheat)
- Translation/Cutout: Work without conditioning but risk data leakage
- Brightness: Linear transform, works without conditioning

v2 mode adds structured masking patterns:
- Non-destructive transforms (can stack): rotation, flip, translation
- Destructive transforms (pick one): cutout OR fixed patterns
- Fixed patterns: checkerboard, grid dropout, coarse dropout, patch dropout
"""

import random
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np

from medgen.models.wrappers import create_zero_init_mlp


# =============================================================================
# Fixed Pattern Definitions (16 total)
# =============================================================================
# These are deterministic masks that the network learns via one-hot embedding.
# Patterns are generated as functions of (H, W) for flexibility.

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


def generate_pattern_mask(pattern_id: int, H: int, W: int) -> torch.Tensor:
    """Generate mask for a fixed pattern ID.

    Pattern IDs (16 total):
        0-3:   Checkerboard (4x4 std, 4x4 offset, 8x8 std, 8x8 offset)
        4-7:   Grid dropout (4x4 25% seed0, 4x4 25% seed1, 4x4 50% seed0, 4x4 50% seed1)
        8-11:  Coarse dropout (patterns 0-3)
        12-15: Patch dropout (8x8 25% seedA, 8x8 25% seedB, 8x8 50% seedA, 8x8 50% seedB)

    Args:
        pattern_id: Pattern index (0-15)
        H, W: Image dimensions

    Returns:
        Mask tensor [H, W] with 0=keep, 1=drop
    """
    if pattern_id < 4:
        # Checkerboard patterns
        if pattern_id == 0:
            return _checkerboard_mask(H, W, grid_size=4, offset=False)
        elif pattern_id == 1:
            return _checkerboard_mask(H, W, grid_size=4, offset=True)
        elif pattern_id == 2:
            return _checkerboard_mask(H, W, grid_size=8, offset=False)
        elif pattern_id == 3:
            return _checkerboard_mask(H, W, grid_size=8, offset=True)

    elif pattern_id < 8:
        # Grid dropout patterns
        if pattern_id == 4:
            return _grid_dropout_mask(H, W, grid_size=4, drop_ratio=0.25, seed=42)
        elif pattern_id == 5:
            return _grid_dropout_mask(H, W, grid_size=4, drop_ratio=0.25, seed=123)
        elif pattern_id == 6:
            return _grid_dropout_mask(H, W, grid_size=4, drop_ratio=0.50, seed=42)
        elif pattern_id == 7:
            return _grid_dropout_mask(H, W, grid_size=4, drop_ratio=0.50, seed=123)

    elif pattern_id < 12:
        # Coarse dropout patterns
        return _coarse_dropout_mask(H, W, pattern_id=pattern_id - 8)

    else:
        # Patch dropout patterns (8x8 patches for 128px image = 16 patches)
        patch_size = max(H // 8, 1)
        if pattern_id == 12:
            return _patch_dropout_mask(H, W, patch_size=patch_size, drop_ratio=0.25, seed=42)
        elif pattern_id == 13:
            return _patch_dropout_mask(H, W, patch_size=patch_size, drop_ratio=0.25, seed=123)
        elif pattern_id == 14:
            return _patch_dropout_mask(H, W, patch_size=patch_size, drop_ratio=0.50, seed=42)
        elif pattern_id == 15:
            return _patch_dropout_mask(H, W, patch_size=patch_size, drop_ratio=0.50, seed=123)

    # Fallback: no masking
    return torch.zeros(H, W)


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


class ScoreAugTransform:
    """Applies transforms to noisy input and target per ScoreAug paper.

    Transforms (from paper):
    - Rotation: 90, 180, 270 degrees (REQUIRES omega conditioning)
    - Translation: +/-10% shift with zero-padding
    - Cutout: Random rectangular region zeroed
    - Brightness: Scale by factor in [1/B, B] range

    Note: Rotation requires omega conditioning because Gaussian noise is
    rotation-invariant, allowing the model to detect rotation from the
    noise pattern and "cheat" by inverting it before denoising.
    """

    def __init__(
        self,
        rotation: bool = True,
        flip: bool = True,
        translation: bool = False,
        cutout: bool = False,
        brightness: bool = False,
        brightness_range: float = 1.2,
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
            rotation: Enable 90, 180, 270 degree rotations (requires omega conditioning)
            flip: Enable horizontal flip (requires omega conditioning for consistency)
            translation: Enable ±40% X, ±20% Y translation with zero-padding
            cutout: Enable random rectangle cutout (10-30% each dimension)
            brightness: Enable brightness scaling (DEPRECATED - do not use)
            brightness_range: Max brightness factor B (DEPRECATED)
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
        self.brightness = brightness
        self.brightness_range = brightness_range
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

        # Cache for pattern masks (generated lazily per image size)
        self._pattern_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

    def sample_transform(self) -> Tuple[str, Dict[str, Any]]:
        """Sample a random transform with equal probability (per paper).

        Identity is always included as one option. All enabled transforms
        have equal probability of being selected.

        Returns:
            Tuple of (transform_type, params_dict)
        """
        # Build list of all enabled transforms + identity
        transforms = ['identity']
        if self.rotation or self.flip:
            transforms.append('spatial')  # Unified D4 symmetries (rotations + flips)
        if self.translation:
            transforms.append('translate')
        if self.cutout:
            transforms.append('cutout')
        if self.brightness:
            transforms.append('brightness')

        # Sample uniformly across all options (per paper)
        transform_type = random.choice(transforms)

        if transform_type == 'identity':
            return 'identity', {}
        elif transform_type == 'spatial':
            # D4 dihedral group: 7 non-identity symmetries
            # Build list based on enabled options
            spatial_options = []
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

            if not spatial_options:
                return 'identity', {}

            return random.choice(spatial_options)
        elif transform_type == 'translate':
            # Asymmetric: ±40% X, ±20% Y (brain is oval, more vertical space taken)
            # Aggressive is fine for ScoreAug - doesn't affect output distribution
            dx = random.uniform(-0.4, 0.4)
            dy = random.uniform(-0.2, 0.2)
            return 'translate', {'dx': dx, 'dy': dy}
        elif transform_type == 'cutout':
            # Random rectangle (not square) - sample width/height independently
            size_x = random.uniform(0.1, 0.3)  # 10-30% of width
            size_y = random.uniform(0.1, 0.3)  # 10-30% of height
            cx = random.uniform(size_x / 2, 1 - size_x / 2)
            cy = random.uniform(size_y / 2, 1 - size_y / 2)
            return 'cutout', {'cx': cx, 'cy': cy, 'size_x': size_x, 'size_y': size_y}
        elif transform_type == 'brightness':
            # Sample scale factor uniformly in log space for symmetric distribution
            log_scale = random.uniform(-1, 1) * abs(self.brightness_range - 1)
            scale = 1.0 + log_scale
            # Clamp to valid range
            scale = max(1.0 / self.brightness_range, min(self.brightness_range, scale))
            return 'brightness', {'scale': scale}

        return 'identity', {}

    def sample_compose_transforms(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Sample multiple transforms independently for compose mode.

        Each enabled transform has compose_prob chance of being applied.
        Order: spatial → translate → cutout → brightness

        Returns:
            List of (transform_type, params) tuples to apply in sequence
        """
        transforms = []

        # Spatial (rotation/flip) - sample one from D4 if triggered
        if (self.rotation or self.flip) and random.random() < self.compose_prob:
            spatial_options = []
            if self.rotation:
                spatial_options.extend([
                    ('rot90', {'k': 1}),
                    ('rot90', {'k': 2}),
                    ('rot90', {'k': 3}),
                ])
            if self.flip:
                spatial_options.extend([
                    ('hflip', {}),
                    ('vflip', {}),
                ])
            if self.rotation and self.flip:
                spatial_options.extend([
                    ('rot90_hflip', {'k': 1}),
                    ('rot90_hflip', {'k': 3}),
                ])
            if spatial_options:
                transforms.append(random.choice(spatial_options))

        # Translation
        if self.translation and random.random() < self.compose_prob:
            dx = random.uniform(-0.4, 0.4)
            dy = random.uniform(-0.2, 0.2)
            transforms.append(('translate', {'dx': dx, 'dy': dy}))

        # Cutout
        if self.cutout and random.random() < self.compose_prob:
            size_x = random.uniform(0.1, 0.3)
            size_y = random.uniform(0.1, 0.3)
            cx = random.uniform(size_x / 2, 1 - size_x / 2)
            cy = random.uniform(size_y / 2, 1 - size_y / 2)
            transforms.append(('cutout', {'cx': cx, 'cy': cy, 'size_x': size_x, 'size_y': size_y}))

        # Brightness
        if self.brightness and random.random() < self.compose_prob:
            log_scale = random.uniform(-1, 1) * abs(self.brightness_range - 1)
            scale = 1.0 + log_scale
            scale = max(1.0 / self.brightness_range, min(self.brightness_range, scale))
            transforms.append(('brightness', {'scale': scale}))

        return transforms

    def sample_v2_transforms(self) -> Tuple[List[Tuple[str, Dict[str, Any]]], Optional[Tuple[str, Dict[str, Any]]]]:
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
            spatial_options = []
            if self.rotation:
                spatial_options.extend([
                    ('rot90', {'k': 1}),
                    ('rot90', {'k': 2}),
                    ('rot90', {'k': 3}),
                ])
            if self.flip:
                spatial_options.extend([
                    ('hflip', {}),
                    ('vflip', {}),
                ])
            if self.rotation and self.flip:
                spatial_options.extend([
                    ('rot90_hflip', {'k': 1}),
                    ('rot90_hflip', {'k': 3}),
                ])
            if spatial_options:
                nondestructive.append(random.choice(spatial_options))

        # Translation (non-destructive since brain is centered)
        if self.translation and random.random() < self.nondestructive_prob:
            dx = random.uniform(-0.4, 0.4)
            dy = random.uniform(-0.2, 0.2)
            nondestructive.append(('translate', {'dx': dx, 'dy': dy}))

        # Sample destructive transform (with destructive_prob)
        destructive = None
        if random.random() < self.destructive_prob:
            # Choose between cutout and fixed patterns
            if random.random() < self.cutout_vs_pattern:
                # Random cutout
                size_x = random.uniform(0.1, 0.3)
                size_y = random.uniform(0.1, 0.3)
                cx = random.uniform(size_x / 2, 1 - size_x / 2)
                cy = random.uniform(size_y / 2, 1 - size_y / 2)
                destructive = ('cutout', {'cx': cx, 'cy': cy, 'size_x': size_x, 'size_y': size_y})
            elif self._enabled_patterns:
                # Fixed pattern (uniform over enabled patterns)
                pattern_id = random.choice(self._enabled_patterns)
                destructive = ('pattern', {'pattern_id': pattern_id})

        return nondestructive, destructive

    def _get_pattern_mask(self, pattern_id: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Get cached pattern mask, generating if needed.

        Args:
            pattern_id: Pattern index (0-15)
            H, W: Image dimensions
            device: Target device

        Returns:
            Mask tensor [H, W] with 0=keep, 1=drop
        """
        cache_key = (pattern_id, H, W)
        if cache_key not in self._pattern_cache:
            mask = generate_pattern_mask(pattern_id, H, W)
            self._pattern_cache[cache_key] = mask
        return self._pattern_cache[cache_key].to(device)

    def _apply_pattern(self, x: torch.Tensor, pattern_id: int) -> torch.Tensor:
        """Apply fixed pattern mask to tensor.

        Args:
            x: Input tensor [B, C, H, W]
            pattern_id: Pattern index (0-15)

        Returns:
            Tensor with pattern regions zeroed
        """
        B, C, H, W = x.shape
        mask = self._get_pattern_mask(pattern_id, H, W, x.device)

        # Expand mask to [1, 1, H, W] for broadcasting
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Zero out masked regions
        return x * (1 - mask)

    def apply(
        self,
        x: torch.Tensor,
        transform_type: str,
        params: Dict[str, Any],
    ) -> torch.Tensor:
        """Apply transform to tensor [B, C, H, W].

        Args:
            x: Input tensor
            transform_type: Type of transform to apply
            params: Transform parameters

        Returns:
            Transformed tensor
        """
        if transform_type == 'identity':
            return x
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
        elif transform_type == 'translate':
            return self._translate(x, params['dx'], params['dy'])
        elif transform_type == 'cutout':
            return self._cutout(x, params['cx'], params['cy'], params['size_x'], params['size_y'])
        elif transform_type == 'brightness':
            return self._brightness(x, params['scale'])
        elif transform_type == 'pattern':
            return self._apply_pattern(x, params['pattern_id'])
        else:
            return x

    def inverse_apply(
        self,
        x: torch.Tensor,
        omega: Optional[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        """Apply inverse transform to tensor.

        For invertible transforms (rotation, brightness), applies the inverse to
        recover original values. For non-invertible transforms (translation,
        cutout), returns None since information is lost.

        Args:
            x: Transformed tensor [B, C, H, W]
            omega: Transform parameters from __call__

        Returns:
            Inverse-transformed tensor, or None if not invertible
        """
        if omega is None:
            return x

        transform_type = omega['type']
        params = omega['params']

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
        elif transform_type == 'brightness':
            # Inverse of multiply by scale is divide by scale
            return x / params['scale']
        elif transform_type in ('translate', 'cutout', 'pattern'):
            # Non-invertible transforms (zero-padded/masked regions lose info)
            return None
        else:
            return x

    def requires_omega(self, omega: Optional[Dict[str, Any]]) -> bool:
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
        # All spatial transforms (D4 symmetries) require omega conditioning
        return omega['type'] in ('rot90', 'hflip', 'vflip', 'rot90_hflip')

    def _translate(self, x: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
        """Translate tensor by (dx, dy) fraction, zero-pad edges.

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

    def _cutout(self, x: torch.Tensor, cx: float, cy: float, size_x: float, size_y: float) -> torch.Tensor:
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

    def _brightness(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        """Scale tensor by brightness factor.

        WARNING: Output is NOT clamped to [0,1] to preserve invertibility
        for score matching. If scale > 1.0, output values may exceed 1.0.
        This is intentional per the ScoreAug paper - clamping would break
        the inverse transform needed for proper score matching.

        Args:
            x: Input tensor [B, C, H, W] in [0, 1] range.
            scale: Brightness scale factor (typically 0.8-1.2).

        Returns:
            Scaled tensor (may have values outside [0,1] if scale != 1.0).
        """
        return x * scale

    def apply_omega(
        self,
        x: torch.Tensor,
        omega: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """Apply transform(s) from omega dict to tensor.

        Handles both single-transform and compose mode omega formats.

        Args:
            x: Input tensor [B, C, H, W]
            omega: Transform parameters (single or compose format), or None for identity

        Returns:
            Transformed tensor
        """
        if omega is None:
            return x

        if omega.get('compose', False):
            # Compose mode: apply all transforms in sequence
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
        omega: Optional[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        """Apply inverse transform(s) from omega dict to tensor.

        Handles both single-transform and compose mode omega formats.
        For compose mode, applies inverses in reverse order.

        Args:
            x: Transformed tensor [B, C, H, W]
            omega: Transform parameters (single or compose format), or None for identity

        Returns:
            Inverse-transformed tensor, or None if any transform is non-invertible
        """
        if omega is None:
            return x

        if omega.get('compose', False):
            # Compose mode: apply inverse transforms in REVERSE order
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
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        """Apply ScoreAug transform to noisy input and target.

        Args:
            noisy_input: [B, C, H, W] - concatenated [noisy_images, seg_mask]
            target: [B, C_out, H, W] - velocity target for RFlow (or noise for DDPM)

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
    mode_id: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply modality-specific intensity scaling to input (per-sample).

    This makes mode conditioning NECESSARY for correct predictions.
    The model sees scaled input but must predict unscaled target.

    Supports mixed modalities within a batch - each sample gets its own
    scale factor based on its mode_id.

    Args:
        x: Input tensor [B, C, H, W]
        mode_id: Mode ID tensor [B] or None (0=bravo, 1=flair, 2=t1_pre, 3=t1_gd)

    Returns:
        Tuple of (scaled_input, scale_factors)
        - scaled_input: x * scale_factors (per-sample scaling)
        - scale_factors: Tensor [B, 1, 1, 1] with per-sample scales (or [1] if mode_id is None)
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
    # Reshape for broadcasting [B, 1, 1, 1]
    scales = scales.view(-1, 1, 1, 1)
    return x * scales, scales


def inverse_mode_intensity_scale(
    x: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Inverse the mode intensity scaling (per-sample).

    Args:
        x: Scaled tensor [B, C, H, W]
        scale: The scale factors that were applied [B, 1, 1, 1] or [1]

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
# Legacy mode (compose=False, v2=False) - uses dims 0-15:
#   [type_onehot(8), rot_k_norm, dx, dy, cx, cy, size_x, size_y, brightness_scale]
#   Types: 0=identity, 1=rotation, 2=hflip, 3=vflip, 4=rot90_hflip, 5=translation, 6=cutout, 7=brightness
#
# Compose mode (compose=True) - uses dims 0-15:
#   [active_mask(4), spatial_type(4), rot_k, dx, dy, cx, cy, size_x, size_y, brightness]
#   Active mask: spatial, translation, cutout, brightness (1=active)
#
# v2 mode (v2=True) - uses dims 0-31:
#   Dims 0-15: Same as compose mode for non-destructive + cutout
#   Dims 16-31: Pattern ID one-hot (16 patterns)
#
# Mode intensity scaling - dims 32-35:
#   Dims 32-35: Mode one-hot (4 modalities) - indicates which scale was applied
#
OMEGA_ENCODING_DIM = 36  # Extended to accommodate pattern IDs + mode scaling


def encode_omega(
    omega: Optional[Dict[str, Any]],
    device: torch.device,
    mode_id: Optional[torch.Tensor] = None,
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
        # v2 mode: encode all transforms (non-destructive + destructive)
        # Format: [active_mask(4), spatial_type(4), rot_k, dx, dy, cx, cy, size_x, size_y, _, pattern_onehot(16)]
        transforms = omega['transforms']

        for t, p in transforms:
            if t in ('rot90', 'hflip', 'vflip', 'rot90_hflip'):
                enc[0, 0] = 1.0  # spatial active
                if t == 'rot90':
                    enc[0, 4] = 1.0  # rot90 type
                    enc[0, 8] = p['k'] / 3.0
                elif t == 'hflip':
                    enc[0, 5] = 1.0  # hflip type
                elif t == 'vflip':
                    enc[0, 6] = 1.0  # vflip type
                elif t == 'rot90_hflip':
                    enc[0, 7] = 1.0  # rot90_hflip type
                    enc[0, 8] = p['k'] / 3.0
            elif t == 'translate':
                enc[0, 1] = 1.0  # translation active
                enc[0, 9] = p['dx']
                enc[0, 10] = p['dy']
            elif t == 'cutout':
                enc[0, 2] = 1.0  # cutout active
                enc[0, 11] = p['cx']
                enc[0, 12] = p['cy']
                enc[0, 13] = p['size_x']
                enc[0, 14] = p['size_y']
            elif t == 'pattern':
                # Fixed pattern: encode as one-hot in dims 16-31
                pattern_id = p['pattern_id']
                enc[0, 3] = 1.0  # pattern active (replaces brightness in v2)
                enc[0, 16 + pattern_id] = 1.0  # pattern one-hot

        return enc

    # Check for compose mode (legacy)
    if omega.get('compose', False):
        # Compose mode: encode all active transforms
        # Format: [active_mask(4), spatial_type(4), rot_k, dx, dy, cx, cy, size_x, size_y, brightness]
        transforms = omega['transforms']

        for t, p in transforms:
            if t in ('rot90', 'hflip', 'vflip', 'rot90_hflip'):
                enc[0, 0] = 1.0  # spatial active
                if t == 'rot90':
                    enc[0, 4] = 1.0  # rot90 type
                    enc[0, 8] = p['k'] / 3.0
                elif t == 'hflip':
                    enc[0, 5] = 1.0  # hflip type
                elif t == 'vflip':
                    enc[0, 6] = 1.0  # vflip type
                elif t == 'rot90_hflip':
                    enc[0, 7] = 1.0  # rot90_hflip type
                    enc[0, 8] = p['k'] / 3.0
            elif t == 'translate':
                enc[0, 1] = 1.0  # translation active
                enc[0, 9] = p['dx']
                enc[0, 10] = p['dy']
            elif t == 'cutout':
                enc[0, 2] = 1.0  # cutout active
                enc[0, 11] = p['cx']
                enc[0, 12] = p['cy']
                enc[0, 13] = p['size_x']
                enc[0, 14] = p['size_y']
            elif t == 'brightness':
                enc[0, 3] = 1.0  # brightness active
                enc[0, 15] = p['scale'] - 1.0

        return enc

    # Single transform mode (legacy)
    t = omega['type']
    p = omega['params']

    if t == 'rot90':
        enc[0, 1] = 1.0  # rotation type
        enc[0, 8] = p['k'] / 3.0  # normalized k (1,2,3 -> 0.33, 0.67, 1.0)
    elif t == 'hflip':
        enc[0, 2] = 1.0  # hflip type (no extra params)
    elif t == 'vflip':
        enc[0, 3] = 1.0  # vflip type (no extra params)
    elif t == 'rot90_hflip':
        enc[0, 4] = 1.0  # rot90_hflip type
        enc[0, 8] = p['k'] / 3.0  # normalized k (1 or 3 -> 0.33 or 1.0)
    elif t == 'translate':
        enc[0, 5] = 1.0  # translation type
        enc[0, 9] = p['dx']  # in [-0.4, 0.4]
        enc[0, 10] = p['dy']  # in [-0.2, 0.2]
    elif t == 'cutout':
        enc[0, 6] = 1.0  # cutout type
        enc[0, 11] = p['cx']  # in [0, 1]
        enc[0, 12] = p['cy']
        enc[0, 13] = p['size_x']  # in [0.1, 0.3]
        enc[0, 14] = p['size_y']  # in [0.1, 0.3]
    elif t == 'brightness':
        enc[0, 7] = 1.0  # brightness type
        enc[0, 15] = p['scale'] - 1.0  # centered around 0

    return enc


class OmegaTimeEmbed(nn.Module):
    """Wrapper around time_embed that adds omega conditioning.

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

        Uses in-place copy to maintain buffer identity for torch.compile.

        Args:
            omega_encoding: Tensor [1, OMEGA_ENCODING_DIM]
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


class ScoreAugModelWrapper(nn.Module):
    """Wrapper to inject omega conditioning into MONAI UNet.

    This implementation is compile-compatible:
    - Replaces time_embed with OmegaTimeEmbed (no hooks)
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
        omega: Optional[Dict[str, Any]] = None,
        mode_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with omega conditioning.

        Args:
            x: Noisy input tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            omega: Optional augmentation parameters for conditioning
            mode_id: Optional mode ID for intensity scaling (0=bravo, 1=flair, etc.)

        Returns:
            Model prediction [B, C_out, H, W]
        """
        # Encode omega + mode_id as (1, 36) - broadcasts to batch in MLP
        # Using fixed shape keeps torch.compile happy
        omega_encoding = encode_omega(omega, x.device, mode_id=mode_id)
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
