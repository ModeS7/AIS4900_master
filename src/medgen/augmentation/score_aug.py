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
from typing import Any

import torch

# Import pattern functions and constants from helper module
from .score_aug_patterns import (
    PATTERN_NAMES,
    PATTERN_NAMES_3D,
    NUM_PATTERNS,
    NUM_PATTERNS_3D,
    generate_pattern_mask,
    generate_pattern_mask_3d,
    _cached_generate_pattern_mask,
    clear_pattern_cache,
)

# Import omega encoding, mode intensity scaling, and wrapper classes
from .score_aug_omega import (
    MODE_INTENSITY_SCALE,
    MODE_INTENSITY_SCALE_INV,
    MODE_INTENSITY_SCALE_3D,
    MODE_INTENSITY_SCALE_INV_3D,
    OMEGA_ENCODING_DIM,
    OMEGA_ENCODING_DIM_3D,
    apply_mode_intensity_scale,
    apply_mode_intensity_scale_3d,
    inverse_mode_intensity_scale,
    inverse_mode_intensity_scale_3d,
    encode_omega,
    encode_omega_3d,
)

from .score_aug_wrapper import (
    OmegaTimeEmbed,
    OmegaTimeEmbed3D,
    ScoreAugModelWrapper,
    ScoreAugModelWrapper3D,
)


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
        Order: spatial -> translate -> cutout

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
# Backwards Compatibility Aliases
# =============================================================================

class ScoreAugTransform3D(ScoreAugTransform):
    """3D ScoreAug transform (backwards compatibility wrapper).

    Equivalent to ScoreAugTransform(spatial_dims=3, ...).
    """

    def __init__(self, **kwargs):
        # Force spatial_dims=3 for 3D compatibility
        kwargs['spatial_dims'] = 3
        super().__init__(**kwargs)
