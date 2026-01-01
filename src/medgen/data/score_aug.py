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
"""

import random
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn


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
    ):
        """Initialize ScoreAug transform.

        Two modes:
        - compose=False (default): One transform sampled with equal probability (per paper)
        - compose=True: Each transform applied independently with compose_prob

        Args:
            rotation: Enable 90, 180, 270 degree rotations (requires omega conditioning)
            flip: Enable horizontal flip (requires omega conditioning for consistency)
            translation: Enable ±40% X, ±20% Y translation with zero-padding
            cutout: Enable random rectangle cutout (10-30% each dimension)
            brightness: Enable brightness scaling (experimental with normalized data)
            brightness_range: Max brightness factor B, scales in [1/B, B]
            compose: If True, apply transforms independently instead of picking one
            compose_prob: Probability for each transform when compose=True (default 0.5)
        """
        self.rotation = rotation
        self.flip = flip
        self.translation = translation
        self.cutout = cutout
        self.brightness = brightness
        self.brightness_range = brightness_range
        self.compose = compose
        self.compose_prob = compose_prob

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
        elif transform_type in ('translate', 'cutout'):
            # Non-invertible transforms (zero-padded regions lose info)
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

        Note: With normalized [0,1] data, scaling up may push values > 1.
        The paper uses this on unnormalized data. Use with caution.

        Args:
            x: Input tensor [B, C, H, W]
            scale: Brightness scale factor

        Returns:
            Scaled tensor (NOT clamped to preserve invertibility)
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
        if self.compose:
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
# Omega Conditioning (compile-compatible implementation)
# =============================================================================

# Omega encoding format:
# Single mode: [type_onehot(8), rot_k_norm, dx, dy, cx, cy, size_x, size_y, brightness_scale]
#   Types: 0=identity, 1=rotation, 2=hflip, 3=vflip, 4=rot90_hflip, 5=translation, 6=cutout, 7=brightness
# Compose mode: [active_mask(4), rot_type(4), rot_k_norm, dx, dy, cx, cy, size_x, size_y, brightness_scale]
#   Active mask: spatial, translation, cutout, brightness (1=active)
#   Rot type: identity, rot90, hflip, vflip, rot90_hflip (one-hot within spatial)
OMEGA_ENCODING_DIM = 16


def encode_omega(
    omega: Optional[Dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    """Encode omega dict into tensor format for MLP.

    All samples in a batch get the same transform, so we return shape (1, 16)
    which broadcasts to (B, 16) in the MLP. This keeps the buffer shape constant
    for torch.compile compatibility.

    Supports both single-transform mode and compose mode.

    Args:
        omega: Transform parameters dict or None
        device: Target device

    Returns:
        Tensor [1, OMEGA_ENCODING_DIM] encoding the transform
    """
    enc = torch.zeros(1, OMEGA_ENCODING_DIM, device=device)

    if omega is None:
        # Identity: type_onehot[0] = 1, rest = 0
        enc[0, 0] = 1.0
        return enc

    # Check for compose mode
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

    # Single transform mode
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

        # MLP that maps omega encoding to embedding
        # Uses larger hidden dim for expressivity
        self.omega_mlp = nn.Sequential(
            nn.Linear(OMEGA_ENCODING_DIM, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Initialize output layer to near-zero so identity starts as no-op
        nn.init.zeros_(self.omega_mlp[-1].weight)
        nn.init.zeros_(self.omega_mlp[-1].bias)

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
    ) -> torch.Tensor:
        """Forward pass with omega conditioning.

        Args:
            x: Noisy input tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            omega: Optional augmentation parameters for conditioning

        Returns:
            Model prediction [B, C_out, H, W]
        """
        # Encode omega as (1, 12) - broadcasts to batch in MLP
        # Using fixed shape keeps torch.compile happy
        omega_encoding = encode_omega(omega, x.device)
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
