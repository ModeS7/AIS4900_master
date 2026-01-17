"""Score Augmentation for 3D Diffusion Models.

Reference: https://arxiv.org/abs/2508.07926

Adapts ScoreAug transforms for 3D volumetric data [B, C, D, H, W].

Key differences from 2D:
- Rotations can be around any of 3 axes (D, H, W)
- Flips along all 3 axes
- 3D translations
- 3D cutout regions

Conditioning requirements (per paper):
- Rotation: REQUIRES omega conditioning (noise is rotation-invariant, model can cheat)
- Translation/Cutout: Work without conditioning but risk data leakage
"""

import random
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn

from medgen.models.wrappers import create_zero_init_mlp


class ScoreAugTransform3D:
    """Applies transforms to 3D noisy input and target per ScoreAug paper.

    Transforms:
    - Rotation: 90, 180, 270 degrees around D/H/W axes (REQUIRES omega conditioning)
    - Flip: Along D/H/W axes (REQUIRES omega conditioning)
    - Translation: +/-20% shift with zero-padding in all 3 dims
    - Cutout: Random 3D rectangular region zeroed
    """

    def __init__(
        self,
        rotation: bool = True,
        flip: bool = True,
        translation: bool = False,
        cutout: bool = False,
        compose: bool = False,
        compose_prob: float = 0.5,
    ):
        """Initialize 3D ScoreAug transform.

        Two modes:
        - compose=False (default): One transform sampled with equal probability
        - compose=True: Each transform applied independently with compose_prob

        Args:
            rotation: Enable 90, 180, 270 degree rotations around axes
            flip: Enable flips along D/H/W axes
            translation: Enable ±20% translation with zero-padding
            cutout: Enable random 3D cutout (10-30% each dimension)
            compose: If True, apply transforms independently
            compose_prob: Probability for each transform when compose=True
        """
        self.rotation = rotation
        self.flip = flip
        self.translation = translation
        self.cutout = cutout
        self.compose = compose
        self.compose_prob = compose_prob

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
            # 3D translation: ±20% in each dimension
            dd = random.uniform(-0.2, 0.2)
            dh = random.uniform(-0.2, 0.2)
            dw = random.uniform(-0.2, 0.2)
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

        # Translation
        if self.translation and random.random() < self.compose_prob:
            dd = random.uniform(-0.2, 0.2)
            dh = random.uniform(-0.2, 0.2)
            dw = random.uniform(-0.2, 0.2)
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
        if self.compose:
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
# Omega Conditioning for 3D (simplified)
# =============================================================================
# Omega encoding format (24 dims):
#   [type_onehot(12), axis_onehot(3), rot_k_norm, dd, dh, dw, cd, ch, cw, size_d, size_h, size_w]
#   Types: 0=identity, 1=rot90_d, 2=rot90_h, 3=rot90_w,
#          4=flip_d, 5=flip_h, 6=flip_w,
#          7=translate, 8=cutout,
#          9-11=reserved

OMEGA_ENCODING_DIM_3D = 24


def encode_omega_3d(
    omega: Optional[Dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    """Encode 3D omega dict into tensor format.

    Args:
        omega: Transform parameters dict or None
        device: Target device

    Returns:
        Tensor [1, OMEGA_ENCODING_DIM_3D] encoding the transform
    """
    enc = torch.zeros(1, OMEGA_ENCODING_DIM_3D, device=device)

    if omega is None:
        enc[0, 0] = 1.0  # identity
        return enc

    # Handle compose mode
    if omega.get('compose', False):
        transforms = omega['transforms']
        for t, p in transforms:
            _encode_single_transform(enc, t, p)
        return enc

    # Single transform
    _encode_single_transform(enc, omega['type'], omega['params'])
    return enc


def _encode_single_transform(
    enc: torch.Tensor,
    transform_type: str,
    params: Dict[str, Any],
) -> None:
    """Encode a single transform into the encoding tensor (in-place).

    Args:
        enc: Encoding tensor [1, OMEGA_ENCODING_DIM_3D]
        transform_type: Transform type string
        params: Transform parameters
    """
    if transform_type == 'rot90_3d':
        axis = params['axis']
        k = params['k']
        if axis == 'd':
            enc[0, 1] = 1.0
            enc[0, 12] = 1.0  # axis=d
        elif axis == 'h':
            enc[0, 2] = 1.0
            enc[0, 13] = 1.0  # axis=h
        elif axis == 'w':
            enc[0, 3] = 1.0
            enc[0, 14] = 1.0  # axis=w
        enc[0, 15] = k / 3.0  # normalized k
    elif transform_type == 'flip_d':
        enc[0, 4] = 1.0
    elif transform_type == 'flip_h':
        enc[0, 5] = 1.0
    elif transform_type == 'flip_w':
        enc[0, 6] = 1.0
    elif transform_type == 'translate':
        enc[0, 7] = 1.0
        enc[0, 16] = params['dd']
        enc[0, 17] = params['dh']
        enc[0, 18] = params['dw']
    elif transform_type == 'cutout':
        enc[0, 8] = 1.0
        enc[0, 19] = params['cd']
        enc[0, 20] = params['ch']
        enc[0, 21] = params['cw']
        enc[0, 22] = params['size_d']
        enc[0, 23] = params['size_h']
        # Note: size_w would need dim 24, but we cap at 24 dims


class OmegaTimeEmbed3D(nn.Module):
    """Wrapper around time_embed that adds 3D omega conditioning.

    Replaces the original time_embed module with one that adds
    omega conditioning to the timestep embedding.
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

        # MLP that maps omega encoding to embedding (zero-init)
        self.omega_mlp = create_zero_init_mlp(OMEGA_ENCODING_DIM_3D, embed_dim)

        # Buffer for omega encoding
        self.register_buffer('_omega_encoding', torch.zeros(1, OMEGA_ENCODING_DIM_3D))

    def set_omega_encoding(self, omega_encoding: torch.Tensor):
        """Set omega encoding for next forward pass."""
        self._omega_encoding.copy_(omega_encoding)

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass: original time_embed + omega embedding."""
        out = self.original(t_emb)
        omega_emb = self.omega_mlp(self._omega_encoding)
        return out + omega_emb


class ScoreAugModelWrapper3D(nn.Module):
    """Wrapper to inject omega conditioning into 3D MONAI UNet.

    Replaces time_embed with OmegaTimeEmbed3D for omega conditioning.
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
        model.time_embed = self.omega_time_embed

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        omega: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Forward pass with omega conditioning.

        Args:
            x: Noisy input tensor [B, C, D, H, W]
            timesteps: Timestep tensor [B]
            omega: Optional augmentation parameters

        Returns:
            Model prediction [B, C_out, D, H, W]
        """
        omega_encoding = encode_omega_3d(omega, x.device)
        self.omega_time_embed.set_omega_encoding(omega_encoding)
        return self.model(x, timesteps)

    def parameters(self, recurse: bool = True):
        """Get all parameters including omega MLP."""
        return self.model.parameters(recurse=recurse)
