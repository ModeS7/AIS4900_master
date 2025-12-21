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
from typing import Tuple, Optional, Dict, Any

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
        translation: bool = False,
        cutout: bool = False,
        brightness: bool = False,
        brightness_range: float = 1.2,
    ):
        """Initialize ScoreAug transform.

        Per paper: one augmentation is randomly selected with equal probability
        across all enabled types, including identity (no transform).

        Args:
            rotation: Enable 90, 180, 270 degree rotations (requires omega conditioning)
            translation: Enable +/-10% translation with zero-padding
            cutout: Enable random square cutout (10-30% of image)
            brightness: Enable brightness scaling (experimental with normalized data)
            brightness_range: Max brightness factor B, scales in [1/B, B]
        """
        self.rotation = rotation
        self.translation = translation
        self.cutout = cutout
        self.brightness = brightness
        self.brightness_range = brightness_range

    def sample_transform(self) -> Tuple[str, Dict[str, Any]]:
        """Sample a random transform with equal probability (per paper).

        Identity is always included as one option. All enabled transforms
        have equal probability of being selected.

        Returns:
            Tuple of (transform_type, params_dict)
        """
        # Build list of all enabled transforms + identity
        transforms = ['identity']
        if self.rotation:
            transforms.append('rotation')  # Will sample k=1,2,3 uniformly
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
        elif transform_type == 'rotation':
            # Sample rotation angle uniformly: 90, 180, 270
            k = random.choice([1, 2, 3])
            return 'rot90', {'k': k}
        elif transform_type == 'translate':
            dx = random.uniform(-0.1, 0.1)
            dy = random.uniform(-0.1, 0.1)
            return 'translate', {'dx': dx, 'dy': dy}
        elif transform_type == 'cutout':
            size = random.uniform(0.1, 0.3)
            cx = random.uniform(size / 2, 1 - size / 2)
            cy = random.uniform(size / 2, 1 - size / 2)
            return 'cutout', {'cx': cx, 'cy': cy, 'size': size}
        elif transform_type == 'brightness':
            # Sample scale factor uniformly in log space for symmetric distribution
            log_scale = random.uniform(-1, 1) * abs(self.brightness_range - 1)
            scale = 1.0 + log_scale
            # Clamp to valid range
            scale = max(1.0 / self.brightness_range, min(self.brightness_range, scale))
            return 'brightness', {'scale': scale}

        return 'identity', {}

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
        elif transform_type == 'translate':
            return self._translate(x, params['dx'], params['dy'])
        elif transform_type == 'cutout':
            return self._cutout(x, params['cx'], params['cy'], params['size'])
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

        Args:
            omega: Transform parameters

        Returns:
            True if omega conditioning is required for this transform
        """
        if omega is None:
            return False
        return omega['type'] == 'rot90'

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

    def _cutout(self, x: torch.Tensor, cx: float, cy: float, size: float) -> torch.Tensor:
        """Apply square cutout centered at (cx, cy) with given size fraction.

        Args:
            x: Input tensor [B, C, H, W]
            cx: Center x as fraction of width (0 to 1)
            cy: Center y as fraction of height (0 to 1)
            size: Size of cutout as fraction of image (0 to 1)

        Returns:
            Tensor with square region zeroed
        """
        B, C, H, W = x.shape

        half_h = int(size * H / 2)
        half_w = int(size * W / 2)
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

# Omega encoding format: [type_onehot(5), rot_k_norm, dx, dy, cx, cy, size, brightness_scale]
# Types: 0=identity, 1=rotation, 2=translation, 3=cutout, 4=brightness
OMEGA_ENCODING_DIM = 12


def encode_omega(
    omega: Optional[Dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    """Encode omega dict into tensor format for MLP.

    All samples in a batch get the same transform, so we return shape (1, 12)
    which broadcasts to (B, 12) in the MLP. This keeps the buffer shape constant
    for torch.compile compatibility.

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

    t = omega['type']
    p = omega['params']

    if t == 'rot90':
        enc[0, 1] = 1.0  # rotation type
        enc[0, 5] = p['k'] / 3.0  # normalized k (1,2,3 -> 0.33, 0.67, 1.0)
    elif t == 'translate':
        enc[0, 2] = 1.0  # translation type
        enc[0, 6] = p['dx']  # already in [-0.1, 0.1]
        enc[0, 7] = p['dy']
    elif t == 'cutout':
        enc[0, 3] = 1.0  # cutout type
        enc[0, 8] = p['cx']  # in [0, 1]
        enc[0, 9] = p['cy']
        enc[0, 10] = p['size']  # in [0.1, 0.3]
    elif t == 'brightness':
        enc[0, 4] = 1.0  # brightness type
        enc[0, 11] = p['scale'] - 1.0  # centered around 0

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


# =============================================================================
# Legacy classes (kept for reference, not used with compile-compatible version)
# =============================================================================

class AugmentationEmbedding(nn.Module):
    """Embed augmentation parameters for model conditioning.

    NOTE: This is the legacy hook-based version. Use OmegaTimeEmbed instead
    for compile-compatible omega conditioning.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation_embed = nn.Embedding(4, embed_dim)
        self.translate_mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.cutout_mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.brightness_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        omega: Optional[Dict[str, Any]],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if omega is None:
            return torch.zeros(batch_size, self.embed_dim, device=device)

        transform_type = omega['type']
        params = omega['params']

        if transform_type == 'rot90':
            rot_idx = params['k']
            idx_tensor = torch.full(
                (batch_size,), rot_idx, dtype=torch.long, device=device
            )
            return self.rotation_embed(idx_tensor)
        elif transform_type == 'translate':
            translate_params = torch.tensor(
                [[params['dx'], params['dy']]],
                dtype=torch.float32,
                device=device,
            ).expand(batch_size, -1)
            return self.translate_mlp(translate_params)
        elif transform_type == 'cutout':
            cutout_params = torch.tensor(
                [[params['cx'], params['cy'], params['size']]],
                dtype=torch.float32,
                device=device,
            ).expand(batch_size, -1)
            return self.cutout_mlp(cutout_params)
        elif transform_type == 'brightness':
            brightness_params = torch.tensor(
                [[params['scale']]],
                dtype=torch.float32,
                device=device,
            ).expand(batch_size, -1)
            return self.brightness_mlp(brightness_params)
        else:
            return torch.zeros(batch_size, self.embed_dim, device=device)
