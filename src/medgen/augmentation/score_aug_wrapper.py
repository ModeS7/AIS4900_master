"""ScoreAug model wrapper and omega time embedding.

Contains nn.Module classes that inject omega conditioning into
MONAI UNet models via time embedding replacement.

Moved from score_aug.py during file split.
"""

from typing import Any

import torch
import torch.nn as nn

from medgen.models.wrappers import create_zero_init_mlp

from .score_aug_omega import OMEGA_ENCODING_DIM, encode_omega


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
# 3D Backwards Compatibility Aliases
# =============================================================================

# OmegaTimeEmbed doesn't depend on spatial_dims, so simple alias is fine
OmegaTimeEmbed3D = OmegaTimeEmbed


class ScoreAugModelWrapper3D(ScoreAugModelWrapper):
    """3D ScoreAug model wrapper (backwards compatibility wrapper).

    Equivalent to ScoreAugModelWrapper(..., spatial_dims=3).
    """

    def __init__(self, model: nn.Module, embed_dim: int = 256, **kwargs):
        # Force spatial_dims=3 for 3D compatibility
        super().__init__(model, embed_dim, spatial_dims=3, **kwargs)
