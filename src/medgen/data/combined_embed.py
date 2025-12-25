"""Combined embedding for omega (ScoreAug) + mode conditioning.

When using both ScoreAug and multi-modality training, we need to combine
both omega and mode embeddings into the time_embed.

This module provides a unified wrapper that handles both conditioning signals.
"""

from typing import Any, Dict, Optional

import torch
from torch import nn

from .mode_embed import MODE_ENCODING_DIM, encode_mode_id
from .score_aug import OMEGA_ENCODING_DIM, encode_omega


class CombinedTimeEmbed(nn.Module):
    """Wrapper around time_embed that adds BOTH omega and mode conditioning.

    Combines the logic from OmegaTimeEmbed and ModeTimeEmbed:
    - omega_mlp: maps omega encoding (16-dim) to embed_dim
    - mode_mlp: maps mode encoding (4-dim) to embed_dim
    - Both use zero-init for neutral start
    - Adds both embeddings to time_embed output

    This is compile-compatible because:
    - No hooks, just module replacement
    - No data-dependent control flow in forward
    - Always adds both embeddings (zero when not set due to init)
    """

    def __init__(self, original_time_embed: nn.Module, embed_dim: int):
        """Initialize combined time embedding.

        Args:
            original_time_embed: The original time_embed module from DiffusionModelUNet
            embed_dim: Output dimension (should match original time_embed output)
        """
        super().__init__()
        self.original = original_time_embed
        self.embed_dim = embed_dim

        # Omega MLP (for ScoreAug conditioning)
        self.omega_mlp = nn.Sequential(
            nn.Linear(OMEGA_ENCODING_DIM, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        nn.init.zeros_(self.omega_mlp[-1].weight)
        nn.init.zeros_(self.omega_mlp[-1].bias)

        # Mode MLP (for modality conditioning)
        self.mode_mlp = nn.Sequential(
            nn.Linear(MODE_ENCODING_DIM, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        nn.init.zeros_(self.mode_mlp[-1].weight)
        nn.init.zeros_(self.mode_mlp[-1].bias)

        # Buffers for current encodings
        self.register_buffer('_omega_encoding', torch.zeros(1, OMEGA_ENCODING_DIM))
        self.register_buffer('_mode_encoding', torch.zeros(1, MODE_ENCODING_DIM))

    def set_omega_encoding(self, omega_encoding: torch.Tensor):
        """Set omega encoding for next forward pass."""
        self._omega_encoding.copy_(omega_encoding)

    def set_mode_encoding(self, mode_encoding: torch.Tensor):
        """Set mode encoding for next forward pass."""
        self._mode_encoding.copy_(mode_encoding)

    def set_encodings(
        self,
        omega_encoding: torch.Tensor,
        mode_encoding: torch.Tensor,
    ):
        """Set both encodings for next forward pass."""
        self._omega_encoding.copy_(omega_encoding)
        self._mode_encoding.copy_(mode_encoding)

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass: original time_embed + omega + mode embeddings.

        Args:
            t_emb: Timestep embedding input [B, C]

        Returns:
            Time embedding with combined conditioning [B, embed_dim]
        """
        # Original time embedding
        out = self.original(t_emb)

        # Add omega embedding
        omega_emb = self.omega_mlp(self._omega_encoding)
        out = out + omega_emb

        # Add mode embedding
        mode_emb = self.mode_mlp(self._mode_encoding)
        out = out + mode_emb

        return out


class CombinedModelWrapper(nn.Module):
    """Wrapper to inject both omega and mode conditioning into MONAI UNet.

    Use this when:
    - training.score_aug.enabled=true AND
    - mode.use_mode_embedding=true

    This implementation is compile-compatible:
    - Replaces time_embed with CombinedTimeEmbed (no hooks)
    - Sets encodings before forward (outside traced graph)
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
        self.combined_time_embed = CombinedTimeEmbed(original_time_embed, embed_dim)

        # Replace the model's time_embed
        model.time_embed = self.combined_time_embed

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        omega: Optional[Dict[str, Any]] = None,
        mode_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with combined omega and mode conditioning.

        Args:
            x: Noisy input tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            omega: Optional ScoreAug parameters for conditioning
            mode_id: Optional mode ID tensor [B] (0=bravo, 1=flair, 2=t1_pre, 3=t1_gd)

        Returns:
            Model prediction [B, C_out, H, W]
        """
        # Encode both conditioning signals
        omega_encoding = encode_omega(omega, x.device)
        mode_encoding = encode_mode_id(mode_id, x.device)

        # Set both encodings
        self.combined_time_embed.set_encodings(omega_encoding, mode_encoding)

        # Call model normally
        return self.model(x, timesteps)

    def parameters(self, recurse: bool = True):
        """Get all parameters including embedding MLPs."""
        return self.model.parameters(recurse=recurse)

    @property
    def parameters_without_embeddings(self):
        """Get model parameters excluding omega and mode embeddings.

        Useful if you want to use different learning rates.
        """
        omega_param_ids = {id(p) for p in self.combined_time_embed.omega_mlp.parameters()}
        mode_param_ids = {id(p) for p in self.combined_time_embed.mode_mlp.parameters()}
        exclude_ids = omega_param_ids | mode_param_ids
        return (p for p in self.model.parameters() if id(p) not in exclude_ids)
