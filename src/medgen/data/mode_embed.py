"""Mode embedding for multi-modality diffusion training.

Adds modality conditioning to the diffusion model via time_embed modification.
Follows the same pattern as OmegaTimeEmbed from score_aug.py.

Mode embedding tells the model which modality (bravo, flair, t1_pre, t1_gd)
it is generating, enabling a single model to handle all modalities.
"""

from typing import Optional

import torch
from torch import nn

# Mode ID mapping
MODE_ID_MAP = {
    'bravo': 0,
    'flair': 1,
    't1_pre': 2,
    't1_gd': 3,
}

# Reverse mapping for inference
ID_TO_MODE = {v: k for k, v in MODE_ID_MAP.items()}

# Encoding dimension (one-hot)
MODE_ENCODING_DIM = 4


def encode_mode_id(
    mode_id: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Encode mode_id to one-hot tensor.

    Args:
        mode_id: Integer tensor [B] with mode IDs (0-3)
        device: Device to create tensor on

    Returns:
        One-hot tensor [1, MODE_ENCODING_DIM] - uses single encoding
        that broadcasts to batch in MLP
    """
    if mode_id is None:
        return torch.zeros(1, MODE_ENCODING_DIM, device=device)

    # Take first mode_id (all should be same in a batch for multi-modality)
    # This is fine because each batch contains one modality
    if mode_id.dim() == 0:
        idx = mode_id.item()
    else:
        idx = mode_id[0].item()

    enc = torch.zeros(1, MODE_ENCODING_DIM, device=device)
    if 0 <= idx < MODE_ENCODING_DIM:
        enc[0, idx] = 1.0

    return enc


class ModeTimeEmbed(nn.Module):
    """Wrapper around time_embed that adds mode conditioning.

    This is compile-compatible because:
    - No hooks, just module replacement
    - No data-dependent control flow in forward
    - Always adds mode embedding (zero for no conditioning due to init)
    """

    def __init__(self, original_time_embed: nn.Module, embed_dim: int):
        """Initialize mode-aware time embedding.

        Args:
            original_time_embed: The original time_embed module from DiffusionModelUNet
            embed_dim: Output dimension (should match original time_embed output)
        """
        super().__init__()
        self.original = original_time_embed
        self.embed_dim = embed_dim

        # MLP that maps mode encoding to embedding
        self.mode_mlp = nn.Sequential(
            nn.Linear(MODE_ENCODING_DIM, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Initialize output layer to near-zero so identity starts as no-op
        nn.init.zeros_(self.mode_mlp[-1].weight)
        nn.init.zeros_(self.mode_mlp[-1].bias)

        # Buffer to store current mode encoding
        # This is set by ModeEmbedModelWrapper before each forward
        self.register_buffer('_mode_encoding', torch.zeros(1, MODE_ENCODING_DIM))

    def set_mode_encoding(self, mode_encoding: torch.Tensor):
        """Set mode encoding for next forward pass.

        Uses in-place copy to maintain buffer identity for torch.compile.

        Args:
            mode_encoding: Tensor [1, MODE_ENCODING_DIM]
        """
        self._mode_encoding.copy_(mode_encoding)

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass: original time_embed + mode embedding.

        Args:
            t_emb: Timestep embedding input [B, C]

        Returns:
            Time embedding with mode conditioning [B, embed_dim]
        """
        # Original time embedding
        out = self.original(t_emb)

        # Add mode embedding (always computed, zero for no mode due to init)
        mode_emb = self.mode_mlp(self._mode_encoding)

        return out + mode_emb


class ModeEmbedModelWrapper(nn.Module):
    """Wrapper to inject mode conditioning into MONAI UNet.

    This implementation is compile-compatible:
    - Replaces time_embed with ModeTimeEmbed (no hooks)
    - Sets mode encoding before forward (outside traced graph)
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

        # Replace time_embed with mode-aware version
        if not hasattr(model, 'time_embed'):
            raise ValueError("Model does not have 'time_embed' attribute")

        original_time_embed = model.time_embed
        self.mode_time_embed = ModeTimeEmbed(original_time_embed, embed_dim)

        # Replace the model's time_embed
        model.time_embed = self.mode_time_embed

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        mode_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with mode conditioning.

        Args:
            x: Noisy input tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            mode_id: Optional mode ID tensor [B] (0=bravo, 1=flair, 2=t1_pre, 3=t1_gd)

        Returns:
            Model prediction [B, C_out, H, W]
        """
        # Encode mode_id as (1, 4) - broadcasts to batch in MLP
        mode_encoding = encode_mode_id(mode_id, x.device)
        self.mode_time_embed.set_mode_encoding(mode_encoding)

        # Call model normally
        return self.model(x, timesteps)

    def parameters(self, recurse: bool = True):
        """Get all parameters including mode MLP."""
        return self.model.parameters(recurse=recurse)

    @property
    def parameters_without_mode(self):
        """Get model parameters excluding mode embedding.

        Useful if you want to use different learning rates.
        """
        mode_param_ids = {id(p) for p in self.mode_time_embed.mode_mlp.parameters()}
        return (p for p in self.model.parameters() if id(p) not in mode_param_ids)
