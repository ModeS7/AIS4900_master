"""Mode embedding for multi-modality diffusion training.

Adds modality conditioning to the diffusion model via time_embed modification.
Follows the same pattern as OmegaTimeEmbed from score_aug.py.

Mode embedding tells the model which modality (bravo, flair, t1_pre, t1_gd)
it is generating, enabling a single model to handle all modalities.
"""

from typing import Optional

import torch
from torch import nn

from .base_embed import create_zero_init_mlp

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
    mode_id: Optional[torch.Tensor],
    device: torch.device,
    batch_size: int = 1,
) -> torch.Tensor:
    """Encode mode_id to one-hot tensor (per-sample encoding).

    Supports mixed modalities within a batch - each sample gets its own
    one-hot encoding based on its mode_id.

    Args:
        mode_id: Optional integer tensor [B] with mode IDs (0-3), or None for zeros.
            Can contain different mode_ids for different samples in the batch.
        device: Device to create tensor on.
        batch_size: Batch size (used when mode_id is None).

    Returns:
        One-hot tensor [B, MODE_ENCODING_DIM] with per-sample encodings.

    Raises:
        ValueError: If mode_id contains invalid values outside [0, MODE_ENCODING_DIM).
    """
    if mode_id is None:
        return torch.zeros(batch_size, MODE_ENCODING_DIM, device=device)

    # Handle scalar mode_id (single sample or uniform batch)
    if mode_id.dim() == 0:
        mode_id = mode_id.unsqueeze(0)

    batch_size = mode_id.shape[0]

    # Validate mode_id range
    if torch.any(mode_id < 0) or torch.any(mode_id >= MODE_ENCODING_DIM):
        invalid = mode_id[(mode_id < 0) | (mode_id >= MODE_ENCODING_DIM)]
        raise ValueError(
            f"Invalid mode_id values: {invalid.tolist()}. "
            f"Expected values in [0, {MODE_ENCODING_DIM - 1}]. "
            f"Valid modes: {list(MODE_ID_MAP.keys())}"
        )

    # Move mode_id to target device (fixes CPU/GPU mismatch)
    mode_id = mode_id.to(device)

    # Create per-sample one-hot encoding [B, MODE_ENCODING_DIM]
    enc = torch.zeros(batch_size, MODE_ENCODING_DIM, device=device)
    enc.scatter_(1, mode_id.unsqueeze(1).long(), 1.0)

    return enc


class ModeTimeEmbed(nn.Module):
    """Wrapper around time_embed that adds per-sample mode conditioning.

    Supports mixed modalities within a batch - each sample can have a different
    mode_id and will receive its own mode embedding.

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

        # MLP that maps mode encoding to embedding (zero-init for neutral start)
        self.mode_mlp = create_zero_init_mlp(MODE_ENCODING_DIM, embed_dim)

        # Store current mode encoding (set before each forward)
        # Using None instead of buffer to support variable batch sizes
        self._mode_encoding: Optional[torch.Tensor] = None

    def set_mode_encoding(self, mode_encoding: torch.Tensor):
        """Set mode encoding for next forward pass.

        Args:
            mode_encoding: Tensor [B, MODE_ENCODING_DIM] with per-sample encodings

        Raises:
            ValueError: If mode_encoding has wrong shape
        """
        if mode_encoding.dim() != 2:
            raise ValueError(
                f"mode_encoding must be 2D [B, {MODE_ENCODING_DIM}], "
                f"got {mode_encoding.dim()}D with shape {mode_encoding.shape}"
            )
        if mode_encoding.shape[1] != MODE_ENCODING_DIM:
            raise ValueError(
                f"mode_encoding.shape[1] must be {MODE_ENCODING_DIM}, "
                f"got {mode_encoding.shape[1]}"
            )
        self._mode_encoding = mode_encoding

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass: original time_embed + per-sample mode embedding.

        Args:
            t_emb: Timestep embedding input [B, C]

        Returns:
            Time embedding with mode conditioning [B, embed_dim]
        """
        # Original time embedding
        out = self.original(t_emb)

        # Add per-sample mode embedding
        if self._mode_encoding is not None:
            mode_emb = self.mode_mlp(self._mode_encoding)  # [B, embed_dim]
            out = out + mode_emb

        return out


class ModeEmbedModelWrapper(nn.Module):
    """Wrapper to inject per-sample mode conditioning into MONAI UNet.

    Supports mixed modalities within a batch - each sample can have a different
    mode_id and will receive its own mode embedding.

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

        # Ensure mode_time_embed is on same device as model
        try:
            device = next(model.parameters()).device
            self.mode_time_embed = self.mode_time_embed.to(device)
            model.time_embed = self.mode_time_embed
        except StopIteration:
            pass  # Model has no parameters, keep on CPU

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        mode_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with per-sample mode conditioning.

        Args:
            x: Noisy input tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            mode_id: Optional mode ID tensor [B] (0=bravo, 1=flair, 2=t1_pre, 3=t1_gd).
                Can contain different mode_ids for different samples.

        Returns:
            Model prediction [B, C_out, H, W]
        """
        # Encode mode_id as [B, 4] - per-sample encoding
        batch_size = x.shape[0]
        mode_encoding = encode_mode_id(mode_id, x.device, batch_size=batch_size)
        self.mode_time_embed.set_mode_encoding(mode_encoding)

        # Call model normally
        return self.model(x, timesteps)

    def parameters(self, recurse: bool = True):
        """Get all parameters including mode MLP."""
        return self.model.parameters(recurse=recurse)
