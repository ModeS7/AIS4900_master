"""Base utilities for conditioning embeddings.

Provides shared MLP construction for omega, mode, and combined time embeddings.
Used by score_aug.py, mode_embed.py, and combined_embed.py.
"""

import torch.nn as nn


def create_zero_init_mlp(input_dim: int, embed_dim: int) -> nn.Sequential:
    """Create MLP with zero-initialized output for neutral conditioning start.

    Architecture: Linear → SiLU → Linear (output zero-init)

    Zero-initialization ensures that when the conditioning is not applied
    (encoding is zeros), the MLP output is also zeros, making it a no-op
    when added to the time embedding. This allows gradual learning of
    conditioning effects.

    Args:
        input_dim: Input encoding dimension (e.g., OMEGA_ENCODING_DIM=16, MODE_ENCODING_DIM=4).
        embed_dim: Output dimension (should match model's time_embed output).

    Returns:
        nn.Sequential with Linear→SiLU→Linear architecture, output layer zero-initialized.

    Example:
        >>> omega_mlp = create_zero_init_mlp(OMEGA_ENCODING_DIM, embed_dim)
        >>> mode_mlp = create_zero_init_mlp(MODE_ENCODING_DIM, embed_dim)
    """
    mlp = nn.Sequential(
        nn.Linear(input_dim, embed_dim),
        nn.SiLU(),
        nn.Linear(embed_dim, embed_dim),
    )
    # Zero-init output layer for neutral start
    nn.init.zeros_(mlp[-1].weight)
    nn.init.zeros_(mlp[-1].bias)
    return mlp
