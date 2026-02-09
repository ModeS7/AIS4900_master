"""Base utilities for conditioning embeddings.

Provides shared MLP construction for omega, mode, and combined time embeddings.
Used by score_aug.py, mode_embed.py, and combined_embed.py.
"""

import torch
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


def create_deep_zero_init_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 512,
    num_hidden_layers: int = 2,
) -> nn.Sequential:
    """Create deeper MLP with zero-initialized output for stronger conditioning.

    Architecture: Linear(in,hidden)→SiLU→[Linear(hidden,hidden)→SiLU]×(n-1)→Linear(hidden,out,zero-init)

    Zero-initialization on the last layer ensures identity at start, same principle
    as create_zero_init_mlp but with configurable depth and width.

    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.
        hidden_dim: Hidden layer width (default: 512).
        num_hidden_layers: Number of hidden layers before output (default: 2).
            Total layers = num_hidden_layers + 1 (output).

    Returns:
        nn.Sequential with deeper MLP, output layer zero-initialized.
    """
    if num_hidden_layers < 1:
        raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")

    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
    ]
    for _ in range(num_hidden_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.SiLU())
    layers.append(nn.Linear(hidden_dim, output_dim))

    mlp = nn.Sequential(*layers)
    # Zero-init output layer for neutral start
    nn.init.zeros_(mlp[-1].weight)
    nn.init.zeros_(mlp[-1].bias)
    return mlp


def create_film_mlp(input_dim: int, embed_dim: int) -> nn.Sequential:
    """Create MLP for FiLM conditioning (outputs gamma and beta).

    Architecture: Linear → SiLU → Linear (outputs 2*embed_dim for gamma, beta)

    Initialization:
    - gamma initialized to 1 (identity scaling)
    - beta initialized to 0 (no shift)
    This ensures FiLM starts as identity transform: gamma*x + beta = 1*x + 0 = x

    Args:
        input_dim: Input encoding dimension (e.g., MODE_ENCODING_DIM=4).
        embed_dim: Feature dimension to modulate.

    Returns:
        nn.Sequential outputting [gamma, beta] concatenated (2*embed_dim).
    """
    mlp = nn.Sequential(
        nn.Linear(input_dim, embed_dim),
        nn.SiLU(),
        nn.Linear(embed_dim, 2 * embed_dim),  # gamma and beta
    )
    # Initialize gamma=1, beta=0 for identity transform at start
    # Output is [gamma, beta] concatenated
    nn.init.zeros_(mlp[-1].weight)
    # Bias: first half (gamma) = 1, second half (beta) = 0
    with torch.no_grad():
        mlp[-1].bias[:embed_dim] = 1.0  # gamma = 1
        mlp[-1].bias[embed_dim:] = 0.0  # beta = 0
    return mlp
