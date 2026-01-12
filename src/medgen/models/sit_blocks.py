"""
Transformer blocks for SiT (Scalable Interpolant Transformers).

Implements adaLN-Zero conditioned transformer blocks with self-attention,
optional cross-attention, and MLP layers.

Reference: https://arxiv.org/abs/2212.09748 (DiT paper for adaLN-Zero)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# DropPath (Stochastic Depth)
# =============================================================================

def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample.

    Randomly drops entire residual branches during training, forcing
    the network to not rely on any single path.

    Reference: https://arxiv.org/abs/1603.09382 (Deep Networks with Stochastic Depth)

    Args:
        x: Input tensor of any shape.
        drop_prob: Probability of dropping the path (0 = no drop, 1 = always drop).
        training: Whether in training mode.

    Returns:
        Tensor of same shape, with paths randomly zeroed and surviving paths scaled.
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    # Create random tensor: shape (batch_size, 1, 1, ...) for broadcasting
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

    # Scale to maintain expected value: E[x] = keep_prob * (x / keep_prob) = x
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    Wrapper module for drop_path function. Applies stochastic depth
    regularization by randomly dropping residual paths during training.

    Typical usage in transformers:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

    Args:
        drop_prob: Probability of dropping the path. Default: 0.0 (no drop).

    Example:
        >>> drop_path = DropPath(0.1)
        >>> x = torch.randn(4, 196, 768)
        >>> out = drop_path(x)  # During training, randomly drops ~10% of samples
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


# =============================================================================
# Modulation Helper
# =============================================================================

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation.

    Args:
        x: [B, N, D] input tensor
        shift: [B, D] shift parameter
        scale: [B, D] scale parameter

    Returns:
        [B, N, D] modulated tensor: x * (1 + scale) + shift
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """Multi-head self-attention layer.

    Args:
        dim: Input/output dimension.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in QKV projection.
        attn_drop: Dropout rate for attention weights.
        proj_drop: Dropout rate for output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input tokens

        Returns:
            [B, N, D] output tokens
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, heads, N, head_dim]

        # Scaled dot-product attention (uses Flash Attention when available)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """Multi-head cross-attention layer.

    Args:
        dim: Query dimension.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in projections.
        attn_drop: Dropout rate for attention weights.
        proj_drop: Dropout rate for output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] query tokens
            context: [B, M, D] context tokens (keys/values)

        Returns:
            [B, N, D] output tokens
        """
        B, N, C = x.shape
        M = context.shape[1]

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # [B, heads, M, head_dim]

        # Scaled dot-product attention
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP block with GELU activation.

    Args:
        in_features: Input dimension.
        hidden_features: Hidden dimension (default: 4x input).
        out_features: Output dimension (default: same as input).
        drop: Dropout rate.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning.

    Uses adaptive layer normalization to inject timestep information,
    with optional cross-attention for spatial conditioning.

    Args:
        hidden_size: Hidden dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        use_cross_attn: Whether to include cross-attention for conditioning.
        drop: Dropout rate.
        drop_path: Stochastic depth rate. Default: 0.0 (no drop).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_cross_attn: bool = False,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, proj_drop=drop)

        # Cross-attention (optional, for conditioning)
        if use_cross_attn:
            self.norm_cross = nn.LayerNorm(hidden_size, eps=1e-6)
            self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, proj_drop=drop)

        # MLP
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(hidden_size, hidden_features=mlp_hidden, drop=drop)

        # DropPath (Stochastic Depth) for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # adaLN-Zero modulation: 6 parameters (shift/scale/gate for attn and mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input tokens
            c: [B, D] conditioning (timestep embedding)
            context: [B, M, D] optional cross-attention context

        Returns:
            [B, N, D] output tokens
        """
        # Get modulation parameters from conditioning
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Self-attention with adaLN modulation + DropPath
        x = x + self.drop_path(
            gate_msa.unsqueeze(1) * self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa)
            )
        )

        # Cross-attention (required if use_cross_attn=True)
        # Note: No DropPath on cross-attention - conditioning is critical
        if self.use_cross_attn:
            if context is None:
                raise ValueError(
                    "context tensor required when use_cross_attn=True. "
                    "Either pass context or create block with use_cross_attn=False"
                )
            x = x + self.cross_attn(self.norm_cross(x), context)

        # MLP with adaLN modulation + DropPath
        x = x + self.drop_path(
            gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm2(x), shift_mlp, scale_mlp)
            )
        )

        return x


class FinalLayer(nn.Module):
    """Final layer of SiT that projects back to patch space.

    Args:
        hidden_size: Input dimension from transformer.
        patch_size: Patch size for unpatchifying.
        out_channels: Output channels (e.g., 4 for latent, or 2*4 for variance prediction).
        spatial_dims: 2 or 3 for 2D/3D.
    """

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        spatial_dims: int = 2,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.patch_size = patch_size

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Output projection: hidden_size -> patch_size^d * out_channels
        if spatial_dims == 2:
            self.linear = nn.Linear(hidden_size, patch_size ** 2 * out_channels, bias=True)
        else:
            self.linear = nn.Linear(hidden_size, patch_size ** 3 * out_channels, bias=True)

        # adaLN modulation for final layer (just shift and scale, no gate)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] final transformer tokens
            c: [B, D] conditioning (timestep embedding)

        Returns:
            [B, N, patch_size^d * out_channels] ready for unpatchify
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
