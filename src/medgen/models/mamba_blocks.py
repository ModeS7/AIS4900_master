"""
Building blocks for LaMamba-Diff: SS2D (multi-directional Mamba) and windowed attention.

SS2D processes spatial features via selective state space models in multiple scan
directions (4 for 2D, 6 for 3D), capturing global context with linear complexity.
WindowAttention provides local spatial detail via Swin-style windowed self-attention.

Reference: LaMamba-Diff (Fu et al., 2024) — https://arxiv.org/abs/2408.02615
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Try to import mamba_ssm CUDA kernels; fall back to pure PyTorch
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_CUDA_AVAILABLE = True
except ImportError:
    MAMBA_CUDA_AVAILABLE = False
    logger.warning(
        "mamba_ssm not installed — SS2D will use slow PyTorch fallback. "
        "Install with: pip install mamba-ssm"
    )


# =============================================================================
# Pure-PyTorch selective scan fallback (slow, for testing only)
# =============================================================================

def _selective_scan_ref(
    u: torch.Tensor,       # [B, D, L]
    delta: torch.Tensor,   # [B, D, L]
    A: torch.Tensor,       # [D, N]
    B: torch.Tensor,       # [B, N, L]
    C: torch.Tensor,       # [B, N, L]
    D: torch.Tensor | None = None,  # [D]
    delta_bias: torch.Tensor | None = None,  # [D]
    delta_softplus: bool = True,
) -> torch.Tensor:         # [B, D, L]
    """Reference implementation of selective scan (Mamba S6).

    This is O(BDNL) and intended only for testing without CUDA kernels.
    The mamba_ssm CUDA kernel is O(BDL) with N fused into the scan.
    """
    batch, dim, seq_len = u.shape
    n_state = A.shape[1]

    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
    if delta_softplus:
        delta = F.softplus(delta)

    # Discretize: A_bar = exp(delta * A), B_bar = delta * B
    # A: [D, N] → [B, D, N, L]
    deltaA = torch.exp(delta.unsqueeze(2) * A.unsqueeze(0).unsqueeze(-1))  # [B, D, N, L]
    deltaB = delta.unsqueeze(2) * B.unsqueeze(1)  # [B, D, N, L]

    # Sequential scan
    x = torch.zeros(batch, dim, n_state, device=u.device, dtype=u.dtype)
    ys = []
    for i in range(seq_len):
        x = deltaA[:, :, :, i] * x + deltaB[:, :, :, i] * u[:, :, i:i+1]
        y = (x * C[:, :, i].unsqueeze(1)).sum(dim=2)  # [B, D]
        ys.append(y)
    y = torch.stack(ys, dim=-1)  # [B, D, L]

    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(-1)
    return y


# =============================================================================
# SS2D: Multi-directional Selective State Space
# =============================================================================

class SS2D(nn.Module):
    """Multi-directional Selective State Space module for 2D/3D spatial data.

    Processes spatial features via Mamba's selective scan in multiple directions:
    - 2D: 4 directions (row-fwd, row-rev, col-fwd, col-rev)
    - 3D: 6 directions (±D, ±H, ±W axis scans)

    Each direction independently scans the flattened spatial sequence through
    the SSM, then results are summed. This captures global context from all
    spatial orientations with linear O(N) complexity per direction.

    Args:
        d_model: Input/output feature dimension.
        d_state: SSM state dimension (1 is sufficient per LaMamba-Diff).
        ssm_ratio: Expansion ratio for inner dimension (d_inner = ssm_ratio * d_model).
        dt_rank: Rank of delta projection ('auto' = ceil(d_model/16)).
        spatial_dims: 2 or 3.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 1,
        ssm_ratio: float = 2.0,
        dt_rank: int | str = 'auto',
        spatial_dims: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == 'auto' else dt_rank
        self.spatial_dims = spatial_dims
        self.K = 6 if spatial_dims == 3 else 4  # number of scan directions

        # Input projection: d_model → 2*d_inner (x and gate z)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Depthwise convolution on x (before SSM)
        conv_cls = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.conv = conv_cls(
            self.d_inner, self.d_inner, kernel_size=3, padding=1,
            groups=self.d_inner, bias=True,
        )

        self.act = nn.SiLU()

        # SSM parameters
        # x → (dt, B, C) projection: shared across all directions
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * d_state, bias=False,
        )

        # dt projection: dt_rank → d_inner, per-direction
        self.dt_projs = nn.Linear(self.dt_rank, self.K * self.d_inner, bias=False)

        # SSM matrices A, D (per-direction)
        # A initialized as negative range (like Mamba: A = -arange)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.K * self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A.clone()))  # [K*D_inner, N]
        self.D = nn.Parameter(torch.ones(self.K * self.d_inner))  # [K*D_inner]

        # dt bias
        self.dt_bias = nn.Parameter(torch.zeros(self.K * self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # LayerNorm before output (stabilizes training)
        self.out_norm = nn.LayerNorm(self.d_inner)

    def _cross_scan(self, x: torch.Tensor) -> torch.Tensor:
        """Create K directional scans from spatial tensor.

        Args:
            x: [B, C, *spatial] tensor (C=d_inner).

        Returns:
            [B, K, C, L] where L = prod(spatial) and K = 4 (2D) or 6 (3D).
        """
        B, C = x.shape[:2]
        spatial = x.shape[2:]
        L = 1
        for s in spatial:
            L *= s

        if self.spatial_dims == 2:
            # 4 directions: row-fwd, row-rev, col-fwd, col-rev
            x_flat = x.reshape(B, C, L)
            x_t = x.permute(0, 1, 3, 2).reshape(B, C, L)  # transpose H,W

            return torch.stack([
                x_flat,
                x_flat.flip(-1),
                x_t,
                x_t.flip(-1),
            ], dim=1)  # [B, 4, C, L]
        else:
            # 6 directions: ±D, ±H, ±W (scan along each axis)
            # D-axis: flatten as-is [D, H, W]
            x_d = x.reshape(B, C, L)
            # H-axis: permute to [H, D, W] then flatten
            x_h = x.permute(0, 1, 3, 2, 4).reshape(B, C, L)
            # W-axis: permute to [W, D, H] then flatten
            x_w = x.permute(0, 1, 4, 2, 3).reshape(B, C, L)

            return torch.stack([
                x_d,
                x_d.flip(-1),
                x_h,
                x_h.flip(-1),
                x_w,
                x_w.flip(-1),
            ], dim=1)  # [B, 6, C, L]

    def _cross_merge(self, ys: torch.Tensor, spatial_shape: tuple[int, ...]) -> torch.Tensor:
        """Merge K directional scan outputs back to spatial layout, then sum.

        Each direction's scan output lives in a different flat order (e.g. col-major
        vs row-major for 2D; (H,D,W) or (W,D,H) vs (D,H,W) for 3D). Before summing,
        each must be reshaped + permuted back to the canonical flat layout so that
        position i of every output corresponds to the same spatial position.

        Args:
            ys: [B, K, C, L] scan outputs.
            spatial_shape: original spatial dims (H, W) for 2D or (D, H, W) for 3D.

        Returns:
            [B, C, L] merged output with all directions aligned to canonical order.
        """
        B = ys.shape[0]
        C = ys.shape[2]
        L = ys.shape[3]

        if self.spatial_dims == 2:
            H, W = spatial_shape
            y0 = ys[:, 0]                                     # row-fwd: canonical [H, W]
            y1 = ys[:, 1].flip(-1)                            # row-rev: unflip
            # col-fwd: scanned [W, H] flat; reshape + transpose back to [H, W]
            y2 = ys[:, 2].reshape(B, C, W, H).permute(0, 1, 3, 2).reshape(B, C, L)
            y3 = ys[:, 3].flip(-1).reshape(B, C, W, H).permute(0, 1, 3, 2).reshape(B, C, L)
            return y0 + y1 + y2 + y3
        else:
            D, H, W = spatial_shape
            y_d_fwd = ys[:, 0]                                # canonical [D, H, W]
            y_d_rev = ys[:, 1].flip(-1)
            # H-axis scanned [H, D, W]; reshape + swap H↔D to get [D, H, W]
            y_h_fwd = ys[:, 2].reshape(B, C, H, D, W).permute(0, 1, 3, 2, 4).reshape(B, C, L)
            y_h_rev = ys[:, 3].flip(-1).reshape(B, C, H, D, W).permute(0, 1, 3, 2, 4).reshape(B, C, L)
            # W-axis scanned [W, D, H]; permute (W,D,H)→(D,H,W) = dims (3,4,2) over B,C
            y_w_fwd = ys[:, 4].reshape(B, C, W, D, H).permute(0, 1, 3, 4, 2).reshape(B, C, L)
            y_w_rev = ys[:, 5].flip(-1).reshape(B, C, W, D, H).permute(0, 1, 3, 4, 2).reshape(B, C, L)
            return y_d_fwd + y_d_rev + y_h_fwd + y_h_rev + y_w_fwd + y_w_rev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, *spatial, C] input (channels-last).

        Returns:
            [B, *spatial, C] output (channels-last).
        """
        spatial_shape = x.shape[1:-1]
        B = x.shape[0]
        L = 1
        for s in spatial_shape:
            L *= s

        # Project and split into x_branch and gate z
        xz = self.in_proj(x)  # [B, *spatial, 2*d_inner]
        x_branch, z = xz.chunk(2, dim=-1)  # each [B, *spatial, d_inner]

        # Rearrange to channels-first for conv
        if self.spatial_dims == 2:
            x_branch = x_branch.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        else:
            x_branch = x_branch.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, D, H, W]

        # Depthwise conv + activation
        x_branch = self.act(self.conv(x_branch))

        # Multi-directional scan
        xs = self._cross_scan(x_branch)  # [B, K, C, L]

        # Reshape for grouped SSM computation
        BK = B * self.K
        xs_flat = xs.reshape(BK, self.d_inner, L)  # [B*K, C, L]

        # Project x to (dt, B, C) for SSM — via channels-last
        x_dbl = self.x_proj(xs_flat.transpose(1, 2))  # [B*K, L, dt_rank + 2*N]
        x_dbl = x_dbl.transpose(1, 2)  # [B*K, dt_rank + 2*N, L]

        dt = x_dbl[:, :self.dt_rank]  # [B*K, dt_rank, L]
        B_ssm = x_dbl[:, self.dt_rank:self.dt_rank + self.d_state]  # [B*K, N, L]
        C_ssm = x_dbl[:, self.dt_rank + self.d_state:]  # [B*K, N, L]

        # Project dt to d_inner
        # Reshape so each direction's dt_proj is applied correctly
        dt = dt.reshape(B, self.K, self.dt_rank, L)
        dt = dt.reshape(B * self.K, self.dt_rank, L)
        dt_weight = self.dt_projs.weight.reshape(self.K, self.d_inner, self.dt_rank)
        dt_out = torch.zeros(B, self.K, self.d_inner, L, device=x.device, dtype=x.dtype)
        for k in range(self.K):
            dt_out[:, k] = torch.einsum('dr,brl->bdl', dt_weight[k], dt[B*k:B*(k+1)])
        dt = dt_out.reshape(BK, self.d_inner, L)

        # SSM parameters — per-direction slices
        A = -torch.exp(self.A_log.float())  # [K*D_inner, N]
        D_param = self.D.float()  # [K*D_inner]
        dt_bias = self.dt_bias.float()  # [K*D_inner]

        # Run selective scan per direction (avoids shape mismatch with grouped params)
        A_per_k = A.reshape(self.K, self.d_inner, self.d_state)
        D_per_k = D_param.reshape(self.K, self.d_inner)
        dt_bias_per_k = dt_bias.reshape(self.K, self.d_inner)

        # xs_flat: [B*K, D_inner, L], dt: [B*K, D_inner, L]
        # B_ssm: [B*K, N, L], C_ssm: [B*K, N, L]
        ys_list = []
        scan_fn = selective_scan_fn if MAMBA_CUDA_AVAILABLE else _selective_scan_ref
        for k in range(self.K):
            y_k = scan_fn(
                xs_flat[B*k:B*(k+1)].float(),
                dt[B*k:B*(k+1)].float(),
                A_per_k[k],          # [D_inner, N]
                B_ssm[B*k:B*(k+1)].float(),
                C_ssm[B*k:B*(k+1)].float(),
                D=D_per_k[k],        # [D_inner]
                delta_bias=dt_bias_per_k[k],  # [D_inner]
                delta_softplus=True,
            )
            ys_list.append(y_k)
        ys = torch.stack(ys_list, dim=1).to(x.dtype)  # [B, K, D_inner, L]

        # Merge directions: ys is already [B, K, D_inner, L] from stack
        y = self._cross_merge(ys, tuple(spatial_shape))  # [B, D_inner, L]

        # Gate and project
        y = y.transpose(1, 2)  # [B, L, C]
        z = self.act(z.reshape(B, L, self.d_inner))
        y = self.out_norm(y * z)
        y = self.out_proj(y)  # [B, L, d_model]

        # Reshape back to spatial
        return y.reshape(B, *spatial_shape, -1)


# =============================================================================
# Window Attention (2D/3D unified)
# =============================================================================

def _to_windows(
    x: torch.Tensor,
    window_size: int,
    spatial_dims: int,
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Partition spatial tensor into non-overlapping windows with padding.

    Args:
        x: [B, *spatial, C] input.
        window_size: Window size (uniform across all spatial dims).
        spatial_dims: 2 or 3.

    Returns:
        (windows, padded_shape, pad_amounts) where:
        - windows: [num_windows * B, window_size^spatial_dims, C]
        - padded_shape: spatial dims after padding
        - pad_amounts: amount padded per dim (for unpadding later)
    """
    spatial = list(x.shape[1:-1])
    C = x.shape[-1]
    B = x.shape[0]
    w = window_size

    # Pad each spatial dim to multiple of window_size
    pads = []
    padded = []
    for s in spatial:
        p = (w - s % w) % w
        pads.append(p)
        padded.append(s + p)

    if any(p > 0 for p in pads):
        # F.pad expects (last_dim_pad, ..., first_dim_pad) in reverse
        pad_args = []
        for p in reversed(pads):
            pad_args.extend([0, p])
        pad_args = [0, 0] + pad_args  # don't pad C dim
        x = F.pad(x, pad_args)

    if spatial_dims == 2:
        Hp, Wp = padded
        nH, nW = Hp // w, Wp // w
        x = x.reshape(B, nH, w, nW, w, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * nH * nW, w * w, C)
    else:
        Dp, Hp, Wp = padded
        nD, nH, nW = Dp // w, Hp // w, Wp // w
        x = x.reshape(B, nD, w, nH, w, nW, w, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(B * nD * nH * nW, w * w * w, C)

    return x, padded, pads


def _from_windows(
    windows: torch.Tensor,
    window_size: int,
    padded_shape: list[int],
    original_shape: list[int],
    spatial_dims: int,
    batch_size: int,
) -> torch.Tensor:
    """Reverse window partition and remove padding.

    Args:
        windows: [num_windows * B, window_size^d, C]
        window_size: Window size.
        padded_shape: Spatial dims after padding.
        original_shape: Original spatial dims (before padding).
        spatial_dims: 2 or 3.
        batch_size: Original batch size B.

    Returns:
        [B, *original_shape, C] output.
    """
    w = window_size
    C = windows.shape[-1]
    B = batch_size

    if spatial_dims == 2:
        Hp, Wp = padded_shape
        nH, nW = Hp // w, Wp // w
        x = windows.reshape(B, nH, nW, w, w, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        H, W = original_shape
        x = x[:, :H, :W, :].contiguous()
    else:
        Dp, Hp, Wp = padded_shape
        nD, nH, nW = Dp // w, Hp // w, Wp // w
        x = windows.reshape(B, nD, nH, nW, w, w, w, C)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, Dp, Hp, Wp, C)
        D, H, W = original_shape
        x = x[:, :D, :H, :W, :].contiguous()

    return x


class WindowAttention(nn.Module):
    """Swin-style windowed multi-head self-attention for 2D/3D.

    Partitions the spatial dimensions into non-overlapping windows, applies
    self-attention within each window, and reassembles. Supports shifted
    windows for cross-window information flow (alternating blocks).

    Args:
        dim: Input feature dimension.
        num_heads: Number of attention heads.
        window_size: Window size (uniform across spatial dims).
        shift_size: Shift for shifted-window attention (0 = no shift).
        spatial_dims: 2 or 3.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        spatial_dims: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.spatial_dims = spatial_dims
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        # Relative position bias
        if spatial_dims == 2:
            num_relative = (2 * window_size - 1) ** 2
        else:
            num_relative = (2 * window_size - 1) ** 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index
        self._register_relative_position_index()

    def _register_relative_position_index(self):
        """Precompute relative position index for the window."""
        w = self.window_size
        if self.spatial_dims == 2:
            coords_h = torch.arange(w)
            coords_w = torch.arange(w)
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # [2, w, w]
            coords = coords.reshape(2, -1)  # [2, w*w]
            relative = coords[:, :, None] - coords[:, None, :]  # [2, N, N]
            relative = relative.permute(1, 2, 0).contiguous()  # [N, N, 2]
            relative[:, :, 0] += w - 1
            relative[:, :, 1] += w - 1
            relative[:, :, 0] *= 2 * w - 1
            index = relative.sum(-1)  # [N, N]
        else:
            coords_d = torch.arange(w)
            coords_h = torch.arange(w)
            coords_w = torch.arange(w)
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))  # [3, w, w, w]
            coords = coords.reshape(3, -1)  # [3, w^3]
            relative = coords[:, :, None] - coords[:, None, :]  # [3, N, N]
            relative = relative.permute(1, 2, 0).contiguous()  # [N, N, 3]
            relative[:, :, 0] += w - 1
            relative[:, :, 1] += w - 1
            relative[:, :, 2] += w - 1
            relative[:, :, 0] *= (2 * w - 1) ** 2
            relative[:, :, 1] *= (2 * w - 1)
            index = relative.sum(-1)  # [N, N]

        self.register_buffer('relative_position_index', index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, *spatial, C] input (channels-last).

        Returns:
            [B, *spatial, C] output (channels-last).
        """
        B = x.shape[0]
        spatial_shape = list(x.shape[1:-1])
        w = self.window_size

        # Apply cyclic shift if needed
        if self.shift_size > 0:
            shift = [-self.shift_size] * self.spatial_dims
            x = torch.roll(x, shifts=shift, dims=list(range(1, 1 + self.spatial_dims)))

        # Partition into windows
        windows, padded_shape, pads = _to_windows(x, w, self.spatial_dims)
        # windows: [num_win * B, win_tokens, C]

        num_tokens = windows.shape[1]

        # QKV
        qkv = self.qkv(windows).reshape(-1, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, nW*B, heads, tokens, head_dim]
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [nW*B, heads, tokens, tokens]

        # Add relative position bias
        N = num_tokens
        bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)]
        bias = bias.reshape(N, N, -1).permute(2, 0, 1)  # [heads, N, N]
        attn = attn + bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(-1, num_tokens, self.dim)

        out = self.proj(out)

        # Reverse windows
        out = _from_windows(out, w, padded_shape, spatial_shape, self.spatial_dims, B)

        # Reverse cyclic shift
        if self.shift_size > 0:
            shift = [self.shift_size] * self.spatial_dims
            out = torch.roll(out, shifts=shift, dims=list(range(1, 1 + self.spatial_dims)))

        return out
