"""Size bin embedding for conditioned segmentation mask generation.

Adds tumor size distribution conditioning to the diffusion model via FiLM
(Feature-wise Linear Modulation) on the time embedding. This provides stronger
conditioning than simple additive embedding while still affecting all UNet layers.

Size bins encode the tumor count per size category:
    - 7 bins: 0-3mm, 3-6mm, 6-10mm, 10-15mm, 15-20mm, 20-30mm, 30+mm
    - Each bin contains a count (0 to max_count_per_bin)

FiLM modulation: out = time_embed * (1 + scale) + shift
    - Zero-initialized projection ensures identity at start (scale=0, shift=0)
    - Conditioning strength grows during training
    - Same principle as adaLN-Zero in DiT
"""

from typing import List, Optional

import torch
from torch import nn

from .base_embed import create_zero_init_mlp

# Default bin configuration (aligned with RANO-BM thresholds)
# 7 bins: 6 bounded + 1 overflow (30+mm)
DEFAULT_BIN_EDGES = [0, 3, 6, 10, 15, 20, 30]  # mm boundaries
DEFAULT_NUM_BINS = 7  # 6 bounded bins + 1 overflow bin (30+)
DEFAULT_MAX_COUNT = 10  # Max tumors per bin for embedding vocab
DEFAULT_EMBEDDING_DIM = 32  # Embedding dim per bin


def encode_size_bins(
    size_bins: Optional[torch.Tensor],
    device: torch.device,
    batch_size: int = 1,
    num_bins: int = DEFAULT_NUM_BINS,
) -> torch.Tensor:
    """Prepare size bins tensor for embedding.

    Args:
        size_bins: Optional integer tensor [B, num_bins] with counts per bin, or None.
        device: Device to create tensor on.
        batch_size: Batch size (used when size_bins is None).
        num_bins: Number of size bins.

    Returns:
        Integer tensor [B, num_bins] ready for embedding lookup.
    """
    if size_bins is None:
        return torch.zeros(batch_size, num_bins, dtype=torch.long, device=device)

    # Ensure long type for embedding lookup
    size_bins = size_bins.long().to(device)

    return size_bins


class SizeBinTimeEmbed(nn.Module):
    """Wrapper around time_embed that applies FiLM conditioning from size bins.

    Uses FiLM (Feature-wise Linear Modulation) to condition the time embedding:
        out = time_embed * (1 + scale) + shift

    Each bin's count is embedded separately, then concatenated and projected
    to produce scale and shift parameters for FiLM modulation.

    Example:
        size_bins = [0, 0, 2, 0, 0, 0, 1]  # 2 tumors in bin 2, 1 in bin 6
        - bin_embeds[2](2) -> vector for "2 tumors of size 6-10mm"
        - bin_embeds[6](1) -> vector for "1 tumor of size 20-30mm"
        - Concatenate → project → [scale, shift] → FiLM modulation

    This is compile-compatible because:
    - No hooks, just module replacement
    - No data-dependent control flow in forward
    - Zero-init ensures identity at start (scale=0, shift=0)
    """

    def __init__(
        self,
        original_time_embed: nn.Module,
        embed_dim: int,
        num_bins: int = DEFAULT_NUM_BINS,
        max_count: int = DEFAULT_MAX_COUNT,
        per_bin_embed_dim: int = DEFAULT_EMBEDDING_DIM,
    ):
        """Initialize size bin conditioned time embedding.

        Args:
            original_time_embed: The original time_embed module from DiffusionModelUNet
            embed_dim: Output dimension (should match original time_embed output)
            num_bins: Number of size bins (default: 7)
            max_count: Maximum count per bin for embedding vocabulary (default: 10)
            per_bin_embed_dim: Embedding dimension per bin before projection
        """
        super().__init__()
        self.original = original_time_embed
        self.embed_dim = embed_dim
        self.num_bins = num_bins
        self.max_count = max_count

        # One embedding layer per bin, each maps count 0-max_count → per_bin_embed_dim
        # +1 for vocabulary since we need 0, 1, 2, ..., max_count
        self.bin_embeddings = nn.ModuleList([
            nn.Embedding(max_count + 1, per_bin_embed_dim)
            for _ in range(num_bins)
        ])

        # Project combined embeddings to FiLM parameters (scale and shift)
        # Input: num_bins * per_bin_embed_dim (concatenated)
        # Output: embed_dim * 2 (scale and shift for FiLM modulation)
        combined_dim = num_bins * per_bin_embed_dim
        self.projection = create_zero_init_mlp(combined_dim, embed_dim * 2)

        # Store current size bins (set before each forward)
        self._size_bins: Optional[torch.Tensor] = None

    def set_size_bins(self, size_bins: torch.Tensor):
        """Set size bins for next forward pass.

        Args:
            size_bins: Integer tensor [B, num_bins] with counts per bin.
                Values should be in [0, max_count].

        Raises:
            ValueError: If size_bins has wrong shape or invalid values.
        """
        if size_bins.dim() != 2:
            raise ValueError(
                f"size_bins must be 2D [B, {self.num_bins}], "
                f"got {size_bins.dim()}D with shape {size_bins.shape}"
            )
        if size_bins.shape[1] != self.num_bins:
            raise ValueError(
                f"size_bins.shape[1] must be {self.num_bins}, "
                f"got {size_bins.shape[1]}"
            )

        # Clamp to valid range (silently handle outliers)
        size_bins = size_bins.clamp(0, self.max_count)

        self._size_bins = size_bins

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass: original time_embed with FiLM conditioning from size bins.

        Args:
            t_emb: Timestep embedding input [B, C]

        Returns:
            Time embedding with size bin conditioning [B, embed_dim]
        """
        # Original time embedding
        out = self.original(t_emb)

        # Add size bin embedding if set
        if self._size_bins is not None:
            # Embed each bin separately and concatenate
            bin_embeds = []
            for i, embed_layer in enumerate(self.bin_embeddings):
                # size_bins[:, i] is [B] with counts for bin i
                count = self._size_bins[:, i].long()  # [B]
                bin_emb = embed_layer(count)  # [B, per_bin_embed_dim]
                bin_embeds.append(bin_emb)

            # Concatenate all bin embeddings
            combined = torch.cat(bin_embeds, dim=1)  # [B, num_bins * per_bin_embed_dim]

            # FiLM: Feature-wise Linear Modulation
            # Project to 2*embed_dim and split into scale and shift
            film_params = self.projection(combined)  # [B, embed_dim * 2]
            scale, shift = film_params.chunk(2, dim=-1)  # Each [B, embed_dim]
            out = out * (1 + scale) + shift

        return out


class SizeBinModelWrapper(nn.Module):
    """Wrapper to inject size bin conditioning into MONAI UNet.

    Use this when:
    - mode.name = seg_conditioned
    - Training diffusion to generate masks conditioned on tumor size distribution

    This implementation is compile-compatible:
    - Replaces time_embed with SizeBinTimeEmbed (no hooks)
    - Sets size_bins before forward (outside traced graph)
    - Forward has fixed control flow
    """

    def __init__(
        self,
        model: nn.Module,
        embed_dim: int = 256,
        num_bins: int = DEFAULT_NUM_BINS,
        max_count: int = DEFAULT_MAX_COUNT,
        per_bin_embed_dim: int = DEFAULT_EMBEDDING_DIM,
    ):
        """Initialize wrapper.

        Args:
            model: MONAI DiffusionModelUNet to wrap
            embed_dim: Embedding dimension (should match model's time_embed output)
            num_bins: Number of size bins
            max_count: Maximum count per bin
            per_bin_embed_dim: Embedding dimension per bin
        """
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim
        self.num_bins = num_bins

        if not hasattr(model, 'time_embed'):
            raise ValueError("Model does not have 'time_embed' attribute")

        original_time_embed = model.time_embed
        self.size_bin_time_embed = SizeBinTimeEmbed(
            original_time_embed,
            embed_dim,
            num_bins=num_bins,
            max_count=max_count,
            per_bin_embed_dim=per_bin_embed_dim,
        )

        # Replace the model's time_embed
        model.time_embed = self.size_bin_time_embed

        # Ensure size_bin_time_embed is on same device as model
        try:
            device = next(model.parameters()).device
            self.size_bin_time_embed = self.size_bin_time_embed.to(device)
            model.time_embed = self.size_bin_time_embed
        except StopIteration:
            pass  # Model has no parameters, keep on CPU

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        size_bins: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with size bin conditioning.

        Args:
            x: Noisy input tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            size_bins: Optional size bin tensor [B, num_bins] with counts per bin.

        Returns:
            Model prediction [B, C_out, H, W]
        """
        batch_size = x.shape[0]

        # Prepare size bins
        if size_bins is not None:
            size_bins = size_bins.to(x.device)
            self.size_bin_time_embed.set_size_bins(size_bins)
        else:
            # Create zeros for no conditioning
            zeros = torch.zeros(batch_size, self.num_bins, dtype=torch.long, device=x.device)
            self.size_bin_time_embed.set_size_bins(zeros)

        # Call model normally
        return self.model(x, timesteps)

    def parameters(self, recurse: bool = True):
        """Get all parameters including embedding layers."""
        return self.model.parameters(recurse=recurse)


def get_bin_label(bin_idx: int, bin_edges: List[float] = None, num_bins: int = None) -> str:
    """Get human-readable label for a bin index.

    Args:
        bin_idx: Bin index (0 to num_bins-1)
        bin_edges: List of bin edges in mm. Default: DEFAULT_BIN_EDGES
        num_bins: Total number of bins. If > len(edges)-1, last bin is overflow.

    Returns:
        String label like "3-6mm" or "30+mm" for overflow bin
    """
    if bin_edges is None:
        bin_edges = DEFAULT_BIN_EDGES
    if num_bins is None:
        num_bins = DEFAULT_NUM_BINS

    n_bounded = len(bin_edges) - 1

    if bin_idx < 0:
        return f"bin_{bin_idx}"

    # Overflow bin (30+)
    if bin_idx >= n_bounded:
        return f"{bin_edges[-1]:.0f}+mm"

    return f"{bin_edges[bin_idx]:.0f}-{bin_edges[bin_idx + 1]:.0f}mm"


def format_size_bins(
    size_bins: torch.Tensor,
    bin_edges: List[float] = None,
) -> str:
    """Format size bins tensor as human-readable string.

    Args:
        size_bins: Integer tensor [num_bins] or [B, num_bins] with counts.
        bin_edges: List of bin edges in mm. Default: DEFAULT_BIN_EDGES

    Returns:
        String like "2x 3-6mm, 1x 20-30mm"
    """
    if bin_edges is None:
        bin_edges = DEFAULT_BIN_EDGES

    if size_bins.dim() == 2:
        size_bins = size_bins[0]  # Take first sample

    parts = []
    for i, count in enumerate(size_bins.tolist()):
        if count > 0:
            label = get_bin_label(i, bin_edges)
            parts.append(f"{count}x {label}")

    return ", ".join(parts) if parts else "empty"
