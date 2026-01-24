"""Combined embedding for omega (ScoreAug) + mode conditioning.

When using both ScoreAug and multi-modality training, we need to combine
both omega and mode embeddings into the time_embed.

This module provides a unified wrapper that handles both conditioning signals,
plus a factory function to create the appropriate wrapper based on config.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .base_embed import create_zero_init_mlp
from .mode_embed import (
    MODE_ENCODING_DIM,
    encode_mode_id,
    ModeEmbedModelWrapper,
    ModeEmbedDropoutModelWrapper,
    NoModeModelWrapper,
    LateModeModelWrapper,
    FiLMModeModelWrapper,
)

# Lazy imports to avoid circular dependency (augmentation imports from wrappers)
# These are imported at runtime when needed
def _get_omega_encoding_dim():
    from medgen.augmentation import OMEGA_ENCODING_DIM
    return OMEGA_ENCODING_DIM

def _get_encode_omega():
    from medgen.augmentation import encode_omega
    return encode_omega

def _get_score_aug_wrapper():
    from medgen.augmentation import ScoreAugModelWrapper
    return ScoreAugModelWrapper

# Cache for OMEGA_ENCODING_DIM (used frequently)
_OMEGA_DIM_CACHE = None

def _omega_dim():
    global _OMEGA_DIM_CACHE
    if _OMEGA_DIM_CACHE is None:
        _OMEGA_DIM_CACHE = _get_omega_encoding_dim()
    return _OMEGA_DIM_CACHE


def create_conditioning_wrapper(
    model: nn.Module,
    use_omega: bool = False,
    use_mode: bool = False,
    embed_dim: int = 256,
    mode_strategy: str = 'full',
    mode_dropout_prob: float = 0.2,
    late_mode_start_level: int = 2,
) -> Tuple[nn.Module, Optional[str]]:
    """Factory function to create appropriate conditioning wrapper.

    Simplifies the if-elif chain in trainer.py by selecting the right
    wrapper based on conditioning flags and mode strategy.

    Args:
        model: MONAI DiffusionModelUNet to wrap.
        use_omega: Whether to use omega (ScoreAug) conditioning.
        use_mode: Whether to use mode (modality) conditioning.
        embed_dim: Embedding dimension for time_embed.
        mode_strategy: Mode embedding strategy (only used if use_mode=True):
            - 'full': Standard mode embedding (original behavior)
            - 'dropout': Randomly drop mode embedding with mode_dropout_prob
            - 'none': No mode embedding (hard parameter sharing)
            - 'late': Late mode conditioning (inject at later UNet levels)
            - 'film': FiLM conditioning (scale and shift instead of additive)
        mode_dropout_prob: Probability of dropping mode embedding (for 'dropout' strategy).
        late_mode_start_level: UNet level to start injecting mode (for 'late' strategy).

    Returns:
        Tuple of (wrapped_model, wrapper_name):
        - wrapped_model: The model with conditioning wrapper applied (or original if no conditioning)
        - wrapper_name: String describing the wrapper type, or None if no wrapper applied.
            Values: "combined", "combined_film", "omega", "mode", "mode_dropout", "no_mode",
            "late_mode", "film", or None

    Example:
        >>> wrapper, wrapper_name = create_conditioning_wrapper(
        ...     model=raw_model,
        ...     use_omega=cfg.training.score_aug.use_omega_conditioning,
        ...     use_mode=cfg.mode.use_mode_embedding,
        ...     mode_strategy=cfg.mode.get('mode_embedding_strategy', 'full'),
        ...     embed_dim=4 * channels[0],
        ... )
        >>> if wrapper_name:
        ...     logger.info(f"Applied {wrapper_name} conditioning wrapper")
    """
    # Handle 'none' strategy - no mode embedding at all
    if use_mode and mode_strategy == 'none':
        if use_omega:
            # Omega only, no mode
            return _get_score_aug_wrapper()(model, embed_dim=embed_dim), "omega"
        else:
            # No conditioning at all, but wrap for API compatibility
            return NoModeModelWrapper(model), "no_mode"

    # Handle combined omega + mode
    if use_omega and use_mode:
        if mode_strategy == 'film':
            return CombinedFiLMModelWrapper(model, embed_dim=embed_dim), "combined_film"
        elif mode_strategy == 'full':
            return CombinedModelWrapper(model, embed_dim=embed_dim), "combined"
        elif mode_strategy == 'dropout':
            raise ValueError(
                "mode_embedding_strategy='dropout' is not supported with omega conditioning. "
                "Use 'full' or 'film' instead, or disable score_aug.use_omega_conditioning."
            )
        elif mode_strategy == 'late':
            raise ValueError(
                "mode_embedding_strategy='late' is not supported with omega conditioning. "
                "Use 'full' or 'film' instead, or disable score_aug.use_omega_conditioning."
            )
        else:
            return CombinedModelWrapper(model, embed_dim=embed_dim), "combined"

    # Handle omega only
    if use_omega:
        return _get_score_aug_wrapper()(model, embed_dim=embed_dim), "omega"

    # Handle mode only (with strategy)
    if use_mode:
        if mode_strategy == 'dropout':
            return ModeEmbedDropoutModelWrapper(
                model, embed_dim=embed_dim, dropout_prob=mode_dropout_prob
            ), "mode_dropout"
        elif mode_strategy == 'late':
            return LateModeModelWrapper(
                model, embed_dim=embed_dim, start_level=late_mode_start_level
            ), "late_mode"
        elif mode_strategy == 'film':
            return FiLMModeModelWrapper(model, embed_dim=embed_dim), "film"
        else:  # 'full' or default
            return ModeEmbedModelWrapper(model, embed_dim=embed_dim), "mode"

    # No conditioning
    return model, None


class CombinedTimeEmbed(nn.Module):
    """Wrapper around time_embed that adds BOTH omega and per-sample mode conditioning.

    Combines the logic from OmegaTimeEmbed and ModeTimeEmbed:
    - omega_mlp: maps omega encoding (16-dim) to embed_dim
    - mode_mlp: maps mode encoding (4-dim) to embed_dim (per-sample)
    - Both use zero-init for neutral start
    - Adds both embeddings to time_embed output

    Supports mixed modalities within a batch - each sample can have a different
    mode_id and will receive its own mode embedding.

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

        # MLPs for conditioning (zero-init for neutral start)
        self.omega_mlp = create_zero_init_mlp(_omega_dim(), embed_dim)
        self.mode_mlp = create_zero_init_mlp(MODE_ENCODING_DIM, embed_dim)

        # Omega uses buffer (same for all samples in batch)
        self.register_buffer('_omega_encoding', torch.zeros(1, _omega_dim()))

        # Mode uses regular tensor (per-sample, variable batch size)
        self._mode_encoding: Optional[torch.Tensor] = None

    def set_omega_encoding(self, omega_encoding: torch.Tensor):
        """Set omega encoding for next forward pass."""
        self._omega_encoding.copy_(omega_encoding)

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

    def set_encodings(
        self,
        omega_encoding: torch.Tensor,
        mode_encoding: torch.Tensor,
    ):
        """Set both encodings for next forward pass."""
        self._omega_encoding.copy_(omega_encoding)
        self._mode_encoding = mode_encoding

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass: original time_embed + omega + per-sample mode embeddings.

        Args:
            t_emb: Timestep embedding input [B, C]

        Returns:
            Time embedding with combined conditioning [B, embed_dim]
        """
        # Original time embedding
        out = self.original(t_emb)

        # Add omega embedding (broadcasts from [1, embed_dim] to [B, embed_dim])
        omega_emb = self.omega_mlp(self._omega_encoding)
        out = out + omega_emb

        # Add per-sample mode embedding
        if self._mode_encoding is not None:
            mode_emb = self.mode_mlp(self._mode_encoding)  # [B, embed_dim]
            out = out + mode_emb

        return out


class CombinedModelWrapper(nn.Module):
    """Wrapper to inject both omega and per-sample mode conditioning into MONAI UNet.

    Supports mixed modalities within a batch - each sample can have a different
    mode_id and will receive its own mode embedding.

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

        # Ensure combined_time_embed is on same device as model
        try:
            device = next(model.parameters()).device
            self.combined_time_embed = self.combined_time_embed.to(device)
            model.time_embed = self.combined_time_embed
        except StopIteration:
            pass  # Model has no parameters, keep on CPU

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        omega: Optional[Dict[str, Any]] = None,
        mode_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with combined omega and per-sample mode conditioning.

        Args:
            x: Noisy input tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            omega: Optional ScoreAug parameters for conditioning
            mode_id: Optional mode ID tensor [B] (0=bravo, 1=flair, 2=t1_pre, 3=t1_gd).
                Can contain different mode_ids for different samples.

        Returns:
            Model prediction [B, C_out, H, W]
        """
        # Encode both conditioning signals
        batch_size = x.shape[0]
        # Pass mode_id to encode_omega to include mode intensity scaling info (dims 32-35)
        omega_encoding = _get_encode_omega()(omega, x.device, mode_id=mode_id)
        mode_encoding = encode_mode_id(mode_id, x.device, batch_size=batch_size)

        # Set both encodings
        self.combined_time_embed.set_encodings(omega_encoding, mode_encoding)

        # Call model normally
        return self.model(x, timesteps)

    def parameters(self, recurse: bool = True):
        """Get all parameters including embedding MLPs."""
        return self.model.parameters(recurse=recurse)


class CombinedFiLMTimeEmbed(nn.Module):
    """Wrapper around time_embed with omega (additive) + FiLM mode conditioning.

    Combines:
    - omega_mlp: maps omega encoding (16-dim) to embed_dim (additive)
    - film_mlp: maps mode encoding (4-dim) to gamma and beta for FiLM

    Order of operations:
    1. out = original(t_emb)
    2. out = out + omega_emb  (additive omega)
    3. out = gamma * out + beta  (FiLM mode modulation)
    """

    def __init__(self, original_time_embed: nn.Module, embed_dim: int):
        """Initialize combined FiLM time embedding.

        Args:
            original_time_embed: The original time_embed module from DiffusionModelUNet
            embed_dim: Output dimension (should match original time_embed output)
        """
        super().__init__()
        self.original = original_time_embed
        self.embed_dim = embed_dim

        # Import FiLM MLP creator
        from .base_embed import create_film_mlp

        # Omega uses additive embedding (zero-init)
        self.omega_mlp = create_zero_init_mlp(_omega_dim(), embed_dim)

        # Mode uses FiLM (gamma/beta initialized to identity)
        self.film_mlp = create_film_mlp(MODE_ENCODING_DIM, embed_dim)

        # Omega uses buffer (same for all samples in batch)
        self.register_buffer('_omega_encoding', torch.zeros(1, _omega_dim()))

        # Mode uses regular tensor (per-sample, variable batch size)
        self._mode_encoding: Optional[torch.Tensor] = None

    def set_omega_encoding(self, omega_encoding: torch.Tensor):
        """Set omega encoding for next forward pass."""
        self._omega_encoding.copy_(omega_encoding)

    def set_mode_encoding(self, mode_encoding: torch.Tensor):
        """Set mode encoding for next forward pass."""
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

    def set_encodings(
        self,
        omega_encoding: torch.Tensor,
        mode_encoding: torch.Tensor,
    ):
        """Set both encodings for next forward pass."""
        self._omega_encoding.copy_(omega_encoding)
        self._mode_encoding = mode_encoding

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass: original + omega (additive) + mode (FiLM).

        Args:
            t_emb: Timestep embedding input [B, C]

        Returns:
            Time embedding with combined conditioning [B, embed_dim]
        """
        # Original time embedding
        out = self.original(t_emb)

        # Add omega embedding (broadcasts from [1, embed_dim] to [B, embed_dim])
        omega_emb = self.omega_mlp(self._omega_encoding)
        out = out + omega_emb

        # Apply FiLM modulation for mode
        if self._mode_encoding is not None:
            film_params = self.film_mlp(self._mode_encoding)  # [B, 2*embed_dim]
            gamma = film_params[:, :self.embed_dim]  # [B, embed_dim]
            beta = film_params[:, self.embed_dim:]   # [B, embed_dim]
            out = gamma * out + beta

        return out


class CombinedFiLMModelWrapper(nn.Module):
    """Wrapper with omega (additive) + FiLM mode conditioning.

    Use this when:
    - training.score_aug.enabled=true AND
    - mode.use_mode_embedding=true AND
    - mode.mode_embedding_strategy=film
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
        self.combined_film_time_embed = CombinedFiLMTimeEmbed(original_time_embed, embed_dim)

        # Replace the model's time_embed
        model.time_embed = self.combined_film_time_embed

        # Ensure on same device as model
        try:
            device = next(model.parameters()).device
            self.combined_film_time_embed = self.combined_film_time_embed.to(device)
            model.time_embed = self.combined_film_time_embed
        except StopIteration:
            pass

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        omega: Optional[Dict[str, Any]] = None,
        mode_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with omega + FiLM mode conditioning.

        Args:
            x: Noisy input tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            omega: Optional ScoreAug parameters for conditioning
            mode_id: Optional mode ID tensor [B] (0=bravo, 1=flair, 2=t1_pre, 3=t1_gd).

        Returns:
            Model prediction [B, C_out, H, W]
        """
        batch_size = x.shape[0]
        omega_encoding = _get_encode_omega()(omega, x.device, mode_id=mode_id)
        mode_encoding = encode_mode_id(mode_id, x.device, batch_size=batch_size)

        self.combined_film_time_embed.set_encodings(omega_encoding, mode_encoding)

        return self.model(x, timesteps)

    def parameters(self, recurse: bool = True):
        """Get all parameters including embedding MLPs."""
        return self.model.parameters(recurse=recurse)
