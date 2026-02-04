"""Mode embedding for multi-modality diffusion training.

Adds modality conditioning to the diffusion model via time_embed modification.
Follows the same pattern as OmegaTimeEmbed from score_aug.py.

Mode embedding tells the model which modality (bravo, flair, t1_pre, t1_gd)
it is generating, enabling a single model to handle all modalities.

Strategies:
    - 'full': Standard mode embedding (original behavior)
    - 'dropout': Randomly drop mode embedding with probability (like CFG training)
    - 'none': No mode embedding (hard parameter sharing, forces shared representations)
    - 'late': Late conditioning (inject mode only in later UNet blocks)
"""


import torch
from torch import nn

from .base_embed import create_film_mlp, create_zero_init_mlp

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
    mode_id: torch.Tensor | None,
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
        self._mode_encoding: torch.Tensor | None = None

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
        mode_id: torch.Tensor | None = None,
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


class ModeEmbedDropoutModelWrapper(nn.Module):
    """Wrapper with mode embedding dropout for regularization.

    During training, randomly drops the mode embedding with probability `dropout_prob`.
    This forces the model to learn shared representations that work without mode info,
    similar to classifier-free guidance training.

    At inference, mode embedding is always used (dropout disabled).
    """

    def __init__(
        self,
        model: nn.Module,
        embed_dim: int = 256,
        dropout_prob: float = 0.2,
    ):
        """Initialize wrapper with dropout.

        Args:
            model: MONAI DiffusionModelUNet to wrap
            embed_dim: Embedding dimension (should match model's time_embed output)
            dropout_prob: Probability of dropping mode embedding during training
        """
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim
        self.dropout_prob = dropout_prob

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
        mode_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with mode embedding dropout.

        During training, randomly drops mode embedding with probability dropout_prob.
        During inference (eval mode), always uses mode embedding.

        Args:
            x: Noisy input tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            mode_id: Optional mode ID tensor [B] (0=bravo, 1=flair, 2=t1_pre, 3=t1_gd).

        Returns:
            Model prediction [B, C_out, H, W]
        """
        batch_size = x.shape[0]

        # During training, randomly drop mode embedding
        if self.training and torch.rand(1).item() < self.dropout_prob:
            # Drop mode: use zeros encoding
            mode_encoding = torch.zeros(
                batch_size, MODE_ENCODING_DIM, device=x.device
            )
        else:
            # Use actual mode encoding
            mode_encoding = encode_mode_id(mode_id, x.device, batch_size=batch_size)

        self.mode_time_embed.set_mode_encoding(mode_encoding)

        # Call model normally
        return self.model(x, timesteps)

    def parameters(self, recurse: bool = True):
        """Get all parameters including mode MLP."""
        return self.model.parameters(recurse=recurse)


class NoModeModelWrapper(nn.Module):
    """Wrapper that ignores mode_id - pure hard parameter sharing.

    This forces the model to learn shared representations across all modalities
    without any mode-specific conditioning. The model cannot distinguish between
    modalities and must learn features that generalize.

    This is expected to provide regularization via multi-task learning
    (hard parameter sharing).
    """

    def __init__(self, model: nn.Module):
        """Initialize wrapper.

        Args:
            model: MONAI DiffusionModelUNet to wrap (used without modification)
        """
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        mode_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass ignoring mode_id.

        Args:
            x: Noisy input tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            mode_id: Ignored - only present for API compatibility.

        Returns:
            Model prediction [B, C_out, H, W]
        """
        # Simply call model without any mode conditioning
        return self.model(x, timesteps)

    def parameters(self, recurse: bool = True):
        """Get all parameters."""
        return self.model.parameters(recurse=recurse)


class LateModeTimeEmbed(nn.Module):
    """Time embedding that stores mode encoding for later injection.

    Unlike ModeTimeEmbed which adds mode to time_embed output,
    this class stores the mode encoding for injection at later UNet levels.
    The actual injection happens via LateModeInjector modules.
    """

    def __init__(self, original_time_embed: nn.Module, embed_dim: int):
        """Initialize late mode time embedding.

        Args:
            original_time_embed: The original time_embed module from DiffusionModelUNet
            embed_dim: Output dimension (should match original time_embed output)
        """
        super().__init__()
        self.original = original_time_embed
        self.embed_dim = embed_dim

        # Store mode encoding for late injection (not used in forward)
        self._mode_encoding: torch.Tensor | None = None

    def set_mode_encoding(self, mode_encoding: torch.Tensor):
        """Set mode encoding for late injection.

        Args:
            mode_encoding: Tensor [B, MODE_ENCODING_DIM] with per-sample encodings
        """
        if mode_encoding.dim() != 2:
            raise ValueError(
                f"mode_encoding must be 2D [B, {MODE_ENCODING_DIM}], "
                f"got {mode_encoding.dim()}D with shape {mode_encoding.shape}"
            )
        self._mode_encoding = mode_encoding

    def get_mode_encoding(self) -> torch.Tensor | None:
        """Get stored mode encoding for injection."""
        return self._mode_encoding

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass: original time_embed only (mode added later).

        Args:
            t_emb: Timestep embedding input [B, C]

        Returns:
            Time embedding WITHOUT mode conditioning [B, embed_dim]
        """
        # Original time embedding only - mode is injected later
        return self.original(t_emb)


class LateModeModelWrapper(nn.Module):
    """Wrapper that injects mode conditioning only at later UNet levels.

    Instead of adding mode to time_embed at the start (which flows through
    all layers), this wrapper injects mode conditioning only at deeper
    resolution levels. This forces early/mid layers to learn shared
    representations while allowing later layers to specialize.

    The mode embedding is added to the residual blocks at specified levels
    via a hook mechanism.

    Args:
        model: MONAI DiffusionModelUNet to wrap
        embed_dim: Embedding dimension
        start_level: UNet level to start injecting mode (0-indexed from input)
            For a 4-level UNet: 0=256, 1=128, 2=64, 3=32 (at 256px input)
            start_level=2 means mode is only added at 64x64 and 32x32 levels
    """

    def __init__(
        self,
        model: nn.Module,
        embed_dim: int = 256,
        start_level: int = 2,
    ):
        """Initialize late mode wrapper.

        Args:
            model: MONAI DiffusionModelUNet to wrap
            embed_dim: Embedding dimension (should match model's time_embed output)
            start_level: First level to inject mode (0=earliest, higher=later)
        """
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim
        self.start_level = start_level

        if not hasattr(model, 'time_embed'):
            raise ValueError("Model does not have 'time_embed' attribute")

        # Replace time_embed with late-mode version (stores encoding for later)
        original_time_embed = model.time_embed
        self.late_time_embed = LateModeTimeEmbed(original_time_embed, embed_dim)
        model.time_embed = self.late_time_embed

        # Create mode MLP for late injection
        self.mode_mlp = create_zero_init_mlp(MODE_ENCODING_DIM, embed_dim)

        # Ensure on same device as model
        try:
            device = next(model.parameters()).device
            self.late_time_embed = self.late_time_embed.to(device)
            self.mode_mlp = self.mode_mlp.to(device)
            model.time_embed = self.late_time_embed
        except StopIteration:
            pass

        # Register hooks for late injection
        self._register_late_injection_hooks()

    def _register_late_injection_hooks(self):
        """Register forward hooks to inject mode at later levels."""
        # MONAI UNet structure: down_blocks, mid_block, up_blocks
        # Each down/up block corresponds to a resolution level

        self._hooks = []

        # Get number of down blocks to determine levels
        if hasattr(self.model, 'down_blocks'):
            num_levels = len(self.model.down_blocks)

            # Register hooks on down blocks at start_level and beyond
            for level in range(self.start_level, num_levels):
                if level < len(self.model.down_blocks):
                    hook = self.model.down_blocks[level].register_forward_hook(
                        self._create_injection_hook()
                    )
                    self._hooks.append(hook)

            # Also inject at middle block (MONAI uses 'middle_block')
            if hasattr(self.model, 'middle_block') and self.model.middle_block is not None:
                hook = self.model.middle_block.register_forward_hook(
                    self._create_injection_hook()
                )
                self._hooks.append(hook)

            # Inject at up blocks (mirrored levels)
            if hasattr(self.model, 'up_blocks'):
                for level in range(num_levels - 1, self.start_level - 1, -1):
                    up_idx = num_levels - 1 - level
                    if up_idx < len(self.model.up_blocks):
                        hook = self.model.up_blocks[up_idx].register_forward_hook(
                            self._create_injection_hook()
                        )
                        self._hooks.append(hook)

    def _create_injection_hook(self):
        """Create a hook function for mode injection.

        Handles both tensor outputs and tuple outputs (MONAI down/up blocks
        return (hidden_states, residual_samples) tuples).
        """
        def hook(module, input, output):
            mode_encoding = self.late_time_embed.get_mode_encoding()
            if mode_encoding is None:
                return output

            # Compute mode embedding
            mode_emb = self.mode_mlp(mode_encoding)  # [B, embed_dim]

            # Handle tuple output (down/up blocks return (hidden, residuals))
            if isinstance(output, tuple):
                hidden = output[0]
                if hidden.dim() == 4 and hidden.shape[1] == self.embed_dim:
                    mode_emb_spatial = mode_emb.view(-1, self.embed_dim, 1, 1)
                    return (hidden + mode_emb_spatial,) + output[1:]
                return output

            # Handle tensor output (middle block)
            if output.dim() == 4:
                mode_emb_spatial = mode_emb.view(-1, self.embed_dim, 1, 1)
                if output.shape[1] == self.embed_dim:
                    return output + mode_emb_spatial
            return output
        return hook

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        mode_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with late mode injection.

        Args:
            x: Noisy input tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            mode_id: Optional mode ID tensor [B] (0=bravo, 1=flair, 2=t1_pre, 3=t1_gd).

        Returns:
            Model prediction [B, C_out, H, W]
        """
        batch_size = x.shape[0]
        mode_encoding = encode_mode_id(mode_id, x.device, batch_size=batch_size)
        self.late_time_embed.set_mode_encoding(mode_encoding)

        # Call model - hooks will inject mode at later levels
        return self.model(x, timesteps)

    def parameters(self, recurse: bool = True):
        """Get all parameters including mode MLP."""
        # Include both model parameters and mode_mlp parameters
        for p in self.model.parameters(recurse=recurse):
            yield p
        for p in self.mode_mlp.parameters():
            yield p

    def __del__(self):
        """Remove hooks on deletion."""
        if hasattr(self, '_hooks'):
            for hook in self._hooks:
                hook.remove()


class FiLMModeTimeEmbed(nn.Module):
    """Time embedding with FiLM (Feature-wise Linear Modulation) mode conditioning.

    Instead of adding mode embedding to time_embed output, FiLM learns to
    scale (gamma) and shift (beta) the features per mode:

        output = gamma * time_embed + beta

    This gives the model more expressive power to adapt features per modality.

    Initialization ensures FiLM starts as identity: gamma=1, beta=0
    """

    def __init__(self, original_time_embed: nn.Module, embed_dim: int):
        """Initialize FiLM time embedding.

        Args:
            original_time_embed: The original time_embed module from DiffusionModelUNet
            embed_dim: Output dimension (should match original time_embed output)
        """
        super().__init__()
        self.original = original_time_embed
        self.embed_dim = embed_dim

        # FiLM MLP: mode encoding -> [gamma, beta]
        self.film_mlp = create_film_mlp(MODE_ENCODING_DIM, embed_dim)

        # Store current mode encoding
        self._mode_encoding: torch.Tensor | None = None

    def set_mode_encoding(self, mode_encoding: torch.Tensor):
        """Set mode encoding for next forward pass.

        Args:
            mode_encoding: Tensor [B, MODE_ENCODING_DIM] with per-sample encodings
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
        """Forward pass: FiLM modulation of time embedding.

        Args:
            t_emb: Timestep embedding input [B, C]

        Returns:
            FiLM-modulated time embedding [B, embed_dim]
        """
        # Original time embedding
        out = self.original(t_emb)

        # Apply FiLM modulation if mode encoding is set
        if self._mode_encoding is not None:
            # Get gamma and beta from FiLM MLP
            film_params = self.film_mlp(self._mode_encoding)  # [B, 2*embed_dim]
            gamma = film_params[:, :self.embed_dim]  # [B, embed_dim]
            beta = film_params[:, self.embed_dim:]   # [B, embed_dim]

            # Apply FiLM: gamma * out + beta
            out = gamma * out + beta

        return out


class FiLMModeModelWrapper(nn.Module):
    """Wrapper to inject FiLM mode conditioning into MONAI UNet.

    FiLM (Feature-wise Linear Modulation) learns per-mode scale and shift
    parameters that modulate the time embedding features. This is more
    expressive than simple additive conditioning.

    Use this when mode.mode_embedding_strategy='film'
    """

    def __init__(self, model: nn.Module, embed_dim: int = 256):
        """Initialize FiLM wrapper.

        Args:
            model: MONAI DiffusionModelUNet to wrap
            embed_dim: Embedding dimension (should match model's time_embed output)
        """
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim

        # Replace time_embed with FiLM version
        if not hasattr(model, 'time_embed'):
            raise ValueError("Model does not have 'time_embed' attribute")

        original_time_embed = model.time_embed
        self.film_time_embed = FiLMModeTimeEmbed(original_time_embed, embed_dim)

        # Replace the model's time_embed
        model.time_embed = self.film_time_embed

        # Ensure film_time_embed is on same device as model
        try:
            device = next(model.parameters()).device
            self.film_time_embed = self.film_time_embed.to(device)
            model.time_embed = self.film_time_embed
        except StopIteration:
            pass

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        mode_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FiLM mode conditioning.

        Args:
            x: Noisy input tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            mode_id: Optional mode ID tensor [B] (0=bravo, 1=flair, 2=t1_pre, 3=t1_gd).

        Returns:
            Model prediction [B, C_out, H, W]
        """
        batch_size = x.shape[0]
        mode_encoding = encode_mode_id(mode_id, x.device, batch_size=batch_size)
        self.film_time_embed.set_mode_encoding(mode_encoding)

        return self.model(x, timesteps)

    def parameters(self, recurse: bool = True):
        """Get all parameters including FiLM MLP."""
        return self.model.parameters(recurse=recurse)
