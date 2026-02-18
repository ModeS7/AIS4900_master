"""Model loading utilities for diffusion models.

This module provides reusable functions for loading trained diffusion models
from checkpoints, with automatic wrapper detection and architecture inference.

Usage:
    from medgen.diffusion import load_diffusion_model, load_diffusion_model_with_metadata

    # Simple loading
    model = load_diffusion_model(
        "path/to/checkpoint.pt",
        device=torch.device("cuda"),
        in_channels=2,
        out_channels=1,
    )

    # Loading with metadata
    result = load_diffusion_model_with_metadata(
        "path/to/checkpoint.pt",
        device=torch.device("cuda"),
    )
    print(f"Wrapper: {result.wrapper_type}, Epoch: {result.epoch}")
"""

import logging
from dataclasses import dataclass
from typing import Any, Literal

import torch
from monai.networks.nets import DiffusionModelUNet
from torch import nn

from medgen.core import (
    DEFAULT_ATTENTION_LEVELS,
    DEFAULT_CHANNELS,
    DEFAULT_NUM_HEAD_CHANNELS,
    DEFAULT_NUM_RES_BLOCKS,
)
from medgen.data import create_conditioning_wrapper
from medgen.models.wrappers.size_bin_embed import SizeBinModelWrapper

logger = logging.getLogger(__name__)


WrapperType = Literal['raw', 'score_aug', 'mode_embed', 'combined', 'size_bin']


@dataclass
class LoadedModel:
    """Result of loading a diffusion model with metadata.

    Attributes:
        model: The loaded model in eval mode.
        config: Full checkpoint config dict (training params, model config, etc.).
        wrapper_type: Detected wrapper type ('raw', 'score_aug', 'mode_embed', 'combined').
        epoch: Training epoch when checkpoint was saved.
        checkpoint_path: Path to the loaded checkpoint.
    """
    model: nn.Module
    config: dict[str, Any]
    wrapper_type: WrapperType
    epoch: int
    checkpoint_path: str


def detect_wrapper_type(state_dict: dict[str, Any]) -> WrapperType:
    """Detect wrapper type from checkpoint state dict keys.

    Wrapper models store the inner model under the 'model.' key prefix,
    and have additional MLP keys for conditioning:
    - omega_mlp: ScoreAug conditioning
    - mode_mlp: Mode (modality) embedding
    - size_bin_time_embed: Size bin conditioning for seg_conditioned mode

    Args:
        state_dict: Model state dictionary from checkpoint.

    Returns:
        Wrapper type: 'raw', 'score_aug', 'mode_embed', 'combined', or 'size_bin'.
    """
    keys = list(state_dict.keys())

    # Check if model is wrapped (inner model stored under 'model.' prefix)
    # Note: Wrappers use self.model, so keys look like 'model.time_embed.0.weight'
    has_model_prefix = any(k.startswith('model.') for k in keys)

    if not has_model_prefix:
        return 'raw'

    # Check for conditioning MLPs
    has_omega_mlp = any('omega_mlp' in k for k in keys)
    has_mode_mlp = any('mode_mlp' in k for k in keys)
    # Size bin wrapper has size_bin_time_embed or model.time_embed.bin_embeddings
    has_size_bin = any('size_bin_time_embed' in k or 'bin_embeddings' in k for k in keys)

    if has_omega_mlp and has_mode_mlp:
        return 'combined'
    elif has_omega_mlp:
        return 'score_aug'
    elif has_mode_mlp:
        return 'mode_embed'
    elif has_size_bin:
        return 'size_bin'

    # Has model prefix but no recognized MLPs - could be a different wrapper
    # Fall back to raw and let strict loading fail if incompatible
    logger.warning(
        "Model has 'model.' prefix but no recognized conditioning MLPs. "
        "Attempting to load as raw model."
    )
    return 'raw'


def load_diffusion_model(
    checkpoint_path: str,
    device: torch.device,
    in_channels: int | None = None,
    out_channels: int | None = None,
    compile_model: bool = False,
    spatial_dims: int = 2,
) -> nn.Module:
    """Load trained diffusion model from checkpoint.

    Automatically detects wrapper type and architecture from checkpoint config.
    Falls back to default architecture if not present in checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file).
        device: Device to load model onto.
        in_channels: Input channels. Required if not in checkpoint config.
        out_channels: Output channels. Required if not in checkpoint config.
        compile_model: Whether to compile with torch.compile.
        spatial_dims: Spatial dimensions (2 for 2D, 3 for 3D).

    Returns:
        Loaded model in eval mode, optionally compiled.

    Raises:
        ValueError: If in_channels/out_channels not provided and not in checkpoint.
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    result = load_diffusion_model_with_metadata(
        checkpoint_path=checkpoint_path,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        compile_model=compile_model,
        spatial_dims=spatial_dims,
    )
    return result.model


def load_diffusion_model_with_metadata(
    checkpoint_path: str,
    device: torch.device,
    in_channels: int | None = None,
    out_channels: int | None = None,
    compile_model: bool = False,
    spatial_dims: int = 2,
) -> LoadedModel:
    """Load trained diffusion model with full metadata.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file).
        device: Device to load model onto.
        in_channels: Input channels. Overrides checkpoint config if provided.
        out_channels: Output channels. Overrides checkpoint config if provided.
        compile_model: Whether to compile with torch.compile.
        spatial_dims: Spatial dimensions (2 for 2D, 3 for 3D).

    Returns:
        LoadedModel with model, config, wrapper_type, epoch, and checkpoint_path.

    Raises:
        ValueError: If channels not resolvable from args or checkpoint.
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    # weights_only=False because checkpoints contain OmegaConf config objects
    # (ListConfig, DictConfig, ContainerMetadata, etc.) - these are our own checkpoints
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Detect wrapper type
    wrapper_type = detect_wrapper_type(state_dict)
    logger.info(f"Detected model type: {wrapper_type}")

    # Extract config and architecture params
    # Architecture can be in multiple places depending on checkpoint format:
    # 1. 'model_config' (standalone key)
    # 2. 'config' (flat format - old seg_conditioned checkpoints)
    # 3. 'config.model' (nested format - new trainer checkpoints)
    config = checkpoint.get('config', {})
    model_config = checkpoint.get('model_config', {})

    # Check for nested model config (config.model)
    nested_model_config = config.get('model', {}) if isinstance(config, dict) else {}

    # Merge: model_config > nested_model_config > config (priority order)
    arch_config = {**config, **nested_model_config, **model_config}

    # Infer channels from state dict weight shapes as fallback
    inferred_in, inferred_out = _infer_channels_from_state_dict(state_dict)

    # Resolve architecture parameters (args > checkpoint config > state dict inference)
    resolved_in_channels = _resolve_channels(
        'in_channels', in_channels, arch_config, inferred_in, required=True
    )
    resolved_out_channels = _resolve_channels(
        'out_channels', out_channels, arch_config, inferred_out, required=True
    )

    channels = arch_config.get('channels', list(DEFAULT_CHANNELS))
    attention_levels = arch_config.get('attention_levels', list(DEFAULT_ATTENTION_LEVELS))
    num_res_blocks = arch_config.get('num_res_blocks', DEFAULT_NUM_RES_BLOCKS)
    num_head_channels = arch_config.get('num_head_channels', DEFAULT_NUM_HEAD_CHANNELS)

    # Use checkpoint spatial_dims if available, else use arg
    resolved_spatial_dims = arch_config.get('spatial_dims', spatial_dims)

    # Compute norm_num_groups: must divide all channel counts
    # Default is 32, but if channels are smaller, compute GCD
    import math
    from functools import reduce
    norm_num_groups = arch_config.get('norm_num_groups')
    if norm_num_groups is None:
        # Find largest divisor that works for all channels (max 32)
        gcd = reduce(math.gcd, channels)
        norm_num_groups = min(gcd, 32)

    # Log architecture info
    if 'channels' in arch_config:
        logger.info(f"Using architecture from checkpoint: channels={channels}, norm_num_groups={norm_num_groups}")
    else:
        logger.info(f"Using default architecture: channels={channels}")

    # Create base model
    base_model = DiffusionModelUNet(
        spatial_dims=resolved_spatial_dims,
        in_channels=resolved_in_channels,
        out_channels=resolved_out_channels,
        channels=tuple(channels),
        attention_levels=tuple(attention_levels),
        num_res_blocks=num_res_blocks,
        num_head_channels=num_head_channels,
        norm_num_groups=norm_num_groups,
    )

    # Wrap model if needed
    # MONAI DiffusionModelUNet time_embed output dim is 4 * channels[0]
    embed_dim = 4 * channels[0]

    if wrapper_type == 'raw':
        model = base_model
    elif wrapper_type in ('score_aug', 'mode_embed', 'combined'):
        model, wrapper_name = create_conditioning_wrapper(
            model=base_model,
            use_omega=(wrapper_type in ('score_aug', 'combined')),
            use_mode=(wrapper_type in ('mode_embed', 'combined')),
            embed_dim=embed_dim,
        )
        logger.info(f"Applied {wrapper_name} conditioning wrapper")
    elif wrapper_type == 'size_bin':
        # Size bin wrapper for seg_conditioned mode
        # Get num_bins from checkpoint config if available
        size_bin_cfg = config.get('size_bin', {})
        num_bins = size_bin_cfg.get('num_bins', 7)
        model = SizeBinModelWrapper(
            model=base_model,
            embed_dim=embed_dim,
            num_bins=num_bins,
        )
        logger.info(f"Applied SizeBinModelWrapper (num_bins={num_bins})")
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper_type}")

    # Load state dict
    model.load_state_dict(state_dict, strict=True)

    # Move to device and set eval mode
    model.to(device)
    model.eval()

    # Optionally compile
    if compile_model:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]

    # Extract epoch
    epoch = checkpoint.get('epoch', 0)

    return LoadedModel(
        model=model,
        config=config,
        wrapper_type=wrapper_type,
        epoch=epoch,
        checkpoint_path=checkpoint_path,
    )


def _infer_channels_from_state_dict(
    state_dict: dict[str, Any],
) -> tuple[int | None, int | None]:
    """Infer in_channels and out_channels from state dict weight shapes.

    Works for both raw and wrapped models (handles 'model.' prefix).

    Returns:
        (in_channels, out_channels) tuple, None if not found.
    """
    in_channels = None
    out_channels = None

    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        # conv_in.conv.weight shape: [features, in_channels, *kernel_size]
        if key.endswith('conv_in.conv.weight'):
            in_channels = tensor.shape[1]
        # out.2.conv.weight shape: [out_channels, features, *kernel_size]
        if key.endswith('out.2.conv.weight'):
            out_channels = tensor.shape[0]

    return in_channels, out_channels


def _resolve_channels(
    name: str,
    arg_value: int | None,
    model_config: dict[str, Any],
    inferred_value: int | None = None,
    required: bool = False,
) -> int:
    """Resolve channel count from argument, checkpoint config, state dict, or raise error.

    Priority: arg_value > checkpoint config > inferred from state dict weights.

    Args:
        name: Parameter name for error messages.
        arg_value: Value passed as argument (highest priority).
        model_config: Model config dict from checkpoint.
        inferred_value: Value inferred from state dict weight shapes.
        required: Whether to raise error if not resolvable.

    Returns:
        Resolved channel count.

    Raises:
        ValueError: If required and not resolvable.
    """
    # Argument takes precedence
    if arg_value is not None:
        return arg_value

    # Check checkpoint config
    if name in model_config:
        return int(model_config[name])

    # Infer from state dict weight shapes
    if inferred_value is not None:
        logger.info(f"Inferred {name}={inferred_value} from checkpoint weight shapes")
        return inferred_value

    # Not found
    if required:
        raise ValueError(
            f"{name} not provided and not found in checkpoint config. "
            f"Either pass {name} explicitly or use a checkpoint that includes architecture info."
        )

    return 0  # Unreachable if required=True
