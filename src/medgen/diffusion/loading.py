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

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional
import logging

import torch
from torch import nn
from monai.networks.nets import DiffusionModelUNet

from medgen.core import (
    DEFAULT_CHANNELS,
    DEFAULT_ATTENTION_LEVELS,
    DEFAULT_NUM_RES_BLOCKS,
    DEFAULT_NUM_HEAD_CHANNELS,
)
from medgen.data import create_conditioning_wrapper


log = logging.getLogger(__name__)


WrapperType = Literal['raw', 'score_aug', 'mode_embed', 'combined']


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
    config: Dict[str, Any]
    wrapper_type: WrapperType
    epoch: int
    checkpoint_path: str


def detect_wrapper_type(state_dict: Dict[str, Any]) -> WrapperType:
    """Detect wrapper type from checkpoint state dict keys.

    Wrapper models store the inner model under the 'model.' key prefix,
    and have additional MLP keys for conditioning:
    - omega_mlp: ScoreAug conditioning
    - mode_mlp: Mode (modality) embedding

    Args:
        state_dict: Model state dictionary from checkpoint.

    Returns:
        Wrapper type: 'raw', 'score_aug', 'mode_embed', or 'combined'.
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

    if has_omega_mlp and has_mode_mlp:
        return 'combined'
    elif has_omega_mlp:
        return 'score_aug'
    elif has_mode_mlp:
        return 'mode_embed'

    # Has model prefix but no recognized MLPs - could be a different wrapper
    # Fall back to raw and let strict loading fail if incompatible
    log.warning(
        "Model has 'model.' prefix but no recognized conditioning MLPs. "
        "Attempting to load as raw model."
    )
    return 'raw'


def load_diffusion_model(
    checkpoint_path: str,
    device: torch.device,
    in_channels: Optional[int] = None,
    out_channels: Optional[int] = None,
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
    in_channels: Optional[int] = None,
    out_channels: Optional[int] = None,
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
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Detect wrapper type
    wrapper_type = detect_wrapper_type(state_dict)
    log.info(f"Detected model type: {wrapper_type}")

    # Extract config and architecture params
    config = checkpoint.get('config', {})
    model_config = checkpoint.get('model_config', {})

    # Resolve architecture parameters (args > checkpoint > defaults)
    resolved_in_channels = _resolve_channels(
        'in_channels', in_channels, model_config, required=True
    )
    resolved_out_channels = _resolve_channels(
        'out_channels', out_channels, model_config, required=True
    )

    channels = model_config.get('channels', list(DEFAULT_CHANNELS))
    attention_levels = model_config.get('attention_levels', list(DEFAULT_ATTENTION_LEVELS))
    num_res_blocks = model_config.get('num_res_blocks', DEFAULT_NUM_RES_BLOCKS)
    num_head_channels = model_config.get('num_head_channels', DEFAULT_NUM_HEAD_CHANNELS)

    # Use checkpoint spatial_dims if available, else use arg
    resolved_spatial_dims = model_config.get('spatial_dims', spatial_dims)

    # Log architecture info
    if 'channels' in model_config:
        log.info(f"Using architecture from checkpoint: channels={channels}")
    else:
        log.info(f"Using default architecture: channels={channels}")

    # Create base model
    base_model = DiffusionModelUNet(
        spatial_dims=resolved_spatial_dims,
        in_channels=resolved_in_channels,
        out_channels=resolved_out_channels,
        channels=tuple(channels),
        attention_levels=tuple(attention_levels),
        num_res_blocks=num_res_blocks,
        num_head_channels=num_head_channels,
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
        log.info(f"Applied {wrapper_name} conditioning wrapper")
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper_type}")

    # Load state dict
    model.load_state_dict(state_dict, strict=True)

    # Move to device and set eval mode
    model.to(device)
    model.eval()

    # Optionally compile
    if compile_model:
        log.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")

    # Extract epoch
    epoch = checkpoint.get('epoch', 0)

    return LoadedModel(
        model=model,
        config=config,
        wrapper_type=wrapper_type,
        epoch=epoch,
        checkpoint_path=checkpoint_path,
    )


def _resolve_channels(
    name: str,
    arg_value: Optional[int],
    model_config: Dict[str, Any],
    required: bool = False,
) -> int:
    """Resolve channel count from argument, checkpoint, or raise error.

    Args:
        name: Parameter name for error messages.
        arg_value: Value passed as argument (highest priority).
        model_config: Model config dict from checkpoint.
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
        return model_config[name]

    # Not found
    if required:
        raise ValueError(
            f"{name} not provided and not found in checkpoint config. "
            f"Either pass {name} explicitly or use a checkpoint that includes architecture info."
        )

    return 0  # Unreachable if required=True
