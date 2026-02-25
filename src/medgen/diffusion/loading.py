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

    # Detect model type: 'unet', 'dit'/'sit', 'hdit', or 'uvit'
    model_type = arch_config.get('model_type')
    if model_type is None:
        model_type = _detect_model_arch_from_state_dict(state_dict)

    if model_type in ('hdit',):
        from medgen.models.hdit import create_hdit

        variant = arch_config.get('variant')
        patch_size = arch_config.get('patch_size', 4)
        conditioning = arch_config.get('conditioning', 'concat')

        if conditioning == 'concat':
            hdit_in_channels = resolved_in_channels
        else:
            hdit_in_channels = resolved_out_channels
        cond_channels = max(0, resolved_in_channels - resolved_out_channels)

        # Infer variant from hidden_size in state_dict
        if variant is None:
            variant = _infer_variant_from_state_dict(state_dict, 'hdit')

        # Infer level_depths from state_dict
        level_depths = _infer_hdit_level_depths(state_dict)

        # Infer input_size and depth_size from pos_embeds
        hdit_input_size = arch_config.get('image_size', 128)
        depth_size = arch_config.get('depth_size')
        hdit_input_size, depth_size = _infer_spatial_from_pos_embeds(
            state_dict, resolved_spatial_dims, patch_size,
            hdit_input_size, depth_size, model_type='hdit',
        )

        base_model = create_hdit(
            variant=variant,
            spatial_dims=resolved_spatial_dims,
            input_size=hdit_input_size,
            patch_size=patch_size,
            in_channels=hdit_in_channels,
            out_channels=resolved_out_channels,
            conditioning=conditioning,
            cond_channels=cond_channels,
            level_depths=level_depths,
            depth_size=depth_size,
        )
        logger.info(
            f"Using HDiT-{variant} from checkpoint: input_size={hdit_input_size}, "
            f"depth_size={depth_size}, patch_size={patch_size}, in_ch={hdit_in_channels}, "
            f"cond_ch={cond_channels}, level_depths={level_depths}"
        )
        model: nn.Module = base_model

    elif model_type in ('uvit',):
        from medgen.models.uvit import create_uvit

        variant = arch_config.get('variant')
        patch_size = arch_config.get('patch_size', 2)
        conditioning = arch_config.get('conditioning', 'concat')

        if conditioning == 'concat':
            uvit_in_channels = resolved_in_channels
        else:
            uvit_in_channels = resolved_out_channels
        cond_channels = max(0, resolved_in_channels - resolved_out_channels)

        # Infer variant from hidden_size in state_dict
        if variant is None:
            variant = _infer_variant_from_state_dict(state_dict, 'uvit')

        # Infer input_size and depth_size from pos_embed
        uvit_input_size = arch_config.get('image_size', 128)
        depth_size = arch_config.get('depth_size')
        uvit_input_size, depth_size = _infer_spatial_from_pos_embeds(
            state_dict, resolved_spatial_dims, patch_size,
            uvit_input_size, depth_size, model_type='uvit',
        )

        base_model = create_uvit(
            variant=variant,
            spatial_dims=resolved_spatial_dims,
            input_size=uvit_input_size,
            patch_size=patch_size,
            in_channels=uvit_in_channels,
            out_channels=resolved_out_channels,
            conditioning=conditioning,
            cond_channels=cond_channels,
            depth_size=depth_size,
        )
        logger.info(
            f"Using UViT-{variant} from checkpoint: input_size={uvit_input_size}, "
            f"depth_size={depth_size}, patch_size={patch_size}, in_ch={uvit_in_channels}, "
            f"cond_ch={cond_channels}"
        )
        model = base_model

    elif model_type in ('dit', 'sit'):
        # Create DiT/SiT model from checkpoint config
        from medgen.models.dit import create_dit

        variant = arch_config.get('variant', 'S')
        patch_size = arch_config.get('patch_size', 2)
        conditioning = arch_config.get('conditioning', 'concat')
        qk_norm = arch_config.get('qk_norm', True)

        # For concat conditioning: in_channels is the FULL input (noise + conditioning)
        # This matches how the factory creates DiT during training (factory.py line 124)
        if conditioning == 'concat':
            dit_in_channels = resolved_in_channels
        else:
            dit_in_channels = resolved_out_channels
        cond_channels = max(0, resolved_in_channels - resolved_out_channels)

        # Infer input_size and depth_size from pos_embed shape in state dict.
        # The checkpoint config may store the Hydra default (e.g. 32 for dit_3d)
        # rather than the actual computed value.
        # pos_embed shape: [1, num_tokens, hidden_size]
        # num_tokens = prod(input_dims / patch_size)
        dit_input_size = arch_config.get('image_size', 32)
        depth_size = arch_config.get('depth_size')
        if 'pos_embed' in state_dict:
            num_tokens = state_dict['pos_embed'].shape[1]
            # For 3D: tokens = (D/p) * (H/p) * (W/p), assume H=W=input_size
            # For 2D: tokens = (H/p) * (W/p)
            if resolved_spatial_dims == 3:
                # Solve: (depth_size/p) * (input_size/p)^2 = num_tokens
                # Find (input_size, depth_size) where depth is a reasonable
                # fraction of spatial size (medical volumes: depth ~ 0.3-0.8 * spatial)
                # Prefer the solution closest to depth/spatial ratio of ~0.5
                best_candidate = None
                best_ratio_diff = float('inf')
                for candidate_s in [8, 16, 32, 64, 128, 256]:
                    s_tokens = candidate_s // patch_size
                    if s_tokens <= 0:
                        continue
                    remaining = num_tokens / (s_tokens * s_tokens)
                    if remaining == int(remaining) and remaining > 0:
                        candidate_d = int(remaining) * patch_size
                        if candidate_d <= 0:
                            continue
                        # Target ratio ~1.0 (depth ≈ spatial); medical
                        # volumes can have depth > spatial (e.g. 160 depth, 128 spatial)
                        ratio_diff = abs(candidate_d / candidate_s - 1.0)
                        if ratio_diff < best_ratio_diff:
                            best_ratio_diff = ratio_diff
                            best_candidate = (candidate_s, candidate_d)
                found = best_candidate is not None
                if found:
                    dit_input_size, depth_size = best_candidate
                if not found:
                    logger.warning(
                        f"Could not infer DiT input_size from pos_embed "
                        f"(num_tokens={num_tokens}, patch_size={patch_size})"
                    )
            else:
                # 2D: tokens = (H/p) * (W/p), assume square
                tokens_per_side = int(num_tokens ** 0.5)
                if tokens_per_side * tokens_per_side == num_tokens:
                    dit_input_size = tokens_per_side * patch_size

        base_model = create_dit(
            variant=variant,
            spatial_dims=resolved_spatial_dims,
            input_size=dit_input_size,
            patch_size=patch_size,
            in_channels=dit_in_channels,
            out_channels=resolved_out_channels,
            conditioning=conditioning,
            cond_channels=cond_channels,
            depth_size=depth_size,
            qk_norm=qk_norm,
        )
        logger.info(
            f"Using DiT-{variant} from checkpoint: input_size={dit_input_size}, "
            f"depth_size={depth_size}, patch_size={patch_size}, in_ch={dit_in_channels}, "
            f"cond_ch={cond_channels}, spatial_dims={resolved_spatial_dims}"
        )

        # DiT models are always 'raw' (no wrapper support)
        model: nn.Module = base_model
    else:
        # Create UNet model
        # Log architecture info
        if 'channels' in arch_config:
            logger.info(f"Using architecture from checkpoint: channels={channels}, norm_num_groups={norm_num_groups}")
        else:
            logger.info(f"Using default architecture: channels={channels}")

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
            size_bin_cfg = config.get('size_bin', {})
            if size_bin_cfg:
                num_bins = size_bin_cfg.get('num_bins', 7)
                max_count = size_bin_cfg.get('max_count', 10)
                per_bin_embed_dim = size_bin_cfg.get('embed_dim', 32)
                projection_hidden_dim = size_bin_cfg.get('projection_hidden_dim', 0)
                projection_num_layers = size_bin_cfg.get('projection_num_layers', 2)
            else:
                num_bins, max_count, per_bin_embed_dim, projection_hidden_dim, projection_num_layers = (
                    _infer_size_bin_params(state_dict)
                )
            model = SizeBinModelWrapper(
                model=base_model,
                embed_dim=embed_dim,
                num_bins=num_bins,
                max_count=max_count,
                per_bin_embed_dim=per_bin_embed_dim,
                projection_hidden_dim=projection_hidden_dim,
                projection_num_layers=projection_num_layers,
            )
            logger.info(
                f"Applied SizeBinModelWrapper (num_bins={num_bins}, embed_dim={per_bin_embed_dim}, "
                f"projection_hidden_dim={projection_hidden_dim}, projection_num_layers={projection_num_layers})"
            )
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


def _detect_model_arch_from_state_dict(state_dict: dict[str, Any]) -> str:
    """Detect model architecture from state dict key patterns.

    Returns:
        'hdit', 'uvit', 'dit', 'sit', or 'unet'.
    """
    keys = set(state_dict.keys())

    # HDiT: has encoder_levels, decoder_levels, mergers, splitters
    if any(k.startswith('encoder_levels.') for k in keys):
        return 'hdit'

    # UViT: has in_blocks, out_blocks, mid_block, decoder_pred
    if any(k.startswith('in_blocks.') for k in keys):
        return 'uvit'

    # DiT/SiT: has blocks.N.attn and final_layer.adaLN_modulation
    if any(k.startswith('blocks.') for k in keys) and any('final_layer' in k for k in keys):
        return 'dit'

    return 'unet'


def _infer_variant_from_state_dict(state_dict: dict[str, Any], model_type: str) -> str:
    """Infer model variant (S/B/L/XL) from hidden_size in state dict."""
    from medgen.models.dit import DIT_VARIANTS

    # Find hidden_size from a known weight shape
    hidden_size = None
    if model_type == 'hdit':
        # encoder_levels.0.0.attn.qkv.weight shape: [3*hidden, hidden]
        for k, v in state_dict.items():
            if 'encoder_levels.0.0.mlp.fc1.weight' in k:
                hidden_size = v.shape[1]
                break
    elif model_type == 'uvit':
        from medgen.models.uvit import UVIT_VARIANTS
        for k, v in state_dict.items():
            if 'in_blocks.0.mlp.fc1.weight' in k:
                hidden_size = v.shape[1]
                break
        if hidden_size is not None:
            for name, cfg in UVIT_VARIANTS.items():
                if cfg['hidden_size'] == hidden_size:
                    return name
        return 'S'
    elif model_type in ('dit', 'sit'):
        for k, v in state_dict.items():
            if 'blocks.0.mlp.fc1.weight' in k:
                hidden_size = v.shape[1]
                break

    if hidden_size is not None:
        for name, cfg in DIT_VARIANTS.items():
            if cfg['hidden_size'] == hidden_size:
                return name

    return 'S'


def _infer_hdit_level_depths(state_dict: dict[str, Any]) -> list[int]:
    """Infer HDiT level_depths from encoder_levels/mid_blocks/decoder_levels keys."""
    encoder_levels: dict[int, set[int]] = {}
    decoder_levels: dict[int, set[int]] = {}
    mid_blocks: set[int] = set()

    for key in state_dict:
        if key.startswith('encoder_levels.'):
            parts = key.split('.')
            level, block = int(parts[1]), int(parts[2])
            encoder_levels.setdefault(level, set()).add(block)
        elif key.startswith('decoder_levels.'):
            parts = key.split('.')
            level, block = int(parts[1]), int(parts[2])
            decoder_levels.setdefault(level, set()).add(block)
        elif key.startswith('mid_blocks.'):
            parts = key.split('.')
            mid_blocks.add(int(parts[1]))

    depths = []
    # Encoder levels (0, 1, ...)
    for level in sorted(encoder_levels):
        depths.append(len(encoder_levels[level]))
    # Mid blocks
    depths.append(len(mid_blocks))
    # Decoder levels (0, 1, ...) - reverse order to match level_depths convention
    for level in sorted(decoder_levels):
        depths.append(len(decoder_levels[level]))

    return depths if depths else [2, 4, 6, 4, 2]


def _infer_spatial_from_pos_embeds(
    state_dict: dict[str, Any],
    spatial_dims: int,
    patch_size: int,
    default_input_size: int,
    default_depth_size: int | None,
    model_type: str = 'dit',
) -> tuple[int, int | None]:
    """Infer input_size and depth_size from positional embedding shapes."""
    # HDiT has multiple pos_embeds (one per level)
    # UViT/DiT have a single pos_embed
    pos_embed_key = None
    if model_type == 'hdit':
        pos_embed_key = 'pos_embeds.0'
    else:
        pos_embed_key = 'pos_embed'

    if pos_embed_key not in state_dict:
        return default_input_size, default_depth_size

    # pos_embed shape: [1, num_tokens, hidden_size] or [num_tokens, hidden_size]
    pos_embed = state_dict[pos_embed_key]
    if pos_embed.ndim == 3:
        num_tokens = pos_embed.shape[1]
    elif pos_embed.ndim == 2:
        num_tokens = pos_embed.shape[0]
    else:
        return default_input_size, default_depth_size

    # For UViT, first token is the time token
    if model_type == 'uvit':
        num_tokens -= 1

    input_size = default_input_size
    depth_size = default_depth_size

    if spatial_dims == 3:
        # Solve: (depth/p) * (size/p)^2 = num_tokens (for level 0 / full resolution)
        best_candidate = None
        best_ratio_diff = float('inf')
        for candidate_s in [8, 16, 32, 64, 128, 256]:
            s_tokens = candidate_s // patch_size
            if s_tokens <= 0:
                continue
            remaining = num_tokens / (s_tokens * s_tokens)
            if remaining == int(remaining) and remaining > 0:
                candidate_d = int(remaining) * patch_size
                if candidate_d <= 0:
                    continue
                # Target ratio ~1.0 (depth ≈ spatial); medical
                # volumes can have depth > spatial (e.g. 160 depth, 128 spatial)
                ratio_diff = abs(candidate_d / candidate_s - 1.0)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_candidate = (candidate_s, candidate_d)
        if best_candidate is not None:
            input_size, depth_size = best_candidate
    else:
        tokens_per_side = int(num_tokens ** 0.5)
        if tokens_per_side * tokens_per_side == num_tokens:
            input_size = tokens_per_side * patch_size

    return input_size, depth_size


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
        # UNet: conv_in.conv.weight shape: [features, in_channels, *kernel_size]
        if key.endswith('conv_in.conv.weight'):
            in_channels = tensor.shape[1]
        # UNet: out.2.conv.weight shape: [out_channels, features, *kernel_size]
        if key.endswith('out.2.conv.weight'):
            out_channels = tensor.shape[0]
        # DiT/HDiT/UViT: x_embedder.proj.weight shape: [hidden, in_channels, *patch]
        if key.endswith('x_embedder.proj.weight') and in_channels is None:
            in_channels = tensor.shape[1]
        # UViT: final_conv.weight shape: [out_channels, out_channels, *kernel]
        if key.endswith('final_conv.weight') and out_channels is None:
            out_channels = tensor.shape[0]

    return in_channels, out_channels


def _infer_size_bin_params(
    state_dict: dict[str, Any],
) -> tuple[int, int, int, int, int]:
    """Infer SizeBinModelWrapper hyperparameters from state dict shapes.

    For legacy checkpoints that don't store size_bin config explicitly.

    Returns:
        (num_bins, max_count, per_bin_embed_dim, projection_hidden_dim, projection_num_layers)
    """
    # Defaults
    num_bins = 7
    max_count = 10
    per_bin_embed_dim = 32
    projection_hidden_dim = 0
    projection_num_layers = 2

    # Infer from bin_embeddings: shape [max_count+1, per_bin_embed_dim]
    bin_embed_keys = sorted(
        k for k in state_dict if 'bin_embeddings' in k and k.endswith('.weight')
    )
    if bin_embed_keys:
        # Count unique bin indices to get num_bins
        # Keys like: model.time_embed.bin_embeddings.0.weight
        bin_indices = set()
        for k in bin_embed_keys:
            parts = k.split('bin_embeddings.')[-1].split('.')
            bin_indices.add(int(parts[0]))
        # Each SizeBinTimeEmbed has num_bins embeddings, but there are two copies
        # (model.time_embed and size_bin_time_embed), so divide by 2
        num_bins = len(bin_indices)

        # Get per_bin_embed_dim and max_count from first embedding weight
        first_weight = state_dict[bin_embed_keys[0]]
        max_count = first_weight.shape[0] - 1  # vocab_size = max_count + 1
        per_bin_embed_dim = first_weight.shape[1]

    # Infer projection architecture from projection layer keys
    # Keys like: model.time_embed.projection.0.weight, .2.weight, .4.weight, etc.
    # Linear layers in Sequential are at even indices (odd = SiLU activations)
    proj_keys = sorted(
        k for k in state_dict
        if 'model.time_embed.projection.' in k and k.endswith('.weight')
    )
    if len(proj_keys) > 2:
        # More than 2 linear layers = deep MLP (create_deep_zero_init_mlp)
        # Total linear layers = num_hidden_layers + 1 (output layer)
        # So: num_hidden_layers = total_linear_layers - 1
        projection_num_layers = len(proj_keys) - 1
        # Hidden dim from first layer output
        projection_hidden_dim = state_dict[proj_keys[0]].shape[0]
    elif len(proj_keys) == 2:
        # Legacy 2-layer MLP (create_zero_init_mlp)
        projection_hidden_dim = 0
        projection_num_layers = 2

    logger.info(
        f"Inferred size_bin params from state_dict: num_bins={num_bins}, "
        f"max_count={max_count}, embed_dim={per_bin_embed_dim}, "
        f"projection_hidden_dim={projection_hidden_dim}, "
        f"projection_num_layers={projection_num_layers}"
    )
    return num_bins, max_count, per_bin_embed_dim, projection_hidden_dim, projection_num_layers


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
