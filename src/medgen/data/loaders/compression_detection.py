"""Compression model detection and loading utilities.

Provides functions to:
- Detect compression type (VAE, VQ-VAE, DC-AE) from checkpoint
- Detect spatial dimensions (2D or 3D) from checkpoint
- Detect scale factor and latent channels
- Load compression models with auto-detection

Performance: Uses load-once pattern to avoid redundant checkpoint loading.
"""

import logging
import os
import re
from typing import Any

import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _validate_checkpoint_path(checkpoint_path: str) -> None:
    """Validate checkpoint path exists and is readable.

    Args:
        checkpoint_path: Path to compression checkpoint.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        PermissionError: If checkpoint file is not readable.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.access(checkpoint_path, os.R_OK):
        raise PermissionError(f"Cannot read checkpoint: {checkpoint_path}")


def _load_checkpoint_dict(checkpoint_path: str, device: str = 'cpu') -> dict:
    """Load checkpoint once and return the dict.

    Internal helper to avoid redundant file I/O.

    Args:
        checkpoint_path: Path to compression checkpoint.
        device: Device to map tensors to.

    Returns:
        Loaded checkpoint dictionary.
    """
    _validate_checkpoint_path(checkpoint_path)
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


# =============================================================================
# Internal detection functions (operate on already-loaded checkpoint dict)
# =============================================================================


def _detect_compression_type_from_dict(checkpoint: dict) -> str:
    """Detect compression model type from already-loaded checkpoint.

    Args:
        checkpoint: Loaded checkpoint dictionary.

    Returns:
        Compression type: 'vae', 'dcae', or 'vqvae'.
    """
    # Check for config in checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        # DC-AE checkpoints have dc_ae or dcae in config
        if 'dc_ae' in config or 'dcae' in config:
            return 'dcae'
        # VQ-VAE has num_embeddings
        if 'num_embeddings' in config or 'vqvae' in config:
            return 'vqvae'

    # Check model state dict keys
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

    # VQ-VAE has quantize layer
    if any('quantize' in k or 'embedding' in k for k in state_dict):
        return 'vqvae'

    # DC-AE has specific layer patterns
    if any('residual_autoencoding' in k.lower() for k in state_dict):
        return 'dcae'

    # Default to VAE
    return 'vae'


def _detect_spatial_dims_from_dict(checkpoint: dict) -> int:
    """Detect spatial dimensions (2D or 3D) from already-loaded checkpoint.

    Args:
        checkpoint: Loaded checkpoint dictionary.

    Returns:
        Spatial dimensions: 2 or 3.
    """
    # Check for spatial_dims in config
    if 'config' in checkpoint:
        config = checkpoint['config']
        if 'spatial_dims' in config:
            return config['spatial_dims']
        # Check nested configs
        for key in ['vae', 'vqvae', 'dcae', 'vae_3d', 'vqvae_3d', 'dcae_3d']:
            if key in config and isinstance(config[key], dict):
                if 'spatial_dims' in config[key]:
                    return config[key]['spatial_dims']

    # Check model state dict for 3D patterns (conv3d weights have 5D shape)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

    # Look for encoder conv layers - 3D convs have 5D weights [out, in, D, H, W]
    for key, value in state_dict.items():
        if 'conv' in key.lower() and 'weight' in key:
            if isinstance(value, torch.Tensor) and value.dim() == 5:
                logger.info(f"Detected 3D compression model from weight shape {value.shape}")
                return 3

    # Default to 2D
    return 2


def _detect_scale_factor_from_dict(checkpoint: dict, compression_type: str) -> int:
    """Detect spatial scale factor from already-loaded checkpoint.

    Args:
        checkpoint: Loaded checkpoint dictionary.
        compression_type: Type of model ('vae', 'dcae', 'vqvae').

    Returns:
        Spatial scale factor (8 for VAE/VQ-VAE, 32/64 for DC-AE).
    """
    config = checkpoint.get('config', {})

    # DC-AE: check for spatial_compression_ratio or f{N} naming
    if compression_type == 'dcae' or (compression_type == 'auto' and 'dcae' in str(config).lower()):
        # Explicit spatial_compression_ratio
        if 'spatial_compression_ratio' in config:
            return config['spatial_compression_ratio']
        # Check nested dcae config
        if 'dcae' in config and isinstance(config['dcae'], dict):
            if 'spatial_compression_ratio' in config['dcae']:
                return config['dcae']['spatial_compression_ratio']
        # Check for f{N} naming pattern (e.g., 'dc-ae-f32c32')
        if 'name' in config:
            match = re.search(r'f(\d+)', str(config['name']))
            if match:
                return int(match.group(1))
        # DC-AE default
        return 32

    # VAE/VQ-VAE: count downsampling stages or use channels length
    if 'channels' in config:
        num_stages = len(config['channels'])
        return 2 ** num_stages  # Typically 2^3 = 8

    # Fallback: count downsample blocks in encoder state_dict
    # Downsample blocks are identified by postconv or standalone conv patterns
    # between ResBlock groups in the encoder
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
    downsample_count = sum(
        1 for k in state_dict
        if k.startswith(('encoder.blocks.', 'model.encoder.blocks.'))
        and k.endswith('.postconv.conv.weight')
    )
    if downsample_count > 0:
        # scale = 2^num_downsamples, +1 for the total levels
        return 2 ** (downsample_count + 1)

    # Default for VAE/VQ-VAE
    return 8


def _detect_latent_channels_from_dict(checkpoint: dict, compression_type: str) -> int:
    """Detect latent channels from already-loaded checkpoint.

    Args:
        checkpoint: Loaded checkpoint dictionary.
        compression_type: Type of model ('vae', 'dcae', 'vqvae').

    Returns:
        Number of latent channels.
    """
    config = checkpoint.get('config', {})

    # Check common attribute names
    if 'latent_channels' in config:
        return config['latent_channels']
    if 'z_channels' in config:
        return config['z_channels']
    if 'embedding_dim' in config:  # VQ-VAE
        return config['embedding_dim']

    # Check nested configs
    for key in ['vae', 'vqvae', 'dcae']:
        if key in config and isinstance(config[key], dict):
            nested = config[key]
            if 'latent_channels' in nested:
                return nested['latent_channels']
            if 'z_channels' in nested:
                return nested['z_channels']

    # Infer from state_dict weights (fallback for DC-AE)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
    # DC-AE encoder.conv_out.weight has shape [latent_channels, hidden, k, k]
    if 'encoder.conv_out.weight' in state_dict:
        return state_dict['encoder.conv_out.weight'].shape[0]
    # With 'model.' prefix
    if 'model.encoder.conv_out.weight' in state_dict:
        return state_dict['model.encoder.conv_out.weight'].shape[0]
    # VAE quant_conv has shape [2*latent_channels, hidden, k, k] (mu + logvar)
    if 'quant_conv.weight' in state_dict:
        return state_dict['quant_conv.weight'].shape[0] // 2
    # MONAI AutoencoderKL: separate mu/logvar convs
    for prefix in ('', 'model.'):
        key = f'{prefix}quant_conv_mu.conv.weight'
        if key in state_dict:
            return state_dict[key].shape[0]

    # Default
    return 4


def _infer_vae_config_from_state_dict(state_dict: dict) -> dict:
    """Infer VAE architecture from state_dict when checkpoint config is missing.

    Parses MONAI AutoencoderKL weight shapes to reconstruct the constructor args.
    This handles old checkpoints that don't have an embedded 'config' key.

    MONAI AutoencoderKL encoder structure (flattened into blocks[]):
      [0]  conv_in
      For each level (except last):
        [..] num_res_blocks × ResBlock
        [..] optional Attention
        [..] Downsample (conv.conv.weight)
      Last level:
        [..] num_res_blocks × ResBlock
        [..] optional Attention
      Mid block:
        [..] ResBlock, Attention, ResBlock  (always present)
      [..]  GroupNorm
      [..]  conv_out

    Args:
        state_dict: Model state dict (already stripped of 'model.' prefix).

    Returns:
        Dict with keys matching AutoencoderKL constructor args.
    """
    # --- latent_channels: from quant_conv_mu [latent_ch, latent_ch, 1, ...] ---
    latent_channels = state_dict['quant_conv_mu.conv.weight'].shape[0]

    # --- spatial_dims: from conv weight ndim (5D = 3D, 4D = 2D) ---
    spatial_dims = state_dict['quant_conv_mu.conv.weight'].ndim - 2

    # --- in_channels: from encoder conv_in [first_ch, in_ch, k, ...] ---
    in_channels = state_dict['encoder.blocks.0.conv.weight'].shape[1]

    # --- Classify each encoder block by its key pattern ---
    block_types: dict[int, str] = {}  # idx -> 'resblock'|'downsample'|'attention'|'other'
    resblock_out_ch: dict[int, int] = {}  # idx -> output channels

    encoder_keys: dict[int, list[str]] = {}
    for key in state_dict:
        if not key.startswith('encoder.blocks.'):
            continue
        parts = key.split('.')
        idx = int(parts[2])
        rest = '.'.join(parts[3:])
        encoder_keys.setdefault(idx, []).append(rest)

    for idx, keys in encoder_keys.items():
        if any('conv1' in k for k in keys):
            block_types[idx] = 'resblock'
            resblock_out_ch[idx] = state_dict[f'encoder.blocks.{idx}.conv2.conv.weight'].shape[0]
        elif any('attn' in k for k in keys):
            block_types[idx] = 'attention'
        elif 'conv.conv.weight' in keys:
            block_types[idx] = 'downsample'
        else:
            block_types[idx] = 'other'  # conv_in, GroupNorm, conv_out

    # --- Split into levels using Downsample blocks as boundaries ---
    # Collect ResBlock indices between downsample boundaries
    sorted_indices = sorted(block_types.keys())
    # Skip block 0 (conv_in)
    resblock_indices = [i for i in sorted_indices if block_types[i] == 'resblock']
    downsample_indices = sorted(i for i in sorted_indices if block_types[i] == 'downsample')

    # Number of downsamples = number of levels - 1
    num_levels = len(downsample_indices) + 1

    # Group ResBlocks into levels (before each downsample = one level)
    levels: list[list[int]] = []
    remaining_resblocks = list(resblock_indices)

    for ds_idx in downsample_indices:
        level_resblocks = [i for i in remaining_resblocks if i < ds_idx]
        levels.append(level_resblocks)
        remaining_resblocks = [i for i in remaining_resblocks if i >= ds_idx]

    # Remaining ResBlocks: last level + mid block (2 extra ResBlocks)
    # The last level has num_res_blocks, mid block has 2
    # num_res_blocks = count from first level (most reliable)
    num_res_blocks = len(levels[0]) if levels else 2
    last_level_resblocks = remaining_resblocks[:num_res_blocks]
    levels.append(last_level_resblocks)
    # (remaining after that are mid block ResBlocks)

    # Extract channel per level from the last ResBlock in each level
    channels = []
    for level_blocks in levels:
        if level_blocks:
            last_rb = level_blocks[-1]
            channels.append(resblock_out_ch[last_rb])

    # --- Detect attention per level (excluding mid block attention) ---
    # Mid block always has attention; we only care about per-level attention
    attention_indices = sorted(i for i in sorted_indices if block_types[i] == 'attention')

    # Find mid block attention: the attention that comes after the last downsample
    # and after the last level's ResBlocks
    last_level_end = levels[-1][-1] if levels[-1] else 0
    mid_attn = {i for i in attention_indices if i > last_level_end}

    # Per-level attention: check if any attention block falls within each level's range
    attention_levels = []
    for level_idx, level_blocks in enumerate(levels):
        if not level_blocks:
            attention_levels.append(False)
            continue
        level_start = level_blocks[0]
        # Level ends at the downsample (or at end for last level)
        if level_idx < len(downsample_indices):
            level_end = downsample_indices[level_idx]
        else:
            level_end = last_level_end + 1
        has_attn = any(
            level_start < ai < level_end and ai not in mid_attn
            for ai in attention_indices
        )
        attention_levels.append(has_attn)

    config = {
        'in_channels': in_channels,
        'out_channels': in_channels,
        'latent_channels': latent_channels,
        'channels': channels,
        'attention_levels': attention_levels,
        'num_res_blocks': num_res_blocks,
        'norm_num_groups': 32,
    }
    if spatial_dims == 3:
        config['spatial_dims'] = 3
    logger.info(f"Inferred VAE config from state_dict: channels={channels}, "
                f"latent_channels={latent_channels}, attention={attention_levels}")
    return config


# =============================================================================
# Public API functions (backward-compatible, load checkpoint internally)
# =============================================================================


def detect_compression_type(checkpoint_path: str) -> str:
    """Detect compression model type from checkpoint.

    Args:
        checkpoint_path: Path to compression checkpoint.

    Returns:
        Compression type: 'vae', 'dcae', or 'vqvae'.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    checkpoint = _load_checkpoint_dict(checkpoint_path)
    return _detect_compression_type_from_dict(checkpoint)


def detect_spatial_dims(checkpoint_path: str) -> int:
    """Detect spatial dimensions (2D or 3D) from checkpoint.

    Args:
        checkpoint_path: Path to compression checkpoint.

    Returns:
        Spatial dimensions: 2 or 3.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    checkpoint = _load_checkpoint_dict(checkpoint_path)
    return _detect_spatial_dims_from_dict(checkpoint)


def detect_scale_factor(checkpoint_path: str, compression_type: str = 'auto') -> int:
    """Detect spatial scale factor from compression checkpoint.

    Args:
        checkpoint_path: Path to compression checkpoint.
        compression_type: Type of model ('auto', 'vae', 'dcae', 'vqvae').

    Returns:
        Spatial scale factor (8 for VAE/VQ-VAE, 32/64 for DC-AE).

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    checkpoint = _load_checkpoint_dict(checkpoint_path)
    if compression_type == 'auto':
        compression_type = _detect_compression_type_from_dict(checkpoint)
    return _detect_scale_factor_from_dict(checkpoint, compression_type)


def detect_latent_channels(checkpoint_path: str, compression_type: str = 'auto') -> int:
    """Detect latent channels from compression checkpoint.

    Args:
        checkpoint_path: Path to compression checkpoint.
        compression_type: Type of model ('auto', 'vae', 'dcae', 'vqvae').

    Returns:
        Number of latent channels.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    checkpoint = _load_checkpoint_dict(checkpoint_path)
    if compression_type == 'auto':
        compression_type = _detect_compression_type_from_dict(checkpoint)
    return _detect_latent_channels_from_dict(checkpoint, compression_type)


def load_compression_model(
    checkpoint_path: str,
    compression_type: str,
    device: torch.device,
    cfg: DictConfig | None = None,
    spatial_dims: Any = 'auto',
) -> tuple[torch.nn.Module, str, int, int, int]:
    """Load compression model from checkpoint.

    Uses load-once pattern: checkpoint is loaded once and all detection
    functions operate on the same dict.

    Args:
        checkpoint_path: Path to compression checkpoint.
        compression_type: Type of model ('auto', 'vae', 'dcae', 'vqvae').
        device: Device to load model to.
        cfg: Optional config for model architecture.
        spatial_dims: Spatial dimensions ('auto', 2, or 3).

    Returns:
        Tuple of (model, detected_type, spatial_dims, scale_factor, latent_channels).

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        ValueError: If compression type is unknown.
    """
    # Load checkpoint once
    checkpoint = _load_checkpoint_dict(checkpoint_path, device=str(device))

    # Auto-detect type if needed (from already-loaded checkpoint)
    if compression_type == 'auto':
        compression_type = _detect_compression_type_from_dict(checkpoint)
        logger.info(f"Auto-detected compression type: {compression_type}")

    # Auto-detect spatial dimensions if needed (from already-loaded checkpoint)
    if spatial_dims == 'auto':
        spatial_dims = _detect_spatial_dims_from_dict(checkpoint)
        logger.info(f"Auto-detected spatial dimensions: {spatial_dims}D")
    else:
        spatial_dims = int(spatial_dims)

    # Detect scale factor and latent channels (from already-loaded checkpoint)
    scale_factor = _detect_scale_factor_from_dict(checkpoint, compression_type)
    latent_channels = _detect_latent_channels_from_dict(checkpoint, compression_type)

    model_config = checkpoint.get('config', {})

    # Get state_dict early (needed for both config inference and weight loading)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    keys_with_prefix = [k for k in state_dict if k.startswith('model.')]
    if keys_with_prefix:
        state_dict = {
            k.replace('model.', '', 1) if k.startswith('model.') else k: v
            for k, v in state_dict.items()
        }
        logger.debug(f"Stripped 'model.' prefix from {len(keys_with_prefix)} state_dict keys")

    if compression_type == 'vae':
        # Infer config from state_dict if checkpoint config is missing
        if not model_config and 'quant_conv_mu.conv.weight' in state_dict:
            model_config = _infer_vae_config_from_state_dict(state_dict)

        from monai.networks.nets import AutoencoderKL

        model = AutoencoderKL(
            spatial_dims=spatial_dims,
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 1),
            channels=tuple(model_config.get('channels', [64, 128, 256])),
            attention_levels=tuple(model_config.get('attention_levels', [False, False, True])),
            latent_channels=model_config.get('latent_channels', 4),
            num_res_blocks=model_config.get('num_res_blocks', 2),
            norm_num_groups=model_config.get('norm_num_groups', 32),
        ).to(device)

    elif compression_type == 'vqvae':
        from monai.networks.nets import VQVAE

        # Get channels to compute default downsample/upsample parameters
        channels = tuple(model_config.get('channels', [64, 128, 256]))
        n_levels = len(channels)

        # Default downsample/upsample parameters if not in config
        # Format: (kernel_size, stride, padding, output_padding) for each level
        default_downsample = tuple((4, 2, 1) for _ in range(n_levels))
        default_upsample = tuple((4, 2, 1, 0) for _ in range(n_levels))

        # Get parameters from config, converting lists to tuples
        downsample_params = model_config.get('downsample_parameters', default_downsample)
        upsample_params = model_config.get('upsample_parameters', default_upsample)

        # Ensure they are tuples of tuples
        if isinstance(downsample_params, list):
            downsample_params = tuple(tuple(p) for p in downsample_params)
        if isinstance(upsample_params, list):
            upsample_params = tuple(tuple(p) for p in upsample_params)

        model = VQVAE(
            spatial_dims=spatial_dims,
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 1),
            channels=channels,
            num_res_layers=model_config.get('num_res_layers', 2),
            num_res_channels=tuple(model_config.get('num_res_channels', list(channels))),
            downsample_parameters=downsample_params,
            upsample_parameters=upsample_params,
            num_embeddings=model_config.get('num_embeddings', 512),
            embedding_dim=model_config.get('embedding_dim', 3),
            commitment_cost=model_config.get('commitment_cost', 0.25),
            decay=model_config.get('decay', 0.99),
            epsilon=model_config.get('epsilon', 1e-5),
        ).to(device)

    elif compression_type == 'dcae':
        # DC-AE model loading
        if spatial_dims == 3:
            # 3D DC-AE: use our custom implementation
            from medgen.models.autoencoder_dc_3d import AutoencoderDC3D

            model = AutoencoderDC3D(
                in_channels=model_config.get('in_channels', 1),
                latent_channels=model_config.get('latent_channels', 32),
                encoder_block_out_channels=tuple(model_config.get('encoder_block_out_channels', [128, 256, 512, 512])),
                decoder_block_out_channels=tuple(model_config.get('decoder_block_out_channels', [512, 512, 256, 128])),
                encoder_layers_per_block=tuple(model_config.get('encoder_layers_per_block', [2, 2, 2, 2])),
                decoder_layers_per_block=tuple(model_config.get('decoder_layers_per_block', [2, 2, 2, 2])),
                depth_factors=tuple(model_config.get('depth_factors', [1, 2, 2, 2])),
                encoder_out_shortcut=model_config.get('encoder_out_shortcut', True),
                decoder_in_shortcut=model_config.get('decoder_in_shortcut', True),
                scaling_factor=model_config.get('scaling_factor', 1.0),
            ).to(device)
        else:
            # 2D DC-AE: use diffusers AutoencoderDC
            from diffusers import AutoencoderDC

            pretrained = model_config.get('pretrained')
            if pretrained:
                # Load from HuggingFace pretrained weights
                model = AutoencoderDC.from_pretrained(pretrained, torch_dtype=torch.float32)

                # Modify input channels if needed (grayscale)
                in_channels = model_config.get('in_channels', 1)
                if model.encoder.conv_in.in_channels != in_channels:
                    old_conv = model.encoder.conv_in
                    new_conv = torch.nn.Conv2d(
                        in_channels, old_conv.out_channels,
                        kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                        padding=old_conv.padding, bias=old_conv.bias is not None
                    )
                    model.encoder.conv_in = new_conv

                    # Also modify decoder output
                    old_conv_out = model.decoder.conv_out.conv
                    new_conv_out = torch.nn.Conv2d(
                        old_conv_out.in_channels, in_channels,
                        kernel_size=old_conv_out.kernel_size, stride=old_conv_out.stride,
                        padding=old_conv_out.padding, bias=old_conv_out.bias is not None
                    )
                    model.decoder.conv_out.conv = new_conv_out

                model = model.to(device)
            else:
                # From-scratch DC-AE: recreate architecture
                # Architecture is same for all variants (f32, f64, f128), only latent_channels differs
                in_channels = model_config.get('in_channels', 1)

                model = AutoencoderDC(
                    in_channels=in_channels,
                    latent_channels=latent_channels,  # Use detected value from checkpoint
                    encoder_block_out_channels=(128, 256, 512, 512, 1024, 1024),
                    decoder_block_out_channels=(128, 256, 512, 512, 1024, 1024),
                    encoder_layers_per_block=(2, 2, 2, 3, 3, 3),
                    decoder_layers_per_block=(3, 3, 3, 3, 3, 3),
                    encoder_qkv_multiscales=((), (), (), (5,), (5,), (5,)),
                    decoder_qkv_multiscales=((), (), (), (5,), (5,), (5,)),
                    encoder_block_types="ResBlock",
                    decoder_block_types="ResBlock",
                    downsample_block_type="pixel_unshuffle",
                    upsample_block_type="pixel_shuffle",
                    encoder_out_shortcut=True,
                    decoder_in_shortcut=True,
                    scaling_factor=model_config.get('scaling_factor', 1.0),
                ).to(device)

    else:
        raise ValueError(f"Unknown compression type: {compression_type}")

    # Load weights (state_dict already extracted and prefix-stripped above)
    model.load_state_dict(state_dict)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    logger.info(
        f"Loaded {compression_type} compression model ({spatial_dims}D) from {checkpoint_path} "
        f"[scale_factor={scale_factor}x, latent_channels={latent_channels}]"
    )

    return model, compression_type, spatial_dims, scale_factor, latent_channels
