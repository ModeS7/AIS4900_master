"""Sample generation for generation metrics.

Contains methods for generating samples with fixed conditioning,
including 2D batched generation and 3D streaming generation.

Moved from generation.py during file split.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import Dataset

from medgen.core.dict_utils import get_with_fallbacks

if TYPE_CHECKING:
    from .generation import GenerationMetrics

logger = logging.getLogger(__name__)


def set_fixed_conditioning(
    self_: GenerationMetrics,
    train_dataset: Dataset,
    num_masks: int = 500,
    seg_channel_idx: int = 1,
) -> None:
    """Load fixed conditioning masks from training dataset.

    Uses the same masks every epoch for reproducible comparisons.

    Args:
        self_: GenerationMetrics instance.
        train_dataset: Training dataset to sample from.
        num_masks: Number of masks to sample.
        seg_channel_idx: Channel index for segmentation mask.
    """
    logger.info(f"Sampling {num_masks} fixed conditioning masks...")

    masks = []
    gt_images = []
    size_bins_list = []  # For seg_conditioned mode
    bin_maps_list = []  # For seg_conditioned_input mode
    attempts = 0
    max_attempts = len(train_dataset)
    samples_without_seg = 0  # Track samples missing seg data

    # Use fixed seed for reproducibility
    rng = torch.Generator()
    rng.manual_seed(42)

    while len(masks) < num_masks and attempts < max_attempts:
        idx = int(torch.randint(0, len(train_dataset), (1,), generator=rng).item())
        data = train_dataset[idx]

        # Handle dict, tuple, or tensor format
        is_seg_conditioned = False  # Track if this sample uses size_bins conditioning
        is_seg_conditioned_input = False  # Track if this sample uses input channel conditioning
        current_size_bins = None
        current_bin_maps = None
        if isinstance(data, dict):
            # Dict format - check multiple key variants
            # Latent dataset: {'latent': ..., 'seg_mask': ..., 'latent_seg': ...}
            # Pixel dataset: {'image': ..., 'seg': ...}

            # Check if this is a latent dataset (has 'latent' key)
            is_latent_data = 'latent' in data

            if is_latent_data:
                # Latent dataset: seg_mask is pixel-space, latent is compressed
                # Can't concatenate - use seg_mask directly for conditioning
                seg_data = data.get('seg_mask')
                if seg_data is None:
                    samples_without_seg += 1
                    attempts += 1
                    continue
                if isinstance(seg_data, np.ndarray):
                    seg_data = torch.from_numpy(seg_data).float()
                tensor = seg_data  # Use seg_mask directly
                local_seg_idx = 0  # Seg is at channel 0
            elif 'size_bins' in data:
                # seg_conditioned dict format: {'image': seg_mask, 'size_bins': ...}
                # The 'image' IS the seg mask (no separate seg key)
                is_seg_conditioned = True
                seg_data = get_with_fallbacks(data, 'image', 'images')
                if seg_data is None:
                    samples_without_seg += 1
                    attempts += 1
                    continue
                if isinstance(seg_data, np.ndarray):
                    seg_data = torch.from_numpy(seg_data).float()
                elif isinstance(seg_data, torch.Tensor):
                    seg_data = seg_data.float()
                tensor = seg_data
                local_seg_idx = 0
                size_bins_data = data['size_bins']
                if isinstance(size_bins_data, np.ndarray):
                    size_bins_data = torch.from_numpy(size_bins_data)
                current_size_bins = size_bins_data.long()
            else:
                # Pixel dataset: image and seg at same resolution
                image = get_with_fallbacks(data, 'image', 'images')
                seg_data = get_with_fallbacks(data, 'seg', 'mask', 'labels')

                if image is None or seg_data is None:
                    samples_without_seg += 1
                    attempts += 1
                    continue
                # Convert to tensors
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image).float()
                if isinstance(seg_data, np.ndarray):
                    seg_data = torch.from_numpy(seg_data).float()
                tensor = torch.cat([image, seg_data], dim=0)
                local_seg_idx = image.shape[0]  # seg follows image channels
        elif isinstance(data, tuple):
            # Handle 2-element (images, seg) or 3-element (seg, size_bins, bin_maps) tuples
            if len(data) == 2:
                first, second = data
                bin_maps = None
            elif len(data) == 3:
                first, second, bin_maps = data
            else:
                raise ValueError(f"Unexpected tuple length: {len(data)}")

            # Convert first element to tensor
            if isinstance(first, torch.Tensor):
                first = first.float()
            elif isinstance(first, np.ndarray):
                first = torch.from_numpy(first).float()
            else:
                first = torch.tensor(first).float()
            # Convert second element to tensor
            if isinstance(second, torch.Tensor):
                second = second.float()
            elif isinstance(second, np.ndarray):
                second = torch.from_numpy(second).float()
            else:
                second = torch.tensor(second).float()

            # seg_conditioned_input mode: (seg, size_bins, bin_maps)
            if bin_maps is not None and second.dim() == 1:
                is_seg_conditioned_input = True
                tensor = first
                current_size_bins = second.long()
                current_bin_maps = bin_maps.float() if isinstance(bin_maps, torch.Tensor) else torch.from_numpy(bin_maps).float()
                local_seg_idx = 0
            # Check if this is seg_conditioned mode: (seg, size_bins)
            # size_bins is 1D, seg is 3D [C, H, W] or 4D [C, D, H, W]
            elif second.dim() == 1 and first.dim() >= 3:
                is_seg_conditioned = True
                # seg_conditioned mode: first element is the seg mask
                tensor = first
                current_size_bins = second.long()  # Store size_bins
                # Override seg_channel_idx for this mode
                local_seg_idx = 0
            else:
                # Standard mode: (images, seg) - concatenate
                tensor = torch.cat([first, second], dim=0)
                local_seg_idx = seg_channel_idx
        else:
            if isinstance(data, torch.Tensor):
                tensor = data.float()
            elif isinstance(data, np.ndarray):
                tensor = torch.from_numpy(data).float()
            else:
                tensor = torch.tensor(data).float()
            local_seg_idx = seg_channel_idx

        # Extract seg mask - use ... to handle both 2D and 3D
        seg = tensor[local_seg_idx:local_seg_idx + 1, ...]

        if seg.sum() > 0:  # Has positive mask
            masks.append(seg)
            if local_seg_idx > 0:
                gt_images.append(tensor[0:local_seg_idx, ...])
            # Save size_bins for seg_conditioned mode
            if is_seg_conditioned and current_size_bins is not None:
                size_bins_list.append(current_size_bins)
            # Save bin_maps for seg_conditioned_input mode
            if is_seg_conditioned_input and current_bin_maps is not None:
                bin_maps_list.append(current_bin_maps)
        attempts += 1

    if len(masks) < num_masks:
        logger.warning(f"Only found {len(masks)} positive masks (requested {num_masks})")

    if len(masks) == 0:
        error_msg = (
            f"No positive masks found in dataset! "
            f"Checked {attempts} samples, {samples_without_seg} had no seg data. "
        )
        if samples_without_seg > 0:
            error_msg += (
                "This usually means the latent cache was built without regional_losses=true. "
                "Delete the cache directory and re-run with training.logging.regional_losses=true "
                "to rebuild it with segmentation masks."
            )
        raise RuntimeError(error_msg)

    self_.fixed_conditioning_masks = torch.stack(masks).to(self_.device)
    if gt_images:
        self_.fixed_gt_images = torch.stack(gt_images).to(self_.device)
    if size_bins_list:
        self_.fixed_size_bins = torch.stack(size_bins_list).to(self_.device)
        logger.info(f"Loaded {len(size_bins_list)} fixed size_bins for seg_conditioned mode")
    if bin_maps_list:
        self_.fixed_bin_maps = torch.stack(bin_maps_list).to(self_.device)
        logger.info(f"Loaded {len(bin_maps_list)} fixed bin_maps for seg_conditioned_input mode")

    logger.info(f"Loaded {len(masks)} fixed conditioning masks")


@torch.no_grad()
def generate_samples(
    self_: GenerationMetrics,
    model: nn.Module,
    strategy: Any,
    mode: Any,
    num_samples: int,
    num_steps: int,
    batch_size: int = 16,
) -> torch.Tensor:
    """Generate samples using fixed conditioning in batches.

    Args:
        self_: GenerationMetrics instance.
        model: Diffusion model.
        strategy: Diffusion strategy (DDPM/RFlow).
        mode: Training mode (bravo/dual/seg).
        num_samples: Number of samples to generate.
        num_steps: Number of denoising steps.
        batch_size: Batch size for generation (to avoid OOM).

    Returns:
        Generated samples [N, C, H, W].
    """
    if self_.fixed_conditioning_masks is None:
        raise RuntimeError("Must call set_fixed_conditioning() first")

    # Detect 3D from tensor dimensions: 3D is [N, C, D, H, W] with ndim=5
    is_3d = self_.fixed_conditioning_masks.ndim == 5

    # 3D generation uses batch_size=1 to avoid OOM at high resolutions
    # (256x256x160 with CFG requires ~50GB per batch of 2 due to dual forward passes)
    if is_3d:
        batch_size = 1

    # Cap batch_size at available masks to avoid errors with small mask counts
    num_available = self_.fixed_conditioning_masks.shape[0]
    batch_size = min(batch_size, num_available)

    # Round up to full batches for torch.compile consistency
    # e.g., 100 samples with batch_size=16 -> 112 samples (7 full batches)
    num_batches = math.ceil(num_samples / batch_size)
    num_to_use = num_batches * batch_size

    # Cap at available masks
    if num_to_use > num_available:
        # Round down to full batches if we don't have enough masks
        num_batches = num_available // batch_size
        num_to_use = num_batches * batch_size
        if num_to_use == 0:
            # If even 1 batch doesn't fit, use all available masks
            num_to_use = num_available
            logger.warning(f"Using all {num_available} masks (fewer than batch_size={batch_size})")

    # Get model config for output channels
    model_config = mode.get_model_config()
    out_channels = model_config['out_channels']
    in_channels = model_config['in_channels']

    model.eval()
    all_samples = []

    # Check if using latent diffusion
    is_latent = self_.space is not None and hasattr(self_.space, 'scale_factor') and self_.space.scale_factor > 1

    # Generate in batches to avoid OOM with CUDA graphs
    for start_idx in range(0, num_to_use, batch_size):
        end_idx = min(start_idx + batch_size, num_to_use)
        masks = self_.fixed_conditioning_masks[start_idx:end_idx]

        # For latent diffusion, encode pixel-space masks to latent space
        # so noise and model input have correct latent dimensions
        if is_latent:
            with torch.no_grad():
                masks = self_.space.encode(masks)

        # Generate noise
        noise = torch.randn_like(masks)

        if out_channels == 2:  # Dual mode
            noise_pre = torch.randn_like(masks)
            noise_gd = torch.randn_like(masks)
            model_input = torch.cat([noise_pre, noise_gd, masks], dim=1)
            batch_bin_maps = None
        elif self_.mode_name == 'seg_conditioned_input' and self_.fixed_bin_maps is not None:
            # seg_conditioned_input mode: pass noise as model_input, bin_maps separately for CFG
            batch_bin_maps = self_.fixed_bin_maps[start_idx:end_idx]
            model_input = noise  # Just noise, bin_maps passed separately
        elif in_channels == 1:  # Unconditional modes (seg, seg_conditioned)
            # No channel concatenation - conditioning via embedding (if any)
            model_input = noise
            batch_bin_maps = None
        else:  # Conditional single channel modes (bravo)
            model_input = torch.cat([noise, masks], dim=1)
            batch_bin_maps = None

        # Get size_bins for this batch if in seg_conditioned mode
        batch_size_bins = None
        if self_.fixed_size_bins is not None:
            batch_size_bins = self_.fixed_size_bins[start_idx:end_idx]

        # Generate samples
        # Get latent channels for proper parsing of conditional input
        latent_ch = self_.space.latent_channels if self_.space is not None else 1
        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            samples = strategy.generate(
                model, model_input, num_steps=num_steps, device=self_.device,
                size_bins=batch_size_bins,
                bin_maps=batch_bin_maps,
                cfg_scale=self_.config.cfg_scale,
                latent_channels=latent_ch,
            )

        # Move to CPU immediately to free GPU memory
        all_samples.append(torch.clamp(samples.float(), 0, 1).cpu())

        # Clear intermediate tensors
        del samples, model_input, noise, masks
        if out_channels == 2:
            del noise_pre, noise_gd

    # Concatenate all samples
    result = torch.cat(all_samples, dim=0)

    # Free list memory
    del all_samples
    torch.cuda.empty_cache()

    # Decode from latent space to pixel space for feature extraction
    result = result.to(self_.device)
    if self_.space is not None and hasattr(self_.space, 'scale_factor') and self_.space.scale_factor > 1:
        result = self_.space.decode(result)

    # Threshold seg mode output at 0.5 to get binary masks
    if self_.is_seg_mode:
        result = (result > 0.5).float()

    return result


def generate_and_extract_features_3d_streaming(
    self_: GenerationMetrics,
    model: nn.Module,
    strategy: Any,
    mode: Any,
    num_samples: int,
    num_steps: int,
    keep_samples_for_diversity: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Generate 3D samples and extract features in a streaming fashion.

    Generates one sample at a time and extracts features immediately to
    avoid memory accumulation. This prevents OOM from keeping all generated
    volumes in memory simultaneously.

    Args:
        self_: GenerationMetrics instance.
        model: Diffusion model.
        strategy: Diffusion strategy (DDPM/RFlow).
        mode: Training mode.
        num_samples: Number of samples to generate.
        num_steps: Number of denoising steps.
        keep_samples_for_diversity: If True, keep samples for diversity metrics.
            Only keeps first 2 samples to limit memory usage.

    Returns:
        Tuple of (resnet_features, biomed_features, samples_for_diversity).
        samples_for_diversity is None if keep_samples_for_diversity is False.
    """
    if self_.fixed_conditioning_masks is None:
        raise RuntimeError("Must call set_fixed_conditioning() first")

    # Get model config
    model_config = mode.get_model_config()
    out_channels = model_config['out_channels']
    in_channels = model_config['in_channels']

    # Cap at available masks
    num_available = self_.fixed_conditioning_masks.shape[0]
    num_to_generate = min(num_samples, num_available)

    model.eval()
    all_resnet_features = []
    all_biomed_features = []
    samples_for_diversity = [] if keep_samples_for_diversity else None

    # Limit diversity samples to 2 for 3D to avoid OOM
    max_diversity_samples = 2

    # Check if using latent diffusion
    is_latent = self_.space is not None and hasattr(self_.space, 'scale_factor') and self_.space.scale_factor > 1

    for idx in range(num_to_generate):
        # Get conditioning for this sample (pixel-space from dataset)
        masks_pixel = self_.fixed_conditioning_masks[idx:idx+1]

        # For latent diffusion, encode masks to latent space
        if is_latent:
            with torch.no_grad():
                masks = self_.space.encode(masks_pixel)
        else:
            masks = masks_pixel

        # Generate noise matching the (possibly encoded) masks shape
        noise = torch.randn_like(masks)

        # Build model input based on mode
        if out_channels == 2:  # Dual mode
            noise_pre = torch.randn_like(masks)
            noise_gd = torch.randn_like(masks)
            model_input = torch.cat([noise_pre, noise_gd, masks], dim=1)
            batch_bin_maps = None
        elif self_.mode_name == 'seg_conditioned_input' and self_.fixed_bin_maps is not None:
            batch_bin_maps = self_.fixed_bin_maps[idx:idx+1]
            model_input = noise
        elif in_channels == 1:
            model_input = noise
            batch_bin_maps = None
        else:
            model_input = torch.cat([noise, masks], dim=1)
            batch_bin_maps = None

        # Get size_bins if available
        batch_size_bins = None
        if self_.fixed_size_bins is not None:
            batch_size_bins = self_.fixed_size_bins[idx:idx+1]

        # Generate sample
        # Get latent channels for proper parsing of conditional input
        latent_ch = self_.space.latent_channels if self_.space is not None else 1
        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            sample = strategy.generate(
                model, model_input, num_steps=num_steps, device=self_.device,
                size_bins=batch_size_bins,
                bin_maps=batch_bin_maps,
                cfg_scale=self_.config.cfg_scale,
                latent_channels=latent_ch,
            )

        # Process sample: clamp, decode if latent, threshold if seg
        sample = torch.clamp(sample.float(), 0, 1)
        if self_.space is not None and hasattr(self_.space, 'scale_factor') and self_.space.scale_factor > 1:
            sample = self_.space.decode(sample)
        if self_.is_seg_mode:
            sample = (sample > 0.5).float()

        # Keep sample for diversity if requested and within limit
        if samples_for_diversity is not None and len(samples_for_diversity) < max_diversity_samples:
            samples_for_diversity.append(sample.cpu())

        # Extract features immediately
        from .generation_computation import extract_features_batched
        resnet_feat = extract_features_batched(self_, sample, self_.resnet)
        biomed_feat = extract_features_batched(self_, sample, self_.biomed)
        all_resnet_features.append(resnet_feat.cpu())
        all_biomed_features.append(biomed_feat.cpu())

        # Clear GPU memory before next iteration
        del sample, model_input, noise, masks, resnet_feat, biomed_feat
        if out_channels == 2:
            del noise_pre, noise_gd
        torch.cuda.empty_cache()

    # Concatenate all features
    gen_resnet = torch.cat(all_resnet_features, dim=0)
    gen_biomed = torch.cat(all_biomed_features, dim=0)

    # Concatenate diversity samples if kept
    diversity_tensor = None
    if samples_for_diversity:
        diversity_tensor = torch.cat(samples_for_diversity, dim=0)
        # Explicitly clear list to free memory immediately
        samples_for_diversity.clear()
        del samples_for_diversity

    del all_resnet_features, all_biomed_features
    torch.cuda.empty_cache()

    return gen_resnet, gen_biomed, diversity_tensor
