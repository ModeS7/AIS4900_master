"""Sample generation utilities for generation metrics.

This module provides utilities for generating samples from diffusion models
with conditioning support. Used by GenerationMetrics for computing generation
quality metrics (KID, CMMD, FID).

Classes:
    ConditionalSampler: Generates samples from diffusion models with conditioning.

Functions:
    sample_conditioning_from_dataset: Extract conditioning masks from a dataset.
"""
import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import Dataset

from medgen.core.dict_utils import get_with_fallbacks

logger = logging.getLogger(__name__)


def sample_conditioning_from_dataset(
    dataset: Dataset,
    num_samples: int,
    seg_channel_idx: int = 1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Sample conditioning masks from a dataset.

    Extracts positive (non-empty) segmentation masks from the dataset
    for use as conditioning during generation.

    Args:
        dataset: Dataset to sample from.
        num_samples: Number of masks to sample.
        seg_channel_idx: Channel index for segmentation mask in combined tensors.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (masks, size_bins, bin_maps) where:
            - masks: [N, 1, H, W] or [N, 1, D, H, W] tensor of seg masks
            - size_bins: Optional [N, num_bins] tensor for seg_conditioned mode
            - bin_maps: Optional [N, 1, H, W] tensor for seg_conditioned_input mode

    Raises:
        RuntimeError: If no positive masks found in dataset.
    """
    masks = []
    size_bins_list = []
    bin_maps_list = []
    attempts = 0
    max_attempts = len(dataset)
    samples_without_seg = 0

    # Use fixed seed for reproducibility
    rng = torch.Generator()
    rng.manual_seed(seed)

    while len(masks) < num_samples and attempts < max_attempts:
        idx = int(torch.randint(0, len(dataset), (1,), generator=rng).item())
        data = dataset[idx]

        # Parse data format
        parsed = _parse_dataset_item(data, seg_channel_idx)
        if parsed is None:
            samples_without_seg += 1
            attempts += 1
            continue

        seg, size_bins, bin_maps = parsed

        if seg.sum() > 0:  # Has positive mask
            masks.append(seg)
            if size_bins is not None:
                size_bins_list.append(size_bins)
            if bin_maps is not None:
                bin_maps_list.append(bin_maps)

        attempts += 1

    if len(masks) < num_samples:
        logger.warning(f"Only found {len(masks)} positive masks (requested {num_samples})")

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

    result_masks = torch.stack(masks)
    result_size_bins = torch.stack(size_bins_list) if size_bins_list else None
    result_bin_maps = torch.stack(bin_maps_list) if bin_maps_list else None

    return result_masks, result_size_bins, result_bin_maps


def _parse_dataset_item(
    data: Any,
    seg_channel_idx: int = 1,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None:
    """Parse a dataset item to extract seg, size_bins, and bin_maps.

    Handles various data formats:
    - Dict: {'image': ..., 'seg': ...} or {'latent': ..., 'seg_mask': ...}
    - Tuple: (images, seg), (seg, size_bins), or (seg, size_bins, bin_maps)
    - Tensor: Combined tensor with seg at seg_channel_idx

    Args:
        data: Dataset item in any supported format.
        seg_channel_idx: Channel index for seg in combined tensors.

    Returns:
        Tuple of (seg, size_bins, bin_maps) or None if seg not found.
        seg is [1, H, W] or [1, D, H, W].
    """
    size_bins = None
    bin_maps = None
    tensor = None
    local_seg_idx = seg_channel_idx

    if isinstance(data, dict):
        # Dict format
        is_latent_data = 'latent' in data

        if is_latent_data:
            seg_data = data.get('seg_mask')
            if seg_data is None:
                return None
            if isinstance(seg_data, np.ndarray):
                seg_data = torch.from_numpy(seg_data).float()
            tensor = seg_data
            local_seg_idx = 0
        else:
            image = get_with_fallbacks(data, 'image', 'images')
            seg_data = get_with_fallbacks(data, 'seg', 'mask', 'labels')

            if image is None or seg_data is None:
                return None
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
            if isinstance(seg_data, np.ndarray):
                seg_data = torch.from_numpy(seg_data).float()
            tensor = torch.cat([image, seg_data], dim=0)
            local_seg_idx = image.shape[0]

    elif isinstance(data, tuple):
        if len(data) == 2:
            first, second = data
            bin_maps_data = None
        elif len(data) == 3:
            first, second, bin_maps_data = data
        else:
            return None

        # Convert to tensors
        first = _to_tensor(first)
        second = _to_tensor(second)

        # Detect format
        if bin_maps_data is not None and second.dim() == 1:
            # seg_conditioned_input mode: (seg, size_bins, bin_maps)
            tensor = first
            size_bins = second.long()
            bin_maps = _to_tensor(bin_maps_data)
            local_seg_idx = 0
        elif second.dim() == 1 and first.dim() >= 3:
            # seg_conditioned mode: (seg, size_bins)
            tensor = first
            size_bins = second.long()
            local_seg_idx = 0
        else:
            # Standard mode: (images, seg)
            tensor = torch.cat([first, second], dim=0)
            local_seg_idx = seg_channel_idx
    else:
        tensor = _to_tensor(data)
        local_seg_idx = seg_channel_idx

    # Extract seg mask
    seg = tensor[local_seg_idx:local_seg_idx + 1, ...]

    return seg, size_bins, bin_maps


def _to_tensor(data: Any) -> torch.Tensor:
    """Convert data to float tensor."""
    if isinstance(data, torch.Tensor):
        return data.float()
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    else:
        return torch.tensor(data).float()


class ConditionalSampler:
    """Generates samples from diffusion models with conditioning.

    Handles the complexity of generating samples with:
    - Seg mask conditioning (bravo mode)
    - Size bin conditioning (seg_conditioned mode)
    - Bin map conditioning (seg_conditioned_input mode)
    - Classifier-free guidance (CFG)
    - Latent space decoding

    Example:
        sampler = ConditionalSampler(device=torch.device('cuda'))
        sampler.set_fixed_conditioning(dataset, num_masks=500)

        samples = sampler.generate_samples(
            model, strategy, mode,
            num_samples=100, num_steps=50,
            cfg_scale=2.0,
        )
    """

    def __init__(
        self,
        device: torch.device,
        space: Any | None = None,
        mode_name: str = 'bravo',
    ) -> None:
        """Initialize sampler.

        Args:
            device: PyTorch device for generation.
            space: Optional DiffusionSpace for latent encoding/decoding.
            mode_name: Training mode name for behavior selection.
        """
        self.device = device
        self.space = space
        self.mode_name = mode_name
        self.is_seg_mode = mode_name in ('seg', 'seg_conditioned', 'seg_conditioned_input')

        # Fixed conditioning (set via set_fixed_conditioning)
        self.fixed_conditioning_masks: torch.Tensor | None = None
        self.fixed_size_bins: torch.Tensor | None = None
        self.fixed_bin_maps: torch.Tensor | None = None

    def set_fixed_conditioning(
        self,
        dataset: Dataset,
        num_masks: int = 500,
        seg_channel_idx: int = 1,
    ) -> None:
        """Load fixed conditioning masks from training dataset.

        Uses the same masks every epoch for reproducible comparisons.

        Args:
            dataset: Training dataset to sample from.
            num_masks: Number of masks to sample.
            seg_channel_idx: Channel index for segmentation mask.
        """
        logger.info(f"Sampling {num_masks} fixed conditioning masks...")

        masks, size_bins, bin_maps = sample_conditioning_from_dataset(
            dataset, num_masks, seg_channel_idx
        )

        self.fixed_conditioning_masks = masks.to(self.device)
        if size_bins is not None:
            self.fixed_size_bins = size_bins.to(self.device)
            logger.info(f"Loaded {len(size_bins)} fixed size_bins for seg_conditioned mode")
        if bin_maps is not None:
            self.fixed_bin_maps = bin_maps.to(self.device)
            logger.info(f"Loaded {len(bin_maps)} fixed bin_maps for seg_conditioned_input mode")

        logger.info(f"Loaded {masks.shape[0]} fixed conditioning masks")

    def get_conditioning_batch(
        self,
        start_idx: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Get a batch of conditioning from fixed masks.

        Args:
            start_idx: Start index in fixed masks.
            batch_size: Batch size.

        Returns:
            Tuple of (masks, size_bins, bin_maps) for the batch.
        """
        end_idx = min(start_idx + batch_size, self.fixed_conditioning_masks.shape[0])
        masks = self.fixed_conditioning_masks[start_idx:end_idx]

        size_bins = None
        if self.fixed_size_bins is not None:
            size_bins = self.fixed_size_bins[start_idx:end_idx]

        bin_maps = None
        if self.fixed_bin_maps is not None:
            bin_maps = self.fixed_bin_maps[start_idx:end_idx]

        return masks, size_bins, bin_maps

    @torch.no_grad()
    def generate_samples(
        self,
        model: nn.Module,
        strategy: Any,
        mode: Any,
        num_samples: int,
        num_steps: int,
        batch_size: int = 16,
        cfg_scale: float = 2.0,
    ) -> torch.Tensor:
        """Generate samples using fixed conditioning.

        Args:
            model: Diffusion model.
            strategy: Diffusion strategy (DDPM/RFlow).
            mode: Training mode object.
            num_samples: Number of samples to generate.
            num_steps: Number of denoising steps.
            batch_size: Batch size for generation.
            cfg_scale: Classifier-free guidance scale.

        Returns:
            Generated samples [N, C, H, W] or [N, C, D, H, W].
        """
        if self.fixed_conditioning_masks is None:
            raise RuntimeError("Must call set_fixed_conditioning() first")

        # Detect 3D from tensor dimensions
        is_3d = self.fixed_conditioning_masks.ndim == 5

        # 3D uses batch_size=1 to avoid OOM
        if is_3d:
            batch_size = 1

        # Cap batch_size at available masks
        num_available = self.fixed_conditioning_masks.shape[0]
        batch_size = min(batch_size, num_available)

        # Round to full batches for torch.compile consistency
        num_batches = math.ceil(num_samples / batch_size)
        num_to_use = num_batches * batch_size

        # Cap at available masks
        if num_to_use > num_available:
            num_batches = num_available // batch_size
            num_to_use = num_batches * batch_size
            if num_to_use == 0:
                num_to_use = num_available
                logger.warning(f"Using all {num_available} masks (fewer than batch_size={batch_size})")

        # Get model config
        model_config = mode.get_model_config()
        out_channels = model_config['out_channels']
        in_channels = model_config['in_channels']

        model.eval()
        all_samples = []

        for start_idx in range(0, num_to_use, batch_size):
            masks, batch_size_bins, batch_bin_maps = self.get_conditioning_batch(
                start_idx, batch_size
            )

            # Generate noise
            noise = torch.randn_like(masks)

            # Build model input based on mode
            if out_channels == 2:  # Dual mode
                noise_pre = torch.randn_like(masks)
                noise_gd = torch.randn_like(masks)
                model_input = torch.cat([noise_pre, noise_gd, masks], dim=1)
                batch_bin_maps = None
            elif self.mode_name == 'seg_conditioned_input' and batch_bin_maps is not None:
                model_input = noise
            elif in_channels == 1:  # Unconditional modes
                model_input = noise
                batch_bin_maps = None
            else:  # Conditional single channel (bravo)
                model_input = torch.cat([noise, masks], dim=1)
                batch_bin_maps = None

            # Get latent channels for proper parsing
            latent_ch = self.space.latent_channels if self.space is not None else 1

            with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                samples = strategy.generate(
                    model, model_input, num_steps=num_steps, device=self.device,
                    size_bins=batch_size_bins,
                    bin_maps=batch_bin_maps,
                    cfg_scale=cfg_scale,
                    latent_channels=latent_ch,
                )

            # Move to CPU immediately to free GPU memory
            # Don't clamp here — latent/wavelet coefficients can be outside [0, 1]
            all_samples.append(samples.float().cpu())

            # Cleanup
            del samples, model_input, noise, masks
            if out_channels == 2:
                del noise_pre, noise_gd

        result = torch.cat(all_samples, dim=0)
        del all_samples
        torch.cuda.empty_cache()

        # Decode from latent/wavelet space if needed
        result = result.to(self.device)
        if self.space is not None and hasattr(self.space, 'scale_factor') and self.space.scale_factor > 1:
            result = self.space.decode(result)

        # Clamp to [0, 1] in pixel space (after decoding)
        result = torch.clamp(result, 0, 1)

        # Threshold seg mode output
        if self.is_seg_mode:
            result = (result > 0.5).float()

        return result
