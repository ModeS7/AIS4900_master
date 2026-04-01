"""Sample generation for generation metrics.

Contains methods for generating samples with fixed conditioning,
including 2D batched generation and 3D streaming generation.

Moved from generation.py during file split.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import Dataset

from medgen.core.dict_utils import get_with_fallbacks
from medgen.data.utils import binarize_seg

if TYPE_CHECKING:
    from .generation import GenerationMetrics

logger = logging.getLogger(__name__)


class StreamingFeatures(NamedTuple):
    """Features extracted during streaming 3D generation."""

    resnet: torch.Tensor
    biomed: torch.Tensor
    resnet_rin: torch.Tensor | None
    resnet_3d: torch.Tensor | None  # tri-planar
    biomed_3d: torch.Tensor | None  # tri-planar
    resnet_rin_3d: torch.Tensor | None  # tri-planar
    diversity_samples: torch.Tensor | None
    # Per-modality features (dual/triple modes)
    per_modality_resnet: dict[str, torch.Tensor] | None = None
    per_modality_biomed: dict[str, torch.Tensor] | None = None
    per_modality_resnet_rin: dict[str, torch.Tensor] | None = None
    # PCA brain shape metrics (3D only)
    pca_mean_error: float | None = None
    pca_pass_rate: float | None = None
    pca_errors: list[float] | None = None


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

                if image is None:
                    samples_without_seg += 1
                    attempts += 1
                    continue

                # For seg mode (seg_channel_idx=0): image IS the seg mask
                if seg_data is None and seg_channel_idx == 0:
                    if isinstance(image, np.ndarray):
                        image = torch.from_numpy(image).float()
                    elif isinstance(image, torch.Tensor):
                        image = image.float()
                    tensor = image
                    local_seg_idx = 0
                elif seg_data is None:
                    samples_without_seg += 1
                    attempts += 1
                    continue
                else:
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

    # Override in_channels from actual model (handles ControlNet stage1 where
    # mode says in=2 but model was built with in=1)
    actual_model = model.module if hasattr(model, 'module') else model
    if hasattr(actual_model, 'conv_in'):
        in_channels = actual_model.conv_in[0].in_channels

    model.eval()
    all_samples = []

    # Check if conditioning must be encoded (latent/wavelet/pixel normalization)
    # Training encodes labels through space.encode(), generation must match
    encode_cond = self_.space is not None and self_.space.encode_conditioning

    # Generate in batches to avoid OOM with CUDA graphs
    for start_idx in range(0, num_to_use, batch_size):
        end_idx = min(start_idx + batch_size, num_to_use)
        masks = self_.fixed_conditioning_masks[start_idx:end_idx]

        # Encode conditioning to match training (pixel norm, latent, wavelet)
        if encode_cond:
            with torch.no_grad():
                masks = self_.space.encode(masks)

        # Generate noise
        noise = torch.randn_like(masks)

        if out_channels >= 2:  # Multi-channel mode (dual/triple)
            if encode_cond:
                raise ValueError(
                    "Multi-channel mode metrics do not support latent-encoded conditioning. "
                    "Use bravo_seg_cond mode for latent conditioning."
                )
            noise_channels = [torch.randn_like(masks) for _ in range(out_channels)]
            model_input = torch.cat([*noise_channels, masks], dim=1)
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
                cfg_mode=self_.config.cfg_mode,
            )

        # Move to CPU immediately to free GPU memory
        # Don't clamp here — latent/wavelet coefficients can be outside [0, 1]
        all_samples.append(samples.float().cpu())

        # Clear intermediate tensors
        del samples, model_input, noise, masks
        if out_channels >= 2:
            del noise_channels

    # Concatenate all samples
    result = torch.cat(all_samples, dim=0)

    # Free list memory
    del all_samples
    torch.cuda.empty_cache()

    # Decode from latent/wavelet space to pixel space for feature extraction
    result = result.to(self_.device)
    if self_.space is not None and self_.space.needs_decode:
        result = self_.space.decode(result)

    # Binarize seg output or clamp to [0, 1] for image output
    if self_.is_seg_mode:
        result = binarize_seg(result)
    else:
        result = torch.clamp(result, 0, 1)

    return result


@torch.no_grad()
def generate_and_extract_features_3d_streaming(
    self_: GenerationMetrics,
    model: nn.Module,
    strategy: Any,
    mode: Any,
    num_samples: int,
    num_steps: int,
    keep_samples_for_diversity: bool = False,
) -> StreamingFeatures:
    """Generate 3D samples and extract features in two memory-efficient phases.

    Phase 1 (Generation): All feature extractors are unloaded to maximize GPU
    memory for Conv3d workspace (~18 GiB at 256x256x160). Samples stored on CPU.

    Phase 2 (Feature extraction): Extractors loaded one at a time to minimize
    GPU footprint. Each extractor processes all samples then is unloaded before
    loading the next.

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
        StreamingFeatures with axial and tri-planar features.
    """
    if self_.fixed_conditioning_masks is None:
        raise RuntimeError("Must call set_fixed_conditioning() first")

    # Get model config
    model_config = mode.get_model_config()
    out_channels = model_config['out_channels']
    in_channels = model_config['in_channels']

    # Override in_channels from actual model (handles ControlNet stage1 where
    # mode says in=2 but model was built with in=1)
    actual_model = model.module if hasattr(model, 'module') else model
    if hasattr(actual_model, 'conv_in'):
        in_channels = actual_model.conv_in[0].in_channels

    # Cap at available masks
    num_available = self_.fixed_conditioning_masks.shape[0]
    num_to_generate = min(num_samples, num_available)

    has_resnet_rin = self_.resnet_rin is not None

    # Check if conditioning must be encoded (latent/wavelet/pixel normalization)
    encode_cond = self_.space is not None and self_.space.encode_conditioning

    # =====================================================================
    # Phase 1: Generate all samples (extractors unloaded for max GPU memory)
    # =====================================================================
    # Unload any extractors loaded from cache_reference_features or prior epochs
    self_.unload_extractors()
    torch.cuda.empty_cache()
    model.eval()
    cpu_samples = []

    for idx in range(num_to_generate):
        masks_pixel = self_.fixed_conditioning_masks[idx:idx+1]

        if encode_cond:
            masks = self_.space.encode(masks_pixel)
        else:
            masks = masks_pixel

        noise = torch.randn_like(masks)

        if out_channels >= 2:  # Multi-channel mode (dual/triple)
            if encode_cond:
                raise ValueError(
                    "Multi-channel mode metrics do not support latent-encoded conditioning. "
                    "Use bravo_seg_cond mode for latent conditioning."
                )
            noise_channels = [torch.randn_like(masks) for _ in range(out_channels)]
            model_input = torch.cat([*noise_channels, masks], dim=1)
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

        batch_size_bins = None
        if self_.fixed_size_bins is not None:
            batch_size_bins = self_.fixed_size_bins[idx:idx+1]

        latent_ch = self_.space.latent_channels if self_.space is not None else 1
        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            sample = strategy.generate(
                model, model_input, num_steps=num_steps, device=self_.device,
                size_bins=batch_size_bins,
                bin_maps=batch_bin_maps,
                cfg_scale=self_.config.cfg_scale,
                latent_channels=latent_ch,
                cfg_mode=self_.config.cfg_mode,
            )

        # Decode and normalize in pixel space
        sample = sample.float()
        if self_.space is not None and self_.space.needs_decode:
            sample = self_.space.decode(sample)
        if self_.is_seg_mode:
            sample = binarize_seg(sample)
        else:
            sample = torch.clamp(sample, 0, 1)

        cpu_samples.append(sample.cpu())
        del sample, model_input, noise, masks
        if out_channels >= 2:
            del noise_channels
        torch.cuda.empty_cache()

    logger.debug(f"[3D GenMetrics] Phase 1 complete: {len(cpu_samples)} samples on CPU")

    # PCA brain shape validation (CPU-only, no GPU needed)
    pca_mean_error = None
    pca_pass_rate = None
    pca_errors: list[float] = []
    if self_.is_seg_mode and hasattr(self_, 'seg_pca') and self_.seg_pca is not None:
        # Seg PCA: check tumor pattern directly on generated seg masks
        for sample_cpu in cpu_samples:
            seg_np = (sample_cpu[0, 0].numpy() > 0.5).astype(np.float32)
            _, error = self_.seg_pca.is_valid(seg_np)
            pca_errors.append(error)
        if pca_errors:
            pca_mean_error = float(np.mean(pca_errors))
            pca_pass_rate = float(np.mean([e <= self_.seg_pca.error_threshold for e in pca_errors]))
            logger.debug(f"[3D GenMetrics] Seg PCA: mean_error={pca_mean_error:.8f}, "
                         f"pass_rate={pca_pass_rate:.0%}")
    elif not self_.is_seg_mode and hasattr(self_, 'brain_pca') and self_.brain_pca is not None:
        # Brain PCA: check brain shape on generated bravo volumes
        from .brain_mask import create_brain_mask
        for sample_cpu in cpu_samples:
            vol_np = sample_cpu[0, 0].numpy()
            brain_mask = create_brain_mask(vol_np, threshold=0.05, fill_holes=True, dilate_pixels=0)
            _, error = self_.brain_pca.is_valid(brain_mask)
            pca_errors.append(error)
        if pca_errors:
            pca_mean_error = float(np.mean(pca_errors))
            pca_pass_rate = float(np.mean([e <= self_.brain_pca.error_threshold for e in pca_errors]))
            logger.debug(f"[3D GenMetrics] Brain PCA: mean_error={pca_mean_error:.6f}, "
                         f"pass_rate={pca_pass_rate:.0%}")

    # Diversity samples (keep first 2 on CPU)
    max_diversity_samples = 2
    diversity_tensor = None
    if keep_samples_for_diversity and cpu_samples:
        kept = cpu_samples[:max_diversity_samples]
        diversity_tensor = torch.cat(kept, dim=0)

    # =====================================================================
    # Phase 2: Extract features one extractor at a time
    # =====================================================================
    from .generation_3d import extract_features_3d_triplanar
    from .generation_computation import extract_features_batched

    chunk_sz = self_.config.feature_batch_size
    orig_d = self_.config.original_depth

    # --- ResNet50 (ImageNet) ---
    logger.debug("[3D GenMetrics] Phase 2a: ResNet50 (ImageNet) features")
    all_resnet = []
    all_resnet_3d = []
    for sample_cpu in cpu_samples:
        sample_gpu = sample_cpu.to(self_.device)
        all_resnet.append(extract_features_batched(self_, sample_gpu, self_.resnet).cpu())
        all_resnet_3d.append(extract_features_3d_triplanar(sample_gpu, self_.resnet, chunk_sz, orig_d).cpu())
        del sample_gpu
    self_.resnet.unload()
    torch.cuda.empty_cache()

    # --- ResNet50 (RadImageNet) ---
    all_resnet_rin = []
    all_resnet_rin_3d = []
    if has_resnet_rin:
        logger.debug("[3D GenMetrics] Phase 2b: ResNet50 (RadImageNet) features")
        for sample_cpu in cpu_samples:
            sample_gpu = sample_cpu.to(self_.device)
            all_resnet_rin.append(extract_features_batched(self_, sample_gpu, self_.resnet_rin).cpu())
            all_resnet_rin_3d.append(extract_features_3d_triplanar(sample_gpu, self_.resnet_rin, chunk_sz, orig_d).cpu())
            del sample_gpu
        self_.resnet_rin.unload()
        torch.cuda.empty_cache()

    # --- BiomedCLIP ---
    logger.debug("[3D GenMetrics] Phase 2c: BiomedCLIP features")
    all_biomed = []
    all_biomed_3d = []
    for sample_cpu in cpu_samples:
        sample_gpu = sample_cpu.to(self_.device)
        all_biomed.append(extract_features_batched(self_, sample_gpu, self_.biomed).cpu())
        all_biomed_3d.append(extract_features_3d_triplanar(sample_gpu, self_.biomed, chunk_sz, orig_d).cpu())
        del sample_gpu
    self_.biomed.unload()
    torch.cuda.empty_cache()

    # --- Per-modality features (dual/triple) ---
    pm_resnet: dict[str, torch.Tensor] | None = None
    pm_biomed: dict[str, torch.Tensor] | None = None
    pm_rin: dict[str, torch.Tensor] | None = None
    if self_.image_keys and self_.cache.per_modality:
        logger.debug("[3D GenMetrics] Phase 2d: Per-modality features")
        pm_resnet_acc: dict[str, list] = {k: [] for k in self_.image_keys}
        pm_biomed_acc: dict[str, list] = {k: [] for k in self_.image_keys}
        pm_rin_acc: dict[str, list] = {k: [] for k in self_.image_keys}

        for sample_cpu in cpu_samples:
            for ch_i, key in enumerate(self_.image_keys):
                single_ch = sample_cpu[:, ch_i:ch_i+1].to(self_.device)
                pm_resnet_acc[key].append(extract_features_batched(self_, single_ch, self_.resnet).cpu())
                pm_biomed_acc[key].append(extract_features_batched(self_, single_ch, self_.biomed).cpu())
                if has_resnet_rin:
                    pm_rin_acc[key].append(extract_features_batched(self_, single_ch, self_.resnet_rin).cpu())
                del single_ch

        pm_resnet = {k: torch.cat(v, dim=0) for k, v in pm_resnet_acc.items()}
        pm_biomed = {k: torch.cat(v, dim=0) for k, v in pm_biomed_acc.items()}
        pm_rin = {k: torch.cat(v, dim=0) for k, v in pm_rin_acc.items()} if has_resnet_rin else None

        # Unload extractors used for per-modality
        self_.resnet.unload()
        self_.biomed.unload()
        if has_resnet_rin:
            self_.resnet_rin.unload()
        torch.cuda.empty_cache()

    del cpu_samples

    # Concatenate all features
    gen_resnet = torch.cat(all_resnet, dim=0)
    gen_resnet_rin = torch.cat(all_resnet_rin, dim=0) if all_resnet_rin else None
    gen_biomed = torch.cat(all_biomed, dim=0)
    gen_resnet_3d = torch.cat(all_resnet_3d, dim=0)
    gen_resnet_rin_3d = torch.cat(all_resnet_rin_3d, dim=0) if all_resnet_rin_3d else None
    gen_biomed_3d = torch.cat(all_biomed_3d, dim=0)

    del all_resnet, all_resnet_rin, all_biomed
    del all_resnet_3d, all_resnet_rin_3d, all_biomed_3d
    torch.cuda.empty_cache()

    return StreamingFeatures(
        resnet=gen_resnet,
        biomed=gen_biomed,
        resnet_rin=gen_resnet_rin,
        resnet_3d=gen_resnet_3d,
        biomed_3d=gen_biomed_3d,
        resnet_rin_3d=gen_resnet_rin_3d,
        diversity_samples=diversity_tensor,
        per_modality_resnet=pm_resnet,
        per_modality_biomed=pm_biomed,
        per_modality_resnet_rin=pm_rin,
        pca_mean_error=pca_mean_error,
        pca_pass_rate=pca_pass_rate,
        pca_errors=pca_errors if pca_mean_error is not None else None,
    )
