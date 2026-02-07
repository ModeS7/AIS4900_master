"""Metric computation for generation metrics.

Contains methods for extracting features, computing distributional
metrics (KID, CMMD, FID), diversity metrics, and size bin adherence.

Moved from generation.py during file split.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .generation import compute_cmmd, compute_fid, compute_kid
from .generation_3d import extract_features_3d
from .quality import (
    compute_lpips_diversity,
    compute_lpips_diversity_3d,
    compute_msssim_diversity,
    compute_msssim_diversity_3d,
)

if TYPE_CHECKING:
    from .generation import GenerationMetrics

logger = logging.getLogger(__name__)


def _validate_conditioning_and_detect_3d(self_: 'GenerationMetrics') -> bool:
    """Validate fixed_conditioning_masks is set. Returns True if 3D."""
    if self_.fixed_conditioning_masks is None:
        raise RuntimeError(
            "fixed_conditioning_masks is None. "
            "Call set_fixed_conditioning() before computing metrics."
        )
    return self_.fixed_conditioning_masks.ndim == 5


def extract_features_batched(
    self_: 'GenerationMetrics',
    samples: torch.Tensor,
    extractor: nn.Module,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Extract features from samples in batches.

    Args:
        self_: GenerationMetrics instance.
        samples: Input samples [N, C, H, W] or [N, C, D, H, W] for 3D.
        extractor: Feature extractor (ResNet50 or BiomedCLIP).
        batch_size: Batch size for extraction (default: config.feature_batch_size).

    Returns:
        Feature tensor [N, D] for 2D or [N*D_orig, D] for 3D (slice-wise, padding removed).
    """
    if batch_size is None:
        batch_size = self_.config.feature_batch_size

    # Handle 3D volumes via slice-wise extraction (2.5D approach)
    is_3d = samples.ndim == 5
    if is_3d:
        # Remove padded slices if original_depth is specified
        if self_.config.original_depth is not None:
            original_depth = self_.config.original_depth
            current_depth = samples.shape[2]  # [N, C, D, H, W]
            if current_depth > original_depth:
                # Remove last (current_depth - original_depth) slices
                samples = samples[:, :, :original_depth, :, :]
                logger.debug(f"Removed {current_depth - original_depth} padded slices for metrics")

        return extract_features_3d(samples, extractor, chunk_size=batch_size)

    # 2D: extract in sub-batches
    all_features = []
    for start_idx in range(0, samples.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, samples.shape[0])
        batch = samples[start_idx:end_idx]
        features = extractor.extract_features(batch)
        all_features.append(features.cpu())
        del features, batch

    result = torch.cat(all_features, dim=0)
    del all_features
    return result


def compute_metrics_against_reference(
    self_: 'GenerationMetrics',
    gen_resnet: torch.Tensor,
    gen_biomed: torch.Tensor,
    ref_resnet: torch.Tensor,
    ref_biomed: torch.Tensor,
    prefix: str = "",
) -> dict[str, float]:
    """Compute KID and CMMD against reference features.

    Args:
        self_: GenerationMetrics instance.
        gen_resnet: Generated ResNet50 features.
        gen_biomed: Generated BiomedCLIP features.
        ref_resnet: Reference ResNet50 features.
        ref_biomed: Reference BiomedCLIP features.
        prefix: Prefix for metric names (e.g., "extended_").

    Returns:
        Dictionary of metric values.
    """
    results = {}

    # KID
    kid_mean, kid_std = compute_kid(
        ref_resnet.to(self_.device),
        gen_resnet.to(self_.device),
    )
    results[f'{prefix}KID_mean'] = kid_mean
    results[f'{prefix}KID_std'] = kid_std

    # CMMD
    cmmd = compute_cmmd(
        ref_biomed.to(self_.device),
        gen_biomed.to(self_.device),
    )
    results[f'{prefix}CMMD'] = cmmd

    return results


def compute_diversity_metrics(
    self_: 'GenerationMetrics',
    samples: torch.Tensor,
    prefix: str = "",
) -> dict[str, float]:
    """Compute diversity metrics for generated samples.

    Measures how different the generated samples are from each other
    using LPIPS and MS-SSIM diversity. Higher values indicate more
    diversity (less mode collapse).

    Args:
        self_: GenerationMetrics instance.
        samples: Generated samples [N, C, H, W] or [B, C, D, H, W].
        prefix: Prefix for metric names (e.g., "extended_").

    Returns:
        Dictionary with diversity metrics.
    """
    results = {}
    is_3d = samples.ndim == 5
    n_samples = samples.shape[0]

    if n_samples < 2:
        logger.debug(f"Need at least 2 samples for diversity (got {n_samples})")
        return results

    try:
        if is_3d:
            # 3D: compare same slice across volumes
            lpips_div = compute_lpips_diversity_3d(
                samples, device=self_.device
            )
            msssim_div = compute_msssim_diversity_3d(samples)
        else:
            # 2D: compare all generated images
            lpips_div = compute_lpips_diversity(
                samples, device=self_.device
            )
            msssim_div = compute_msssim_diversity(samples)

        # Use special prefix format for separate TensorBoard section
        # "Diversity/" prefix tells log_generation to use Generation_Diversity/ section
        results[f'Diversity/{prefix}LPIPS'] = lpips_div
        results[f'Diversity/{prefix}MSSSIM'] = msssim_div

    except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as e:
        logger.warning(f"Diversity computation failed: {e}")

    return results


def compute_size_bin_adherence(
    self_: 'GenerationMetrics',
    generated_masks: torch.Tensor,
    conditioning_bins: torch.Tensor,
    prefix: str = "",
) -> dict[str, float]:
    """Compute size bin adherence metrics for seg_conditioned mode.

    Measures how well generated masks match their conditioning size bins.

    Args:
        self_: GenerationMetrics instance.
        generated_masks: Generated binary masks [N, 1, H, W].
        conditioning_bins: Target size bin counts [N, num_bins].
        prefix: Prefix for metric names (e.g., "extended_").

    Returns:
        Dictionary with metrics:
        - SizeBin/{prefix}exact_match: % of bins that exactly match
        - SizeBin/{prefix}MAE: Mean absolute error per bin
        - SizeBin/{prefix}correlation: Spearman correlation
    """
    from scipy.stats import spearmanr

    from medgen.data.loaders.datasets import DEFAULT_BIN_EDGES, compute_size_bins

    results = {}

    # Get bin config
    bin_edges = self_.config.size_bin_edges or DEFAULT_BIN_EDGES
    fov_mm = self_.config.size_bin_fov_mm
    image_size = generated_masks.shape[-1]  # H or W
    pixel_spacing_mm = fov_mm / image_size
    # Get num_bins from the conditioning tensor shape
    num_bins = conditioning_bins.shape[-1]

    # Compute actual size bins from generated masks
    actual_bins_list = []
    for i in range(generated_masks.shape[0]):
        mask_np = generated_masks[i].squeeze().cpu().numpy()
        actual_bins = compute_size_bins(
            mask_np, bin_edges, pixel_spacing_mm, num_bins=num_bins
        )
        actual_bins_list.append(actual_bins)

    actual_bins = torch.tensor(np.stack(actual_bins_list), dtype=torch.long)
    target_bins = conditioning_bins.cpu()

    # 1. Exact match rate (per bin average)
    exact_matches = (actual_bins == target_bins).float().mean().item()
    results[f'SizeBin/{prefix}exact_match'] = exact_matches

    # 2. Mean Absolute Error (per sample, then averaged)
    mae = (actual_bins.float() - target_bins.float()).abs().mean().item()
    results[f'SizeBin/{prefix}MAE'] = mae

    # 3. Spearman correlation (flatten and compute)
    actual_flat = actual_bins.flatten().numpy()
    target_flat = target_bins.flatten().numpy()

    # Handle edge case: constant arrays
    if np.std(actual_flat) > 0 and np.std(target_flat) > 0:
        corr, _ = spearmanr(actual_flat, target_flat)
        results[f'SizeBin/{prefix}correlation'] = corr
    else:
        results[f'SizeBin/{prefix}correlation'] = 0.0

    return results


def compute_epoch_metrics(
    self_: 'GenerationMetrics',
    model: nn.Module,
    strategy: Any,
    mode: Any,
) -> dict[str, float]:
    """Compute quick generation metrics (every epoch).

    Uses fewer samples and steps for fast feedback.

    Args:
        self_: GenerationMetrics instance.
        model: Diffusion model.
        strategy: Diffusion strategy.
        mode: Training mode.

    Returns:
        Dictionary with KID/CMMD vs train and val.
    """
    from .generation_sampling import generate_and_extract_features_3d_streaming, generate_samples

    # Preserve RNG state (per pitfall #42)
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(self_.device)

    try:
        is_3d = _validate_conditioning_and_detect_3d(self_)

        if is_3d:
            # 3D: Generate and extract features one at a time (streaming)
            gen_resnet, gen_biomed, diversity_samples = generate_and_extract_features_3d_streaming(
                self_, model, strategy, mode,
                self_.config.samples_per_epoch,
                self_.config.steps_per_epoch,
                keep_samples_for_diversity=True,
            )

            # Compute diversity metrics from kept samples (limited to 2 for 3D)
            diversity_metrics = {}
            if diversity_samples is not None and diversity_samples.shape[0] >= 2:
                diversity_metrics = compute_diversity_metrics(self_, diversity_samples, prefix="")
                del diversity_samples

            # Size bin metrics not supported in streaming mode for 3D
            size_bin_metrics = {}
        else:
            # 2D: Use batched approach (more efficient for small samples)
            samples = generate_samples(
                self_, model, strategy, mode,
                self_.config.samples_per_epoch,
                self_.config.steps_per_epoch,
            )

            # Extract features in batches
            gen_resnet = extract_features_batched(self_, samples, self_.resnet)
            gen_biomed = extract_features_batched(self_, samples, self_.biomed)

            # Compute diversity metrics (2D - every epoch)
            diversity_metrics = compute_diversity_metrics(self_, samples, prefix="")

            # Compute size bin adherence for seg_conditioned mode
            size_bin_metrics = {}
            if self_.mode_name == 'seg_conditioned' and self_.fixed_size_bins is not None:
                size_bin_metrics = compute_size_bin_adherence(
                    self_, samples, self_.fixed_size_bins[:samples.shape[0]], prefix=""
                )

            # Free samples
            del samples
            torch.cuda.empty_cache()

        results = {}

        # Add diversity metrics
        results.update(diversity_metrics)

        # Add size bin metrics
        results.update(size_bin_metrics)

        # Metrics vs train
        train_metrics = compute_metrics_against_reference(
            self_, gen_resnet, gen_biomed,
            self_.cache.train_resnet, self_.cache.train_biomed,
            prefix="",
        )
        for key, value in train_metrics.items():
            results[f'{key}_train'] = value

        # Metrics vs val
        val_metrics = compute_metrics_against_reference(
            self_, gen_resnet, gen_biomed,
            self_.cache.val_resnet, self_.cache.val_biomed,
            prefix="",
        )
        for key, value in val_metrics.items():
            results[f'{key}_val'] = value

        return results

    finally:
        # Restore RNG state
        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state(cuda_rng_state, self_.device)


def compute_extended_metrics(
    self_: 'GenerationMetrics',
    model: nn.Module,
    strategy: Any,
    mode: Any,
) -> dict[str, float]:
    """Compute extended generation metrics (every N epochs).

    Uses more samples and steps for detailed analysis.

    Args:
        self_: GenerationMetrics instance.
        model: Diffusion model.
        strategy: Diffusion strategy.
        mode: Training mode.

    Returns:
        Dictionary with extended_ prefixed KID/CMMD metrics.
    """
    from .generation_sampling import generate_and_extract_features_3d_streaming, generate_samples

    # Preserve RNG state
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(self_.device)

    try:
        is_3d = _validate_conditioning_and_detect_3d(self_)

        if is_3d:
            # 3D: Generate and extract features one at a time (streaming)
            gen_resnet, gen_biomed, diversity_samples = generate_and_extract_features_3d_streaming(
                self_, model, strategy, mode,
                self_.config.samples_extended,
                self_.config.steps_extended,
                keep_samples_for_diversity=True,
            )

            # Compute diversity metrics from kept samples (limited to 2 for 3D)
            diversity_metrics = {}
            if diversity_samples is not None and diversity_samples.shape[0] >= 2:
                diversity_metrics = compute_diversity_metrics(self_, diversity_samples, prefix="extended_")
                del diversity_samples

            # Size bin metrics not supported in streaming mode for 3D
            size_bin_metrics = {}
        else:
            # 2D: Use batched approach (more efficient for small samples)
            samples = generate_samples(
                self_, model, strategy, mode,
                self_.config.samples_extended,
                self_.config.steps_extended,
            )

            # Extract features in batches
            gen_resnet = extract_features_batched(self_, samples, self_.resnet)
            gen_biomed = extract_features_batched(self_, samples, self_.biomed)

            # Compute diversity metrics (2D - extended)
            diversity_metrics = compute_diversity_metrics(self_, samples, prefix="extended_")

            # Compute size bin adherence for seg_conditioned mode
            size_bin_metrics = {}
            if self_.mode_name == 'seg_conditioned' and self_.fixed_size_bins is not None:
                size_bin_metrics = compute_size_bin_adherence(
                    self_, samples, self_.fixed_size_bins[:samples.shape[0]], prefix="extended_"
                )

            # Free samples
            del samples
            torch.cuda.empty_cache()

        results = {}

        # Add diversity metrics
        results.update(diversity_metrics)

        # Add size bin metrics
        results.update(size_bin_metrics)

        # Metrics vs train
        train_metrics = compute_metrics_against_reference(
            self_, gen_resnet, gen_biomed,
            self_.cache.train_resnet, self_.cache.train_biomed,
            prefix="extended_",
        )
        for key, value in train_metrics.items():
            results[f'{key}_train'] = value

        # Metrics vs val
        val_metrics = compute_metrics_against_reference(
            self_, gen_resnet, gen_biomed,
            self_.cache.val_resnet, self_.cache.val_biomed,
            prefix="extended_",
        )
        for key, value in val_metrics.items():
            results[f'{key}_val'] = value

        return results

    finally:
        # Restore RNG state
        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state(cuda_rng_state, self_.device)


def compute_metrics_from_samples(
    self_: 'GenerationMetrics',
    samples: torch.Tensor,
    extended: bool = False,
) -> dict[str, float]:
    """Compute metrics from pre-generated samples.

    This method is useful when samples are generated externally (e.g., 3D volumes).
    Handles both 2D [N, C, H, W] and 3D [N, C, D, H, W] samples.

    Args:
        self_: GenerationMetrics instance.
        samples: Generated samples tensor.
        extended: If True, use "extended_" prefix for metric names.

    Returns:
        Dictionary with KID/CMMD vs train and val.
    """
    if self_.cache.train_resnet is None:
        logger.warning("Reference features not cached, cannot compute metrics")
        return {}

    prefix = "extended_" if extended else ""
    is_3d = samples.ndim == 5
    n_samples = samples.shape[0]

    # Extract features
    if is_3d:
        gen_resnet = extract_features_3d(samples, self_.resnet, chunk_size=self_.config.feature_batch_size)
        gen_biomed = extract_features_3d(samples, self_.biomed, chunk_size=self_.config.feature_batch_size)
    else:
        gen_resnet = extract_features_batched(self_, samples, self_.resnet)
        gen_biomed = extract_features_batched(self_, samples, self_.biomed)

    # Compute diversity metrics (3D: only when 2+ volumes, typically 4+ for extended/test)
    # For 3D, diversity is computed per-slice across volumes, so we need multiple volumes
    diversity_metrics = {}
    if n_samples >= 2:
        diversity_metrics = compute_diversity_metrics(self_, samples, prefix=prefix)

    # Free samples
    del samples
    torch.cuda.empty_cache()

    results = {}

    # Add diversity metrics
    results.update(diversity_metrics)

    # Metrics vs train
    train_metrics = compute_metrics_against_reference(
        self_, gen_resnet, gen_biomed,
        self_.cache.train_resnet, self_.cache.train_biomed,
        prefix=prefix,
    )
    for key, value in train_metrics.items():
        results[f'{key}_train'] = value

    # Metrics vs val
    if self_.cache.val_resnet is not None:
        val_metrics = compute_metrics_against_reference(
            self_, gen_resnet, gen_biomed,
            self_.cache.val_resnet, self_.cache.val_biomed,
            prefix=prefix,
        )
        for key, value in val_metrics.items():
            results[f'{key}_val'] = value

    return results


def compute_test_metrics(
    self_: 'GenerationMetrics',
    model: nn.Module,
    strategy: Any,
    mode: Any,
    test_loader: DataLoader | None = None,
) -> dict[str, float]:
    """Compute full test generation metrics with FID.

    Uses the most samples and steps for final evaluation.
    Optionally compares against test set if loader provided.

    Args:
        self_: GenerationMetrics instance.
        model: Diffusion model.
        strategy: Diffusion strategy.
        mode: Training mode.
        test_loader: Optional test data loader for FID vs test.

    Returns:
        Dictionary with FID, KID, CMMD test metrics.
    """
    from .generation_sampling import generate_and_extract_features_3d_streaming, generate_samples

    is_3d = _validate_conditioning_and_detect_3d(self_)

    if is_3d:
        # 3D: Generate and extract features one at a time (streaming)
        gen_resnet, gen_biomed, diversity_samples = generate_and_extract_features_3d_streaming(
            self_, model, strategy, mode,
            self_.config.samples_test,
            self_.config.steps_test,
            keep_samples_for_diversity=True,
        )

        # Compute diversity metrics from kept samples (limited to 2 for 3D)
        diversity_metrics = {}
        if diversity_samples is not None and diversity_samples.shape[0] >= 2:
            diversity_metrics = compute_diversity_metrics(self_, diversity_samples, prefix="")
            del diversity_samples
    else:
        # 2D: Use batched approach (more efficient for small samples)
        samples = generate_samples(
            self_, model, strategy, mode,
            self_.config.samples_test,
            self_.config.steps_test,
        )

        # Extract features in batches
        gen_resnet = extract_features_batched(self_, samples, self_.resnet)
        gen_biomed = extract_features_batched(self_, samples, self_.biomed)

        # Compute diversity metrics (2D - test)
        diversity_metrics = compute_diversity_metrics(self_, samples, prefix="")

        # Free samples
        del samples
        torch.cuda.empty_cache()

    results = {}

    # Add diversity metrics
    results.update(diversity_metrics)

    # If test loader provided, compute metrics vs test
    if test_loader is not None:
        # Extract test features
        test_resnet = self_.cache._extract_features_from_loader(
            test_loader, self_.resnet, "test_resnet",
            max_samples=self_.config.samples_test,
        )
        test_biomed = self_.cache._extract_features_from_loader(
            test_loader, self_.biomed, "test_biomed",
            max_samples=self_.config.samples_test,
        )

        # FID (only for test)
        fid = compute_fid(test_resnet, gen_resnet)
        results['FID'] = fid

        # KID and CMMD vs test
        test_metrics = compute_metrics_against_reference(
            self_, gen_resnet, gen_biomed,
            test_resnet, test_biomed,
            prefix="",
        )
        results.update(test_metrics)
    else:
        # Use val as reference if no test loader
        val_metrics = compute_metrics_against_reference(
            self_, gen_resnet, gen_biomed,
            self_.cache.val_resnet, self_.cache.val_biomed,
            prefix="",
        )
        results.update(val_metrics)

        # FID vs val
        fid = compute_fid(
            self_.cache.val_resnet,
            gen_resnet,
        )
        results['FID'] = fid

    return results
