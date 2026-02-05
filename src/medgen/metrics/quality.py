"""
Quality metrics for image comparison.

Provides PSNR, MS-SSIM, and LPIPS metrics used by DiffusionTrainer and VAETrainer.

Metrics:
- PSNR: Peak Signal-to-Noise Ratio (works with any dimensions)
- MS-SSIM: Multi-Scale Structural Similarity (2D and 3D via MONAI)
- LPIPS: Learned Perceptual Image Patch Similarity (2D only, uses pretrained networks)

Caching:
- Metric instances are cached using functools.lru_cache with size limits
- Use clear_metric_caches() to clear all caches when needed
"""
import logging
import threading
import traceback
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from monai.metrics import MultiScaleSSIMMetric

logger = logging.getLogger(__name__)

# =============================================================================
# MS-SSIM Constants
# =============================================================================

# MS-SSIM scale thresholds (minimum image size for each scale count)
# With kernel size 11, minimum size = 11 * 2^(num_scales-1) + 1
MSSSIM_5_SCALE_MIN_SIZE = 176  # 5 scales: 11 * 16 + 1 = 177
MSSSIM_4_SCALE_MIN_SIZE = 88   # 4 scales: 11 * 8 + 1 = 89
MSSSIM_3_SCALE_MIN_SIZE = 44   # 3 scales: 11 * 4 + 1 = 45
MSSSIM_2_SCALE_MIN_SIZE = 22   # 2 scales: 11 * 2 + 1 = 23

# MS-SSIM weights for different scale counts
MSSSIM_5_SCALE_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
MSSSIM_4_SCALE_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.3695)
MSSSIM_3_SCALE_WEIGHTS = (0.0448, 0.2856, 0.6696)
MSSSIM_2_SCALE_WEIGHTS = (0.5, 0.5)

# =============================================================================
# PSNR Constants
# =============================================================================

PSNR_MSE_EPSILON = 1e-10

# =============================================================================
# 3D Processing Constants
# =============================================================================

LPIPS_3D_CHUNK_SIZE = 32


class _WarningFlags:
    """Thread-safe warning flags to rate-limit log messages."""

    def __init__(self):
        self._lock = threading.Lock()
        self._msssim_warned = False
        self._lpips_warned = False

    def warn_msssim_once(self, log, message: str) -> bool:
        """Log warning if not already warned. Returns True if logged."""
        with self._lock:
            if not self._msssim_warned:
                self._msssim_warned = True
                log.warning(message)
                return True
        return False

    def warn_lpips_once(self, log, message: str) -> bool:
        """Log warning if not already warned. Returns True if logged."""
        with self._lock:
            if not self._lpips_warned:
                self._lpips_warned = True
                log.warning(message)
                return True
        return False

    def reset_msssim(self) -> None:
        with self._lock:
            self._msssim_warned = False

    def reset_lpips(self) -> None:
        with self._lock:
            self._lpips_warned = False

    def reset_all(self) -> None:
        with self._lock:
            self._msssim_warned = False
            self._lpips_warned = False


_warning_flags = _WarningFlags()

# Lock for torch.compile operations (not thread-safe)
_compile_lock = threading.Lock()


def reset_msssim_nan_warning() -> None:
    """Reset MS-SSIM NaN warning flag. Call at start of each validation run."""
    _warning_flags.reset_msssim()


def reset_lpips_nan_warning() -> None:
    """Reset LPIPS NaN warning flag. Call at start of each validation run."""
    _warning_flags.reset_lpips()


def clear_metric_caches() -> None:
    """Clear all cached metric instances and warning flags.

    Call this when:
    - Running multiple training runs in the same process
    - Switching between GPU devices
    - Memory cleanup is needed

    This clears both MS-SSIM and LPIPS cached instances (via lru_cache),
    and resets NaN warning flags.
    """
    _get_msssim_metric.cache_clear()
    _get_lpips_metric.cache_clear()
    _warning_flags.reset_all()
    logger.debug("Cleared metric caches (MS-SSIM, LPIPS)")


def _get_weights_for_size(min_size: int) -> tuple[float, ...]:
    """Get MS-SSIM weights based on image size.

    MS-SSIM requires minimum image size for each scale (halved at each level).
    With kernel size 11, minimum size = 11 * 2^(num_scales-1) + 1.

    Args:
        min_size: Minimum spatial dimension of the image.

    Returns:
        Tuple of weights for MS-SSIM scales.
    """
    if min_size > MSSSIM_5_SCALE_MIN_SIZE:
        # 5 scales (default) - needs 177+ pixels
        return MSSSIM_5_SCALE_WEIGHTS
    elif min_size > MSSSIM_4_SCALE_MIN_SIZE:
        # 4 scales - needs 89+ pixels
        return MSSSIM_4_SCALE_WEIGHTS
    elif min_size > MSSSIM_3_SCALE_MIN_SIZE:
        # 3 scales - needs 45+ pixels
        return MSSSIM_3_SCALE_WEIGHTS
    else:
        # 2 scales - minimum for very small images
        return MSSSIM_2_SCALE_WEIGHTS


@lru_cache(maxsize=8)
def _get_msssim_metric(
    spatial_dims: int,
    weights: tuple[float, ...],
    device_str: str,
) -> 'MultiScaleSSIMMetric':
    """Get or create cached MS-SSIM metric instance.

    Uses lru_cache with maxsize=8 for automatic eviction of old entries.
    Cache key is (spatial_dims, weights, device_str).

    Args:
        spatial_dims: Number of spatial dimensions (2 or 3).
        weights: Tuple of MS-SSIM scale weights (determines number of scales).
        device_str: String representation of device (for hashability).

    Returns:
        MONAI MultiScaleSSIMMetric instance.
    """
    from monai.metrics import MultiScaleSSIMMetric

    metric = MultiScaleSSIMMetric(
        spatial_dims=spatial_dims,
        data_range=1.0,
        weights=weights,
    )
    return metric


def compute_psnr(
    generated: torch.Tensor,
    reference: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    """Compute PSNR between generated and reference images.

    Pure PyTorch implementation for GPU acceleration.
    Works with any tensor shape (2D, 3D, any number of channels).

    Args:
        generated: Generated images in [0, 1] range.
        reference: Reference images in [0, 1] range.
        data_range: Maximum pixel value (default 1.0 for normalized images).

    Returns:
        Average PSNR in dB across batch.
    """
    with torch.no_grad():
        gen = torch.clamp(generated.float(), 0, 1)
        ref = torch.clamp(reference.float(), 0, 1)

        mse = torch.mean((gen - ref) ** 2)

        # Avoid log(0)
        if mse < 1e-10:
            return 100.0

        psnr = 20 * torch.log10(torch.tensor(data_range, device=mse.device) / torch.sqrt(mse))
        return float(psnr.item())


def compute_msssim(
    generated: torch.Tensor,
    reference: torch.Tensor,
    data_range: float = 1.0,
    spatial_dims: int = 2,
) -> float:
    """Compute MS-SSIM between generated and reference images.

    Uses MONAI's MultiScaleSSIMMetric for native 2D/3D support.
    Multi-Scale Structural Similarity measures perceptual quality at multiple
    scales, combining structural similarity with multi-resolution analysis.

    Replaces both SSIM and LPIPS as a single metric that works in 2D and 3D.

    Args:
        generated: Generated images [B, C, H, W] or [B, C, D, H, W] in [0, 1] range.
        reference: Reference images [B, C, H, W] or [B, C, D, H, W] in [0, 1] range.
        data_range: Value range of input images (default: 1.0 for [0, 1]).
        spatial_dims: Number of spatial dimensions (2 or 3).

    Returns:
        Average MS-SSIM across batch (higher is better, 1.0 = identical).
    """
    try:
        with torch.no_grad():
            # Ensure float32 and clamp to valid range
            gen = torch.clamp(generated.float(), 0, data_range)
            ref = torch.clamp(reference.float(), 0, data_range)

            # Normalize to [0, 1] if data_range != 1.0
            if data_range != 1.0:
                gen = gen / data_range
                ref = ref / data_range

            # Get minimum spatial dimension for weight selection
            if spatial_dims == 2:
                min_size = min(gen.shape[2], gen.shape[3])
            else:  # 3D
                min_size = min(gen.shape[2], gen.shape[3], gen.shape[4])

            # Get weights and cached metric
            weights = _get_weights_for_size(min_size)
            metric = _get_msssim_metric(spatial_dims, weights, str(gen.device))

            # Compute MS-SSIM
            # MONAI returns [B, C] tensor, we want scalar mean
            result = metric(gen, ref)

            # Handle NaN values (can occur with edge cases like early epoch garbage)
            if torch.isnan(result).any():
                _warning_flags.warn_msssim_once(
                    logger,
                    "MS-SSIM returned NaN values, replacing with 0 (logging once per validation)"
                )
                result = torch.nan_to_num(result, nan=0.0)

            return float(result.mean().item())

    except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as e:
        # Log full traceback at debug level for debugging, summary at warning level
        logger.warning(f"MS-SSIM computation failed: {e}")
        logger.debug(f"MS-SSIM traceback:\n{traceback.format_exc()}")
        return 0.0


@lru_cache(maxsize=4)
def _get_lpips_metric(
    device_str: str,
    network_type: str = "radimagenet_resnet50",
    cache_dir: str | None = None,
    use_compile: bool = True,
) -> torch.nn.Module:
    """Get or create cached LPIPS metric instance.

    Uses lru_cache with maxsize=4 for automatic eviction of old entries.
    Uses MONAI's PerceptualLoss with pretrained networks.
    Optionally uses torch.compile for faster inference.

    Args:
        device_str: String representation of device (for hashability).
        network_type: Pretrained network type. Options:
            - "radimagenet_resnet50" (default, medical imaging)
            - "resnet50" (ImageNet)
            - "vgg" (VGG-based, classic LPIPS)
            - "alex" (AlexNet-based)
        cache_dir: Optional cache directory for model weights.
        use_compile: Whether to apply torch.compile for speedup (default: True).

    Returns:
        MONAI PerceptualLoss module configured as metric.
    """
    from monai.losses import PerceptualLoss

    device = torch.device(device_str)
    metric = PerceptualLoss(
        spatial_dims=2,
        network_type=network_type,
        cache_dir=cache_dir,
        pretrained=True,
    ).to(device)
    metric.eval()

    # Apply torch.compile for faster inference
    # reduce-overhead is best for repeated small batch inference
    # Use lock since torch.compile is not thread-safe
    if use_compile:
        with _compile_lock:
            try:
                metric = torch.compile(metric, mode="reduce-overhead")
                logger.debug("LPIPS metric compiled with torch.compile")
            except RuntimeError as e:
                logger.warning(f"torch.compile failed for LPIPS, using uncompiled: {e}")

    return metric


def compute_lpips(
    generated: torch.Tensor,
    reference: torch.Tensor,
    device: torch.device | None = None,
    network_type: str = "radimagenet_resnet50",
    cache_dir: str | None = None,
) -> float:
    """Compute LPIPS (perceptual distance) between generated and reference images.

    Uses MONAI's PerceptualLoss with pretrained feature extractors.
    Lower values indicate more similar images (0 = identical).

    Note: Only works with 2D images. For 3D, use MS-SSIM instead.

    Args:
        generated: Generated images [B, C, H, W] in [0, 1] range.
        reference: Reference images [B, C, H, W] in [0, 1] range.
        device: Device for computation (defaults to input tensor device).
        network_type: Pretrained network type:
            - "radimagenet_resnet50" (default, best for medical imaging)
            - "resnet50" (ImageNet pretrained)
            - "vgg", "alex" (classic LPIPS networks)
        cache_dir: Optional cache directory for model weights.

    Returns:
        Average LPIPS score across batch (lower is better, 0 = identical).
    """
    try:
        with torch.no_grad():
            # Determine device
            if device is None:
                device = generated.device

            # Ensure float32 and clamp to valid range
            gen = torch.clamp(generated.float(), 0, 1).to(device)
            ref = torch.clamp(reference.float(), 0, 1).to(device)

            # Get cached metric with lock protection for thread safety
            # The lru_cache + torch.compile combination is not thread-safe
            with _compile_lock:
                metric = _get_lpips_metric(str(device), network_type, cache_dir)

            # Handle channel count (pretrained networks expect 3 channels)
            num_channels = gen.shape[1]
            if num_channels == 1:
                # Grayscale: repeat to 3 channels (standard grayscale→RGB)
                gen = gen.repeat(1, 3, 1, 1)
                ref = ref.repeat(1, 3, 1, 1)
                result = metric(gen, ref)
            elif num_channels == 2:
                # Dual channel: compute per-channel LPIPS and average
                # This matches PerceptualLoss in losses.py for consistency
                gen_0 = gen[:, 0:1].repeat(1, 3, 1, 1)
                ref_0 = ref[:, 0:1].repeat(1, 3, 1, 1)
                gen_1 = gen[:, 1:2].repeat(1, 3, 1, 1)
                ref_1 = ref[:, 1:2].repeat(1, 3, 1, 1)
                result = (metric(gen_0, ref_0) + metric(gen_1, ref_1)) / 2.0
            elif num_channels == 3:
                # Already 3 channels, use directly
                result = metric(gen, ref)
            else:
                # More than 3 channels: compute per-channel and average
                total_lpips = 0.0
                for ch in range(num_channels):
                    gen_ch = gen[:, ch:ch+1].repeat(1, 3, 1, 1)
                    ref_ch = ref[:, ch:ch+1].repeat(1, 3, 1, 1)
                    total_lpips += metric(gen_ch, ref_ch)
                result = total_lpips / num_channels

            # Handle NaN values (can occur with edge cases)
            if isinstance(result, torch.Tensor) and torch.isnan(result).any():
                _warning_flags.warn_lpips_once(
                    logger,
                    "LPIPS returned NaN values, replacing with 0 (logging once per validation)"
                )
                result = torch.nan_to_num(result, nan=0.0)

            return float(result.item()) if isinstance(result, torch.Tensor) else float(result)

    except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as e:
        # Log full traceback at debug level for debugging, summary at warning level
        logger.warning(f"LPIPS computation failed: {e}")
        logger.debug(f"LPIPS traceback:\n{traceback.format_exc()}")
        return 0.0


# =============================================================================
# 3D-Specific Metric Functions (Intentionally Separate)
# =============================================================================
# NOTE: The following _3d functions use fundamentally different algorithms:
# - compute_lpips_3d(): Uses 2.5D slice-by-slice computation because LPIPS
#   relies on 2D pretrained networks (ImageNet/RadImageNet). Cannot unify.
# - compute_fid_3d(), compute_kid_3d(), compute_cmmd_3d(): Use 2.5D feature
#   extraction from InceptionV3/CLIP, which are 2D pretrained networks.
# These are NOT candidates for unification - they solve the "3D metrics with
# 2D pretrained networks" problem using a fundamentally different approach.


def compute_lpips_3d(
    generated: torch.Tensor,
    reference: torch.Tensor,
    device: torch.device | None = None,
    network_type: str = "radimagenet_resnet50",
    chunk_size: int = 32,
) -> float:
    """Compute LPIPS slice-by-slice for 3D volumes (batched for efficiency).

    Since LPIPS uses 2D pretrained networks, this reshapes all depth slices
    into a batch and computes LPIPS in chunked batches for memory efficiency.
    Previous version ran the network D times per volume; this runs ~D/chunk_size times.

    Args:
        generated: Generated volumes [B, C, D, H, W] in [0, 1] range.
        reference: Reference volumes [B, C, D, H, W] in [0, 1] range.
        device: Device for computation (defaults to input tensor device).
        network_type: Pretrained network type (default: radimagenet_resnet50).
        chunk_size: Number of slices to process per forward pass (default: 32).
            Higher values are faster but use more GPU memory.

    Returns:
        Average LPIPS score across all slices (lower is better, 0 = identical).
    """
    B, C, D, H, W = generated.shape
    if D == 0:
        return 0.0

    if device is None:
        device = generated.device

    # Reshape: [B, C, D, H, W] -> [B*D, C, H, W]
    # permute to [B, D, C, H, W] then reshape
    gen_flat = generated.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
    ref_flat = reference.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

    total_slices = B * D
    total_lpips = 0.0

    # Process in chunks to avoid OOM
    for start in range(0, total_slices, chunk_size):
        end = min(start + chunk_size, total_slices)
        gen_chunk = gen_flat[start:end]
        ref_chunk = ref_flat[start:end]

        # compute_lpips returns a scalar (mean over batch)
        # We need the sum, so multiply by chunk size
        chunk_lpips = compute_lpips(gen_chunk, ref_chunk, device=device, network_type=network_type)
        total_lpips += chunk_lpips * (end - start)

    return total_lpips / total_slices


def compute_msssim_2d_slicewise(
    generated: torch.Tensor,
    reference: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    """Compute MS-SSIM slice-by-slice for 3D volumes (batched for efficiency).

    Reshapes all depth slices into a batch dimension and computes 2D MS-SSIM
    in a single call. Previous version ran the metric D times (once per slice);
    this runs once with B*D batch size for ~4x throughput improvement.

    Args:
        generated: Generated volumes [B, C, D, H, W] in [0, 1] range.
        reference: Reference volumes [B, C, D, H, W] in [0, 1] range.
        data_range: Value range of input images (default: 1.0).

    Returns:
        Average MS-SSIM across all slices (higher is better, 1.0 = identical).
    """
    B, C, D, H, W = generated.shape
    if D == 0:
        return 0.0

    # Reshape: [B, C, D, H, W] -> [B*D, C, H, W]
    # permute to [B, D, C, H, W] then reshape to batch all slices together
    gen_flat = generated.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
    ref_flat = reference.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

    # Compute MS-SSIM in single batched call
    return compute_msssim(gen_flat, ref_flat, data_range, spatial_dims=2)


# =============================================================================
# Segmentation Metrics
# =============================================================================

def compute_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
    apply_sigmoid: bool = True,
) -> float:
    """Compute Dice coefficient between prediction and target masks.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Works with logits (applies sigmoid) or probabilities.

    Args:
        pred: Predicted mask (logits or probabilities) [B, C, H, W].
        target: Binary target mask [B, C, H, W] with values in {0, 1}.
        threshold: Threshold for binarization (default 0.5).
        smooth: Smoothing factor to avoid division by zero.
        apply_sigmoid: Whether to apply sigmoid to pred (True for logits,
            False if pred is already probabilities in [0, 1]).

    Returns:
        Dice coefficient (0-1, higher is better, 1.0 = perfect overlap).

    Example:
        >>> logits = model(images)  # Raw model output
        >>> dice = compute_dice(logits, target_masks, apply_sigmoid=True)
    """
    with torch.no_grad():
        pred = pred.float()
        target = target.float()

        # Apply sigmoid if input is logits
        if apply_sigmoid:
            pred = torch.sigmoid(pred)

        # Binarize prediction
        pred_binary = (pred > threshold).float()

        # Compute Dice
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)

        return dice.item()


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
    apply_sigmoid: bool = True,
) -> float:
    """Compute Intersection over Union (IoU/Jaccard) between prediction and target masks.

    IoU = |A ∩ B| / |A ∪ B|

    Works with logits (applies sigmoid) or probabilities.

    Args:
        pred: Predicted mask (logits or probabilities) [B, C, H, W].
        target: Binary target mask [B, C, H, W] with values in {0, 1}.
        threshold: Threshold for binarization (default 0.5).
        smooth: Smoothing factor to avoid division by zero.
        apply_sigmoid: Whether to apply sigmoid to pred (True for logits,
            False if pred is already probabilities in [0, 1]).

    Returns:
        IoU score (0-1, higher is better, 1.0 = perfect overlap).

    Example:
        >>> logits = model(images)  # Raw model output
        >>> iou = compute_iou(logits, target_masks, apply_sigmoid=True)
    """
    with torch.no_grad():
        pred = pred.float()
        target = target.float()

        # Apply sigmoid if input is logits
        if apply_sigmoid:
            pred = torch.sigmoid(pred)

        # Binarize prediction
        pred_binary = (pred > threshold).float()

        # Compute IoU
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)

        return iou.item()


# =============================================================================
# Diversity Metrics (sample-to-sample variation)
# =============================================================================

@torch.no_grad()
def compute_lpips_diversity(
    samples: torch.Tensor,
    device: torch.device | None = None,
    network_type: str = "radimagenet_resnet50",
) -> float:
    """Compute mean pairwise LPIPS diversity between generated samples.

    Measures how different the generated samples are from each other.
    Higher values indicate more diversity (less mode collapse).

    Args:
        samples: Generated images [N, C, H, W] in [0, 1] range.
        device: Device for computation (defaults to input tensor device).
        network_type: Pretrained network type (default: radimagenet_resnet50).

    Returns:
        Mean pairwise LPIPS distance (higher = more diverse).
    """
    n = samples.shape[0]
    if n < 2:
        logger.warning(f"Need at least 2 samples for diversity (got {n})")
        return 0.0

    if device is None:
        device = samples.device

    # Compute all pairwise LPIPS distances
    total_lpips = 0.0
    num_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Get pair of samples
            img_i = samples[i:i+1]  # [1, C, H, W]
            img_j = samples[j:j+1]  # [1, C, H, W]

            lpips_dist = compute_lpips(img_i, img_j, device=device, network_type=network_type)
            total_lpips += lpips_dist
            num_pairs += 1

    return total_lpips / num_pairs if num_pairs > 0 else 0.0


@torch.no_grad()
def compute_msssim_diversity(
    samples: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    """Compute mean pairwise MS-SSIM diversity between generated samples.

    Uses 1 - MS-SSIM as diversity metric (since MS-SSIM measures similarity).
    Higher values indicate more diversity (less mode collapse).

    Args:
        samples: Generated images [N, C, H, W] in [0, 1] range.
        data_range: Value range of input images (default: 1.0).

    Returns:
        Mean pairwise (1 - MS-SSIM) distance (higher = more diverse).
    """
    n = samples.shape[0]
    if n < 2:
        logger.warning(f"Need at least 2 samples for diversity (got {n})")
        return 0.0

    # Compute all pairwise MS-SSIM distances
    total_diversity = 0.0
    num_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Get pair of samples
            img_i = samples[i:i+1]  # [1, C, H, W]
            img_j = samples[j:j+1]  # [1, C, H, W]

            msssim = compute_msssim(img_i, img_j, data_range=data_range, spatial_dims=2)
            diversity = 1.0 - msssim  # Convert similarity to diversity
            total_diversity += diversity
            num_pairs += 1

    return total_diversity / num_pairs if num_pairs > 0 else 0.0


@torch.no_grad()
def compute_lpips_diversity_3d(
    volumes: torch.Tensor,
    device: torch.device | None = None,
    network_type: str = "radimagenet_resnet50",
) -> float:
    """Compute mean pairwise LPIPS diversity for 3D volumes (same-slice comparison).

    Compares the same slice index across different volumes, then averages.
    This measures generation diversity without mixing anatomical variation.

    For B volumes with D slices each:
    - For each slice index d in [0, D):
      - Compare all pairs of slice d across the B volumes
    - Average across all slice indices

    Args:
        volumes: Generated volumes [B, C, D, H, W] in [0, 1] range.
        device: Device for computation (defaults to input tensor device).
        network_type: Pretrained network type (default: radimagenet_resnet50).

    Returns:
        Mean pairwise LPIPS distance across same-slice comparisons (higher = more diverse).
    """
    B, C, D, H, W = volumes.shape
    if B < 2:
        logger.warning(f"Need at least 2 volumes for diversity (got {B})")
        return 0.0

    if device is None:
        device = volumes.device

    total_diversity = 0.0
    num_slices = 0

    for d in range(D):
        # Get slice d from all volumes: [B, C, H, W]
        slices = volumes[:, :, d, :, :]

        # Compute pairwise LPIPS for this slice across volumes
        slice_diversity = compute_lpips_diversity(slices, device=device, network_type=network_type)
        total_diversity += slice_diversity
        num_slices += 1

    return total_diversity / num_slices if num_slices > 0 else 0.0


@torch.no_grad()
def compute_msssim_diversity_3d(
    volumes: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    """Compute mean pairwise MS-SSIM diversity for 3D volumes (same-slice comparison).

    Compares the same slice index across different volumes, then averages.
    This measures generation diversity without mixing anatomical variation.

    For B volumes with D slices each:
    - For each slice index d in [0, D):
      - Compare all pairs of slice d across the B volumes
    - Average across all slice indices

    Args:
        volumes: Generated volumes [B, C, D, H, W] in [0, 1] range.
        data_range: Value range of input images (default: 1.0).

    Returns:
        Mean pairwise (1 - MS-SSIM) distance across same-slice comparisons (higher = more diverse).
    """
    B, C, D, H, W = volumes.shape
    if B < 2:
        logger.warning(f"Need at least 2 volumes for diversity (got {B})")
        return 0.0

    total_diversity = 0.0
    num_slices = 0

    for d in range(D):
        # Get slice d from all volumes: [B, C, H, W]
        slices = volumes[:, :, d, :, :]

        # Compute pairwise MS-SSIM diversity for this slice across volumes
        slice_diversity = compute_msssim_diversity(slices, data_range=data_range)
        total_diversity += slice_diversity
        num_slices += 1

    return total_diversity / num_slices if num_slices > 0 else 0.0
