"""
Generation quality metrics for diffusion models.

Tracks distributional distance between generated and real images using:
- KID: Kernel Inception Distance (unbiased, small sample friendly)
- CMMD: CLIP Maximum Mean Discrepancy (domain-aware via BiomedCLIP)
- FID: Fréchet Inception Distance (for final test evaluation)

These metrics compare feature distributions rather than individual samples,
enabling detection of mode collapse and overfitting during training.

Usage:
    config = GenerationMetricsConfig(enabled=True)
    gen_metrics = GenerationMetrics(config, device, run_dir)

    # At training start
    gen_metrics.set_fixed_conditioning(train_dataset, num_masks=500)
    gen_metrics.cache_reference_features(train_loader, val_loader, experiment_id)

    # Every epoch
    results = gen_metrics.compute_epoch_metrics(model, strategy, mode)

    # Every N epochs
    extended = gen_metrics.compute_extended_metrics(model, strategy, mode)

    # Final test
    test_results = gen_metrics.compute_test_metrics(model, strategy, mode, test_loader)

Reference:
- KID: https://arxiv.org/abs/1801.01401 (MMD-based, unbiased)
- CMMD: CLIP-based MMD for domain-specific comparison
- FID: https://arxiv.org/abs/1706.08500 (Fréchet distance on Inception features)
"""
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

from .feature_extractors import ResNet50Features, BiomedCLIPFeatures
from .quality import (
    compute_lpips_diversity,
    compute_msssim_diversity,
    compute_lpips_diversity_3d,
    compute_msssim_diversity_3d,
)
from medgen.data.loaders.seg_conditioned import compute_size_bins, DEFAULT_BIN_EDGES

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GenerationMetricsConfig:
    """Configuration for generation quality metrics.

    Attributes:
        enabled: Whether generation metrics are enabled.
        samples_per_epoch: Number of samples for quick metrics (every epoch).
        samples_extended: Number of samples for extended metrics (every N epochs).
        samples_test: Number of samples for final test evaluation.
        steps_per_epoch: Denoising steps for quick metrics.
        steps_extended: Denoising steps for extended metrics.
        steps_test: Denoising steps for test evaluation.
        cache_dir: Directory for caching reference features.
        feature_batch_size: Batch size for feature extraction.
        cfg_scale: CFG scale for generation (requires CFG dropout during training).
        original_depth: Original depth before padding (for 3D). Padded slices excluded from metrics.
        size_bin_edges: Bin edges in mm for size bin adherence (seg_conditioned mode).
        size_bin_fov_mm: Field of view in mm for computing pixel spacing.
    """
    enabled: bool = True
    samples_per_epoch: int = 100
    samples_extended: int = 500
    samples_test: int = 1000
    steps_per_epoch: int = 10
    steps_extended: int = 25
    steps_test: int = 50
    cache_dir: str = ".cache/generation_features"
    feature_batch_size: int = 16  # Set by trainer to match training.batch_size
    cfg_scale: float = 2.0  # CFG scale for generation (requires CFG dropout during training)
    original_depth: Optional[int] = None  # Original depth before padding (for 3D metrics)
    # Size bin adherence config (for seg_conditioned mode)
    size_bin_edges: Optional[List[float]] = None  # Bin edges in mm (default from loader)
    size_bin_fov_mm: float = 240.0  # Field of view in mm


# =============================================================================
# Metric Computation Functions
# =============================================================================

def compute_kid(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
    subset_size: int = 100,
    num_subsets: int = 50,
) -> Tuple[float, float]:
    """Compute Kernel Inception Distance between real and generated features.

    KID uses a polynomial kernel k(x,y) = (x^T y / d + 1)^3 and computes
    an unbiased MMD estimate. Unlike FID, KID has unbiased estimator and
    works well with small sample sizes.

    Args:
        real_features: Real image features [N, D].
        generated_features: Generated image features [M, D].
        subset_size: Size of random subsets for MMD estimation.
        num_subsets: Number of random subsets to average over.

    Returns:
        Tuple of (kid_mean, kid_std).
    """
    real_features = real_features.float()
    generated_features = generated_features.float()

    n_real = real_features.shape[0]
    n_gen = generated_features.shape[0]
    d = real_features.shape[1]

    # Adjust subset size if we don't have enough samples
    subset_size = min(subset_size, n_real, n_gen)

    if subset_size < 2:
        logger.warning(f"Not enough samples for KID (real={n_real}, gen={n_gen})")
        return 0.0, 0.0

    def polynomial_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Polynomial kernel k(x,y) = (x^T y / d + 1)^3."""
        return ((x @ y.T) / d + 1) ** 3

    kid_values = []

    for _ in range(num_subsets):
        # Random subset indices
        real_idx = torch.randperm(n_real)[:subset_size]
        gen_idx = torch.randperm(n_gen)[:subset_size]

        real_subset = real_features[real_idx]
        gen_subset = generated_features[gen_idx]

        # Compute kernel matrices
        k_rr = polynomial_kernel(real_subset, real_subset)
        k_gg = polynomial_kernel(gen_subset, gen_subset)
        k_rg = polynomial_kernel(real_subset, gen_subset)

        # Unbiased MMD estimate
        # MMD^2 = E[k(r,r')] + E[k(g,g')] - 2*E[k(r,g)]
        # Unbiased: exclude diagonal for same-set terms
        n = subset_size

        # Sum excluding diagonal
        k_rr_sum = k_rr.sum() - k_rr.trace()
        k_gg_sum = k_gg.sum() - k_gg.trace()

        mmd_squared = (
            k_rr_sum / (n * (n - 1)) +
            k_gg_sum / (n * (n - 1)) -
            2 * k_rg.mean()
        )

        kid_values.append(float(mmd_squared.item()))

    kid_mean = np.mean(kid_values)
    kid_std = np.std(kid_values)

    return kid_mean, kid_std


def compute_cmmd(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
    kernel_bandwidth: Optional[float] = None,
) -> float:
    """Compute CLIP Maximum Mean Discrepancy with RBF kernel.

    CMMD uses an RBF (Gaussian) kernel on CLIP embeddings. The bandwidth
    is set using the median heuristic if not provided.

    Args:
        real_features: Real image CLIP features [N, D].
        generated_features: Generated image CLIP features [M, D].
        kernel_bandwidth: RBF kernel bandwidth (sigma). If None, uses median heuristic.

    Returns:
        CMMD value (lower is better).
    """
    real_features = real_features.float()
    generated_features = generated_features.float()

    # L2 normalize features (CLIP embeddings are typically normalized)
    real_features = F.normalize(real_features, p=2, dim=1)
    generated_features = F.normalize(generated_features, p=2, dim=1)

    # Median heuristic for bandwidth if not provided
    if kernel_bandwidth is None:
        # Compute pairwise distances for a random subset
        n_sample = min(500, real_features.shape[0], generated_features.shape[0])
        real_sample = real_features[:n_sample]
        gen_sample = generated_features[:n_sample]

        # Pairwise squared distances
        all_features = torch.cat([real_sample, gen_sample], dim=0)
        dists = torch.cdist(all_features, all_features, p=2)
        median_dist = torch.median(dists[dists > 0])
        kernel_bandwidth = float(median_dist.item())

        # Avoid very small bandwidth
        kernel_bandwidth = max(kernel_bandwidth, 0.1)

    def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """Compute RBF kernel matrix."""
        # Pairwise squared distances
        xx = (x ** 2).sum(dim=1, keepdim=True)
        yy = (y ** 2).sum(dim=1, keepdim=True)
        xy = x @ y.T
        distances = xx + yy.T - 2 * xy
        return torch.exp(-distances / (2 * sigma ** 2))

    # Compute kernel matrices
    k_rr = rbf_kernel(real_features, real_features, kernel_bandwidth)
    k_gg = rbf_kernel(generated_features, generated_features, kernel_bandwidth)
    k_rg = rbf_kernel(real_features, generated_features, kernel_bandwidth)

    # Unbiased MMD^2 estimate
    n, m = real_features.shape[0], generated_features.shape[0]

    # Zero diagonal for unbiased estimate
    k_rr_no_diag = k_rr.clone()
    k_rr_no_diag.fill_diagonal_(0)
    k_gg_no_diag = k_gg.clone()
    k_gg_no_diag.fill_diagonal_(0)

    mmd_squared = (
        k_rr_no_diag.sum() / (n * (n - 1)) +
        k_gg_no_diag.sum() / (m * (m - 1)) -
        2 * k_rg.sum() / (n * m)
    )

    # Return sqrt for CMMD (like standard deviation)
    return float(torch.sqrt(torch.clamp(mmd_squared, min=0)).item())


def compute_fid(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
) -> float:
    """Compute Fréchet Inception Distance between real and generated features.

    FID measures the Fréchet distance between two multivariate Gaussians
    fitted to the feature distributions.

    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*sqrt(Sigma_r @ Sigma_g))

    Args:
        real_features: Real image features [N, D].
        generated_features: Generated image features [M, D].

    Returns:
        FID value (lower is better).
    """
    real_features = real_features.float().cpu().numpy()
    generated_features = generated_features.float().cpu().numpy()

    # Compute mean and covariance
    mu_r = np.mean(real_features, axis=0)
    mu_g = np.mean(generated_features, axis=0)

    sigma_r = np.cov(real_features, rowvar=False)
    sigma_g = np.cov(generated_features, rowvar=False)

    # Handle edge case of single sample
    if real_features.shape[0] == 1:
        sigma_r = np.zeros_like(sigma_r)
    if generated_features.shape[0] == 1:
        sigma_g = np.zeros_like(sigma_g)

    # Compute FID
    diff = mu_r - mu_g
    diff_squared = np.sum(diff ** 2)

    # Matrix square root using scipy
    try:
        from scipy import linalg
        covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)

        # Handle numerical issues
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        trace_term = np.trace(sigma_r + sigma_g - 2 * covmean)

        fid = diff_squared + trace_term
        return float(fid)

    except Exception as e:
        logger.warning(f"FID computation failed: {e}")
        return float('inf')


# =============================================================================
# Reference Feature Cache
# =============================================================================

class ReferenceFeatureCache:
    """Caches reference features from train/val datasets.

    Features are extracted once at training start and reused for all epochs.
    Cached to disk for fast loading on training resumption.

    Args:
        resnet_extractor: ResNet50 feature extractor.
        biomed_extractor: BiomedCLIP feature extractor.
        cache_dir: Directory for caching features.
        device: PyTorch device.
        batch_size: Batch size for feature extraction.
    """

    def __init__(
        self,
        resnet_extractor: ResNet50Features,
        biomed_extractor: BiomedCLIPFeatures,
        cache_dir: Path,
        device: torch.device,
        batch_size: int = 32,
    ) -> None:
        self.resnet = resnet_extractor
        self.biomed = biomed_extractor
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.batch_size = batch_size

        # Feature storage
        self.train_resnet: Optional[torch.Tensor] = None
        self.train_biomed: Optional[torch.Tensor] = None
        self.val_resnet: Optional[torch.Tensor] = None
        self.val_biomed: Optional[torch.Tensor] = None

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _extract_features_from_loader(
        self,
        dataloader: DataLoader,
        extractor: nn.Module,
        name: str,
        max_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract features from a dataloader, filtering for positive masks.

        Handles both 2D [B, C, H, W] and 3D [B, C, D, H, W] inputs.
        For 3D, extracts features slice-wise using extract_features_3d.

        Args:
            dataloader: DataLoader providing batches.
            extractor: Feature extractor module.
            name: Name for logging.
            max_samples: Maximum samples to process.

        Returns:
            Feature tensor [N, D].
        """
        all_features = []
        sample_count = 0

        for batch in dataloader:
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch.get('images', batch.get('image'))
                masks = batch.get('seg', batch.get('mask', batch.get('labels')))
            elif isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    images, masks = batch
                else:
                    images = batch[0]
                    masks = None
            else:
                images = batch
                masks = None

            images = images.to(self.device)
            is_3d = images.ndim == 5  # [B, C, D, H, W]

            # Filter for positive masks (has tumor)
            if masks is not None:
                masks = masks.to(self.device)
                # Check if mask has any positive pixels per sample
                # Handle both 2D (dim 1,2,3) and 3D (dim 1,2,3,4)
                sum_dims = tuple(range(1, masks.ndim))
                positive_mask = masks.sum(dim=sum_dims) > 0
                if positive_mask.sum() == 0:
                    continue
                images = images[positive_mask]

            if is_3d:
                # 3D: extract slice-wise features
                features = extract_features_3d(images, extractor, chunk_size=self.batch_size)
                all_features.append(features.cpu())
                # For 3D, sample_count is number of slices
                sample_count += features.shape[0]
            else:
                # 2D: extract in sub-batches
                for i in range(0, images.shape[0], self.batch_size):
                    sub_batch = images[i:i + self.batch_size]
                    features = extractor.extract_features(sub_batch)
                    all_features.append(features.cpu())
                    sample_count += sub_batch.shape[0]

                    if max_samples and sample_count >= max_samples:
                        break

            if max_samples and sample_count >= max_samples:
                break

        if not all_features:
            logger.warning(f"No features extracted for {name}")
            return torch.empty(0)

        features = torch.cat(all_features, dim=0)
        logger.info(f"Extracted {features.shape[0]} {name} features")
        return features

    def extract_and_cache(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        experiment_id: str,
    ) -> None:
        """Extract and cache reference features.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            experiment_id: Unique experiment identifier for cache file.
        """
        cache_file = self.cache_dir / f"{experiment_id}_reference_features.pt"

        if cache_file.exists():
            logger.info(f"Loading cached reference features from {cache_file}")
            cached = torch.load(cache_file, map_location='cpu', weights_only=True)
            # Support both old (inception) and new (resnet) cache format
            self.train_resnet = cached.get('train_resnet', cached.get('train_inception'))
            self.train_biomed = cached['train_biomed']
            self.val_resnet = cached.get('val_resnet', cached.get('val_inception'))
            self.val_biomed = cached['val_biomed']
            logger.info(
                f"Loaded: train={self.train_resnet.shape[0]}, "
                f"val={self.val_resnet.shape[0]} samples"
            )
            return

        logger.info("Extracting reference features (this happens once per experiment)...")

        # Extract train features
        logger.info("  Extracting train ResNet50 features...")
        self.train_resnet = self._extract_features_from_loader(
            train_loader, self.resnet, "train_resnet"
        )

        logger.info("  Extracting train BiomedCLIP features...")
        self.train_biomed = self._extract_features_from_loader(
            train_loader, self.biomed, "train_biomed"
        )

        # Extract val features
        logger.info("  Extracting val ResNet50 features...")
        self.val_resnet = self._extract_features_from_loader(
            val_loader, self.resnet, "val_resnet"
        )

        logger.info("  Extracting val BiomedCLIP features...")
        self.val_biomed = self._extract_features_from_loader(
            val_loader, self.biomed, "val_biomed"
        )

        # Cache to disk
        torch.save({
            'train_resnet': self.train_resnet,
            'train_biomed': self.train_biomed,
            'val_resnet': self.val_resnet,
            'val_biomed': self.val_biomed,
        }, cache_file)
        logger.info(f"Cached features to {cache_file}")


# =============================================================================
# Main Generation Metrics Class
# =============================================================================

class GenerationMetrics:
    """Tracks generation quality during diffusion training.

    Computes KID and CMMD metrics comparing generated samples against
    train and validation distributions to detect overfitting.

    Args:
        config: GenerationMetricsConfig instance.
        device: PyTorch device.
        run_dir: Run directory for caching.
        space: Optional DiffusionSpace for latent decoding.
        mode_name: Training mode name (seg, bravo, dual, etc.).

    Example:
        config = GenerationMetricsConfig(enabled=True)
        metrics = GenerationMetrics(config, device, run_dir, mode_name='bravo')

        # At training start
        metrics.set_fixed_conditioning(train_dataset, num_masks=500)
        metrics.cache_reference_features(train_loader, val_loader, "exp1")

        # Every epoch
        results = metrics.compute_epoch_metrics(model, strategy, mode)
        # Returns: {'KID_train_mean': 0.05, 'KID_val_mean': 0.06, ...}
    """

    def __init__(
        self,
        config: GenerationMetricsConfig,
        device: torch.device,
        run_dir: Path,
        space: Optional[Any] = None,
        mode_name: str = 'bravo',
    ) -> None:
        self.config = config
        self.device = device
        self.run_dir = Path(run_dir)
        self.space = space  # DiffusionSpace for latent decoding
        self.mode_name = mode_name
        self.is_seg_mode = mode_name in ('seg', 'seg_conditioned')

        # Initialize feature extractors (lazy-loaded)
        self.resnet = ResNet50Features(device, cache_dir=Path(config.cache_dir))
        self.biomed = BiomedCLIPFeatures(device, cache_dir=config.cache_dir)

        # Reference feature cache
        self.cache = ReferenceFeatureCache(
            self.resnet,
            self.biomed,
            Path(config.cache_dir),
            device,
            config.feature_batch_size,
        )

        # Fixed conditioning masks (loaded once, used every epoch)
        self.fixed_conditioning_masks: Optional[torch.Tensor] = None
        self.fixed_gt_images: Optional[torch.Tensor] = None
        self.fixed_size_bins: Optional[torch.Tensor] = None  # For seg_conditioned mode

        if self.is_seg_mode:
            logger.info(f"GenerationMetrics: {mode_name} mode - will threshold output at 0.5")

    def set_fixed_conditioning(
        self,
        train_dataset: Dataset,
        num_masks: int = 500,
        seg_channel_idx: int = 1,
    ) -> None:
        """Load fixed conditioning masks from training dataset.

        Uses the same masks every epoch for reproducible comparisons.

        Args:
            train_dataset: Training dataset to sample from.
            num_masks: Number of masks to sample.
            seg_channel_idx: Channel index for segmentation mask.
        """
        logger.info(f"Sampling {num_masks} fixed conditioning masks...")

        masks = []
        gt_images = []
        size_bins_list = []  # For seg_conditioned mode
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
            current_size_bins = None
            if isinstance(data, dict):
                # Dict format - check multiple key variants
                # Latent dataset: {'latent': ..., 'seg_mask': ..., 'latent_seg': ...}
                # Pixel dataset: {'image': ..., 'seg': ...}
                image = data.get('image', data.get('images', data.get('latent')))
                seg_data = data.get('seg', data.get('mask', data.get('labels', data.get('seg_mask'))))

                # For latent bravo_seg_cond mode, use seg_mask (pixel-space) for conditioning
                # since generation metrics compare actual tumor masks, not latent representations
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
                is_seg_conditioned_input = bin_maps is not None and second.dim() == 1
                if is_seg_conditioned_input:
                    tensor = first
                    current_size_bins = second.long()
                    # bin_maps are for input conditioning, not needed for metrics sampling
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
                if isinstance(data, tuple) and is_seg_conditioned:
                    size_bins_list.append(current_size_bins)
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

        self.fixed_conditioning_masks = torch.stack(masks).to(self.device)
        if gt_images:
            self.fixed_gt_images = torch.stack(gt_images).to(self.device)
        if size_bins_list:
            self.fixed_size_bins = torch.stack(size_bins_list).to(self.device)
            logger.info(f"Loaded {len(size_bins_list)} fixed size_bins for seg_conditioned mode")

        logger.info(f"Loaded {len(masks)} fixed conditioning masks")

    def cache_reference_features(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        experiment_id: str,
    ) -> None:
        """Cache reference features from train and val datasets.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            experiment_id: Unique experiment identifier.
        """
        self.cache.extract_and_cache(train_loader, val_loader, experiment_id)

    @torch.no_grad()
    def _generate_samples(
        self,
        model: nn.Module,
        strategy: Any,
        mode: Any,
        num_samples: int,
        num_steps: int,
        batch_size: int = 16,
    ) -> torch.Tensor:
        """Generate samples using fixed conditioning in batches.

        Args:
            model: Diffusion model.
            strategy: Diffusion strategy (DDPM/RFlow).
            mode: Training mode (bravo/dual/seg).
            num_samples: Number of samples to generate.
            num_steps: Number of denoising steps.
            batch_size: Batch size for generation (to avoid OOM).

        Returns:
            Generated samples [N, C, H, W].
        """
        if self.fixed_conditioning_masks is None:
            raise RuntimeError("Must call set_fixed_conditioning() first")

        # Detect 3D from tensor dimensions: 3D is [N, C, D, H, W] with ndim=5
        is_3d = self.fixed_conditioning_masks.ndim == 5

        # 3D generation uses batch_size=1 to avoid OOM at high resolutions
        # (256x256x160 with CFG requires ~50GB per batch of 2 due to dual forward passes)
        if is_3d:
            batch_size = 1

        # Cap batch_size at available masks to avoid errors with small mask counts
        num_available = self.fixed_conditioning_masks.shape[0]
        batch_size = min(batch_size, num_available)

        # Round up to full batches for torch.compile consistency
        # e.g., 100 samples with batch_size=16 → 112 samples (7 full batches)
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

        # Generate in batches to avoid OOM with CUDA graphs
        for start_idx in range(0, num_to_use, batch_size):
            end_idx = min(start_idx + batch_size, num_to_use)
            masks = self.fixed_conditioning_masks[start_idx:end_idx]

            # Generate noise
            noise = torch.randn_like(masks)

            if out_channels == 2:  # Dual mode
                noise_pre = torch.randn_like(masks)
                noise_gd = torch.randn_like(masks)
                model_input = torch.cat([noise_pre, noise_gd, masks], dim=1)
            elif in_channels == 1:  # Unconditional modes (seg, seg_conditioned)
                # No channel concatenation - conditioning via embedding (if any)
                model_input = noise
            else:  # Conditional single channel modes (bravo)
                model_input = torch.cat([noise, masks], dim=1)

            # Get size_bins for this batch if in seg_conditioned mode
            batch_size_bins = None
            if self.fixed_size_bins is not None:
                batch_size_bins = self.fixed_size_bins[start_idx:end_idx]

            # Generate samples
            with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                samples = strategy.generate(
                    model, model_input, num_steps=num_steps, device=self.device,
                    size_bins=batch_size_bins,
                    cfg_scale=self.config.cfg_scale,
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
        result = result.to(self.device)
        if self.space is not None and hasattr(self.space, 'scale_factor') and self.space.scale_factor > 1:
            result = self.space.decode(result)

        # Threshold seg mode output at 0.5 to get binary masks
        if self.is_seg_mode:
            result = (result > 0.5).float()

        return result

    def _extract_features_batched(
        self,
        samples: torch.Tensor,
        extractor: nn.Module,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract features from samples in batches.

        Args:
            samples: Input samples [N, C, H, W] or [N, C, D, H, W] for 3D.
            extractor: Feature extractor (ResNet50 or BiomedCLIP).
            batch_size: Batch size for extraction (default: config.feature_batch_size).

        Returns:
            Feature tensor [N, D] for 2D or [N*D_orig, D] for 3D (slice-wise, padding removed).
        """
        if batch_size is None:
            batch_size = self.config.feature_batch_size

        # Handle 3D volumes via slice-wise extraction (2.5D approach)
        is_3d = samples.ndim == 5
        if is_3d:
            # Remove padded slices if original_depth is specified
            if self.config.original_depth is not None:
                original_depth = self.config.original_depth
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

    def _compute_metrics_against_reference(
        self,
        gen_resnet: torch.Tensor,
        gen_biomed: torch.Tensor,
        ref_resnet: torch.Tensor,
        ref_biomed: torch.Tensor,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Compute KID and CMMD against reference features.

        Args:
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
            ref_resnet.to(self.device),
            gen_resnet.to(self.device),
        )
        results[f'{prefix}KID_mean'] = kid_mean
        results[f'{prefix}KID_std'] = kid_std

        # CMMD
        cmmd = compute_cmmd(
            ref_biomed.to(self.device),
            gen_biomed.to(self.device),
        )
        results[f'{prefix}CMMD'] = cmmd

        return results

    def _compute_diversity_metrics(
        self,
        samples: torch.Tensor,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Compute diversity metrics for generated samples.

        Measures how different the generated samples are from each other
        using LPIPS and MS-SSIM diversity. Higher values indicate more
        diversity (less mode collapse).

        Args:
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
                    samples, device=self.device
                )
                msssim_div = compute_msssim_diversity_3d(samples)
            else:
                # 2D: compare all generated images
                lpips_div = compute_lpips_diversity(
                    samples, device=self.device
                )
                msssim_div = compute_msssim_diversity(samples)

            # Use special prefix format for separate TensorBoard section
            # "Diversity/" prefix tells log_generation to use Generation_Diversity/ section
            results[f'Diversity/{prefix}LPIPS'] = lpips_div
            results[f'Diversity/{prefix}MSSSIM'] = msssim_div

        except Exception as e:
            logger.warning(f"Diversity computation failed: {e}")

        return results

    def _compute_size_bin_adherence(
        self,
        generated_masks: torch.Tensor,
        conditioning_bins: torch.Tensor,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Compute size bin adherence metrics for seg_conditioned mode.

        Measures how well generated masks match their conditioning size bins.

        Args:
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

        results = {}

        # Get bin config
        bin_edges = self.config.size_bin_edges or DEFAULT_BIN_EDGES
        fov_mm = self.config.size_bin_fov_mm
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
        self,
        model: nn.Module,
        strategy: Any,
        mode: Any,
    ) -> Dict[str, float]:
        """Compute quick generation metrics (every epoch).

        Uses fewer samples and steps for fast feedback.

        Args:
            model: Diffusion model.
            strategy: Diffusion strategy.
            mode: Training mode.

        Returns:
            Dictionary with KID/CMMD vs train and val.
        """
        # Preserve RNG state (per pitfall #42)
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(self.device)

        try:
            # Generate samples
            samples = self._generate_samples(
                model, strategy, mode,
                self.config.samples_per_epoch,
                self.config.steps_per_epoch,
            )

            # Extract features in batches
            gen_resnet = self._extract_features_batched(samples, self.resnet)
            gen_biomed = self._extract_features_batched(samples, self.biomed)

            # Compute diversity metrics (2D - every epoch)
            diversity_metrics = self._compute_diversity_metrics(samples, prefix="")

            # Compute size bin adherence for seg_conditioned mode
            size_bin_metrics = {}
            if self.mode_name == 'seg_conditioned' and self.fixed_size_bins is not None:
                size_bin_metrics = self._compute_size_bin_adherence(
                    samples, self.fixed_size_bins[:samples.shape[0]], prefix=""
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
            train_metrics = self._compute_metrics_against_reference(
                gen_resnet, gen_biomed,
                self.cache.train_resnet, self.cache.train_biomed,
                prefix="",
            )
            for key, value in train_metrics.items():
                results[f'{key}_train'] = value

            # Metrics vs val
            val_metrics = self._compute_metrics_against_reference(
                gen_resnet, gen_biomed,
                self.cache.val_resnet, self.cache.val_biomed,
                prefix="",
            )
            for key, value in val_metrics.items():
                results[f'{key}_val'] = value

            return results

        finally:
            # Restore RNG state
            torch.set_rng_state(rng_state)
            torch.cuda.set_rng_state(cuda_rng_state, self.device)

    def compute_extended_metrics(
        self,
        model: nn.Module,
        strategy: Any,
        mode: Any,
    ) -> Dict[str, float]:
        """Compute extended generation metrics (every N epochs).

        Uses more samples and steps for detailed analysis.

        Args:
            model: Diffusion model.
            strategy: Diffusion strategy.
            mode: Training mode.

        Returns:
            Dictionary with extended_ prefixed KID/CMMD metrics.
        """
        # Preserve RNG state
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(self.device)

        try:
            # Generate samples
            samples = self._generate_samples(
                model, strategy, mode,
                self.config.samples_extended,
                self.config.steps_extended,
            )

            # Extract features in batches
            gen_resnet = self._extract_features_batched(samples, self.resnet)
            gen_biomed = self._extract_features_batched(samples, self.biomed)

            # Compute diversity metrics (2D - extended)
            diversity_metrics = self._compute_diversity_metrics(samples, prefix="extended_")

            # Compute size bin adherence for seg_conditioned mode
            size_bin_metrics = {}
            if self.mode_name == 'seg_conditioned' and self.fixed_size_bins is not None:
                size_bin_metrics = self._compute_size_bin_adherence(
                    samples, self.fixed_size_bins[:samples.shape[0]], prefix="extended_"
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
            train_metrics = self._compute_metrics_against_reference(
                gen_resnet, gen_biomed,
                self.cache.train_resnet, self.cache.train_biomed,
                prefix="extended_",
            )
            for key, value in train_metrics.items():
                results[f'{key}_train'] = value

            # Metrics vs val
            val_metrics = self._compute_metrics_against_reference(
                gen_resnet, gen_biomed,
                self.cache.val_resnet, self.cache.val_biomed,
                prefix="extended_",
            )
            for key, value in val_metrics.items():
                results[f'{key}_val'] = value

            return results

        finally:
            # Restore RNG state
            torch.set_rng_state(rng_state)
            torch.cuda.set_rng_state(cuda_rng_state, self.device)

    def compute_metrics_from_samples(
        self,
        samples: torch.Tensor,
        extended: bool = False,
    ) -> Dict[str, float]:
        """Compute metrics from pre-generated samples.

        This method is useful when samples are generated externally (e.g., 3D volumes).
        Handles both 2D [N, C, H, W] and 3D [N, C, D, H, W] samples.

        Args:
            samples: Generated samples tensor.
            extended: If True, use "extended_" prefix for metric names.

        Returns:
            Dictionary with KID/CMMD vs train and val.
        """
        if self.cache.train_resnet is None:
            logger.warning("Reference features not cached, cannot compute metrics")
            return {}

        prefix = "extended_" if extended else ""
        is_3d = samples.ndim == 5
        n_samples = samples.shape[0]

        # Extract features
        if is_3d:
            gen_resnet = extract_features_3d(samples, self.resnet, chunk_size=self.config.feature_batch_size)
            gen_biomed = extract_features_3d(samples, self.biomed, chunk_size=self.config.feature_batch_size)
        else:
            gen_resnet = self._extract_features_batched(samples, self.resnet)
            gen_biomed = self._extract_features_batched(samples, self.biomed)

        # Compute diversity metrics (3D: only when 2+ volumes, typically 4+ for extended/test)
        # For 3D, diversity is computed per-slice across volumes, so we need multiple volumes
        diversity_metrics = {}
        if n_samples >= 2:
            diversity_metrics = self._compute_diversity_metrics(samples, prefix=prefix)

        # Free samples
        del samples
        torch.cuda.empty_cache()

        results = {}

        # Add diversity metrics
        results.update(diversity_metrics)

        # Metrics vs train
        train_metrics = self._compute_metrics_against_reference(
            gen_resnet, gen_biomed,
            self.cache.train_resnet, self.cache.train_biomed,
            prefix=prefix,
        )
        for key, value in train_metrics.items():
            results[f'{key}_train'] = value

        # Metrics vs val
        if self.cache.val_resnet is not None:
            val_metrics = self._compute_metrics_against_reference(
                gen_resnet, gen_biomed,
                self.cache.val_resnet, self.cache.val_biomed,
                prefix=prefix,
            )
            for key, value in val_metrics.items():
                results[f'{key}_val'] = value

        return results

    def compute_test_metrics(
        self,
        model: nn.Module,
        strategy: Any,
        mode: Any,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """Compute full test generation metrics with FID.

        Uses the most samples and steps for final evaluation.
        Optionally compares against test set if loader provided.

        Args:
            model: Diffusion model.
            strategy: Diffusion strategy.
            mode: Training mode.
            test_loader: Optional test data loader for FID vs test.

        Returns:
            Dictionary with FID, KID, CMMD test metrics.
        """
        # Generate samples
        samples = self._generate_samples(
            model, strategy, mode,
            self.config.samples_test,
            self.config.steps_test,
        )

        # Extract features in batches
        gen_resnet = self._extract_features_batched(samples, self.resnet)
        gen_biomed = self._extract_features_batched(samples, self.biomed)

        # Compute diversity metrics (2D - test)
        diversity_metrics = self._compute_diversity_metrics(samples, prefix="")

        # Free samples
        del samples
        torch.cuda.empty_cache()

        results = {}

        # Add diversity metrics
        results.update(diversity_metrics)

        # If test loader provided, compute metrics vs test
        if test_loader is not None:
            # Extract test features
            test_resnet = self.cache._extract_features_from_loader(
                test_loader, self.resnet, "test_resnet",
                max_samples=self.config.samples_test,
            )
            test_biomed = self.cache._extract_features_from_loader(
                test_loader, self.biomed, "test_biomed",
                max_samples=self.config.samples_test,
            )

            # FID (only for test)
            fid = compute_fid(test_resnet, gen_resnet)
            results['FID'] = fid

            # KID and CMMD vs test
            test_metrics = self._compute_metrics_against_reference(
                gen_resnet, gen_biomed,
                test_resnet, test_biomed,
                prefix="",
            )
            results.update(test_metrics)
        else:
            # Use val as reference if no test loader
            val_metrics = self._compute_metrics_against_reference(
                gen_resnet, gen_biomed,
                self.cache.val_resnet, self.cache.val_biomed,
                prefix="",
            )
            results.update(val_metrics)

            # FID vs val
            fid = compute_fid(
                self.cache.val_resnet,
                gen_resnet,
            )
            results['FID'] = fid

        return results


# =============================================================================
# 3D Slice-Wise Feature Extraction (2.5D Approach)
# =============================================================================

def volumes_to_slices(volumes: torch.Tensor) -> torch.Tensor:
    """Reshape 3D volumes to 2D slices for feature extraction.

    Converts [B, C, D, H, W] → [B*D, C, H, W] by treating each depth slice
    as an independent 2D sample. This enables using 2D feature extractors
    (ResNet50, BiomedCLIP) on 3D volumes.

    Args:
        volumes: 5D tensor [B, C, D, H, W] with B batches of 3D volumes.

    Returns:
        4D tensor [B*D, C, H, W] with all slices batched together.

    Example:
        >>> volumes = torch.randn(2, 1, 160, 256, 256)  # 2 volumes
        >>> slices = volumes_to_slices(volumes)
        >>> slices.shape  # (320, 1, 256, 256) - 320 slices
    """
    if volumes.dim() != 5:
        raise ValueError(f"Expected 5D tensor [B,C,D,H,W], got {volumes.dim()}D")

    B, C, D, H, W = volumes.shape
    # Permute to [B, D, C, H, W] then reshape to [B*D, C, H, W]
    return volumes.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)


def extract_features_3d(
    volumes: torch.Tensor,
    extractor: Union[ResNet50Features, BiomedCLIPFeatures],
    chunk_size: int = 64,
) -> torch.Tensor:
    """Extract features from 3D volumes slice-wise (2.5D approach).

    Reshapes 5D volumes to 4D slices and extracts features in chunks
    to avoid GPU OOM. The resulting features represent the distribution
    of 2D slices within the 3D volumes.

    Args:
        volumes: 5D tensor [B, C, D, H, W] in [0, 1] range.
        extractor: 2D feature extractor (ResNet50Features or BiomedCLIPFeatures).
        chunk_size: Number of slices per forward pass (memory vs speed tradeoff).

    Returns:
        Feature tensor [B*D, feat_dim] with features for each slice.

    Example:
        >>> volumes = torch.randn(1, 1, 160, 256, 256).cuda()
        >>> resnet = ResNet50Features(device)
        >>> features = extract_features_3d(volumes, resnet)
        >>> features.shape  # (160, 2048) - one feature per slice
    """
    # Reshape to slices
    slices = volumes_to_slices(volumes)  # [B*D, C, H, W]
    total_slices = slices.shape[0]

    all_features = []
    for start in range(0, total_slices, chunk_size):
        end = min(start + chunk_size, total_slices)
        chunk = slices[start:end]
        features = extractor.extract_features(chunk)
        all_features.append(features.cpu())
        del features, chunk

    return torch.cat(all_features, dim=0)


def compute_kid_3d(
    real_volumes: torch.Tensor,
    generated_volumes: torch.Tensor,
    extractor: ResNet50Features,
    subset_size: int = 100,
    num_subsets: int = 50,
    chunk_size: int = 64,
) -> Tuple[float, float]:
    """Compute KID for 3D volumes using 2.5D slice-wise approach.

    Extracts ResNet50 features from all slices of real and generated
    volumes, then computes KID on the slice feature distributions.

    Args:
        real_volumes: Real 3D volumes [B, C, D, H, W] in [0, 1] range.
        generated_volumes: Generated 3D volumes [B, C, D, H, W] in [0, 1] range.
        extractor: ResNet50 feature extractor.
        subset_size: Size of random subsets for KID estimation.
        num_subsets: Number of random subsets to average over.
        chunk_size: Slices per forward pass for feature extraction.

    Returns:
        Tuple of (kid_mean, kid_std).
    """
    # Extract features slice-wise
    real_features = extract_features_3d(real_volumes, extractor, chunk_size)
    gen_features = extract_features_3d(generated_volumes, extractor, chunk_size)

    # Compute KID on slice features
    return compute_kid(real_features, gen_features, subset_size, num_subsets)


def compute_cmmd_3d(
    real_volumes: torch.Tensor,
    generated_volumes: torch.Tensor,
    extractor: BiomedCLIPFeatures,
    kernel_bandwidth: Optional[float] = None,
    chunk_size: int = 64,
) -> float:
    """Compute CMMD for 3D volumes using 2.5D slice-wise approach.

    Extracts BiomedCLIP features from all slices of real and generated
    volumes, then computes CMMD on the slice feature distributions.

    Args:
        real_volumes: Real 3D volumes [B, C, D, H, W] in [0, 1] range.
        generated_volumes: Generated 3D volumes [B, C, D, H, W] in [0, 1] range.
        extractor: BiomedCLIP feature extractor.
        kernel_bandwidth: RBF kernel bandwidth (None = median heuristic).
        chunk_size: Slices per forward pass for feature extraction.

    Returns:
        CMMD value (lower is better).
    """
    # Extract features slice-wise
    real_features = extract_features_3d(real_volumes, extractor, chunk_size)
    gen_features = extract_features_3d(generated_volumes, extractor, chunk_size)

    # Compute CMMD on slice features
    return compute_cmmd(real_features, gen_features, kernel_bandwidth)


def compute_fid_3d(
    real_volumes: torch.Tensor,
    generated_volumes: torch.Tensor,
    extractor: ResNet50Features,
    chunk_size: int = 64,
) -> float:
    """Compute FID for 3D volumes using 2.5D slice-wise approach.

    Extracts ResNet50 features from all slices of real and generated
    volumes, then computes FID on the slice feature distributions.

    Args:
        real_volumes: Real 3D volumes [B, C, D, H, W] in [0, 1] range.
        generated_volumes: Generated 3D volumes [B, C, D, H, W] in [0, 1] range.
        extractor: ResNet50 feature extractor.
        chunk_size: Slices per forward pass for feature extraction.

    Returns:
        FID value (lower is better).
    """
    # Extract features slice-wise
    real_features = extract_features_3d(real_volumes, extractor, chunk_size)
    gen_features = extract_features_3d(generated_volumes, extractor, chunk_size)

    # Compute FID on slice features
    return compute_fid(real_features, gen_features)
