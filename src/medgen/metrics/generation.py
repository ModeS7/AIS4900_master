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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from medgen.core.dict_utils import get_with_fallbacks

from .feature_extractors import BiomedCLIPFeatures, ResNet50Features

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
    enabled: bool = False
    samples_per_epoch: int = 100
    samples_extended: int = 500
    samples_test: int = 1000
    steps_per_epoch: int = 10
    steps_extended: int = 25
    steps_test: int = 50
    cache_dir: str = ".cache/generation_features"
    feature_batch_size: int = 16  # Set by trainer to match training.batch_size
    cfg_scale: float = 2.0  # CFG scale for generation (requires CFG dropout during training)
    original_depth: int | None = None  # Original depth before padding (for 3D metrics)
    # Size bin adherence config (for seg_conditioned mode)
    size_bin_edges: list[float] | None = None  # Bin edges in mm (default from loader)
    size_bin_fov_mm: float = 240.0  # Field of view in mm

    @classmethod
    def from_hydra(cls, cfg: DictConfig, spatial_dims: int = 2) -> 'GenerationMetricsConfig':
        """Extract generation metrics config from Hydra DictConfig.

        Handles 3D sample capping, feature_batch_size derivation,
        cache_dir resolution, and size bin config for seg_conditioned mode.

        Args:
            cfg: Hydra configuration object.
            spatial_dims: Spatial dimensions (2 or 3).

        Returns:
            GenerationMetricsConfig (enabled=False if not configured).
        """
        gen_cfg = cfg.training.get('generation_metrics', {})
        if not gen_cfg.get('enabled', False):
            return cls(enabled=False)

        # Feature batch size: default to training batch_size (for torch.compile)
        batch_size = cfg.training.get('batch_size', 16)
        feature_batch_size = gen_cfg.get('feature_batch_size', None)
        if feature_batch_size is None:
            if spatial_dims == 3:
                feature_batch_size = max(32, batch_size * 16)
            else:
                feature_batch_size = batch_size

        # Cache dir: use paths.cache_dir if available
        cache_dir = gen_cfg.get('cache_dir', None)
        if cache_dir is None:
            base_cache = getattr(cfg.paths, 'cache_dir', '.cache')
            cache_dir = f"{base_cache}/generation_features"

        # 3D volumes: cap sample counts to avoid OOM
        if spatial_dims == 3:
            samples_per_epoch = min(gen_cfg.get('samples_per_epoch', 1), 2)
            samples_extended = min(gen_cfg.get('samples_extended', 4), 4)
            samples_test = min(gen_cfg.get('samples_test', 10), 10)
        else:
            samples_per_epoch = gen_cfg.get('samples_per_epoch', 100)
            samples_extended = gen_cfg.get('samples_extended', 500)
            samples_test = gen_cfg.get('samples_test', 1000)

        # Original depth for 3D (exclude padded slices from metrics)
        original_depth = None
        if spatial_dims == 3:
            original_depth = cfg.volume.get('original_depth', None)

        # Size bin config for seg_conditioned mode
        mode_name = cfg.mode.name
        size_bin_edges = None
        size_bin_fov_mm = 240.0
        if mode_name == 'seg_conditioned':
            size_bin_cfg = cfg.mode.get('size_bins', {})
            size_bin_edges = list(size_bin_cfg.get('edges', [0, 3, 6, 10, 15, 20, 30]))
            size_bin_fov_mm = float(size_bin_cfg.get('fov_mm', 240.0))

        # DDPM uses DDIM for inference which needs more steps than RFlow
        strategy_name = cfg.strategy.name
        step_mult = 2 if strategy_name == 'ddpm' else 1

        return cls(
            enabled=True,
            samples_per_epoch=samples_per_epoch,
            samples_extended=samples_extended,
            samples_test=samples_test,
            steps_per_epoch=gen_cfg.get('steps_per_epoch', 10) * step_mult,
            steps_extended=gen_cfg.get('steps_extended', 25) * step_mult,
            steps_test=gen_cfg.get('steps_test', 50) * step_mult,
            cache_dir=cache_dir,
            feature_batch_size=feature_batch_size,
            cfg_scale=gen_cfg.get('cfg_scale', 2.0),
            original_depth=original_depth,
            size_bin_edges=size_bin_edges,
            size_bin_fov_mm=size_bin_fov_mm,
        )


# =============================================================================
# Metric Computation Functions
# =============================================================================

def compute_kid(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
    subset_size: int = 100,
    num_subsets: int = 50,
) -> tuple[float, float]:
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
    kernel_bandwidth: float | None = None,
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

    # Guard: unbiased MMD estimator requires at least 2 samples
    if n < 2 or m < 2:
        logger.warning(f"CMMD requires >= 2 samples (got real={n}, gen={m})")
        return 0.0

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

    except (ValueError, np.linalg.LinAlgError) as e:
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
        self.train_resnet: torch.Tensor | None = None
        self.train_biomed: torch.Tensor | None = None
        self.val_resnet: torch.Tensor | None = None
        self.val_biomed: torch.Tensor | None = None

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _extract_features_from_loader(
        self,
        dataloader: DataLoader,
        extractor: nn.Module,
        name: str,
        max_samples: int | None = None,
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
                images = get_with_fallbacks(batch, 'image', 'images')
                masks = get_with_fallbacks(batch, 'seg', 'mask', 'labels')
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
            self.train_resnet = get_with_fallbacks(cached, 'train_resnet', 'train_inception')
            self.train_biomed = cached['train_biomed']
            self.val_resnet = get_with_fallbacks(cached, 'val_resnet', 'val_inception')
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

# Import helper functions for delegation
from .generation_computation import (
    compute_diversity_metrics,
    compute_epoch_metrics,
    compute_extended_metrics,
    compute_metrics_against_reference,
    compute_metrics_from_samples,
    compute_size_bin_adherence,
    compute_test_metrics,
    extract_features_batched,
)
from .generation_sampling import (
    generate_and_extract_features_3d_streaming,
    generate_samples,
    set_fixed_conditioning,
)


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
        space: Any | None = None,
        mode_name: str = 'bravo',
    ) -> None:
        self.config = config
        self.device = device
        self.run_dir = Path(run_dir)
        self.space = space  # DiffusionSpace for latent decoding
        self.mode_name = mode_name
        self.is_seg_mode = mode_name in ('seg', 'seg_conditioned', 'seg_conditioned_input')

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
        self.fixed_conditioning_masks: torch.Tensor | None = None
        self.fixed_gt_images: torch.Tensor | None = None
        self.fixed_size_bins: torch.Tensor | None = None  # For seg_conditioned mode
        self.fixed_bin_maps: torch.Tensor | None = None  # For seg_conditioned_input mode

        if self.is_seg_mode:
            logger.info(f"GenerationMetrics: {mode_name} mode - will threshold output at 0.5")

    # --- Thin wrappers delegating to helper modules ---

    def set_fixed_conditioning(self, train_dataset: Dataset, num_masks: int = 500, seg_channel_idx: int = 1) -> None:
        return set_fixed_conditioning(self, train_dataset, num_masks, seg_channel_idx)

    def cache_reference_features(self, train_loader: DataLoader, val_loader: DataLoader, experiment_id: str) -> None:
        self.cache.extract_and_cache(train_loader, val_loader, experiment_id)

    @torch.no_grad()
    def _generate_samples(self, model, strategy, mode, num_samples, num_steps, batch_size=16):
        return generate_samples(self, model, strategy, mode, num_samples, num_steps, batch_size)

    def _generate_and_extract_features_3d_streaming(self, model, strategy, mode, num_samples, num_steps, keep_samples_for_diversity=False):
        return generate_and_extract_features_3d_streaming(self, model, strategy, mode, num_samples, num_steps, keep_samples_for_diversity)

    def _extract_features_batched(self, samples, extractor, batch_size=None):
        return extract_features_batched(self, samples, extractor, batch_size)

    def _compute_metrics_against_reference(self, gen_resnet, gen_biomed, ref_resnet, ref_biomed, prefix=""):
        return compute_metrics_against_reference(self, gen_resnet, gen_biomed, ref_resnet, ref_biomed, prefix)

    def _compute_diversity_metrics(self, samples, prefix=""):
        return compute_diversity_metrics(self, samples, prefix)

    def _compute_size_bin_adherence(self, generated_masks, conditioning_bins, prefix=""):
        return compute_size_bin_adherence(self, generated_masks, conditioning_bins, prefix)

    def compute_epoch_metrics(self, model, strategy, mode):
        return compute_epoch_metrics(self, model, strategy, mode)

    def compute_extended_metrics(self, model, strategy, mode):
        return compute_extended_metrics(self, model, strategy, mode)

    def compute_metrics_from_samples(self, samples, extended=False):
        return compute_metrics_from_samples(self, samples, extended)

    def compute_test_metrics(self, model, strategy, mode, test_loader=None):
        return compute_test_metrics(self, model, strategy, mode, test_loader)


# =============================================================================
# Re-exports from helper modules for backward compatibility
# =============================================================================

from .generation_3d import (  # noqa: F401
    compute_cmmd_3d,
    compute_fid_3d,
    compute_kid_3d,
    extract_features_3d,
    volumes_to_slices,
)
