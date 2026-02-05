"""Generation metric computation functions (FID, KID, CMMD).

This module contains standalone metric computation functions extracted from
generation.py for computing distributional distance metrics:

- compute_kid(): Kernel Inception Distance (unbiased, small sample friendly)
- compute_cmmd(): CLIP Maximum Mean Discrepancy (domain-aware)
- compute_fid(): Frechet Inception Distance (classic GAN metric)

Also includes 3D slice-wise variants and feature extraction utilities.

Reference:
- KID: https://arxiv.org/abs/1801.01401 (MMD-based, unbiased)
- CMMD: CLIP-based MMD for domain-specific comparison
- FID: https://arxiv.org/abs/1706.08500 (Frechet distance on Inception features)
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@torch.no_grad()
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


@torch.no_grad()
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


@torch.no_grad()
def compute_fid(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
) -> float:
    """Compute Frechet Inception Distance between real and generated features.

    FID measures the Frechet distance between two multivariate Gaussians
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
# 3D Slice-Wise Feature Extraction (2.5D Approach)
# =============================================================================

def volumes_to_slices(volumes: torch.Tensor) -> torch.Tensor:
    """Reshape 3D volumes to 2D slices for feature extraction.

    Converts [B, C, D, H, W] -> [B*D, C, H, W] by treating each depth slice
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
    extractor: "ResNet50Features | BiomedCLIPFeatures",
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


@torch.no_grad()
def compute_kid_3d(
    real_volumes: torch.Tensor,
    generated_volumes: torch.Tensor,
    extractor: "ResNet50Features",
    subset_size: int = 100,
    num_subsets: int = 50,
    chunk_size: int = 64,
) -> tuple[float, float]:
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


@torch.no_grad()
def compute_cmmd_3d(
    real_volumes: torch.Tensor,
    generated_volumes: torch.Tensor,
    extractor: "BiomedCLIPFeatures",
    kernel_bandwidth: float | None = None,
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


@torch.no_grad()
def compute_fid_3d(
    real_volumes: torch.Tensor,
    generated_volumes: torch.Tensor,
    extractor: "ResNet50Features",
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


# Type hints for forward references
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .feature_extractors import BiomedCLIPFeatures, ResNet50Features
