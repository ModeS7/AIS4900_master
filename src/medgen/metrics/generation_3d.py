"""3D slice-wise (2.5D) generation metrics.

Extracts features from 3D volumes by reshaping to 2D axial slices and
computing metrics on the slice distributions.

Moved from generation.py during file split.
"""

import torch

from .feature_extractors import BiomedCLIPFeatures, ResNet50Features
from .generation import compute_cmmd, compute_fid, compute_kid


def volumes_to_slices(volumes: torch.Tensor) -> torch.Tensor:
    """Reshape 3D volumes to 2D axial slices for feature extraction.

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
    extractor: ResNet50Features | BiomedCLIPFeatures,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Extract features from 3D volumes slice-wise (2.5D approach).

    Reshapes 5D volumes to 4D axial slices and extracts features in chunks
    to avoid GPU OOM. The resulting features represent the distribution
    of 2D axial slices within the 3D volumes.

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
    # Reshape to axial slices
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


def compute_cmmd_3d(
    real_volumes: torch.Tensor,
    generated_volumes: torch.Tensor,
    extractor: BiomedCLIPFeatures,
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
