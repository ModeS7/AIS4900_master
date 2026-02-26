"""3D slice-wise (2.5D) generation metrics.

Extracts features from 3D volumes by reshaping to 2D slices and
computing metrics on the slice distributions.

Moved from generation.py during file split.
"""

import torch

from .feature_extractors import BiomedCLIPFeatures, ResNet50Features
from .generation import compute_cmmd, compute_fid, compute_kid


def volumes_to_slices(volumes: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Reshape 3D volumes to 2D slices along a given spatial axis.

    Args:
        volumes: 5D tensor [B, C, D, H, W] with B batches of 3D volumes.
        axis: Spatial axis to slice along.
            0 = axial (D axis)  -> [B*D, C, H, W]
            1 = coronal (H axis) -> [B*H, C, D, W]
            2 = sagittal (W axis) -> [B*W, C, D, H]

    Returns:
        4D tensor with all slices batched together.

    Example:
        >>> volumes = torch.randn(2, 1, 160, 256, 256)
        >>> volumes_to_slices(volumes, axis=0).shape  # (320, 1, 256, 256)
        >>> volumes_to_slices(volumes, axis=1).shape  # (512, 1, 160, 256)
        >>> volumes_to_slices(volumes, axis=2).shape  # (512, 1, 160, 256)
    """
    if volumes.dim() != 5:
        raise ValueError(f"Expected 5D tensor [B,C,D,H,W], got {volumes.dim()}D")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0 (axial), 1 (coronal), or 2 (sagittal), got {axis}")

    # Spatial axis in the 5D tensor is axis + 2 (skip B, C dims)
    spatial_dim = axis + 2
    B, C = volumes.shape[:2]
    n_slices = volumes.shape[spatial_dim]

    # Move the slice axis to position 2 (after B, C), then merge B and slice dims
    # e.g. axis=1: [B,C,D,H,W] -> [B,C,H,D,W] via moveaxis(3,2) -> [B*H, C, D, W]
    moved = volumes.moveaxis(spatial_dim, 2)
    return moved.reshape(B * n_slices, C, moved.shape[3], moved.shape[4])


def extract_features_3d(
    volumes: torch.Tensor,
    extractor: ResNet50Features | BiomedCLIPFeatures,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Extract features from 3D volumes using multi-view 2.5D slicing.

    Slices volumes along all 3 anatomical planes (axial, coronal, sagittal),
    extracts 2D features from each slice, and concatenates them. This gives
    a more comprehensive feature distribution than axial-only slicing.

    For a volume [B, 1, 160, 256, 256]:
    - Axial:    B*160 slices of [256, 256]
    - Coronal:  B*256 slices of [160, 256]
    - Sagittal: B*256 slices of [160, 256]
    - Total:    B*672 feature vectors

    Args:
        volumes: 5D tensor [B, C, D, H, W] in [0, 1] range.
        extractor: 2D feature extractor (ResNet50Features or BiomedCLIPFeatures).
        chunk_size: Number of slices per forward pass (memory vs speed tradeoff).

    Returns:
        Feature tensor [N, feat_dim] with features from all views concatenated.
    """
    all_features = []
    for axis in range(3):
        slices = volumes_to_slices(volumes, axis=axis)
        total_slices = slices.shape[0]

        for start in range(0, total_slices, chunk_size):
            end = min(start + chunk_size, total_slices)
            chunk = slices[start:end]
            features = extractor.extract_features(chunk)
            all_features.append(features.cpu())
            del features, chunk

        del slices

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
