"""Fréchet Wavelet Distance (FWD) for image generation evaluation.

Domain-agnostic generation metric that computes per-frequency-band Fréchet
distances using wavelet packet decomposition. Unlike FID (which uses
ImageNet-pretrained features), FWD operates directly in the wavelet domain
and provides frequency-decomposed quality scores.

Reference:
    Veeramacheneni et al., "Fréchet Wavelet Distance", ICLR 2025.
    https://arxiv.org/abs/2312.15289

For 3D volumes, operates slice-wise (axial slices) — consistent with how
FID/KID/CMMD are computed in this project.

Usage:
    from medgen.metrics.fwd import compute_fwd, compute_fwd_3d

    # 2D: compare two sets of images
    fwd_score, per_band = compute_fwd(real_images, gen_images)

    # 3D: compare two sets of volumes (extracts axial slices)
    fwd_score, per_band = compute_fwd_3d(real_volumes, gen_volumes, trim_slices=10)
"""
import logging

import numpy as np
import torch
from scipy import linalg

logger = logging.getLogger(__name__)


def _wavelet_packet_transform(
    images: torch.Tensor,
    wavelet: str = "haar",
    max_level: int = 4,
    log_scale: bool = False,
) -> torch.Tensor:
    """Decompose images into wavelet packets.

    Args:
        images: [B, C, H, W] tensor in [0, 1].
        wavelet: Wavelet family (default: haar).
        max_level: Decomposition depth. Level L produces 4^L packets.
            Use 4 for 256x256, 3 for 128x128.
        log_scale: If True, apply log(|coeffs| + eps).

    Returns:
        [B, num_packets, C * H' * W'] flattened packet features.
    """
    from ptwt import WaveletPacket2D

    B, C, H, W = images.shape

    # Process each sample
    all_packets = []
    for i in range(B):
        wp = WaveletPacket2D(images[i:i+1], wavelet, maxlevel=max_level)

        # get_level returns list-of-lists of string keys
        # Flatten and sort for consistent ordering across samples
        level_nodes = wp.get_level(max_level)
        node_keys = sorted(key for sublist in level_nodes for key in sublist)

        packets = []
        for node_key in node_keys:
            coeff = wp[node_key]  # [1, C, H', W']
            if log_scale:
                coeff = torch.log(torch.abs(coeff) + 1e-12)
            packets.append(coeff.flatten(1))  # [1, C*H'*W']

        # Stack: [num_packets, C*H'*W']
        all_packets.append(torch.stack([p.squeeze(0) for p in packets], dim=0))

    # [B, num_packets, feat_dim]
    return torch.stack(all_packets, dim=0)


def _compute_statistics(
    packet_features: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-packet mean and covariance.

    Args:
        packet_features: [B, num_packets, feat_dim] tensor.

    Returns:
        (mu, sigma) where mu is [P, D] and sigma is [P, D, D].
    """
    feats = packet_features.cpu().double().numpy()
    B, P, D = feats.shape

    mu = np.zeros((P, D))
    sigma = np.zeros((P, D, D))

    for p in range(P):
        mu[p] = feats[:, p, :].mean(axis=0)
        sigma[p] = np.cov(feats[:, p, :], rowvar=False)

    return mu, sigma


def _frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute Fréchet distance between two multivariate Gaussians.

    Same formula as FID but applied per wavelet packet.
    """
    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # Numerical error might give complex numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(
        diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    )


def compute_fwd(
    real_images: torch.Tensor,
    gen_images: torch.Tensor,
    wavelet: str = "haar",
    max_level: int | None = None,
    log_scale: bool = False,
    batch_size: int = 64,
) -> tuple[float, dict[int, float]]:
    """Compute Fréchet Wavelet Distance between two image sets.

    Args:
        real_images: [N, C, H, W] tensor in [0, 1].
        gen_images: [M, C, H, W] tensor in [0, 1].
        wavelet: Wavelet family (default: haar).
        max_level: Decomposition depth. Auto-detected from image size if None.
        log_scale: Apply log to coefficients.
        batch_size: Processing batch size.

    Returns:
        (fwd_score, per_band_scores) where fwd_score is the average FD
        across all packets, and per_band_scores maps packet_idx -> FD.
    """
    H = real_images.shape[2]
    if max_level is None:
        # Level 4 for 256, level 3 for 128, level 2 for 64
        max_level = max(2, int(np.log2(H)) - 4)

    num_packets = 4 ** max_level

    # Extract wavelet features in batches
    def _extract_batched(images: torch.Tensor) -> torch.Tensor:
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            feats = _wavelet_packet_transform(batch, wavelet, max_level, log_scale)
            all_feats.append(feats)
        return torch.cat(all_feats, dim=0)

    logger.info(f"FWD: extracting wavelet packets (level={max_level}, {num_packets} packets)...")
    real_feats = _extract_batched(real_images)  # [N, P, D]
    gen_feats = _extract_batched(gen_images)    # [M, P, D]

    # Compute statistics
    mu_real, sigma_real = _compute_statistics(real_feats)
    mu_gen, sigma_gen = _compute_statistics(gen_feats)

    # Per-packet Fréchet distance
    per_band = {}
    for p in range(num_packets):
        per_band[p] = _frechet_distance(
            mu_real[p], sigma_real[p], mu_gen[p], sigma_gen[p]
        )

    fwd_score = float(np.mean(list(per_band.values())))
    return fwd_score, per_band


def compute_fwd_3d(
    real_volumes: list[np.ndarray],
    gen_volumes: list[np.ndarray],
    trim_slices: int = 10,
    wavelet: str = "haar",
    max_level: int | None = None,
    log_scale: bool = False,
) -> tuple[float, dict[int, float]]:
    """Compute FWD between two sets of 3D volumes using axial slices.

    Args:
        real_volumes: List of [D, H, W] numpy arrays in [0, 1].
        gen_volumes: List of [D, H, W] numpy arrays in [0, 1].
        trim_slices: Number of end slices to skip (padding).
        wavelet: Wavelet family.
        max_level: Decomposition depth.
        log_scale: Apply log to coefficients.

    Returns:
        (fwd_score, per_band_scores).
    """
    def _volumes_to_slices(volumes: list[np.ndarray]) -> torch.Tensor:
        slices = []
        for vol in volumes:
            D = vol.shape[0] - trim_slices if trim_slices > 0 else vol.shape[0]
            for d in range(D):
                slices.append(vol[d])
        # [N_slices, 1, H, W]
        return torch.from_numpy(np.stack(slices)).unsqueeze(1).float()

    logger.info(f"FWD 3D: extracting slices from {len(real_volumes)} real + "
                f"{len(gen_volumes)} generated volumes...")

    real_slices = _volumes_to_slices(real_volumes)
    gen_slices = _volumes_to_slices(gen_volumes)

    logger.info(f"FWD 3D: {real_slices.shape[0]} real slices, {gen_slices.shape[0]} generated slices")

    return compute_fwd(real_slices, gen_slices, wavelet, max_level, log_scale)
