"""Morphological comparison metrics for generated segmentation masks.

Compares generated binary masks against real masks by extracting tumor
statistics (volume, Feret diameter, spatial location, count) and computing
distributional distances using Wasserstein distance.

Used as an alternative to FID/KID for evaluating segmentation mask generation,
since ImageNet features are poorly suited for binary masks.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance

logger = logging.getLogger(__name__)


# Default voxel spacing for BrainMetShare: [D, H, W] in mm
DEFAULT_VOXEL_SPACING = (1.0, 0.9375, 0.9375)


@dataclass
class TumorStats:
    """Statistics for a single tumor (connected component)."""
    volume_mm3: float
    feret_mm: float
    centroid_d: float  # normalized [0, 1] in depth axis
    centroid_h: float  # normalized [0, 1] in height axis
    centroid_w: float  # normalized [0, 1] in width axis


@dataclass
class MorphologicalScore:
    """Morphological comparison score between real and generated masks."""
    total: float          # weighted sum of all components
    volume_dist: float    # Wasserstein on tumor volume distribution
    feret_dist: float     # Wasserstein on Feret diameter distribution
    spatial_d_dist: float # Wasserstein on centroid depth distribution
    spatial_h_dist: float # Wasserstein on centroid height distribution
    spatial_w_dist: float # Wasserstein on centroid width distribution
    count_dist: float     # Wasserstein on tumors-per-volume distribution
    n_real_tumors: int
    n_gen_tumors: int


def extract_tumor_stats_3d(
    mask: np.ndarray,
    voxel_spacing: tuple[float, float, float] = DEFAULT_VOXEL_SPACING,
    min_voxels: int = 3,
) -> list[TumorStats]:
    """Extract tumor statistics from a single 3D binary mask.

    Args:
        mask: Binary 3D volume [D, H, W], values 0 or 1.
        voxel_spacing: Voxel size in mm as (D, H, W).
        min_voxels: Minimum voxels for a region to count as a tumor.

    Returns:
        List of TumorStats for each connected component.
    """
    if mask.max() == 0:
        return []

    binary = (mask > 0.5).astype(np.uint8)
    labeled, n_components = ndimage.label(binary)

    if n_components == 0:
        return []

    voxel_vol = float(np.prod(voxel_spacing))
    shape = np.array(mask.shape, dtype=np.float64)
    stats = []

    for i in range(1, n_components + 1):
        component = (labeled == i)
        n_voxels = component.sum()

        if n_voxels < min_voxels:
            continue

        # Volume in mm³
        volume_mm3 = float(n_voxels) * voxel_vol

        # Feret diameter
        coords = np.argwhere(component)
        if len(coords) >= 2:
            spacing_arr = np.array(voxel_spacing)
            scaled = coords * spacing_arr
            if len(scaled) > 2000:
                rng = np.random.default_rng(seed=len(scaled))
                idx = rng.choice(len(scaled), 2000, replace=False)
                scaled = scaled[idx]
            dists = pdist(scaled)
            feret_mm = float(dists.max()) if len(dists) > 0 else min(voxel_spacing)
        else:
            feret_mm = min(voxel_spacing)

        # Centroid normalized to [0, 1]
        centroid = coords.mean(axis=0)
        centroid_norm = centroid / shape

        stats.append(TumorStats(
            volume_mm3=volume_mm3,
            feret_mm=feret_mm,
            centroid_d=float(centroid_norm[0]),
            centroid_h=float(centroid_norm[1]),
            centroid_w=float(centroid_norm[2]),
        ))

    return stats


def compute_morphological_score(
    real_masks: list[np.ndarray],
    generated_masks: list[np.ndarray],
    voxel_spacing: tuple[float, float, float] = DEFAULT_VOXEL_SPACING,
    weights: dict[str, float] | None = None,
) -> MorphologicalScore:
    """Compare generated masks against real masks using morphological statistics.

    Extracts tumor statistics from both sets and computes Wasserstein distance
    on each distributional property. Returns a weighted combination as the
    total score (lower = more similar to real data).

    Args:
        real_masks: List of real binary 3D masks [D, H, W].
        generated_masks: List of generated binary 3D masks [D, H, W].
        voxel_spacing: Voxel size in mm as (D, H, W).
        weights: Per-component weights. Keys: volume, feret, spatial_d,
            spatial_h, spatial_w, count. Default: equal weights.

    Returns:
        MorphologicalScore with total and per-component distances.
    """
    if weights is None:
        weights = {
            'volume': 1.0,
            'feret': 1.0,
            'spatial_d': 1.0,
            'spatial_h': 1.0,
            'spatial_w': 1.0,
            'count': 1.0,
        }

    # Extract stats from all masks
    real_stats: list[TumorStats] = []
    real_counts: list[int] = []
    for mask in real_masks:
        stats = extract_tumor_stats_3d(mask, voxel_spacing)
        real_stats.extend(stats)
        real_counts.append(len(stats))

    gen_stats: list[TumorStats] = []
    gen_counts: list[int] = []
    for mask in generated_masks:
        stats = extract_tumor_stats_3d(mask, voxel_spacing)
        gen_stats.extend(stats)
        gen_counts.append(len(stats))

    logger.info(
        f"Morphological: {len(real_stats)} real tumors ({len(real_masks)} volumes), "
        f"{len(gen_stats)} generated tumors ({len(generated_masks)} volumes)"
    )

    # Handle degenerate cases
    if len(real_stats) == 0 or len(gen_stats) == 0:
        logger.warning("No tumors found in real or generated masks — returning max score")
        return MorphologicalScore(
            total=float('inf'), volume_dist=float('inf'), feret_dist=float('inf'),
            spatial_d_dist=float('inf'), spatial_h_dist=float('inf'),
            spatial_w_dist=float('inf'), count_dist=float('inf'),
            n_real_tumors=len(real_stats), n_gen_tumors=len(gen_stats),
        )

    # Extract distributions
    real_vols = np.array([s.volume_mm3 for s in real_stats])
    gen_vols = np.array([s.volume_mm3 for s in gen_stats])

    real_ferets = np.array([s.feret_mm for s in real_stats])
    gen_ferets = np.array([s.feret_mm for s in gen_stats])

    real_cd = np.array([s.centroid_d for s in real_stats])
    gen_cd = np.array([s.centroid_d for s in gen_stats])

    real_ch = np.array([s.centroid_h for s in real_stats])
    gen_ch = np.array([s.centroid_h for s in gen_stats])

    real_cw = np.array([s.centroid_w for s in real_stats])
    gen_cw = np.array([s.centroid_w for s in gen_stats])

    # Compute Wasserstein distances
    # Log-transform volumes and ferets for better distribution comparison
    # (tumor sizes are log-normal)
    vol_dist = wasserstein_distance(np.log1p(real_vols), np.log1p(gen_vols))
    feret_dist = wasserstein_distance(np.log1p(real_ferets), np.log1p(gen_ferets))

    # Spatial: centroids already normalized to [0, 1]
    spatial_d = wasserstein_distance(real_cd, gen_cd)
    spatial_h = wasserstein_distance(real_ch, gen_ch)
    spatial_w = wasserstein_distance(real_cw, gen_cw)

    # Count distribution
    count_dist = wasserstein_distance(
        np.array(real_counts, dtype=np.float64),
        np.array(gen_counts, dtype=np.float64),
    )

    # Weighted total
    total = (
        weights.get('volume', 1.0) * vol_dist +
        weights.get('feret', 1.0) * feret_dist +
        weights.get('spatial_d', 1.0) * spatial_d +
        weights.get('spatial_h', 1.0) * spatial_h +
        weights.get('spatial_w', 1.0) * spatial_w +
        weights.get('count', 1.0) * count_dist
    )

    logger.info(
        f"Morphological score: {total:.4f} "
        f"(vol={vol_dist:.4f}, feret={feret_dist:.4f}, "
        f"spatial=[{spatial_d:.4f}, {spatial_h:.4f}, {spatial_w:.4f}], "
        f"count={count_dist:.4f})"
    )

    return MorphologicalScore(
        total=total,
        volume_dist=vol_dist,
        feret_dist=feret_dist,
        spatial_d_dist=spatial_d,
        spatial_h_dist=spatial_h,
        spatial_w_dist=spatial_w,
        count_dist=count_dist,
        n_real_tumors=len(real_stats),
        n_gen_tumors=len(gen_stats),
    )
