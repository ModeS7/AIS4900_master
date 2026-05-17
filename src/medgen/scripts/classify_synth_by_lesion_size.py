"""Classify synthetic 3D brain-mets volumes by their MAX lesion-size bucket.

For each synthetic sample subdirectory (`<pool>/<XXXXX>/seg.nii.gz`), this
script runs 26-connectivity 3D connected-component analysis, computes each
component's Feret diameter (matching the RANO-BM size bins used by
`SegRegionalMetricsTracker`), and classifies the volume by its **largest**
lesion's bucket:

    tiny    : max diameter <  10 mm
    small   : max diameter < 20 mm  (and contains at least one small lesion)
    medium  : max diameter < 30 mm  (and contains at least one medium lesion)
    large   : max diameter >= 30 mm (and contains at least one large lesion)

(A volume with only tiny lesions is "tiny". A volume with tiny + small is
"small". A volume with tiny + medium + small is "medium". And so on.)

The exp8 family uses this to build datasets that prioritise tiny-only synth
volumes — the lesion-size class where exp3 has 36% detection rate and where
the bulk of the BrainMetShare test-set lesions live.

Output manifest JSON:

    {
        "pool_dir": "/abs/path/to/exp48c_handoff_exp32",
        "thresholds_mm": {"tiny": [0, 10], "small": [10, 20], ...},
        "voxel_spacing_mm": [1.0, 0.9375, 0.9375],
        "n_volumes": 525,
        "volumes": {
            "00000": {
                "max_bucket": "tiny",
                "lesion_counts": {"tiny": 7, "small": 0, "medium": 0, "large": 0},
                "max_diameter_mm": 6.81
            },
            ...
        },
        "bucket_counts": {"tiny": 134, "small": 281, "medium": 92, "large": 18}
    }

Usage:

    python -m medgen.scripts.classify_synth_by_lesion_size \\
        --pool-dir /cluster/.../generated/seg_candidates_525/exp48c_handoff_exp32 \\
        --output manifest.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

import nibabel as nib
import numpy as np
from scipy.ndimage import label as scipy_label

from medgen.metrics.constants import (
    TUMOR_SIZE_CATEGORIES,
    TUMOR_SIZE_THRESHOLDS_MM,
)

logger = logging.getLogger(__name__)


# BrainMetShare canonical spacing (D, H, W) = (1.0, 0.9375, 0.9375) mm,
# matching the diffusion config and `compute_voxel_size()`.
DEFAULT_VOXEL_SPACING_MM: tuple[float, float, float] = (1.0, 0.9375, 0.9375)

# 26-connectivity in 3D matches the tracker, BraTS-Mets, and Ottesen 2023.
_STRUCTURE_3D = np.ones((3, 3, 3), dtype=np.uint8)


def _classify_bucket(diameter_mm: float) -> str:
    """Return the RANO-BM bucket name for a Feret diameter (mm)."""
    for name, (low, high) in TUMOR_SIZE_THRESHOLDS_MM.items():
        if low <= diameter_mm < high:
            return name
    return 'large'  # >= 30mm


def _feret_diameter_3d(lesion_mask: np.ndarray,
                       voxel_spacing_mm: tuple[float, float, float]) -> float:
    """Compute Feret (longest pairwise distance) diameter of a 3D lesion.

    Uses the centroid-distance approximation: pairwise distance between all
    foreground voxels, scaled by voxel spacing. For brain mets at this voxel
    count this is fast enough; if N voxels gets very large we'd sub-sample
    to the convex hull, but typical lesions here are <500 voxels.
    """
    coords = np.argwhere(lesion_mask).astype(np.float64)
    if coords.shape[0] == 0:
        return 0.0
    if coords.shape[0] == 1:
        # Single voxel: Feret = longest body diagonal of that voxel
        return float(np.linalg.norm(np.asarray(voxel_spacing_mm)))
    # Scale by spacing
    coords *= np.asarray(voxel_spacing_mm)[np.newaxis, :]
    # Pairwise distances. O(N^2) but N is small per lesion.
    diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists_sq = (diffs * diffs).sum(axis=2)
    return float(np.sqrt(dists_sq.max()))


def classify_volume(
    seg_path: str,
    voxel_spacing_mm: tuple[float, float, float] = DEFAULT_VOXEL_SPACING_MM,
) -> dict[str, Any]:
    """Classify a single seg.nii.gz by its max lesion-size bucket.

    Returns dict with keys: max_bucket, lesion_counts (per-bucket counts),
    max_diameter_mm. If the seg is empty, max_bucket='empty', counts all 0.
    """
    seg = (nib.load(seg_path).get_fdata() > 0.5).astype(np.uint8)
    labeled, n_lesions = scipy_label(seg, structure=_STRUCTURE_3D)

    counts = {b: 0 for b in TUMOR_SIZE_CATEGORIES}
    max_diameter_mm = 0.0
    max_bucket = 'empty'

    for lid in range(1, n_lesions + 1):
        lesion = (labeled == lid)
        d_mm = _feret_diameter_3d(lesion, voxel_spacing_mm)
        bucket = _classify_bucket(d_mm)
        counts[bucket] += 1
        if d_mm > max_diameter_mm:
            max_diameter_mm = d_mm
            max_bucket = bucket

    return {
        'max_bucket': max_bucket,
        'lesion_counts': counts,
        'max_diameter_mm': round(max_diameter_mm, 4),
        'n_lesions': int(n_lesions),
    }


def classify_pool(
    pool_dir: str,
    voxel_spacing_mm: tuple[float, float, float] = DEFAULT_VOXEL_SPACING_MM,
) -> dict[str, Any]:
    """Classify every synth subdirectory in `pool_dir` containing seg.nii.gz."""
    if not os.path.isdir(pool_dir):
        raise FileNotFoundError(f"Synth pool not found: {pool_dir}")

    sample_ids = sorted(
        d for d in os.listdir(pool_dir)
        if os.path.isdir(os.path.join(pool_dir, d))
        and os.path.exists(os.path.join(pool_dir, d, 'seg.nii.gz'))
    )
    if not sample_ids:
        raise RuntimeError(f"No seg.nii.gz files found under {pool_dir}")

    logger.info(f"Classifying {len(sample_ids)} volumes in {pool_dir}")
    volumes: dict[str, dict[str, Any]] = {}
    bucket_counts: dict[str, int] = {b: 0 for b in TUMOR_SIZE_CATEGORIES}
    bucket_counts['empty'] = 0

    for i, sid in enumerate(sample_ids):
        seg_path = os.path.join(pool_dir, sid, 'seg.nii.gz')
        info = classify_volume(seg_path, voxel_spacing_mm=voxel_spacing_mm)
        volumes[sid] = info
        bucket_counts[info['max_bucket']] = bucket_counts.get(info['max_bucket'], 0) + 1
        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(sample_ids)} classified")

    return {
        'pool_dir': os.path.abspath(pool_dir),
        'thresholds_mm': {
            k: [v[0], v[1] if np.isfinite(v[1]) else None]
            for k, v in TUMOR_SIZE_THRESHOLDS_MM.items()
        },
        'voxel_spacing_mm': list(voxel_spacing_mm),
        'n_volumes': len(sample_ids),
        'volumes': volumes,
        'bucket_counts': bucket_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool-dir', required=True,
                        help='Synth pool root (contains XXXXX/{bravo,seg}.nii.gz)')
    parser.add_argument('--output', required=True,
                        help='Output manifest JSON path')
    parser.add_argument(
        '--voxel-spacing', default=','.join(map(str, DEFAULT_VOXEL_SPACING_MM)),
        help='Voxel spacing in mm as D,H,W (default: 1.0,0.9375,0.9375)',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    voxel_spacing = tuple(float(x) for x in args.voxel_spacing.split(','))
    if len(voxel_spacing) != 3:
        raise ValueError(
            f"Expected 3 voxel-spacing values, got {len(voxel_spacing)}"
        )

    manifest = classify_pool(args.pool_dir, voxel_spacing_mm=voxel_spacing)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest written to {args.output}")
    logger.info(f"Bucket counts: {manifest['bucket_counts']}")


if __name__ == '__main__':
    main()
