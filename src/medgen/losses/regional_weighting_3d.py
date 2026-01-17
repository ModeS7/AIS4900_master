"""Region-weighted loss computation for 3D diffusion training.

Computes per-voxel weight maps based on tumor size using RANO-BM clinical
thresholds. Smaller tumors receive higher weights to improve reconstruction
quality for clinically important small lesions.

Usage:
    weight_computer = RegionalWeightComputer3D(
        volume_size=(256, 256, 160),
        weights={'tiny': 2.5, 'small': 1.8, 'medium': 1.4, 'large': 1.2},
    )

    # In training loop:
    weight_map = weight_computer(seg_mask)  # [B, 1, D, H, W]
    mse = (pred - target) ** 2
    weighted_loss = (mse * weight_map).mean()
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from scipy.ndimage import label as scipy_label
from skimage.measure import regionprops

from medgen.metrics import TUMOR_SIZE_THRESHOLDS_MM

logger = logging.getLogger(__name__)


def compute_feret_diameter_3d_regionprop(region) -> float:
    """Compute approximate 3D Feret diameter from regionprop.

    Uses the bounding box diagonal as approximation since skimage
    doesn't have native 3D Feret diameter.

    Args:
        region: A regionprop from skimage.measure.regionprops.

    Returns:
        Approximate Feret diameter in pixels.
    """
    # Get bounding box: (min_slice, min_row, min_col, max_slice, max_row, max_col)
    bbox = region.bbox
    d = bbox[3] - bbox[0]  # depth extent
    h = bbox[4] - bbox[1]  # height extent
    w = bbox[5] - bbox[2]  # width extent

    # Bounding box diagonal as upper bound approximation
    diagonal = np.sqrt(d**2 + h**2 + w**2)
    return diagonal


class RegionalWeightComputer3D:
    """Compute per-voxel weight maps based on tumor size for 3D volumes.

    Uses 3D connected component analysis to identify individual tumors,
    classifies each by Feret diameter (bounding box diagonal approximation),
    and assigns weights based on RANO-BM clinical thresholds.

    Args:
        volume_size: Volume size (H, W, D) in voxels.
        weights: Dict mapping size category to weight value.
            Default: {'tiny': 2.5, 'small': 1.8, 'medium': 1.4, 'large': 1.2}
        background_weight: Weight for background voxels. Default: 1.0.
        fov_mm: Field of view in millimeters. Default: 240.0.
        device: PyTorch device. Default: cuda if available.

    Example:
        >>> computer = RegionalWeightComputer3D(volume_size=(256, 256, 160))
        >>> mask = torch.zeros(2, 1, 160, 256, 256)  # Batch of 2
        >>> mask[0, 0, 50:60, 100:120, 100:120] = 1  # Small tumor
        >>> weights = computer(mask)  # [2, 1, 160, 256, 256]
    """

    def __init__(
        self,
        volume_size: Tuple[int, int, int],  # (H, W, D)
        weights: Optional[Dict[str, float]] = None,
        background_weight: float = 1.0,
        fov_mm: float = 240.0,
        device: Optional[torch.device] = None,
    ):
        self.volume_size = volume_size
        self.weights = weights or {
            'tiny': 2.5,
            'small': 1.8,
            'medium': 1.4,
            'large': 1.2,
        }
        self.background_weight = background_weight
        self.fov_mm = fov_mm
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Compute mm per voxel (assume isotropic in H/W, possibly different in D)
        # Use H dimension as reference for RANO-BM thresholds (axial plane)
        self.mm_per_pixel = fov_mm / volume_size[0]

    def _classify_tumor_size(self, diameter_mm: float) -> str:
        """Classify tumor by Feret diameter using RANO-BM thresholds."""
        for size_name, (low, high) in TUMOR_SIZE_THRESHOLDS_MM.items():
            if low <= diameter_mm < high:
                return size_name
        return 'large'

    def __call__(self, mask: Tensor) -> Tensor:
        """Compute per-voxel weight map from 3D segmentation mask.

        Args:
            mask: Binary segmentation mask [B, 1, D, H, W] or [B, D, H, W].

        Returns:
            Weight map [B, 1, D, H, W] with same dtype and device as input.
        """
        # Handle different input shapes
        if mask.dim() == 4:
            mask = mask.unsqueeze(1)

        batch_size = mask.shape[0]
        depth, height, width = mask.shape[2], mask.shape[3], mask.shape[4]

        # Initialize with background weight
        weight_map = torch.full(
            (batch_size, 1, depth, height, width),
            self.background_weight,
            dtype=mask.dtype,
            device=mask.device,
        )

        # Process each sample in the batch
        for i in range(batch_size):
            mask_np = mask[i, 0].cpu().numpy() > 0.5

            # Find 3D connected components
            labeled, num_tumors = scipy_label(mask_np)

            if num_tumors == 0:
                continue

            # Get region properties (3D)
            regions = regionprops(labeled)

            for region in regions:
                # Skip tiny fragments (<10 voxels for 3D)
                if region.area < 10:
                    continue

                # Get approximate 3D Feret diameter
                feret_px = compute_feret_diameter_3d_regionprop(region)
                feret_mm = feret_px * self.mm_per_pixel

                # Classify and get weight
                size_cat = self._classify_tumor_size(feret_mm)
                weight = self.weights.get(size_cat, 1.0)

                # Create mask for this tumor
                tumor_mask = (labeled == region.label)
                weight_map[i, 0][torch.from_numpy(tumor_mask).to(mask.device)] = weight

        return weight_map


def create_regional_weight_computer_3d(cfg) -> Optional[RegionalWeightComputer3D]:
    """Create RegionalWeightComputer3D from config.

    Args:
        cfg: Training config with regional_weighting section.

    Returns:
        RegionalWeightComputer3D if enabled, None otherwise.
    """
    rw_cfg = cfg.training.get('regional_weighting', {})

    if not rw_cfg.get('enabled', False):
        return None

    weights = rw_cfg.get('weights', {})
    if not weights:
        weights = {
            'tiny': 2.5,
            'small': 1.8,
            'medium': 1.4,
            'large': 1.2,
        }

    volume_size = (
        cfg.volume.get('height', 256),
        cfg.volume.get('width', 256),
        cfg.volume.get('depth', 160),
    )

    return RegionalWeightComputer3D(
        volume_size=volume_size,
        weights=dict(weights),
        background_weight=rw_cfg.get('background_weight', 1.0),
        fov_mm=rw_cfg.get('fov_mm', 240.0),
    )
