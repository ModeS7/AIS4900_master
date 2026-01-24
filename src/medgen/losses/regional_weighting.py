"""Region-weighted loss computation for diffusion training.

Computes per-pixel/voxel weight maps based on tumor size using RANO-BM clinical
thresholds. Smaller tumors receive higher weights to improve reconstruction
quality for clinically important small lesions.

Supports both 2D images and 3D volumes via spatial_dims parameter.

Usage:
    # 2D
    weight_computer = RegionalWeightComputer(
        spatial_dims=2,
        image_size=256,
        weights={'tiny': 2.5, 'small': 1.8, 'medium': 1.4, 'large': 1.2},
    )
    weight_map = weight_computer(seg_mask)  # [B, 1, H, W]

    # 3D
    weight_computer = RegionalWeightComputer(
        spatial_dims=3,
        volume_size=(256, 256, 160),
        weights={'tiny': 2.5, 'small': 1.8, 'medium': 1.4, 'large': 1.2},
    )
    weight_map = weight_computer(seg_mask)  # [B, 1, D, H, W]
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


def _compute_feret_diameter_3d(region) -> float:
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


class RegionalWeightComputer:
    """Compute per-pixel/voxel weight maps based on tumor size.

    Uses connected component analysis to identify individual tumors,
    classifies each by Feret diameter (longest edge-to-edge distance),
    and assigns weights based on RANO-BM clinical thresholds.

    Supports both 2D images [B, 1, H, W] and 3D volumes [B, 1, D, H, W].

    Args:
        spatial_dims: Number of spatial dimensions (2 or 3).
        image_size: Image size in pixels for 2D (e.g., 256). Required if spatial_dims=2.
        volume_size: Volume size (H, W, D) for 3D. Required if spatial_dims=3.
        weights: Dict mapping size category to weight value.
            Default: {'tiny': 2.5, 'small': 1.8, 'medium': 1.4, 'large': 1.2}
        background_weight: Weight for background pixels/voxels. Default: 1.0.
        fov_mm: Field of view in millimeters. Default: 240.0.
        device: PyTorch device. Default: cuda if available.

    Example:
        >>> # 2D
        >>> computer = RegionalWeightComputer(spatial_dims=2, image_size=256)
        >>> mask = torch.zeros(2, 1, 256, 256)
        >>> mask[0, 0, 100:120, 100:120] = 1  # Small tumor
        >>> weights = computer(mask)  # [2, 1, 256, 256]

        >>> # 3D
        >>> computer = RegionalWeightComputer(spatial_dims=3, volume_size=(256, 256, 160))
        >>> mask = torch.zeros(2, 1, 160, 256, 256)
        >>> mask[0, 0, 50:60, 100:120, 100:120] = 1  # Small tumor
        >>> weights = computer(mask)  # [2, 1, 160, 256, 256]
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        image_size: Optional[int] = None,
        volume_size: Optional[Tuple[int, int, int]] = None,
        weights: Optional[Dict[str, float]] = None,
        background_weight: float = 1.0,
        fov_mm: float = 240.0,
        device: Optional[torch.device] = None,
    ):
        self.spatial_dims = spatial_dims
        self.weights = weights or {
            'tiny': 2.5,
            'small': 1.8,
            'medium': 1.4,
            'large': 1.2,
        }
        self.background_weight = background_weight
        self.fov_mm = fov_mm
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if spatial_dims == 2:
            if image_size is None:
                raise ValueError("image_size required for spatial_dims=2")
            self.image_size = image_size
            self.mm_per_pixel = fov_mm / image_size
        elif spatial_dims == 3:
            if volume_size is None:
                raise ValueError("volume_size required for spatial_dims=3")
            self.volume_size = volume_size
            # Use H dimension as reference for RANO-BM thresholds (axial plane)
            self.mm_per_pixel = fov_mm / volume_size[0]
        else:
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

        # Min area threshold (pixels for 2D, voxels for 3D)
        self._min_area = 5 if spatial_dims == 2 else 10

    def _classify_tumor_size(self, diameter_mm: float) -> str:
        """Classify tumor by Feret diameter using RANO-BM thresholds."""
        for size_name, (low, high) in TUMOR_SIZE_THRESHOLDS_MM.items():
            if low <= diameter_mm < high:
                return size_name
        return 'large'

    def _get_feret_diameter(self, region) -> float:
        """Get Feret diameter for a region.

        For 2D, uses skimage's native feret_diameter_max.
        For 3D, uses bounding box diagonal approximation.
        """
        if self.spatial_dims == 2:
            return region.feret_diameter_max
        else:
            return _compute_feret_diameter_3d(region)

    def __call__(self, mask: Tensor) -> Tensor:
        """Compute per-pixel/voxel weight map from segmentation mask.

        Args:
            mask: Binary segmentation mask.
                2D: [B, 1, H, W] or [B, H, W]
                3D: [B, 1, D, H, W] or [B, D, H, W]

        Returns:
            Weight map with same shape as input (with channel dim).
        """
        # Handle different input shapes
        expected_dims = self.spatial_dims + 2  # B, C, spatial dims
        if mask.dim() == expected_dims - 1:
            mask = mask.unsqueeze(1)

        batch_size = mask.shape[0]

        if self.spatial_dims == 2:
            height, width = mask.shape[2], mask.shape[3]
            weight_map = torch.full(
                (batch_size, 1, height, width),
                self.background_weight,
                dtype=mask.dtype,
                device=mask.device,
            )
        else:
            depth, height, width = mask.shape[2], mask.shape[3], mask.shape[4]
            weight_map = torch.full(
                (batch_size, 1, depth, height, width),
                self.background_weight,
                dtype=mask.dtype,
                device=mask.device,
            )

        # Process each sample in the batch
        for i in range(batch_size):
            mask_np = mask[i, 0].cpu().numpy() > 0.5

            # Find connected components
            labeled, num_tumors = scipy_label(mask_np)

            if num_tumors == 0:
                continue

            # Get region properties
            regions = regionprops(labeled)

            for region in regions:
                # Skip tiny fragments
                if region.area < self._min_area:
                    continue

                # Get Feret diameter
                feret_px = self._get_feret_diameter(region)
                feret_mm = feret_px * self.mm_per_pixel

                # Classify and get weight
                size_cat = self._classify_tumor_size(feret_mm)
                weight = self.weights.get(size_cat, 1.0)

                # Create mask for this tumor
                tumor_mask = (labeled == region.label)
                weight_map[i, 0][torch.from_numpy(tumor_mask).to(mask.device)] = weight

        return weight_map


# Backwards compatibility aliases
class RegionalWeightComputer3D(RegionalWeightComputer):
    """3D regional weight computer (backwards compatibility wrapper).

    Equivalent to RegionalWeightComputer(spatial_dims=3, ...).
    """

    def __init__(self, volume_size=None, **kwargs):
        kwargs['spatial_dims'] = 3
        kwargs['volume_size'] = volume_size
        super().__init__(**kwargs)


def create_regional_weight_computer(
    cfg,
    spatial_dims: Optional[int] = None,
) -> Optional[RegionalWeightComputer]:
    """Create RegionalWeightComputer from config.

    Auto-detects 2D vs 3D from config if spatial_dims not specified.

    Args:
        cfg: Training config with regional_weighting section.
        spatial_dims: Override spatial dimensions (2 or 3).

    Returns:
        RegionalWeightComputer if enabled, None otherwise.
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

    # Auto-detect spatial dims from config
    if spatial_dims is None:
        spatial_dims = 3 if 'volume' in cfg else 2

    if spatial_dims == 2:
        return RegionalWeightComputer(
            spatial_dims=2,
            image_size=cfg.get('model', {}).get('image_size', 256),
            weights=dict(weights),
            background_weight=rw_cfg.get('background_weight', 1.0),
            fov_mm=rw_cfg.get('fov_mm', 240.0),
        )
    else:
        volume_size = (
            cfg.volume.get('height', 256),
            cfg.volume.get('width', 256),
            cfg.volume.get('depth', 160),
        )
        return RegionalWeightComputer(
            spatial_dims=3,
            volume_size=volume_size,
            weights=dict(weights),
            background_weight=rw_cfg.get('background_weight', 1.0),
            fov_mm=rw_cfg.get('fov_mm', 240.0),
        )


# Backwards compatibility alias
create_regional_weight_computer_3d = create_regional_weight_computer
