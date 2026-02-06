"""Unit tests for spatial utilities, metric dispatch, and 3D generation helpers.

Modules covered:
- core/spatial_utils.py
- metrics/dispatch.py
- metrics/generation_3d.py (volumes_to_slices only)
"""

import pytest
import torch
import torch.nn.functional as F
from unittest.mock import patch, MagicMock


# ============================================================================
# core/spatial_utils.py
# ============================================================================

class TestBroadcastToSpatial:
    """Tests for broadcast_to_spatial()."""

    def test_2d_shape(self):
        from medgen.core.spatial_utils import broadcast_to_spatial
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = broadcast_to_spatial(t, spatial_dims=2)
        assert result.shape == (4, 1, 1, 1)

    def test_3d_shape(self):
        from medgen.core.spatial_utils import broadcast_to_spatial
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = broadcast_to_spatial(t, spatial_dims=3)
        assert result.shape == (4, 1, 1, 1, 1)

    def test_values_preserved(self):
        from medgen.core.spatial_utils import broadcast_to_spatial
        t = torch.tensor([10.0, 20.0])
        result = broadcast_to_spatial(t, spatial_dims=2)
        assert torch.allclose(result.squeeze(), t)

    def test_single_element(self):
        from medgen.core.spatial_utils import broadcast_to_spatial
        t = torch.tensor([5.0])
        result = broadcast_to_spatial(t, spatial_dims=3)
        assert result.shape == (1, 1, 1, 1, 1)
        assert result.item() == 5.0


class TestGetSpatialSumDims:
    """Tests for get_spatial_sum_dims()."""

    def test_2d_returns_2_3(self):
        from medgen.core.spatial_utils import get_spatial_sum_dims
        assert get_spatial_sum_dims(2) == (2, 3)

    def test_3d_returns_2_3_4(self):
        from medgen.core.spatial_utils import get_spatial_sum_dims
        assert get_spatial_sum_dims(3) == (2, 3, 4)


class TestExtractCenterSlice:
    """Tests for extract_center_slice()."""

    def test_2d_passthrough(self):
        from medgen.core.spatial_utils import extract_center_slice
        t = torch.randn(2, 1, 64, 64)
        result = extract_center_slice(t, spatial_dims=2)
        assert result is t  # exact same object

    def test_3d_extracts_center(self):
        from medgen.core.spatial_utils import extract_center_slice
        t = torch.randn(2, 1, 16, 64, 64)
        result = extract_center_slice(t, spatial_dims=3)
        assert result.shape == (2, 1, 64, 64)

    def test_3d_odd_depth(self):
        from medgen.core.spatial_utils import extract_center_slice
        t = torch.randn(2, 1, 15, 64, 64)
        result = extract_center_slice(t, spatial_dims=3)
        # 15 // 2 = 7
        assert result.shape == (2, 1, 64, 64)
        assert torch.equal(result, t[:, :, 7, :, :])

    def test_preserves_values(self):
        from medgen.core.spatial_utils import extract_center_slice
        t = torch.randn(2, 1, 16, 64, 64)
        result = extract_center_slice(t, spatial_dims=3)
        expected = t[:, :, 8, :, :]
        assert torch.equal(result, expected)


class TestGetPoolingFn:
    """Tests for get_pooling_fn()."""

    def test_max_pool_2d(self):
        from medgen.core.spatial_utils import get_pooling_fn
        assert get_pooling_fn(spatial_dims=2, pool_type='max') is F.max_pool2d

    def test_max_pool_3d(self):
        from medgen.core.spatial_utils import get_pooling_fn
        assert get_pooling_fn(spatial_dims=3, pool_type='max') is F.max_pool3d

    def test_avg_pool_2d(self):
        from medgen.core.spatial_utils import get_pooling_fn
        assert get_pooling_fn(spatial_dims=2, pool_type='avg') is F.avg_pool2d

    def test_avg_pool_3d(self):
        from medgen.core.spatial_utils import get_pooling_fn
        assert get_pooling_fn(spatial_dims=3, pool_type='avg') is F.avg_pool3d


# ============================================================================
# metrics/dispatch.py
# ============================================================================

class TestComputeLpipsDispatch:
    """Tests for compute_lpips_dispatch()."""

    @patch('medgen.metrics.quality.compute_lpips', return_value=0.1)
    def test_routes_to_2d(self, mock_fn):
        from medgen.metrics.dispatch import compute_lpips_dispatch
        pred = torch.randn(2, 1, 64, 64)
        gt = torch.randn(2, 1, 64, 64)
        result = compute_lpips_dispatch(pred, gt, spatial_dims=2)
        mock_fn.assert_called_once()
        assert result == 0.1

    @patch('medgen.metrics.quality.compute_lpips_3d', return_value=0.2)
    def test_routes_to_3d(self, mock_fn):
        from medgen.metrics.dispatch import compute_lpips_dispatch
        pred = torch.randn(2, 1, 16, 64, 64)
        gt = torch.randn(2, 1, 16, 64, 64)
        result = compute_lpips_dispatch(pred, gt, spatial_dims=3)
        mock_fn.assert_called_once()
        assert result == 0.2


class TestComputeMsssimDispatch:
    """Tests for compute_msssim_dispatch()."""

    @patch('medgen.metrics.quality.compute_msssim', return_value=0.95)
    def test_routes_to_2d(self, mock_fn):
        from medgen.metrics.dispatch import compute_msssim_dispatch
        pred = torch.randn(2, 1, 64, 64)
        gt = torch.randn(2, 1, 64, 64)
        result = compute_msssim_dispatch(pred, gt, spatial_dims=2)
        mock_fn.assert_called_once_with(pred, gt, spatial_dims=2)
        assert result == 0.95

    @patch('medgen.metrics.quality.compute_msssim_2d_slicewise', return_value=0.90)
    def test_routes_to_3d_slicewise(self, mock_fn):
        from medgen.metrics.dispatch import compute_msssim_dispatch
        pred = torch.randn(2, 1, 16, 64, 64)
        gt = torch.randn(2, 1, 16, 64, 64)
        result = compute_msssim_dispatch(pred, gt, spatial_dims=3, mode='slicewise')
        mock_fn.assert_called_once_with(pred, gt)
        assert result == 0.90

    @patch('medgen.metrics.quality.compute_msssim', return_value=0.85)
    def test_routes_to_3d_volumetric(self, mock_fn):
        from medgen.metrics.dispatch import compute_msssim_dispatch
        pred = torch.randn(2, 1, 16, 64, 64)
        gt = torch.randn(2, 1, 16, 64, 64)
        result = compute_msssim_dispatch(pred, gt, spatial_dims=3, mode='volumetric')
        mock_fn.assert_called_once_with(pred, gt, spatial_dims=3)
        assert result == 0.85


class TestCreateLpipsFn:
    """Tests for create_lpips_fn()."""

    def test_returns_correct_fn(self):
        from medgen.metrics.dispatch import create_lpips_fn
        from medgen.metrics.quality import compute_lpips, compute_lpips_3d
        assert create_lpips_fn(2) is compute_lpips
        assert create_lpips_fn(3) is compute_lpips_3d


# ============================================================================
# metrics/generation_3d.py
# ============================================================================

class TestVolumesToSlices:
    """Tests for volumes_to_slices()."""

    def test_reshapes_5d_to_4d(self):
        from medgen.metrics.generation_3d import volumes_to_slices
        vol = torch.randn(2, 1, 16, 64, 64)
        result = volumes_to_slices(vol)
        assert result.shape == (32, 1, 64, 64)

    def test_rejects_non_5d(self):
        from medgen.metrics.generation_3d import volumes_to_slices
        t = torch.randn(2, 1, 64, 64)
        with pytest.raises(ValueError, match="Expected 5D"):
            volumes_to_slices(t)

    def test_preserves_content(self):
        from medgen.metrics.generation_3d import volumes_to_slices
        vol = torch.randn(2, 1, 16, 64, 64)
        result = volumes_to_slices(vol)
        # First volume's first slice (depth=0) should be output[0]
        expected = vol[0, :, 0, :, :]  # [C, H, W]
        assert torch.equal(result[0], expected)
