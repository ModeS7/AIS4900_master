"""Tests for diversity metrics: LPIPS and MS-SSIM diversity for 2D and 3D."""

import pytest
import torch

from medgen.metrics.quality import (
    compute_lpips_diversity,
    compute_msssim_diversity,
    compute_lpips_diversity_3d,
    compute_msssim_diversity_3d,
)


@pytest.mark.usefixtures("lpips_available")
class TestLPIPSDiversity:
    """Test compute_lpips_diversity functions for 2D and 3D."""

    @pytest.mark.timeout(60)
    @pytest.mark.slow
    @pytest.mark.parametrize("shape,func,desc", [
        ((1, 1, 64, 64), compute_lpips_diversity, "2D"),
        ((1, 1, 4, 64, 64), compute_lpips_diversity_3d, "3D"),
    ])
    def test_single_sample_returns_zero(self, shape, func, desc):
        """Need >= 2 samples for diversity."""
        samples = torch.rand(*shape)
        diversity = func(samples)
        assert diversity == 0.0

    @pytest.mark.timeout(60)
    @pytest.mark.slow
    @pytest.mark.parametrize("shape,repeat_dims,func,desc", [
        ((1, 1, 64, 64), (4, 1, 1, 1), compute_lpips_diversity, "2D"),
        ((1, 1, 4, 64, 64), (2, 1, 1, 1, 1), compute_lpips_diversity_3d, "3D"),
    ])
    def test_identical_samples_returns_near_zero(self, shape, repeat_dims, func, desc):
        """No diversity in identical samples."""
        sample = torch.rand(*shape)
        samples = sample.repeat(*repeat_dims)
        diversity = func(samples)
        assert diversity < 0.01

    @pytest.mark.timeout(60)
    @pytest.mark.slow
    @pytest.mark.parametrize("shape,func,desc", [
        ((4, 1, 64, 64), compute_lpips_diversity, "2D"),
        ((2, 1, 4, 64, 64), compute_lpips_diversity_3d, "3D"),
    ])
    def test_diverse_samples_returns_positive(self, shape, func, desc):
        """Random samples have positive diversity."""
        samples = torch.rand(*shape)
        diversity = func(samples)
        assert diversity > 0.0

    @pytest.mark.timeout(60)
    @pytest.mark.slow
    @pytest.mark.parametrize("shape,func,desc", [
        ((3, 1, 64, 64), compute_lpips_diversity, "2D"),
        ((2, 1, 4, 64, 64), compute_lpips_diversity_3d, "3D"),
    ])
    def test_output_is_float(self, shape, func, desc):
        """Returns Python float with valid value."""
        samples = torch.rand(*shape)
        diversity = func(samples)
        assert isinstance(diversity, float)
        assert not torch.isnan(torch.tensor(diversity)), f"LPIPS diversity is NaN for {desc}"
        assert not torch.isinf(torch.tensor(diversity)), f"LPIPS diversity is inf for {desc}"
        assert diversity >= 0.0


class TestMSSSIMDiversity:
    """Test compute_msssim_diversity functions for 2D and 3D."""

    @pytest.mark.parametrize("shape,func,desc", [
        ((1, 1, 64, 64), compute_msssim_diversity, "2D"),
        ((1, 1, 16, 64, 64), compute_msssim_diversity_3d, "3D"),
    ])
    def test_single_sample_returns_zero(self, shape, func, desc):
        """Need >= 2 samples."""
        samples = torch.rand(*shape)
        diversity = func(samples)
        assert diversity == 0.0

    @pytest.mark.parametrize("shape,repeat_dims,func,desc", [
        ((1, 1, 64, 64), (4, 1, 1, 1), compute_msssim_diversity, "2D"),
        ((1, 1, 16, 64, 64), (3, 1, 1, 1, 1), compute_msssim_diversity_3d, "3D"),
    ])
    def test_identical_samples_returns_near_zero(self, shape, repeat_dims, func, desc):
        """1 - MS-SSIM(identical) = 0."""
        sample = torch.rand(*shape)
        samples = sample.repeat(*repeat_dims)
        diversity = func(samples)
        assert diversity < 0.01

    @pytest.mark.parametrize("shape,func,desc", [
        ((4, 1, 64, 64), compute_msssim_diversity, "2D"),
        ((3, 1, 16, 64, 64), compute_msssim_diversity_3d, "3D"),
    ])
    def test_diverse_samples_returns_positive(self, shape, func, desc):
        """Different samples have diversity."""
        samples = torch.rand(*shape)
        diversity = func(samples)
        assert diversity > 0.0

    @pytest.mark.parametrize("shape,func,desc", [
        ((3, 1, 64, 64), compute_msssim_diversity, "2D"),
        ((2, 1, 8, 64, 64), compute_msssim_diversity_3d, "3D"),
    ])
    def test_output_is_float(self, shape, func, desc):
        """Returns Python float with valid value."""
        samples = torch.rand(*shape)
        diversity = func(samples)
        assert isinstance(diversity, float)
        assert not torch.isnan(torch.tensor(diversity)), f"MS-SSIM diversity is NaN for {desc}"
        assert not torch.isinf(torch.tensor(diversity)), f"MS-SSIM diversity is inf for {desc}"
        assert diversity >= 0.0
