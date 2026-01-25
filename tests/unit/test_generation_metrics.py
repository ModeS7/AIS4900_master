"""Tests for generation quality metrics: KID, CMMD, FID, GenerationMetrics class."""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from medgen.metrics.generation import (
    compute_kid,
    compute_cmmd,
    compute_fid,
    GenerationMetricsConfig,
    GenerationMetrics,
    volumes_to_slices,
)


class TestKID:
    """Test compute_kid function."""

    def test_identical_distributions_low_kid(self):
        """Same features should have KID ~0."""
        features = torch.randn(100, 2048)
        kid_mean, kid_std = compute_kid(features, features)
        assert kid_mean < 0.01

    def test_different_distributions_higher_kid(self):
        """Different distributions have positive KID."""
        real = torch.randn(100, 2048)
        fake = torch.randn(100, 2048) + 5.0  # Shifted distribution
        kid_mean, kid_std = compute_kid(real, fake)
        assert kid_mean > 0.0

    def test_returns_mean_and_std(self):
        """Returns (kid_mean, kid_std) tuple."""
        features = torch.randn(50, 2048)
        result = compute_kid(features, features)
        assert isinstance(result, tuple)
        assert len(result) == 2
        kid_mean, kid_std = result
        assert isinstance(kid_mean, float)
        assert isinstance(kid_std, float)
        assert not torch.isnan(torch.tensor(kid_mean))
        assert not torch.isnan(torch.tensor(kid_std))
        # Note: Unbiased MMD estimator can produce small negative values for identical distributions
        assert kid_std >= 0.0  # Std is always non-negative

    def test_subset_size_capped(self):
        """subset_size capped at min(n_real, n_gen)."""
        # Only 20 samples but subset_size=100
        real = torch.randn(20, 2048)
        fake = torch.randn(20, 2048)
        # Should not crash, will use min(20, 20)
        kid_mean, kid_std = compute_kid(real, fake, subset_size=100)
        assert isinstance(kid_mean, float)
        assert not torch.isnan(torch.tensor(kid_mean))

    def test_too_few_samples_returns_zeros(self):
        """< 2 samples returns (0, 0)."""
        real = torch.randn(1, 2048)
        fake = torch.randn(1, 2048)
        kid_mean, kid_std = compute_kid(real, fake)
        assert kid_mean == 0.0
        assert kid_std == 0.0

    def test_num_subsets_parameter(self):
        """num_subsets controls number of random samples."""
        real = torch.randn(100, 2048)
        fake = torch.randn(100, 2048)
        # Just verify it runs with different num_subsets
        kid1, _ = compute_kid(real, fake, num_subsets=10)
        kid2, _ = compute_kid(real, fake, num_subsets=50)
        assert isinstance(kid1, float)
        assert isinstance(kid2, float)


class TestCMMD:
    """Test compute_cmmd function."""

    def test_identical_distributions_low_cmmd(self):
        """Same features have CMMD ~0."""
        features = torch.randn(100, 512)
        cmmd = compute_cmmd(features, features)
        assert cmmd < 0.1

    def test_different_distributions_higher_cmmd(self):
        """Different distributions have higher CMMD."""
        real = torch.randn(100, 512)
        fake = torch.randn(100, 512) + 3.0
        cmmd = compute_cmmd(real, fake)
        assert cmmd > 0.0

    def test_custom_bandwidth(self):
        """Custom kernel_bandwidth parameter works."""
        real = torch.randn(50, 512)
        fake = torch.randn(50, 512)
        cmmd = compute_cmmd(real, fake, kernel_bandwidth=1.0)
        assert isinstance(cmmd, float)
        assert not torch.isnan(torch.tensor(cmmd))
        assert cmmd >= 0.0  # CMMD is non-negative

    def test_cmmd_returns_float(self):
        """Returns single float value."""
        real = torch.randn(50, 512)
        fake = torch.randn(50, 512)
        cmmd = compute_cmmd(real, fake)
        assert isinstance(cmmd, float)
        assert not torch.isnan(torch.tensor(cmmd))
        assert cmmd >= 0.0


class TestFID:
    """Test compute_fid function."""

    def test_identical_distributions_low_fid(self):
        """Same features have FID ~0."""
        features = torch.randn(100, 2048)
        fid = compute_fid(features, features)
        assert fid < 1.0

    def test_different_distributions_higher_fid(self):
        """Different distributions have higher FID."""
        real = torch.randn(100, 2048)
        fake = torch.randn(100, 2048) + 5.0
        fid = compute_fid(real, fake)
        assert fid > 0.0

    def test_single_sample_edge_case(self):
        """Single sample uses zero covariance."""
        real = torch.randn(1, 2048)
        fake = torch.randn(1, 2048)
        # Should not crash
        fid = compute_fid(real, fake)
        assert isinstance(fid, float)
        assert not torch.isnan(torch.tensor(fid))
        assert fid >= 0.0  # FID is non-negative

    def test_fid_returns_float(self):
        """Returns single float value."""
        real = torch.randn(50, 2048)
        fake = torch.randn(50, 2048)
        fid = compute_fid(real, fake)
        assert isinstance(fid, float)
        assert not torch.isnan(torch.tensor(fid))
        assert fid >= 0.0


class TestGenerationMetricsConfig:
    """Test GenerationMetricsConfig dataclass."""

    def test_default_values(self):
        """Default config has sensible values."""
        config = GenerationMetricsConfig()
        assert config.enabled is True
        assert config.samples_per_epoch > 0
        assert config.samples_extended > config.samples_per_epoch
        assert config.steps_per_epoch > 0

    def test_all_fields_settable(self):
        """All config fields can be set."""
        config = GenerationMetricsConfig(
            enabled=False,
            samples_per_epoch=50,
            samples_extended=200,
            samples_test=500,
            steps_per_epoch=5,
            steps_extended=20,
            steps_test=40,
        )
        assert config.enabled is False
        assert config.samples_per_epoch == 50
        assert config.samples_test == 500


class TestVolumesToSlices:
    """Test volumes_to_slices helper."""

    def test_reshapes_5d_to_4d(self):
        """[B, C, D, H, W] -> [B*D, C, H, W]."""
        volumes = torch.randn(2, 1, 16, 64, 64)
        slices = volumes_to_slices(volumes)
        assert slices.shape == (32, 1, 64, 64)  # 2*16 = 32

    def test_preserves_content(self):
        """Reshaping preserves actual values."""
        volumes = torch.randn(2, 1, 4, 32, 32)
        slices = volumes_to_slices(volumes)

        # First slice of first batch should match
        expected = volumes[0, :, 0, :, :]
        actual = slices[0]
        assert torch.allclose(expected, actual)


class TestGenerationMetricsInit:
    """Test GenerationMetrics initialization."""

    def test_is_seg_mode_detection(self):
        """seg and seg_conditioned detected as seg modes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()

            metrics_seg = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='seg'
            )
            assert metrics_seg.is_seg_mode is True

            metrics_seg_cond = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='seg_conditioned'
            )
            assert metrics_seg_cond.is_seg_mode is True

            metrics_bravo = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='bravo'
            )
            assert metrics_bravo.is_seg_mode is False


class TestSetFixedConditioning:
    """Test set_fixed_conditioning method."""

    def test_tuple_format(self, mock_bravo_dataset):
        """Handles (image, seg) tuple format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            metrics = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='bravo'
            )

            metrics.set_fixed_conditioning(mock_bravo_dataset, num_masks=5)

            assert metrics.fixed_conditioning_masks is not None
            assert len(metrics.fixed_conditioning_masks) <= 5

    def test_seg_conditioned_stores_size_bins(self, mock_seg_conditioned_dataset):
        """REGRESSION: size_bins stored for seg_conditioned mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            metrics = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='seg_conditioned'
            )

            metrics.set_fixed_conditioning(mock_seg_conditioned_dataset, num_masks=5)

            # Key regression test: fixed_size_bins must be populated
            assert metrics.fixed_size_bins is not None
            assert metrics.fixed_size_bins.shape[0] == metrics.fixed_conditioning_masks.shape[0]

    def test_positive_masks_only(self):
        """Only samples with positive masks are kept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            metrics = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='bravo'
            )

            # Create dataset with all positive masks
            class DatasetWithPositiveMasks:
                def __len__(self):
                    return 20

                def __getitem__(self, idx):
                    torch.manual_seed(idx)
                    image = torch.rand(1, 64, 64)
                    seg = torch.ones(1, 64, 64)  # All positive
                    return (image, seg)

            dataset = DatasetWithPositiveMasks()
            metrics.set_fixed_conditioning(dataset, num_masks=10)

            # Should find 10 positive masks
            assert metrics.fixed_conditioning_masks.shape[0] == 10

    def test_empty_masks_raise_error(self):
        """Empty masks cause RuntimeError when stacking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            metrics = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='bravo'
            )

            # Create dataset with all empty masks
            class DatasetWithEmptyMasks:
                def __len__(self):
                    return 10

                def __getitem__(self, idx):
                    torch.manual_seed(idx)
                    image = torch.rand(1, 64, 64)
                    seg = torch.zeros(1, 64, 64)  # All empty
                    return (image, seg)

            dataset = DatasetWithEmptyMasks()
            # Should raise RuntimeError because no positive masks found
            with pytest.raises(RuntimeError, match="stack expects a non-empty TensorList"):
                metrics.set_fixed_conditioning(dataset, num_masks=10)


class TestGenerateSamplesBasic:
    """Basic tests for _generate_samples behavior.

    Note: Full integration tests for size_bins are in test_trainer_msssim_3d.py
    """

    def test_is_seg_mode_flag(self, mock_seg_conditioned_dataset):
        """is_seg_mode flag is set correctly for seg_conditioned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            metrics = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='seg_conditioned'
            )

            # Verify the flag is set
            assert metrics.is_seg_mode is True

    def test_fixed_size_bins_shape(self, mock_seg_conditioned_dataset):
        """fixed_size_bins has correct shape after set_fixed_conditioning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            metrics = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='seg_conditioned'
            )

            metrics.set_fixed_conditioning(mock_seg_conditioned_dataset, num_masks=5)

            # Verify shape: [num_masks, 7] for 7 size bins
            assert metrics.fixed_size_bins.shape[1] == 7
            assert metrics.fixed_size_bins.dtype == torch.long
