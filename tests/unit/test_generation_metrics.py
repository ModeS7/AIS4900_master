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


class TestExtractFeaturesBatched3D:
    """REGRESSION TEST: 3D feature extraction must use slice-wise approach.

    Bug introduced in commit 49c5cc1 (2026-01-20) when 3D trainer was unified.
    The unified trainer started using _extract_features_batched() directly,
    which didn't handle 3D volumes, causing KID to fail with "Not enough samples".

    Fix: _extract_features_batched() must detect 3D (ndim==5) and use
    extract_features_3d() for slice-wise feature extraction.
    """

    def test_3d_returns_slice_features(self):
        """REGRESSION: 3D volumes produce D features, not 1 feature per volume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            metrics = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='bravo'
            )

            # Mock extractor that returns 2048-dim features
            mock_extractor = Mock()
            mock_extractor.extract_features = Mock(
                side_effect=lambda x: torch.randn(x.shape[0], 2048)
            )

            # 3D volume: [1, 1, 16, 64, 64] (1 volume, 16 depth slices)
            samples_3d = torch.randn(1, 1, 16, 64, 64)

            features = metrics._extract_features_batched(samples_3d, mock_extractor)

            # CRITICAL: Must return 16 features (one per slice), NOT 1
            assert features.shape[0] == 16, \
                f"3D extraction should produce 16 features (one per slice), got {features.shape[0]}"
            assert features.shape[1] == 2048

    def test_3d_removes_padding(self):
        """REGRESSION: Padded slices must be removed before feature extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Config with original_depth=12 (meaning 4 slices are padding)
            config = GenerationMetricsConfig(original_depth=12)
            metrics = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='bravo'
            )

            # Mock extractor
            mock_extractor = Mock()
            mock_extractor.extract_features = Mock(
                side_effect=lambda x: torch.randn(x.shape[0], 2048)
            )

            # 3D volume: [1, 1, 16, 64, 64] (padded to 16 from original 12)
            samples_3d = torch.randn(1, 1, 16, 64, 64)

            features = metrics._extract_features_batched(samples_3d, mock_extractor)

            # Should return 12 features (original depth), not 16
            assert features.shape[0] == 12, \
                f"Should exclude padded slices: expected 12 features, got {features.shape[0]}"

    def test_2d_unchanged(self):
        """2D extraction behavior unchanged (no regression)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            metrics = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='bravo'
            )

            # Mock extractor
            mock_extractor = Mock()
            mock_extractor.extract_features = Mock(
                side_effect=lambda x: torch.randn(x.shape[0], 2048)
            )

            # 2D samples: [8, 1, 64, 64]
            samples_2d = torch.randn(8, 1, 64, 64)

            features = metrics._extract_features_batched(samples_2d, mock_extractor)

            # Should return 8 features (one per image)
            assert features.shape[0] == 8
            assert features.shape[1] == 2048


class TestSizeBinAdherence:
    """Tests for size bin adherence metrics (seg_conditioned mode).

    REGRESSION: These tests ensure the size bin adherence metric continues
    to work correctly for seg_conditioned mode.
    """

    def test_config_has_size_bin_fields(self):
        """GenerationMetricsConfig has size bin config fields."""
        config = GenerationMetricsConfig(
            size_bin_edges=[0, 5, 10, 20],
            size_bin_fov_mm=200.0,
        )
        assert config.size_bin_edges == [0, 5, 10, 20]
        assert config.size_bin_fov_mm == 200.0

    def test_config_default_size_bin_values(self):
        """Default config has None/default for size bin fields."""
        config = GenerationMetricsConfig()
        assert config.size_bin_edges is None
        assert config.size_bin_fov_mm == 240.0

    def test_compute_size_bin_adherence_perfect_match(self):
        """Perfect match between generated and conditioning returns 1.0 exact match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from medgen.data.loaders.seg_conditioned import compute_size_bins, DEFAULT_BIN_EDGES
            import numpy as np

            config = GenerationMetricsConfig(
                size_bin_edges=list(DEFAULT_BIN_EDGES),
                size_bin_fov_mm=240.0,
            )
            metrics = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='seg_conditioned'
            )

            # Create masks with known tumor sizes
            masks = torch.zeros(2, 1, 256, 256)

            # Add small circular tumor to mask 0 (~8mm diameter)
            center, radius = 128, 4
            y, x = torch.meshgrid(torch.arange(256), torch.arange(256), indexing='ij')
            circle = ((x - center)**2 + (y - center)**2) < radius**2
            masks[0, 0] = circle.float()

            # Compute actual bins from masks
            pixel_spacing = 240.0 / 256
            actual_bins_list = []
            for i in range(2):
                mask_np = masks[i].squeeze().numpy()
                bins = compute_size_bins(mask_np, DEFAULT_BIN_EDGES, pixel_spacing)
                actual_bins_list.append(bins)

            conditioning = torch.tensor(np.stack(actual_bins_list), dtype=torch.long)

            # Compute adherence (should be perfect)
            results = metrics._compute_size_bin_adherence(masks, conditioning, prefix="")

            assert results['SizeBin/exact_match'] == 1.0
            assert results['SizeBin/MAE'] == 0.0

    def test_compute_size_bin_adherence_mismatch(self):
        """Mismatch between generated and conditioning detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from medgen.data.loaders.seg_conditioned import DEFAULT_BIN_EDGES

            config = GenerationMetricsConfig(
                size_bin_edges=list(DEFAULT_BIN_EDGES),
                size_bin_fov_mm=240.0,
            )
            metrics = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='seg_conditioned'
            )

            # Empty mask (all zeros)
            masks = torch.zeros(1, 1, 256, 256)

            # Conditioning that expects tumors
            conditioning = torch.zeros(1, 6, dtype=torch.long)
            conditioning[0, 2] = 3  # Expect 3 tumors in bin 2

            results = metrics._compute_size_bin_adherence(masks, conditioning, prefix="")

            # Should not be perfect match
            assert results['SizeBin/exact_match'] < 1.0
            assert results['SizeBin/MAE'] > 0.0

    def test_compute_size_bin_adherence_extended_prefix(self):
        """Extended prefix applied to metric names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from medgen.data.loaders.seg_conditioned import DEFAULT_BIN_EDGES

            config = GenerationMetricsConfig(
                size_bin_edges=list(DEFAULT_BIN_EDGES),
                size_bin_fov_mm=240.0,
            )
            metrics = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='seg_conditioned'
            )

            masks = torch.zeros(1, 1, 256, 256)
            conditioning = torch.zeros(1, 6, dtype=torch.long)

            results = metrics._compute_size_bin_adherence(masks, conditioning, prefix="extended_")

            assert 'SizeBin/extended_exact_match' in results
            assert 'SizeBin/extended_MAE' in results
            assert 'SizeBin/extended_correlation' in results

    def test_size_bin_adherence_uses_default_bins_when_none(self):
        """Uses DEFAULT_BIN_EDGES when config.size_bin_edges is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from medgen.data.loaders.seg_conditioned import DEFAULT_BIN_EDGES

            # Config with None size_bin_edges
            config = GenerationMetricsConfig(size_bin_edges=None)
            metrics = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='seg_conditioned'
            )

            masks = torch.zeros(1, 1, 256, 256)
            conditioning = torch.zeros(1, len(DEFAULT_BIN_EDGES) - 1, dtype=torch.long)

            # Should not crash - uses DEFAULT_BIN_EDGES
            results = metrics._compute_size_bin_adherence(masks, conditioning, prefix="")

            assert 'SizeBin/exact_match' in results
            assert 'SizeBin/MAE' in results

    def test_correlation_handles_constant_arrays(self):
        """Correlation returns 0.0 for constant arrays (no variance)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from medgen.data.loaders.seg_conditioned import DEFAULT_BIN_EDGES

            config = GenerationMetricsConfig(
                size_bin_edges=list(DEFAULT_BIN_EDGES),
                size_bin_fov_mm=240.0,
            )
            metrics = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='seg_conditioned'
            )

            # All zeros - constant arrays
            masks = torch.zeros(2, 1, 256, 256)
            conditioning = torch.zeros(2, 6, dtype=torch.long)

            results = metrics._compute_size_bin_adherence(masks, conditioning, prefix="")

            # Correlation should be 0.0 for constant arrays (not NaN)
            assert results['SizeBin/correlation'] == 0.0
