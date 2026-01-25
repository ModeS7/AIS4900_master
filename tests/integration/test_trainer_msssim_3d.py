"""
Regression tests for MS-SSIM-3D bug fix.

BUG: MS-SSIM-3D returned None for 3D diffusion models.
FIX: _compute_volume_3d_msssim now calls _compute_volume_3d_msssim_native for 3D.

BUG: Generation metrics for seg_conditioned didn't pass size_bins.
FIX: Added fixed_size_bins storage and batch slicing in _generate_samples.
"""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from medgen.metrics.generation import GenerationMetrics, GenerationMetricsConfig


class TestMSSSIM3DRegression:
    """Regression tests for MS-SSIM-3D fix."""

    def test_compute_msssim_3d_works(self):
        """Native 3D MS-SSIM computation works."""
        from medgen.metrics.quality import compute_msssim

        # Identical volumes should have high MS-SSIM
        # Need larger volumes for 3D MS-SSIM (11x11x11 kernel)
        vol1 = torch.rand(1, 1, 32, 128, 128)
        vol2 = vol1.clone()

        msssim = compute_msssim(vol1, vol2, spatial_dims=3)

        # Must not be None
        assert msssim is not None
        # Identical volumes should have high MS-SSIM
        assert msssim > 0.99

    def test_3d_msssim_returns_valid_float(self):
        """3D MS-SSIM returns valid float, not None."""
        from medgen.metrics.quality import compute_msssim

        # Need larger volumes for 3D MS-SSIM (11x11x11 kernel)
        vol1 = torch.rand(1, 1, 32, 128, 128)
        vol2 = torch.rand(1, 1, 32, 128, 128)

        msssim = compute_msssim(vol1, vol2, spatial_dims=3)

        assert isinstance(msssim, float)
        assert 0.0 <= msssim <= 1.0


class TestSizeBinsRegression:
    """Regression tests for size_bins generation bug."""

    def test_set_fixed_conditioning_stores_size_bins(self):
        """REGRESSION: fixed_size_bins populated for seg_conditioned."""

        class MockSegConditionedDataset:
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                torch.manual_seed(idx)
                seg = (torch.rand(1, 64, 64) > 0.3).float()  # Positive mask
                size_bins = torch.randint(0, 5, (7,))
                return (seg, size_bins)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            metrics = GenerationMetrics(
                config,
                torch.device('cpu'),
                Path(tmpdir),
                mode_name='seg_conditioned'
            )

            metrics.set_fixed_conditioning(MockSegConditionedDataset(), num_masks=5)

            # CRITICAL: This was the bug - fixed_size_bins was not being stored
            assert metrics.fixed_size_bins is not None, \
                "REGRESSION: fixed_size_bins should be stored for seg_conditioned mode"

            # Size should match masks
            assert metrics.fixed_size_bins.shape[0] == metrics.fixed_conditioning_masks.shape[0]

            # Should be proper shape [num_masks, 7]
            assert metrics.fixed_size_bins.shape[1] == 7

    def test_fixed_size_bins_correct_dtype(self):
        """size_bins should be long dtype for embedding lookup."""

        class MockDataset:
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                torch.manual_seed(idx)
                seg = (torch.rand(1, 64, 64) > 0.3).float()
                size_bins = torch.randint(0, 5, (7,))
                return (seg, size_bins)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            metrics = GenerationMetrics(
                config,
                torch.device('cpu'),
                Path(tmpdir),
                mode_name='seg_conditioned'
            )

            metrics.set_fixed_conditioning(MockDataset(), num_masks=5)

            # Size bins should be long for embedding lookup
            assert metrics.fixed_size_bins.dtype == torch.long


class TestModeDetection:
    """Test mode detection for seg_conditioned."""

    def test_seg_conditioned_is_seg_mode(self):
        """seg_conditioned should be detected as seg mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            metrics = GenerationMetrics(
                config,
                torch.device('cpu'),
                Path(tmpdir),
                mode_name='seg_conditioned'
            )

            assert metrics.is_seg_mode is True

    def test_seg_is_seg_mode(self):
        """seg should be detected as seg mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            metrics = GenerationMetrics(
                config,
                torch.device('cpu'),
                Path(tmpdir),
                mode_name='seg'
            )

            assert metrics.is_seg_mode is True

    def test_bravo_is_not_seg_mode(self):
        """bravo should not be detected as seg mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            metrics = GenerationMetrics(
                config,
                torch.device('cpu'),
                Path(tmpdir),
                mode_name='bravo'
            )

            assert metrics.is_seg_mode is False

    def test_dual_is_not_seg_mode(self):
        """dual should not be detected as seg mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            metrics = GenerationMetrics(
                config,
                torch.device('cpu'),
                Path(tmpdir),
                mode_name='dual'
            )

            assert metrics.is_seg_mode is False
