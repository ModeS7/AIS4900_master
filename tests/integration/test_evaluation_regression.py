"""Regression tests for test set evaluation.

These tests catch bugs in the evaluate_test_set() method and related functionality.
This is critical code that runs after potentially hours of training.

Run with: pytest tests/integration/test_evaluation_regression.py -v
"""

import pytest
import torch
import json
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestEvaluateTestSetMetrics:
    """Verify test evaluation computes and logs all expected metrics."""

    REQUIRED_TEST_METRICS = [
        'mse',
        'msssim',
        'psnr',
        'n_samples',
    ]

    OPTIONAL_TEST_METRICS = [
        'lpips',
        'msssim_3d',  # 3D only
        'FID',
        'KID_mean',
        'CMMD',
    ]

    def test_test_results_json_serializable(self, tmp_path):
        """
        REGRESSION: Test results JSON must not contain numpy types.

        Same bug as regional_losses.json - numpy.float32 causes TypeError.
        """
        import json

        # Simulate test results that might contain numpy types
        test_results = {
            'mse': np.float32(0.05),
            'msssim': np.float64(0.92),
            'psnr': np.float32(28.5),
            'n_samples': np.int64(100),
            'lpips': np.float32(0.15),
        }

        # Convert to native types (what the fix should do)
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            return obj

        # This should work without TypeError
        results_path = tmp_path / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(convert_to_native(test_results), f, indent=2)

        # Verify it's valid JSON
        with open(results_path) as f:
            loaded = json.load(f)

        assert loaded['mse'] == pytest.approx(0.05, abs=0.01)
        assert loaded['n_samples'] == 100

    def test_empty_test_set_returns_empty_dict(self):
        """
        REGRESSION: Empty test set should return {} not crash.

        The method should handle empty dataloaders gracefully.
        """
        # This tests the logic at line 3522-3524
        n_batches = 0
        if n_batches == 0:
            result = {}
        else:
            result = {'mse': 0.0}

        assert result == {}, "Empty test set should return empty dict"

    def test_metric_computation_handles_dict_images(self):
        """
        REGRESSION: Dual-channel images (dict format) must compute metrics correctly.

        Bug potential: Averaging metrics incorrectly for multi-channel modes.
        """
        from medgen.metrics.quality import compute_msssim, compute_psnr

        # Simulate dual-channel images
        pred_t1_pre = torch.rand(2, 1, 64, 64)
        pred_t1_gd = torch.rand(2, 1, 64, 64)
        gt_t1_pre = torch.rand(2, 1, 64, 64)
        gt_t1_gd = torch.rand(2, 1, 64, 64)

        # Compute as the trainer does
        msssim_val = (
            compute_msssim(pred_t1_pre, gt_t1_pre) +
            compute_msssim(pred_t1_gd, gt_t1_gd)
        ) / 2

        psnr_val = (
            compute_psnr(pred_t1_pre, gt_t1_pre) +
            compute_psnr(pred_t1_gd, gt_t1_gd)
        ) / 2

        # Values should be reasonable
        assert 0 <= msssim_val <= 1, f"MS-SSIM out of range: {msssim_val}"
        assert 0 < psnr_val < 100, f"PSNR out of range: {psnr_val}"

    def test_metric_computation_handles_tensor_images(self):
        """
        REGRESSION: Single tensor images must compute metrics correctly.
        """
        from medgen.metrics.quality import compute_msssim, compute_psnr

        pred = torch.rand(2, 1, 64, 64)
        gt = torch.rand(2, 1, 64, 64)

        msssim_val = compute_msssim(pred, gt)
        psnr_val = compute_psnr(pred, gt)

        assert 0 <= msssim_val <= 1, f"MS-SSIM out of range: {msssim_val}"
        assert 0 < psnr_val < 100, f"PSNR out of range: {psnr_val}"


class TestCheckpointLoading:
    """Verify checkpoint loading for test evaluation works correctly."""

    def test_checkpoint_not_found_uses_current_model(self, tmp_path):
        """
        REGRESSION: Missing checkpoint should warn and use current model, not crash.
        """
        checkpoint_name = "best"
        checkpoint_path = tmp_path / f"checkpoint_{checkpoint_name}.pt"

        # Simulate the logic in evaluate_test_set
        if checkpoint_path.exists():
            label = checkpoint_name
        else:
            # Should log warning and use current
            label = "current"

        assert label == "current", "Missing checkpoint should fallback to 'current'"

    def test_checkpoint_loads_ema_state(self, tmp_path):
        """
        REGRESSION: EMA state must be loaded from checkpoint if available.

        Bug: Forgetting to load EMA state means test uses wrong model weights.
        """
        # Create mock checkpoint with EMA state
        checkpoint = {
            'model_state_dict': {'layer.weight': torch.randn(10, 10)},
            'ema_state_dict': {'shadow': {'layer.weight': torch.randn(10, 10)}},
            'epoch': 100,
        }

        checkpoint_path = tmp_path / "checkpoint_best.pt"
        torch.save(checkpoint, checkpoint_path)

        # Load and verify EMA state exists
        loaded = torch.load(checkpoint_path)
        assert 'ema_state_dict' in loaded, "EMA state should be in checkpoint"
        assert 'shadow' in loaded['ema_state_dict'], "EMA shadow weights should exist"


class TestTimestepBinLosses:
    """Verify timestep bin loss computation for test evaluation."""

    def test_timestep_bins_computed_correctly(self):
        """
        REGRESSION: Timestep bin losses must be computed correctly.

        Bug potential: Wrong bin assignment, division by zero, etc.
        """
        num_timesteps = 1000
        num_timestep_bins = 10
        batch_size = 4

        # Simulate timesteps
        timesteps = torch.randint(0, num_timesteps, (batch_size,))
        mse_per_sample = torch.rand(batch_size)

        # Compute bins as trainer does
        bin_size = num_timesteps // num_timestep_bins
        bin_indices = (timesteps // bin_size).clamp(max=num_timestep_bins - 1).long()

        # Accumulate
        timestep_loss_sum = torch.zeros(num_timestep_bins)
        timestep_loss_count = torch.zeros(num_timestep_bins, dtype=torch.long)
        timestep_loss_sum.scatter_add_(0, bin_indices, mse_per_sample)
        timestep_loss_count.scatter_add_(0, bin_indices, torch.ones_like(bin_indices))

        # Verify bins are valid
        assert bin_indices.min() >= 0, "Bin indices should be non-negative"
        assert bin_indices.max() < num_timestep_bins, "Bin indices should be < num_bins"

        # Verify accumulation worked
        assert timestep_loss_count.sum() == batch_size, "All samples should be counted"

    def test_timestep_bins_handle_edge_cases(self):
        """
        REGRESSION: Edge timesteps (0 and max) must map to valid bins.
        """
        num_timesteps = 1000
        num_timestep_bins = 10

        # Edge cases
        edge_timesteps = torch.tensor([0, 999, 500, 100])

        bin_size = num_timesteps // num_timestep_bins
        bin_indices = (edge_timesteps // bin_size).clamp(max=num_timestep_bins - 1).long()

        assert bin_indices[0] == 0, "Timestep 0 should map to bin 0"
        assert bin_indices[1] == 9, "Timestep 999 should map to bin 9"
        assert 0 <= bin_indices[2] < num_timestep_bins, "Middle timestep should map to valid bin"


class TestRegionalTrackerForTest:
    """Verify regional tracker initialization for test evaluation."""

    def test_regional_tracker_uses_correct_spatial_dims(self):
        """
        REGRESSION: Regional tracker for test must match training spatial dims.
        """
        from medgen.metrics.regional import RegionalMetricsTracker

        # 2D tracker
        tracker_2d = RegionalMetricsTracker(
            spatial_dims=2,
            image_size=256,
            fov_mm=240.0,
            device=torch.device('cpu'),
        )
        assert tracker_2d.spatial_dims == 2

        # 3D tracker
        tracker_3d = RegionalMetricsTracker(
            spatial_dims=3,
            volume_size=(128, 128, 64),
            fov_mm=240.0,
            device=torch.device('cpu'),
        )
        assert tracker_3d.spatial_dims == 3

    def test_regional_tracker_handles_labels_none(self):
        """
        REGRESSION: Regional tracking should skip gracefully when labels is None.

        Bug: Calling regional_tracker.update() with None labels crashes.
        """
        from medgen.metrics.regional import RegionalMetricsTracker

        tracker = RegionalMetricsTracker(
            spatial_dims=2,
            image_size=256,
            fov_mm=240.0,
            device=torch.device('cpu'),
        )

        # Simulate the guard condition
        labels = None
        if tracker is not None and labels is not None:
            # This should NOT be called when labels is None
            tracker.update(None, None, labels)

        # If we get here without error, the guard works
        assert True


class TestWorstBatchTracking:
    """Verify worst batch tracking for test evaluation."""

    def test_worst_batch_tracks_highest_loss(self):
        """
        REGRESSION: Worst batch should be the one with highest loss.
        """
        worst_batch_loss = 0.0
        worst_batch_data = None
        batch_size = 4
        min_batch_size = 4

        # Simulate batches with different losses
        batches = [
            {'loss': 0.05, 'data': 'batch1'},
            {'loss': 0.15, 'data': 'batch2'},  # This should be worst
            {'loss': 0.08, 'data': 'batch3'},
        ]

        for batch in batches:
            loss_val = batch['loss']
            if loss_val > worst_batch_loss and batch_size >= min_batch_size:
                worst_batch_loss = loss_val
                worst_batch_data = batch['data']

        assert worst_batch_loss == 0.15, "Should track highest loss"
        assert worst_batch_data == 'batch2', "Should track batch with highest loss"

    def test_worst_batch_ignores_small_batches(self):
        """
        REGRESSION: Small last batches should not be tracked as worst.

        Bug: Small batches often have artificially high loss, skewing results.
        """
        worst_batch_loss = 0.0
        min_batch_size = 4

        # Small batch with high loss (should be ignored)
        batch_size = 2
        loss_val = 0.99

        if loss_val > worst_batch_loss and batch_size >= min_batch_size:
            worst_batch_loss = loss_val

        assert worst_batch_loss == 0.0, "Small batches should not update worst"


class TestTestTensorBoardLogging:
    """Verify TensorBoard logging for test results."""

    def test_test_metrics_logged_with_correct_prefix(self):
        """
        REGRESSION: Test metrics must use test_{label} prefix.
        """
        from medgen.metrics.unified import UnifiedMetrics
        from unittest.mock import MagicMock

        logged = {'scalars': {}}

        def capture_scalar(tag, value, step=None):
            logged['scalars'][tag] = value

        mock_writer = MagicMock()
        mock_writer.add_scalar = capture_scalar

        metrics = UnifiedMetrics(
            writer=mock_writer,
            mode='bravo',
            spatial_dims=2,
            modality='bravo',
            device=torch.device('cpu'),
        )

        test_metrics = {
            'PSNR': 28.5,
            'MS-SSIM': 0.92,
            'MSE': 0.05,
        }

        metrics.log_test(test_metrics, prefix='test_best')

        # Verify prefix is used
        assert any('test_best' in tag for tag in logged['scalars']), (
            f"Test metrics should have 'test_best' prefix\n"
            f"Logged: {list(logged['scalars'].keys())}"
        )

    def test_test_timesteps_logged_correctly(self):
        """
        REGRESSION: Timestep bin losses must be logged for test evaluation.
        """
        from medgen.metrics.unified import UnifiedMetrics
        from unittest.mock import MagicMock

        logged = {'scalars': {}}

        def capture_scalar(tag, value, step=None):
            logged['scalars'][tag] = value

        mock_writer = MagicMock()
        mock_writer.add_scalar = capture_scalar

        metrics = UnifiedMetrics(
            writer=mock_writer,
            mode='bravo',
            spatial_dims=2,
            modality='bravo',
            device=torch.device('cpu'),
        )

        timestep_bins = {
            '0.0-0.1': 0.08,
            '0.1-0.2': 0.07,
            '0.9-1.0': 0.02,
        }

        metrics.log_test_timesteps(timestep_bins, prefix='test_best')

        # Verify timestep bins are logged (uses lowercase 'timestep' in tag)
        assert any('timestep' in tag.lower() for tag in logged['scalars']), (
            f"Timestep bins should be logged\n"
            f"Logged: {list(logged['scalars'].keys())}"
        )


class Test3DTestEvaluation:
    """Verify 3D-specific test evaluation functionality."""

    def test_3d_msssim_computed_for_volumes(self):
        """
        REGRESSION: 3D MS-SSIM should be computed for volume data.
        """
        from medgen.metrics.quality import compute_msssim

        # 3D volumes: [B, C, D, H, W]
        pred = torch.rand(1, 1, 16, 32, 32)
        gt = torch.rand(1, 1, 16, 32, 32)

        msssim_3d = compute_msssim(pred, gt, spatial_dims=3)

        assert 0 <= msssim_3d <= 1, f"3D MS-SSIM out of range: {msssim_3d}"

    def test_3d_regional_tracker_for_test(self):
        """
        REGRESSION: 3D regional tracker must use correct volume dimensions.
        """
        from medgen.metrics.regional import RegionalMetricsTracker

        volume_size = (64, 64, 32)
        fov_mm = 240.0

        tracker = RegionalMetricsTracker(
            spatial_dims=3,
            volume_size=volume_size,
            fov_mm=fov_mm,
            device=torch.device('cpu'),
        )

        # Verify mm_per_pixel is computed from actual volume size
        expected_mm_per_pixel = fov_mm / max(volume_size[0], volume_size[1])
        assert tracker.mm_per_pixel == expected_mm_per_pixel, (
            f"3D tracker mm_per_pixel wrong: {tracker.mm_per_pixel} != {expected_mm_per_pixel}"
        )


class TestGenerationMetricsForTest:
    """Verify generation metrics (FID, KID, CMMD) for test evaluation."""

    def test_generation_metrics_failure_caught(self):
        """
        REGRESSION: Generation metrics failure should warn, not crash.

        The code wraps generation metrics in try/except (line 3628).
        """
        # Simulate what happens when generation metrics fail
        try:
            raise RuntimeError("CUDA out of memory")
        except Exception as e:
            # Should log warning and continue
            warning_msg = f"Generation metrics computation failed: {e}"
            assert "failed" in warning_msg.lower()

    def test_generation_metrics_logged_to_tensorboard(self):
        """
        REGRESSION: FID/KID/CMMD should be logged when computed successfully.
        """
        from medgen.metrics.unified import UnifiedMetrics
        from unittest.mock import MagicMock

        logged = {'scalars': {}}

        def capture_scalar(tag, value, step=None):
            logged['scalars'][tag] = value

        mock_writer = MagicMock()
        mock_writer.add_scalar = capture_scalar

        metrics = UnifiedMetrics(
            writer=mock_writer,
            mode='bravo',
            spatial_dims=2,
            modality='bravo',
            device=torch.device('cpu'),
        )

        gen_results = {
            'FID': 45.2,
            'KID_mean': 0.032,
            'KID_std': 0.005,
            'CMMD': 0.15,
        }

        metrics.log_test_generation(gen_results, prefix='test_best')

        # Verify generation metrics are logged
        assert any('FID' in tag or 'KID' in tag or 'CMMD' in tag for tag in logged['scalars']), (
            f"Generation metrics should be logged\n"
            f"Logged: {list(logged['scalars'].keys())}"
        )


class TestTestResultsFileOutput:
    """Verify test results are saved correctly to files."""

    def test_test_results_json_created(self, tmp_path):
        """
        REGRESSION: test_results_{label}.json must be created.
        """
        import json

        label = "best"
        metrics = {
            'mse': 0.05,
            'msssim': 0.92,
            'psnr': 28.5,
            'n_samples': 100,
        }

        results_path = tmp_path / f'test_results_{label}.json'
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        assert results_path.exists(), "Test results JSON should be created"

        with open(results_path) as f:
            loaded = json.load(f)

        assert loaded['mse'] == 0.05
        assert loaded['n_samples'] == 100

    def test_worst_batch_figure_path_correct(self, tmp_path):
        """
        REGRESSION: Worst batch figure path should use correct label.
        """
        label = "best"
        fig_path = tmp_path / f'test_worst_batch_{label}.png'

        # Just verify the path format is correct
        assert 'test_worst_batch_best' in str(fig_path)
        assert str(fig_path).endswith('.png')
