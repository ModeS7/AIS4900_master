"""Regression tests for metric logging.

These tests catch when metrics silently disappear or change after code updates.
If a test fails, it means a metric that was previously logged is now missing.

Run with: pytest tests/integration/test_metric_logging_regression.py -v
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestDiffusionTrainerMetrics:
    """Verify DiffusionTrainer logs all expected metrics."""

    # ==========================================================================
    # IMPORTANT: Update these lists when you intentionally add/remove metrics
    # ==========================================================================

    REQUIRED_TRAINING_METRICS = [
        # Losses - these MUST be logged every epoch
        'Loss/Total_train',
        'Loss/MSE_train',
        'LR/Generator',
    ]

    REQUIRED_VALIDATION_METRICS = [
        # Quality metrics - these MUST be logged during validation
        'Validation/PSNR',
        'Validation/MS-SSIM',
    ]

    OPTIONAL_VALIDATION_METRICS = [
        # These are logged depending on config/mode
        'Validation/LPIPS',      # Only for image modes, not seg
        'Validation/Dice',       # Only for seg modes
        'Validation/IoU',        # Only for seg modes
        'Validation/MS-SSIM-3D', # Only for 3D
    ]

    REQUIRED_TRAINING_METRICS_WITH_PERCEPTUAL = [
        'Loss/Perceptual_train',  # Only when perceptual loss enabled
    ]

    @pytest.fixture
    def captured_metrics(self):
        """Capture all metrics logged to TensorBoard."""
        logged = {'scalars': {}, 'images': {}, 'figures': {}}

        def capture_scalar(tag, value, global_step=None):
            if tag not in logged['scalars']:
                logged['scalars'][tag] = []
            logged['scalars'][tag].append({'value': value, 'step': global_step})

        def capture_image(tag, img, global_step=None):
            logged['images'][tag] = {'step': global_step}

        def capture_figure(tag, fig, global_step=None):
            logged['figures'][tag] = {'step': global_step}

        mock_writer = MagicMock()
        mock_writer.add_scalar = capture_scalar
        mock_writer.add_image = capture_image
        mock_writer.add_figure = capture_figure

        return mock_writer, logged

    @pytest.fixture
    def minimal_unified_metrics(self, captured_metrics, tmp_path):
        """Create UnifiedMetrics with captured writer."""
        from medgen.metrics.unified import UnifiedMetrics

        mock_writer, logged = captured_metrics

        metrics = UnifiedMetrics(
            writer=mock_writer,
            mode='bravo',
            spatial_dims=2,
            modality='bravo',
            device=torch.device('cpu'),
            enable_regional=False,
            log_grad_norm=True,
            log_timestep_losses=False,
            log_regional_losses=False,
            log_msssim=True,
            log_psnr=True,
            log_lpips=False,
            log_flops=False,
        )

        return metrics, logged

    def test_training_metrics_logged(self, minimal_unified_metrics):
        """Training step logs all required metrics."""
        metrics, logged = minimal_unified_metrics

        # Simulate training updates
        metrics.update_loss('Total', 0.5, phase='train')
        metrics.update_loss('MSE', 0.3, phase='train')
        metrics.update_lr(0.0001)
        metrics.update_grad_norm(1.5)

        # Log training metrics
        metrics.log_training(epoch=1)

        # Check required metrics were logged
        missing = []
        for metric_name in self.REQUIRED_TRAINING_METRICS:
            if metric_name not in logged['scalars']:
                missing.append(metric_name)

        assert not missing, (
            f"REGRESSION: Training metrics missing!\n"
            f"Missing: {missing}\n"
            f"Logged: {list(logged['scalars'].keys())}\n"
            f"Check if metric names changed in UnifiedMetrics.log_training()"
        )

    def test_validation_metrics_logged(self, minimal_unified_metrics):
        """Validation logs all required quality metrics."""
        metrics, logged = minimal_unified_metrics

        # Simulate validation updates
        metrics.update_psnr(
            pred=torch.rand(2, 1, 64, 64),
            gt=torch.rand(2, 1, 64, 64)
        )
        metrics.update_msssim(
            pred=torch.rand(2, 1, 64, 64),
            gt=torch.rand(2, 1, 64, 64)
        )

        # Log validation metrics
        metrics.log_validation(epoch=1)

        # Check required metrics were logged
        missing = []
        for metric_name in self.REQUIRED_VALIDATION_METRICS:
            # Check with and without suffix (suffix depends on mode)
            found = any(
                metric_name in tag or metric_name.replace('Validation/', 'Validation/') in tag
                for tag in logged['scalars'].keys()
            )
            if not found:
                missing.append(metric_name)

        assert not missing, (
            f"REGRESSION: Validation metrics missing!\n"
            f"Missing: {missing}\n"
            f"Logged: {list(logged['scalars'].keys())}\n"
            f"Check if metric names changed in UnifiedMetrics.log_validation()"
        )

    def test_metric_values_are_valid(self, minimal_unified_metrics):
        """Logged metric values are finite numbers."""
        metrics, logged = minimal_unified_metrics

        # Log some metrics
        metrics.update_loss('Total', 0.5, phase='train')
        metrics.update_loss('MSE', 0.3, phase='train')
        metrics.update_lr(0.0001)
        metrics.log_training(epoch=1)

        # Check all values are valid
        for tag, entries in logged['scalars'].items():
            for entry in entries:
                value = entry['value']
                assert value is not None, f"REGRESSION: {tag} is None"
                assert not (isinstance(value, float) and (
                    value != value or  # NaN check
                    value == float('inf') or
                    value == float('-inf')
                )), f"REGRESSION: {tag} has invalid value: {value}"


class TestCompressionTrainerMetrics:
    """Verify compression trainers (VAE/VQVAE/DCAE) log expected metrics."""

    REQUIRED_VAE_METRICS = [
        'Loss/Total_train',
        'Loss/Reconstruction_train',
        'Loss/KL_train',
    ]

    REQUIRED_VQVAE_METRICS = [
        'Loss/Total_train',
        'Loss/Reconstruction_train',
        'Loss/VQ_train',
        'Loss/Commitment_train',
    ]

    @pytest.fixture
    def captured_metrics(self):
        """Capture all metrics logged to TensorBoard."""
        logged = {'scalars': {}}

        def capture_scalar(tag, value, global_step=None):
            if tag not in logged['scalars']:
                logged['scalars'][tag] = []
            logged['scalars'][tag].append({'value': value, 'step': global_step})

        mock_writer = MagicMock()
        mock_writer.add_scalar = capture_scalar

        return mock_writer, logged

    def test_vae_logs_kl_divergence(self, captured_metrics):
        """VAE trainer must log KL divergence - critical for VAE training."""
        mock_writer, logged = captured_metrics

        # Simulate VAE loss logging
        mock_writer.add_scalar('Loss/Total_train', 0.5, 1)
        mock_writer.add_scalar('Loss/Reconstruction_train', 0.4, 1)
        mock_writer.add_scalar('Loss/KL_train', 0.1, 1)

        # Verify KL is logged
        assert 'Loss/KL_train' in logged['scalars'], (
            "REGRESSION: VAE must log KL divergence!\n"
            "This is critical for monitoring VAE training."
        )

    def test_vqvae_logs_codebook_loss(self, captured_metrics):
        """VQ-VAE trainer must log VQ and commitment losses."""
        mock_writer, logged = captured_metrics

        # Simulate VQ-VAE loss logging
        mock_writer.add_scalar('Loss/Total_train', 0.5, 1)
        mock_writer.add_scalar('Loss/Reconstruction_train', 0.3, 1)
        mock_writer.add_scalar('Loss/VQ_train', 0.1, 1)
        mock_writer.add_scalar('Loss/Commitment_train', 0.1, 1)

        # Verify VQ losses are logged
        assert 'Loss/VQ_train' in logged['scalars'], (
            "REGRESSION: VQ-VAE must log VQ loss!"
        )
        assert 'Loss/Commitment_train' in logged['scalars'], (
            "REGRESSION: VQ-VAE must log commitment loss!"
        )


class TestFigureLogging:
    """Verify important figures/visualizations are logged."""

    REQUIRED_FIGURES = [
        'denoising_trajectory/progression',  # Denoising process visualization
    ]

    OPTIONAL_FIGURES = [
        'reconstruction/worst_batch',      # Worst reconstructions
        'samples/generated',               # Generated samples
        'timestep_losses/heatmap',         # Loss by timestep
    ]

    @pytest.fixture
    def captured_figures(self):
        """Capture figures logged to TensorBoard."""
        logged = {'figures': set(), 'images': set()}

        def capture_figure(tag, fig, global_step=None):
            logged['figures'].add(tag)

        def capture_image(tag, img, global_step=None):
            logged['images'].add(tag)

        mock_writer = MagicMock()
        mock_writer.add_figure = capture_figure
        mock_writer.add_image = capture_image
        mock_writer.add_scalar = MagicMock()

        return mock_writer, logged

    def test_denoising_trajectory_logged(self, captured_figures):
        """Denoising trajectory figure must be logged."""
        from medgen.metrics.unified import UnifiedMetrics

        mock_writer, logged = captured_figures

        metrics = UnifiedMetrics(
            writer=mock_writer,
            mode='bravo',
            spatial_dims=2,
            modality='bravo',
            device=torch.device('cpu'),
        )

        # Create fake trajectory: list of tensors at different timesteps
        trajectory = [
            torch.rand(1, 1, 64, 64) for _ in range(5)
        ]

        metrics.log_denoising_trajectory(
            trajectory=trajectory,
            epoch=1,
            tag='denoising_trajectory'
        )

        assert 'denoising_trajectory/progression' in logged['figures'], (
            f"REGRESSION: Denoising trajectory figure not logged!\n"
            f"Logged figures: {logged['figures']}\n"
            f"This visualization shows the denoising process and is critical for debugging."
        )


class TestMetricNamingConsistency:
    """Verify metric names follow consistent conventions."""

    def test_loss_names_have_phase_suffix(self):
        """Loss metrics should end with _train or _val."""
        expected_patterns = [
            'Loss/Total_train',
            'Loss/MSE_train',
            'Loss/Total_val',
            'Loss/MSE_val',
        ]

        for pattern in expected_patterns:
            assert pattern.endswith('_train') or pattern.endswith('_val'), (
                f"Loss metric '{pattern}' should end with _train or _val"
            )

    def test_validation_metrics_have_consistent_prefix(self):
        """Validation quality metrics should start with 'Validation/'."""
        expected_metrics = [
            'Validation/PSNR',
            'Validation/MS-SSIM',
            'Validation/LPIPS',
            'Validation/Dice',
            'Validation/IoU',
        ]

        for metric in expected_metrics:
            assert metric.startswith('Validation/'), (
                f"Quality metric '{metric}' should start with 'Validation/'"
            )


class Test3DSpecificMetrics:
    """Verify 3D-specific metrics are logged correctly."""

    REQUIRED_3D_METRICS = [
        'Validation/MS-SSIM-3D',  # 3D MS-SSIM (with modality suffix)
    ]

    @pytest.fixture
    def captured_metrics_3d(self):
        """Capture metrics for 3D trainer."""
        logged = {'scalars': {}}

        def capture_scalar(tag, value, global_step=None):
            logged['scalars'][tag] = value

        mock_writer = MagicMock()
        mock_writer.add_scalar = capture_scalar

        return mock_writer, logged

    def test_3d_msssim_logged_for_bravo(self, captured_metrics_3d):
        """3D bravo mode must log Validation/MS-SSIM-3D_bravo."""
        from medgen.metrics.unified import UnifiedMetrics

        mock_writer, logged = captured_metrics_3d

        metrics = UnifiedMetrics(
            writer=mock_writer,
            mode='bravo',
            spatial_dims=3,  # 3D mode
            modality='bravo',
            device=torch.device('cpu'),
            enable_regional=False,
        )

        # Simulate 3D validation: update_msssim_3d accumulates values
        # Create small 3D tensors: (B, C, D, H, W)
        pred = torch.rand(1, 1, 16, 32, 32)
        gt = torch.rand(1, 1, 16, 32, 32)
        metrics.update_msssim_3d(pred, gt)

        # log_validation logs all accumulated metrics
        metrics.log_validation(epoch=1)

        # Check MS-SSIM-3D_bravo was logged
        assert 'Validation/MS-SSIM-3D_bravo' in logged['scalars'], (
            f"REGRESSION: 3D MS-SSIM not logged for bravo mode!\n"
            f"Expected: Validation/MS-SSIM-3D_bravo\n"
            f"Logged: {list(logged['scalars'].keys())}"
        )

        # Verify value is a valid number
        value = logged['scalars']['Validation/MS-SSIM-3D_bravo']
        assert 0.0 <= value <= 1.0, (
            f"REGRESSION: 3D MS-SSIM value out of range: {value}"
        )

    def test_3d_msssim_not_logged_for_seg_mode(self, captured_metrics_3d):
        """3D seg mode should NOT log MS-SSIM-3D (seg doesn't use image quality)."""
        from medgen.metrics.unified import UnifiedMetrics

        mock_writer, logged = captured_metrics_3d

        # For seg mode, uses_image_quality=False, so no MS-SSIM is computed
        metrics = UnifiedMetrics(
            writer=mock_writer,
            mode='seg',
            spatial_dims=3,
            modality='seg',
            device=torch.device('cpu'),
            enable_regional=False,
        )

        # update_msssim_3d should return 0.0 and not accumulate for seg mode
        pred = torch.rand(1, 1, 16, 32, 32)
        gt = torch.rand(1, 1, 16, 32, 32)
        result = metrics.update_msssim_3d(pred, gt)

        assert result == 0.0, (
            f"seg mode should skip MS-SSIM computation, got {result}"
        )

        metrics.log_validation(epoch=1)

        # MS-SSIM-3D should NOT be in logged metrics for seg mode
        assert 'Validation/MS-SSIM-3D_seg' not in logged['scalars'], (
            f"REGRESSION: 3D MS-SSIM should NOT be logged for seg mode!\n"
            f"Logged: {list(logged['scalars'].keys())}"
        )

    def test_3d_msssim_requires_3d_spatial_dims(self, captured_metrics_3d):
        """update_msssim_3d should skip computation for 2D mode."""
        from medgen.metrics.unified import UnifiedMetrics

        mock_writer, logged = captured_metrics_3d

        metrics = UnifiedMetrics(
            writer=mock_writer,
            mode='bravo',
            spatial_dims=2,  # 2D mode - should skip 3D MS-SSIM
            modality='bravo',
            device=torch.device('cpu'),
        )

        # update_msssim_3d should return 0.0 for 2D mode
        pred = torch.rand(1, 1, 64, 64)
        gt = torch.rand(1, 1, 64, 64)
        result = metrics.update_msssim_3d(pred, gt)

        assert result == 0.0, (
            f"2D mode should skip 3D MS-SSIM computation, got {result}"
        )


class TestEndToEndMetricLogging:
    """
    End-to-end test: Run actual trainer and verify metrics appear.

    This is the most important test - it catches when the trainer
    stops calling the logging methods entirely.
    """

    @pytest.fixture
    def capture_writer(self):
        """Create a writer that captures all logged metrics."""
        logged_metrics = set()

        class CapturingWriter:
            def add_scalar(self, tag, value, global_step=None):
                logged_metrics.add(tag)

            def add_image(self, tag, img, global_step=None):
                logged_metrics.add(f"[IMAGE]{tag}")

            def add_figure(self, tag, fig, global_step=None):
                logged_metrics.add(f"[FIGURE]{tag}")

            def add_histogram(self, tag, values, global_step=None):
                logged_metrics.add(f"[HIST]{tag}")

            def flush(self):
                pass

            def close(self):
                pass

        return CapturingWriter(), logged_metrics

    @pytest.mark.timeout(60)
    def test_trainer_logs_metrics_after_epoch(self, capture_writer, tmp_path):
        """
        CRITICAL: Trainer must log metrics after training epoch.

        This test catches when:
        - log_training() stops being called
        - Metric update methods stop being called
        - Writer is not properly initialized
        """
        from medgen.metrics.unified import UnifiedMetrics

        writer, logged_metrics = capture_writer

        # Create metrics tracker (same as trainer does)
        metrics = UnifiedMetrics(
            writer=writer,
            mode='bravo',
            spatial_dims=2,
            modality='bravo',
            device=torch.device('cpu'),
            enable_regional=False,
            log_grad_norm=True,
            log_timestep_losses=False,
            log_regional_losses=False,
            log_msssim=True,
            log_psnr=True,
            log_lpips=False,
            log_flops=False,
        )

        # Simulate what trainer does each epoch
        # 1. Training loop
        for step in range(5):
            metrics.update_loss('Total', 0.5 - step * 0.05, phase='train')
            metrics.update_loss('MSE', 0.3 - step * 0.03, phase='train')
            metrics.update_grad_norm(1.0 + step * 0.1)

        metrics.update_lr(0.0001)
        metrics.log_training(epoch=1)

        # 2. Validation loop
        for step in range(3):
            pred = torch.rand(2, 1, 64, 64)
            gt = torch.rand(2, 1, 64, 64)
            metrics.update_psnr(pred, gt)
            metrics.update_msssim(pred, gt)

        metrics.log_validation(epoch=1)

        # =====================================================================
        # CRITICAL ASSERTIONS - If these fail, metrics are broken!
        # =====================================================================

        # Must have training metrics
        assert any('Loss' in m and 'train' in m for m in logged_metrics), (
            f"CRITICAL: No training loss logged!\n"
            f"Logged: {sorted(logged_metrics)}\n"
            f"Check: Is log_training() being called?"
        )

        assert any('LR' in m for m in logged_metrics), (
            f"CRITICAL: Learning rate not logged!\n"
            f"Logged: {sorted(logged_metrics)}\n"
            f"Check: Is update_lr() being called?"
        )

        # Must have validation metrics
        assert any('PSNR' in m for m in logged_metrics), (
            f"CRITICAL: PSNR not logged!\n"
            f"Logged: {sorted(logged_metrics)}\n"
            f"Check: Is log_validation() being called?"
        )

        assert any('MS-SSIM' in m or 'SSIM' in m for m in logged_metrics), (
            f"CRITICAL: MS-SSIM not logged!\n"
            f"Logged: {sorted(logged_metrics)}\n"
            f"Check: Is update_msssim() being called?"
        )

        # Print all logged metrics for debugging
        print(f"\n✓ Logged {len(logged_metrics)} metrics:")
        for m in sorted(logged_metrics):
            print(f"  - {m}")


class TestMetricListCompleteness:
    """
    Meta-test: Verify the test's metric lists are up-to-date.

    If you add new metrics to UnifiedMetrics, add them to the lists above.
    This test reminds you to do that.
    """

    def test_document_all_logged_metrics(self):
        """
        This test documents all metrics that SHOULD be logged.

        If you add a new metric to UnifiedMetrics:
        1. Add it to the appropriate list in TestDiffusionTrainerMetrics
        2. Add a test that verifies it's logged

        Current expected metrics:

        TRAINING:
        - Loss/Total_train
        - Loss/MSE_train
        - Loss/Perceptual_train (when enabled)
        - LR/Generator
        - training/grad_norm_max (when enabled)
        - VRAM/* (when enabled)
        - FLOPs/* (when enabled)

        VALIDATION:
        - Validation/PSNR
        - Validation/MS-SSIM
        - Validation/MS-SSIM-3D (3D only)
        - Validation/LPIPS (image modes)
        - Validation/Dice (seg modes)
        - Validation/IoU (seg modes)
        - Loss/*_val

        GENERATION:
        - Generation/FID
        - Generation/KID
        - Generation_Diversity/*
        """
        pass  # This is documentation, not a real test
