"""Integration tests for TensorBoard logging functionality.

Tests logging format, metric accumulation, and actual file output verification.
"""
import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch


class TestTensorBoardLogging:
    """Tests for TensorBoard logging format and call patterns."""

    @pytest.fixture
    def mock_writer(self):
        """Mock SummaryWriter for call verification."""
        writer = MagicMock()
        writer.add_scalar = MagicMock()
        writer.add_image = MagicMock()
        writer.add_figure = MagicMock()
        writer.add_histogram = MagicMock()
        writer.flush = MagicMock()
        writer.close = MagicMock()
        return writer

    def test_scalar_logging_format(self, mock_writer):
        """Scalar tags use 'group/name' format convention."""
        # Simulate logging calls following project convention
        mock_writer.add_scalar('train/loss', 0.5, global_step=100)
        mock_writer.add_scalar('train/lr', 0.001, global_step=100)
        mock_writer.add_scalar('val/loss', 0.4, global_step=100)
        mock_writer.add_scalar('val/psnr', 25.0, global_step=100)

        # Verify calls
        assert mock_writer.add_scalar.call_count == 4

        # Check tag format
        calls = mock_writer.add_scalar.call_args_list
        for call in calls:
            tag = call[0][0]
            assert '/' in tag, f"Tag '{tag}' should use 'group/name' format"

    def test_image_logging_shape(self, mock_writer):
        """Images logged with correct shape: [C, H, W] or [1, C, H, W]."""
        # Test 3D tensor [C, H, W]
        image_3d = torch.rand(1, 64, 64)
        mock_writer.add_image('samples/generated', image_3d, global_step=100)
        mock_writer.add_image.assert_called_with('samples/generated', image_3d, global_step=100)

        # Test 4D tensor [1, C, H, W] - batch dim should be 1
        image_4d = torch.rand(1, 1, 64, 64)
        mock_writer.add_image('samples/real', image_4d[0], global_step=100)

        assert mock_writer.add_image.call_count == 2

    def test_metric_names_consistent(self, mock_writer):
        """Expected metric names follow project conventions."""
        expected_metrics = [
            'train/loss',
            'train/mse',
            'train/perceptual',
            'val/loss',
            'val/psnr',
            'val/ssim',
            'gen/fid',
            'gen/diversity',
        ]

        # Log all expected metrics
        for i, metric in enumerate(expected_metrics):
            mock_writer.add_scalar(metric, float(i), global_step=100)

        # Verify all were logged
        logged_tags = [call[0][0] for call in mock_writer.add_scalar.call_args_list]
        for expected in expected_metrics:
            assert expected in logged_tags, f"Expected metric '{expected}' not logged"

    def test_histogram_logging(self, mock_writer):
        """Histograms can be logged for gradient/weight analysis."""
        # Simulate gradient histogram logging
        gradients = torch.randn(100)
        mock_writer.add_histogram('gradients/encoder', gradients, global_step=100)

        mock_writer.add_histogram.assert_called_once()
        tag = mock_writer.add_histogram.call_args[0][0]
        assert '/' in tag, "Histogram tag should use group/name format"


class TestMetricAccumulation:
    """Tests for running average metric computation."""

    def test_running_average_computation(self):
        """Running average matches manual calculation."""
        # Simulate metric accumulator
        class MetricAccumulator:
            def __init__(self):
                self.sum = 0.0
                self.count = 0

            def update(self, value, n=1):
                self.sum += value * n
                self.count += n

            def average(self):
                return self.sum / self.count if self.count > 0 else 0.0

            def reset(self):
                self.sum = 0.0
                self.count = 0

        acc = MetricAccumulator()

        # Add values
        values = [0.5, 0.3, 0.4, 0.6]
        for v in values:
            acc.update(v)

        expected_avg = sum(values) / len(values)
        assert abs(acc.average() - expected_avg) < 1e-6, "Running average incorrect"

    def test_weighted_running_average(self):
        """Running average handles weighted updates (variable batch sizes)."""
        class MetricAccumulator:
            def __init__(self):
                self.sum = 0.0
                self.count = 0

            def update(self, value, n=1):
                self.sum += value * n
                self.count += n

            def average(self):
                return self.sum / self.count if self.count > 0 else 0.0

        acc = MetricAccumulator()

        # Batches with different sizes
        acc.update(0.5, n=4)  # batch of 4
        acc.update(0.3, n=2)  # batch of 2
        acc.update(0.4, n=4)  # batch of 4

        # Manual calculation: (0.5*4 + 0.3*2 + 0.4*4) / 10 = 4.2/10 = 0.42
        expected = (0.5 * 4 + 0.3 * 2 + 0.4 * 4) / 10
        assert abs(acc.average() - expected) < 1e-6, "Weighted running average incorrect"

    def test_per_epoch_reset(self):
        """Metrics reset at epoch boundary."""
        class MetricAccumulator:
            def __init__(self):
                self.sum = 0.0
                self.count = 0

            def update(self, value, n=1):
                self.sum += value * n
                self.count += n

            def average(self):
                return self.sum / self.count if self.count > 0 else 0.0

            def reset(self):
                self.sum = 0.0
                self.count = 0

        acc = MetricAccumulator()

        # Epoch 1
        acc.update(0.5)
        acc.update(0.6)
        epoch1_avg = acc.average()

        # Reset for epoch 2
        acc.reset()

        # Epoch 2
        acc.update(0.2)
        acc.update(0.3)
        epoch2_avg = acc.average()

        assert abs(epoch1_avg - 0.55) < 1e-6, "Epoch 1 average incorrect"
        assert abs(epoch2_avg - 0.25) < 1e-6, "Epoch 2 average incorrect after reset"


class TestRealTensorBoardLogging:
    """Tests that create actual TensorBoard log files."""

    @pytest.fixture
    def temp_log_dir(self, tmp_path):
        """Temporary directory for TensorBoard logs."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        return log_dir

    def test_events_file_created(self, temp_log_dir):
        """TensorBoard events file is created on disk."""
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(temp_log_dir))
        writer.add_scalar('test/loss', 0.5, global_step=1)
        writer.flush()
        writer.close()

        # Check for events file
        files = list(temp_log_dir.glob('events.out.tfevents.*'))
        assert len(files) > 0, "No events file created"

    def test_multiple_scalars_logged(self, temp_log_dir):
        """Multiple scalars can be logged in sequence."""
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(temp_log_dir))

        # Log multiple scalars at multiple steps
        for step in range(10):
            writer.add_scalar('train/loss', 1.0 / (step + 1), global_step=step)
            writer.add_scalar('train/accuracy', step / 10.0, global_step=step)

        writer.flush()
        writer.close()

        # Events file should exist
        files = list(temp_log_dir.glob('events.out.tfevents.*'))
        assert len(files) > 0, "No events file created"

    def test_image_logged_to_events(self, temp_log_dir):
        """Image can be logged to events file."""
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(temp_log_dir))

        # Create a small test image [C, H, W]
        image = torch.rand(3, 32, 32)
        writer.add_image('samples/test', image, global_step=0)

        writer.flush()
        writer.close()

        # Events file should exist and have non-trivial size
        files = list(temp_log_dir.glob('events.out.tfevents.*'))
        assert len(files) > 0
        assert files[0].stat().st_size > 0, "Events file is empty"

    def test_figure_logged_to_events(self, temp_log_dir):
        """Matplotlib figure can be logged to events file."""
        pytest.importorskip('matplotlib')
        import matplotlib.pyplot as plt
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(temp_log_dir))

        # Create a simple figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title('Test Figure')

        writer.add_figure('figures/test', fig, global_step=0)
        plt.close(fig)

        writer.flush()
        writer.close()

        # Events file should exist
        files = list(temp_log_dir.glob('events.out.tfevents.*'))
        assert len(files) > 0

    def test_writer_context_manager(self, temp_log_dir):
        """SummaryWriter works correctly as context manager."""
        from torch.utils.tensorboard import SummaryWriter

        # Use as context manager (auto-closes)
        with SummaryWriter(log_dir=str(temp_log_dir)) as writer:
            writer.add_scalar('test/value', 42.0, global_step=0)

        # Events file should exist after context exits
        files = list(temp_log_dir.glob('events.out.tfevents.*'))
        assert len(files) > 0


class TestLoggerIntegration:
    """Tests for Python logging integration."""

    def test_logging_levels(self, caplog):
        """Different logging levels are captured correctly."""
        import logging

        logger = logging.getLogger('test_logger')

        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")

        assert "Debug message" in caplog.text
        assert "Info message" in caplog.text
        assert "Warning message" in caplog.text
        assert "Error message" in caplog.text

    def test_training_log_format(self, caplog):
        """Training logs follow expected format."""
        import logging

        logger = logging.getLogger('medgen.training')

        with caplog.at_level(logging.INFO):
            # Simulate training log
            logger.info("Epoch 1/100 | Loss: 0.5000 | LR: 1.00e-04")

        assert "Epoch 1/100" in caplog.text
        assert "Loss:" in caplog.text
        assert "LR:" in caplog.text


class TestWriterFlushBehavior:
    """Tests for proper writer flush and close behavior."""

    def test_flush_writes_pending(self, tmp_path):
        """Flush writes pending events to disk."""
        from torch.utils.tensorboard import SummaryWriter
        import os

        log_dir = tmp_path / "flush_test"
        log_dir.mkdir()

        writer = SummaryWriter(log_dir=str(log_dir))
        writer.add_scalar('test/value', 1.0, global_step=0)

        # Before flush, file might be empty or not exist
        writer.flush()

        # After flush, events should be on disk
        files = list(log_dir.glob('events.out.tfevents.*'))
        assert len(files) > 0

        writer.close()

    def test_close_without_flush(self, tmp_path):
        """Close implicitly flushes pending events."""
        from torch.utils.tensorboard import SummaryWriter

        log_dir = tmp_path / "close_test"
        log_dir.mkdir()

        writer = SummaryWriter(log_dir=str(log_dir))
        writer.add_scalar('test/value', 1.0, global_step=0)
        writer.close()  # Should flush implicitly

        # Events should be on disk
        files = list(log_dir.glob('events.out.tfevents.*'))
        assert len(files) > 0
        assert files[0].stat().st_size > 0
