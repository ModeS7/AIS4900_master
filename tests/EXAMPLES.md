# Test Examples

Copy-paste templates for common test patterns in MedGen.

---

## Unit Test: Pure Function

```python
"""Test mathematical/pure functions."""
import pytest
import torch

from medgen.metrics.quality import compute_psnr


class TestComputePSNR:
    """Tests for compute_psnr function."""

    def test_identical_images_returns_100(self):
        """Identical images should have PSNR of 100 (capped)."""
        images = torch.rand(2, 1, 64, 64)
        result = compute_psnr(images, images)
        assert result == pytest.approx(100.0, abs=0.01)

    def test_different_images_returns_finite(self):
        """Different images should have finite PSNR."""
        img1 = torch.rand(2, 1, 64, 64)
        img2 = torch.rand(2, 1, 64, 64)
        result = compute_psnr(img1, img2)
        assert 0 < result < 100

    def test_batch_size_one(self):
        """Should work with batch size 1."""
        images = torch.rand(1, 1, 64, 64)
        result = compute_psnr(images, images)
        assert result == pytest.approx(100.0, abs=0.01)

    def test_empty_batch_raises(self):
        """Empty batch should raise ValueError."""
        images = torch.rand(0, 1, 64, 64)
        with pytest.raises(ValueError):
            compute_psnr(images, images)
```

---

## Unit Test: Metric Class

```python
"""Test metric tracker classes."""
import pytest
from unittest.mock import MagicMock, patch

from medgen.pipeline.metrics.tracker import MetricsTracker


class TestMetricsTracker:
    """Tests for MetricsTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create tracker with mocked writer."""
        with patch("medgen.pipeline.metrics.tracker.SummaryWriter"):
            return MetricsTracker(log_dir="/tmp/test")

    def test_log_scalar_writes_to_tensorboard(self, tracker):
        """log_scalar should call writer.add_scalar."""
        tracker.log_scalar("loss", 0.5, step=100)
        tracker.writer.add_scalar.assert_called_once_with("loss", 0.5, 100)

    def test_compute_epoch_summary_returns_dict(self, tracker):
        """compute_epoch_summary should return all logged metrics."""
        tracker.log_scalar("loss", 0.5, step=1)
        tracker.log_scalar("loss", 0.3, step=2)
        summary = tracker.compute_epoch_summary()
        assert "loss" in summary
        assert summary["loss"] == pytest.approx(0.4, abs=0.01)
```

---

## Integration Test: Trainer

```python
"""Test trainer with real components."""
import pytest
import torch

from medgen.pipeline.trainer import DiffusionTrainer


class TestDiffusionTrainerIntegration:
    """Integration tests for DiffusionTrainer."""

    @pytest.fixture
    def minimal_config(self, tmp_path):
        """Create minimal config for testing."""
        return {
            "model": {"image_size": 64, "spatial_dims": 2},
            "training": {"epochs": 1, "batch_size": 2},
            "paths": {"output_dir": str(tmp_path)},
            "mode": "seg",
            "strategy": "ddpm",
        }

    @pytest.mark.timeout(30)
    def test_train_one_epoch_saves_checkpoint(self, minimal_config, tmp_path):
        """Training should save checkpoint after epoch."""
        trainer = DiffusionTrainer(minimal_config)
        trainer.train()

        assert (tmp_path / "latest.pt").exists()

    @pytest.mark.timeout(30)
    def test_resume_from_checkpoint(self, minimal_config, tmp_path):
        """Should resume training from checkpoint."""
        trainer = DiffusionTrainer(minimal_config)
        trainer.train()
        initial_step = trainer.global_step

        # Resume
        trainer2 = DiffusionTrainer(minimal_config)
        trainer2.resume(tmp_path / "latest.pt")

        assert trainer2.global_step == initial_step

    @pytest.mark.gpu
    @pytest.mark.timeout(60)
    def test_train_on_gpu(self, minimal_config):
        """Training should work on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")

        minimal_config["device"] = "cuda"
        trainer = DiffusionTrainer(minimal_config)
        trainer.train()

        assert trainer.model.device.type == "cuda"
```

---

## Property Test: Mathematical Properties

```python
"""Test mathematical properties with hypothesis."""
import pytest
from hypothesis import given, strategies as st, settings
import torch

from medgen.metrics.quality import compute_ssim


class TestSSIMProperties:
    """Property-based tests for SSIM."""

    @given(st.integers(min_value=1, max_value=8))
    @settings(max_examples=10)
    def test_ssim_symmetric(self, batch_size):
        """SSIM(a, b) == SSIM(b, a)."""
        a = torch.rand(batch_size, 1, 64, 64)
        b = torch.rand(batch_size, 1, 64, 64)
        assert compute_ssim(a, b) == pytest.approx(compute_ssim(b, a), abs=1e-5)

    @given(st.integers(min_value=1, max_value=8))
    @settings(max_examples=10)
    def test_ssim_identity(self, batch_size):
        """SSIM(a, a) == 1.0."""
        a = torch.rand(batch_size, 1, 64, 64)
        assert compute_ssim(a, a) == pytest.approx(1.0, abs=1e-5)

    @given(st.integers(min_value=1, max_value=8))
    @settings(max_examples=10)
    def test_ssim_bounded(self, batch_size):
        """SSIM should be in [-1, 1]."""
        a = torch.rand(batch_size, 1, 64, 64)
        b = torch.rand(batch_size, 1, 64, 64)
        result = compute_ssim(a, b)
        assert -1.0 <= result <= 1.0
```

---

## Regression Test: Bug Fix

```python
"""Regression test for specific bug fix."""
import pytest
import torch

from medgen.pipeline.strategies import RFlowStrategy


class TestRFlowTimestepBug:
    """Regression test for pitfall #42: continuous timesteps."""

    def test_continuous_timesteps_in_zero_one_range(self):
        """
        Bug: RFlow was using discrete timesteps [0, 999] instead of continuous [0, 1].
        Fix: Added use_discrete_timesteps=false and proper scaling.

        Reference: docs/common-pitfalls.md #42
        """
        strategy = RFlowStrategy(num_train_timesteps=1000, use_discrete_timesteps=False)

        # Sample timesteps
        timesteps = strategy.sample_timesteps(batch_size=100)

        # Should be continuous floats in [0, 1], not discrete integers
        assert timesteps.dtype == torch.float32
        assert timesteps.min() >= 0.0
        assert timesteps.max() <= 1.0

        # Should NOT be integers
        assert not torch.all(timesteps == timesteps.round())


class TestBF16PrecisionBug:
    """Regression test for pitfall #15: BF16 precision loss."""

    def test_loss_computed_in_float32(self):
        """
        Bug: Computing loss in BF16 caused precision issues.
        Fix: Always call .float() before loss computation.

        Reference: docs/common-pitfalls.md #15
        """
        pred = torch.randn(2, 1, 64, 64, dtype=torch.bfloat16)
        target = torch.randn(2, 1, 64, 64, dtype=torch.bfloat16)

        # The fix: convert to float32 before loss
        loss = torch.nn.functional.mse_loss(pred.float(), target.float())

        assert loss.dtype == torch.float32
        assert not torch.isnan(loss)
```

---

## E2E Test: CLI Command

```python
"""End-to-end test for CLI commands."""
import pytest
import subprocess
from pathlib import Path


class TestTrainCLI:
    """E2E tests for train.py CLI."""

    @pytest.mark.e2e
    @pytest.mark.timeout(120)
    def test_train_help_exits_zero(self):
        """--help should work without errors."""
        result = subprocess.run(
            ["python", "-m", "medgen.scripts.train", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()

    @pytest.mark.e2e
    @pytest.mark.timeout(300)
    def test_train_debug_mode(self, tmp_path):
        """Fast debug training should complete."""
        result = subprocess.run(
            [
                "python", "-m", "medgen.scripts.train",
                "training=fast_debug",
                f"paths.output_dir={tmp_path}",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0
        assert (tmp_path / "latest.pt").exists()

    @pytest.mark.e2e
    @pytest.mark.gpu
    @pytest.mark.timeout(600)
    def test_full_training_run(self, tmp_path, golden_checkpoint_dir):
        """Full training run should produce valid checkpoint."""
        result = subprocess.run(
            [
                "python", "-m", "medgen.scripts.train",
                "mode=seg",
                "strategy=ddpm",
                "training.epochs=2",
                f"paths.output_dir={tmp_path}",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0

        # Verify checkpoint is loadable
        import torch
        checkpoint = torch.load(tmp_path / "latest.pt")
        assert "model_state_dict" in checkpoint
        assert "epoch" in checkpoint
```

---

## Common Assertions

```python
"""Reusable assertion helpers."""
import torch
import pytest


def assert_valid_metric(value: float, name: str, min_val: float = 0.0, max_val: float = float("inf")):
    """Assert metric is valid (finite, in range)."""
    assert isinstance(value, (int, float)), f"{name} should be numeric, got {type(value)}"
    assert not torch.isnan(torch.tensor(value)), f"{name} should not be NaN"
    assert not torch.isinf(torch.tensor(value)), f"{name} should not be infinite"
    assert min_val <= value <= max_val, f"{name}={value} not in [{min_val}, {max_val}]"


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple, name: str = "tensor"):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, f"{name} shape {tensor.shape} != expected {expected_shape}"


def assert_tensor_range(tensor: torch.Tensor, min_val: float, max_val: float, name: str = "tensor"):
    """Assert tensor values are in expected range."""
    assert tensor.min() >= min_val, f"{name} min {tensor.min()} < {min_val}"
    assert tensor.max() <= max_val, f"{name} max {tensor.max()} > {max_val}"


def assert_no_nan_inf(tensor: torch.Tensor, name: str = "tensor"):
    """Assert tensor has no NaN or Inf values."""
    assert not torch.isnan(tensor).any(), f"{name} contains NaN"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf"


def assert_gradient_exists(model: torch.nn.Module, name: str = "model"):
    """Assert at least one parameter has gradients."""
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, f"{name} has no gradients after backward()"
```

---

## Marker Examples

```python
"""Examples of pytest markers."""
import pytest


@pytest.mark.slow
def test_large_model_training():
    """Mark slow tests that take >10 seconds."""
    pass


@pytest.mark.gpu
def test_cuda_operations():
    """Mark tests requiring GPU."""
    pass


@pytest.mark.e2e
def test_full_pipeline():
    """Mark end-to-end tests."""
    pass


@pytest.mark.baseline
def test_metric_baseline():
    """Mark baseline comparison tests."""
    pass


@pytest.mark.benchmark
def test_training_throughput(benchmark):
    """Mark performance benchmark tests."""
    pass


@pytest.mark.parametrize("mode", ["seg", "bravo", "dual"])
def test_all_modes(mode):
    """Parametrize tests across multiple values."""
    pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_gpu_specific():
    """Skip conditionally."""
    pass
```

---

## Fixture Usage Examples

```python
"""Examples of using fixtures."""
import pytest


def test_with_device(device):
    """Use device fixture for CPU/GPU portability."""
    tensor = torch.rand(2, 2).to(device)
    assert tensor.device.type == device


def test_with_factory(make_images):
    """Use factory fixture for flexible tensor creation."""
    images = make_images(batch_size=4, channels=3, height=128, width=128)
    assert images.shape == (4, 3, 128, 128)


def test_with_config(minimal_diffusion_config):
    """Use pre-built config fixture."""
    assert minimal_diffusion_config["model"]["image_size"] == 64


def test_with_tmp_path(tmp_path):
    """Use pytest's tmp_path for file operations."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello")
    assert file_path.read_text() == "hello"
```
