"""Performance benchmarks for training components using pytest-benchmark.

These benchmarks measure the performance of core training operations like
noise addition and timestep sampling.

Run with:
    pytest tests/benchmarks/test_training_perf.py -v --benchmark-only
    pytest tests/benchmarks/test_training_perf.py --benchmark-compare -m "not slow"
"""
import pytest
import torch

# Skip if pytest-benchmark not installed
pytest.importorskip("pytest_benchmark")


@pytest.mark.benchmark(group="ddpm-strategy")
class TestDDPMStrategyPerformance:
    """Benchmark DDPM diffusion strategy operations."""

    @pytest.fixture
    def ddpm_strategy(self):
        """Create DDPM strategy for benchmarking."""
        from medgen.diffusion import DDPMStrategy

        strategy = DDPMStrategy()
        strategy.setup_scheduler(num_timesteps=1000, image_size=64)
        return strategy

    def test_add_noise(self, benchmark, ddpm_strategy):
        """Benchmark noise addition."""
        images = torch.randn(4, 1, 64, 64)
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, 1000, (4,))
        result = benchmark(ddpm_strategy.add_noise, images, noise, timesteps)
        assert result.shape == images.shape

    def test_sample_timesteps(self, benchmark, ddpm_strategy):
        """Benchmark timestep sampling."""
        images = torch.randn(32, 1, 64, 64)
        result = benchmark(ddpm_strategy.sample_timesteps, images)
        assert result.shape == (32,)


@pytest.mark.benchmark(group="rflow-strategy")
class TestRFlowStrategyPerformance:
    """Benchmark RFlow diffusion strategy operations."""

    @pytest.fixture
    def rflow_strategy(self):
        """Create RFlow strategy for benchmarking."""
        from medgen.diffusion import RFlowStrategy

        strategy = RFlowStrategy()
        strategy.setup_scheduler(num_timesteps=1000, image_size=64)
        return strategy

    def test_add_noise(self, benchmark, rflow_strategy):
        """Benchmark velocity-based noise addition."""
        images = torch.randn(4, 1, 64, 64)
        noise = torch.randn_like(images)
        timesteps = torch.rand(4) * 1000  # Continuous timesteps
        result = benchmark(rflow_strategy.add_noise, images, noise, timesteps)
        assert result.shape == images.shape

    def test_sample_timesteps(self, benchmark, rflow_strategy):
        """Benchmark continuous timestep sampling."""
        images = torch.randn(32, 1, 64, 64)
        result = benchmark(rflow_strategy.sample_timesteps, images)
        assert result.shape == (32,)


@pytest.mark.benchmark(group="tensor-ops")
class TestTensorOperationsPerformance:
    """Benchmark common tensor operations used in training."""

    def test_channel_concatenation(self, benchmark):
        """Benchmark channel concatenation (common in conditional diffusion)."""
        noise = torch.randn(8, 1, 256, 256)
        conditioning = torch.randn(8, 1, 256, 256)
        result = benchmark(torch.cat, [noise, conditioning], dim=1)
        assert result.shape == (8, 2, 256, 256)

    def test_clamp_to_range(self, benchmark):
        """Benchmark clamping to [0, 1] range (common post-generation)."""
        images = torch.randn(8, 1, 256, 256)
        result = benchmark(torch.clamp, images, 0, 1)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_mse_loss(self, benchmark):
        """Benchmark MSE loss computation (common training loss)."""
        pred = torch.randn(8, 1, 64, 64)
        target = torch.randn(8, 1, 64, 64)
        result = benchmark(torch.nn.functional.mse_loss, pred, target)
        assert result.ndim == 0  # Scalar

    @pytest.mark.timeout(120)
    @pytest.mark.slow
    def test_batch_normalization(self, benchmark):
        """Benchmark batch normalization (common layer)."""
        bn = torch.nn.BatchNorm2d(64)
        bn.eval()  # Use eval mode for deterministic benchmark
        x = torch.randn(8, 64, 64, 64)
        result = benchmark(bn, x)
        assert result.shape == x.shape
