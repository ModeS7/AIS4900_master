"""Unit test specific fixtures - mocks and synthetic data."""
from unittest.mock import Mock, patch

import pytest
import torch

from medgen.metrics import quality


class _FastPerceptualMetric(torch.nn.Module):
    """Deterministic stand-in for MONAI's pretrained perceptual backbone."""

    def forward(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(generated - reference))


@pytest.fixture
def lpips_available():
    """Use a fast local perceptual metric for unit-level behavior tests.

    Unit tests verify channel handling, aggregation, and return contracts. Loading
    MONAI's real pretrained RadImageNet backbone here makes the CPU suite depend on
    network availability and can take longer than the per-test timeout.
    """
    quality.clear_metric_caches()
    metric = _FastPerceptualMetric().eval()
    with patch.object(quality, "_get_lpips_metric", return_value=metric):
        yield
    quality.clear_metric_caches()


@pytest.fixture
def mock_model_dynamic():
    """Factory for mock model that returns tensor matching input batch size."""
    def _create(output_channels=1):
        def forward(x, *args, **kwargs):
            batch_size = x.shape[0]
            spatial = x.shape[2:]
            return torch.randn(batch_size, output_channels, *spatial)
        model = Mock(side_effect=forward)
        model.eval = Mock(return_value=model)
        model.to = Mock(return_value=model)
        return model
    return _create


@pytest.fixture
def mock_scheduler():
    """Mock scheduler for strategy tests."""
    scheduler = Mock()
    scheduler.num_train_timesteps = 1000
    scheduler.add_noise = Mock(side_effect=lambda x, n, t: x + n * 0.1)
    return scheduler


@pytest.fixture
def mock_trainer_config():
    """Minimal config dict for unit testing trainer components."""
    return {
        'model': {'in_channels': 1, 'out_channels': 1},
        'training': {'learning_rate': 1e-4, 'batch_size': 4},
    }
