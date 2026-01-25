"""Unit test specific fixtures - mocks and synthetic data."""
import pytest
from unittest.mock import Mock
import torch


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
