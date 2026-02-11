"""Unit test specific fixtures - mocks and synthetic data."""
import pytest
from unittest.mock import Mock
import torch

from medgen.metrics.quality import compute_lpips


@pytest.fixture(scope="session")
def lpips_available():
    """Check if LPIPS model weights can be downloaded.

    Skips tests when the RadImageNet weights are unavailable
    (Google Drive rate-limiting, network errors, etc.).
    """
    try:
        compute_lpips(torch.rand(2, 1, 64, 64), torch.rand(2, 1, 64, 64))
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        # Google Drive rate-limiting, network errors, download failures
        skip_patterns = ["FileURLRetrievalError", "Too many users", "gdown", "urlopen"]
        if any(p in err for p in skip_patterns):
            pytest.skip(f"LPIPS model weights unavailable: {err}")
        raise


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
