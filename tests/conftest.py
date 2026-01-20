"""Pytest configuration and shared fixtures."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(scope="session")
def device():
    """Get test device (GPU if available)."""
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope="function")
def deterministic_seed():
    """Set deterministic seed for reproducible tests."""
    import torch
    import numpy as np
    import random

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed
