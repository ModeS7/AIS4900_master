"""Pytest configuration and shared fixtures."""

import pytest
import torch
from unittest.mock import Mock


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
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


# ============================================================================
# Synthetic Data Fixtures
# ============================================================================

@pytest.fixture
def cpu_device():
    """CPU device for tests that don't need GPU."""
    return torch.device('cpu')


@pytest.fixture
def synthetic_images_2d():
    """Synthetic 2D images [B, C, H, W]."""
    return torch.rand(4, 1, 64, 64)


@pytest.fixture
def synthetic_images_3d():
    """Synthetic 3D volumes [B, C, D, H, W]."""
    return torch.rand(2, 1, 16, 64, 64)


@pytest.fixture
def synthetic_masks_2d():
    """Binary masks [B, 1, H, W]."""
    return (torch.rand(4, 1, 64, 64) > 0.7).float()


@pytest.fixture
def synthetic_masks_3d():
    """Binary masks [B, 1, D, H, W]."""
    return (torch.rand(2, 1, 16, 64, 64) > 0.7).float()


# ============================================================================
# Mock Model Fixtures
# ============================================================================

@pytest.fixture
def mock_diffusion_model():
    """Mock diffusion model for testing.

    Returns dynamic batch size matching input rather than fixed size.
    """
    def forward(x, *args, **kwargs):
        # Match input batch and return single channel (noise/velocity prediction)
        return torch.randn_like(x[:, :1])

    model = Mock(side_effect=forward)
    model.eval = Mock(return_value=model)
    model.to = Mock(return_value=model)
    return model


@pytest.fixture
def mock_size_bin_model():
    """Mock model with size_bin_time_embed attribute (SizeBinModelWrapper).

    Returns dynamic batch size matching input rather than fixed size.
    """
    def forward(x, *args, **kwargs):
        return torch.randn_like(x[:, :1])

    model = Mock(side_effect=forward)
    model.size_bin_time_embed = Mock()  # Key attribute for detection
    model.eval = Mock(return_value=model)
    model.to = Mock(return_value=model)
    return model


@pytest.fixture
def mock_omega_model():
    """Mock model that accepts omega/mode_id (ScoreAug/ModeEmbed)."""
    def forward(x, timesteps, omega=None, mode_id=None):
        return torch.randn_like(x[:, :1])  # Return noise/velocity prediction

    model = Mock(side_effect=forward)
    model.eval = Mock(return_value=model)
    model.to = Mock(return_value=model)
    return model


# ============================================================================
# Strategy Fixtures
# ============================================================================

@pytest.fixture
def ddpm_strategy():
    """Configured DDPMStrategy for testing."""
    from medgen.diffusion import DDPMStrategy
    strategy = DDPMStrategy()
    strategy.setup_scheduler(num_timesteps=100, image_size=64)
    return strategy


@pytest.fixture
def rflow_strategy_2d():
    """Configured RFlowStrategy for 2D."""
    from medgen.diffusion import RFlowStrategy
    strategy = RFlowStrategy()
    strategy.setup_scheduler(
        num_timesteps=100,
        image_size=64,
        spatial_dims=2,
    )
    return strategy


@pytest.fixture
def rflow_strategy_3d():
    """Configured RFlowStrategy for 3D."""
    from medgen.diffusion import RFlowStrategy
    strategy = RFlowStrategy()
    strategy.setup_scheduler(
        num_timesteps=100,
        image_size=64,
        depth_size=16,
        spatial_dims=3,
    )
    return strategy


# ============================================================================
# Dataset Fixtures
# ============================================================================

@pytest.fixture
def mock_seg_conditioned_dataset():
    """Mock dataset returning (seg, size_bins) tuples for seg_conditioned mode."""
    class MockDataset:
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            torch.manual_seed(idx)  # Reproducible per-item
            seg = (torch.rand(1, 64, 64) > 0.5).float()
            size_bins = torch.randint(0, 5, (7,))
            return (seg, size_bins)

    return MockDataset()


@pytest.fixture
def mock_bravo_dataset():
    """Mock dataset returning (image, seg) tuples for bravo mode."""
    class MockDataset:
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            torch.manual_seed(idx)
            image = torch.rand(1, 64, 64)
            seg = (torch.rand(1, 64, 64) > 0.5).float()
            return (image, seg)

    return MockDataset()


@pytest.fixture
def mock_dict_dataset():
    """Mock dataset returning dict format {'image': ..., 'seg': ...}."""
    class MockDataset:
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            torch.manual_seed(idx)
            return {
                'image': torch.rand(1, 64, 64),
                'seg': (torch.rand(1, 64, 64) > 0.5).float(),
            }

    return MockDataset()


# ============================================================================
# Factory Fixtures (Dynamic Tensor Creation)
# ============================================================================

@pytest.fixture
def make_images():
    """Factory for synthetic images with configurable dimensions."""
    def _make(batch_size=4, channels=1, height=64, width=64, depth=None):
        if depth is not None:
            return torch.rand(batch_size, channels, depth, height, width)
        return torch.rand(batch_size, channels, height, width)
    return _make


@pytest.fixture
def make_masks():
    """Factory for binary masks with configurable sparsity."""
    def _make(batch_size=4, channels=1, height=64, width=64, depth=None, threshold=0.7):
        shape = (batch_size, channels, depth, height, width) if depth else \
                (batch_size, channels, height, width)
        return (torch.rand(*shape) > threshold).float()
    return _make


# ============================================================================
# Auto-Skip GPU Tests
# ============================================================================

@pytest.fixture(autouse=True)
def skip_gpu_tests_without_cuda(request):
    """Auto-skip @pytest.mark.gpu tests when CUDA unavailable."""
    if request.node.get_closest_marker('gpu'):
        if not torch.cuda.is_available():
            pytest.skip('GPU not available')


# =============================================================================
# State Isolation Fixtures (autouse - run for every test)
# =============================================================================

@pytest.fixture(autouse=True)
def isolate_test_state():
    """Ensure each test starts with clean metric caches.

    Runs before and after EVERY test automatically.
    """
    from medgen.metrics.quality import clear_metric_caches
    clear_metric_caches()
    yield
    clear_metric_caches()


@pytest.fixture(autouse=True)
def isolate_cuda_state():
    """Clear CUDA state after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.fixture(autouse=True)
def isolate_rng_state():
    """Preserve RNG state across tests.

    Prevents one test's random usage from affecting another.
    """
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    yield
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)


@pytest.fixture(autouse=True)
def isolate_environment():
    """Preserve environment variables."""
    import os
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs.

    Automatically cleaned up after test.
    """
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


@pytest.fixture(scope="session")
def shared_temp_dir(tmp_path_factory):
    """Shared temp directory for session-scoped resources.

    Use sparingly - only for expensive-to-create resources.
    """
    return tmp_path_factory.mktemp("shared")