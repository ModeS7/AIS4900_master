"""Integration test specific fixtures - real components."""
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_checkpoint_dir():
    """Temporary directory for checkpoint tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_run_dir():
    """Temporary directory simulating runs/ folder."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "test_run"
        run_dir.mkdir()
        (run_dir / "checkpoints").mkdir()
        (run_dir / "logs").mkdir()
        yield run_dir


@pytest.fixture
def mini_trainer_config():
    """Minimal config for fast integration tests."""
    return {
        'training.epochs': 1,
        'training.batch_size': 2,
        'model.image_size': 32,
    }


@pytest.fixture(scope="module")
def configs_dir():
    """Path to actual configs directory."""
    return Path(__file__).parent.parent.parent / "configs"
