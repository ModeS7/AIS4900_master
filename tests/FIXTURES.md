# Test Fixtures

## Overview

This document describes all test fixtures available in the MedGen test suite.

---

## Global Fixtures (`tests/conftest.py`)

### Device Management

```python
@pytest.fixture
def device():
    """Returns 'cuda' if available, else 'cpu'."""
```

### Random Seed Control

```python
@pytest.fixture
def seed():
    """Returns fixed seed (42) for reproducibility."""

@pytest.fixture(autouse=True)
def set_seed(seed):
    """Auto-use: Sets random seed before each test."""
```

### Tensor Factories

```python
@pytest.fixture
def make_images():
    """Factory for creating test image tensors.

    Usage:
        images = make_images(batch_size=4, channels=1, height=64, width=64)

    Returns:
        Callable that creates normalized image tensors in [0, 1]
    """

@pytest.fixture
def make_3d_images():
    """Factory for creating 3D test image tensors.

    Usage:
        images = make_3d_images(batch=2, channels=1, depth=32, height=64, width=64)
    """

@pytest.fixture
def make_segmentation():
    """Factory for creating binary segmentation masks.

    Usage:
        masks = make_segmentation(batch_size=4, height=64, width=64)

    Returns:
        Binary tensors with values in {0, 1}
    """
```

### Mock Factories

```python
@pytest.fixture
def mock_model():
    """Returns a mock nn.Module that returns input unchanged."""

@pytest.fixture
def mock_config():
    """Returns a minimal OmegaConf DictConfig for testing."""
```

---

## Unit Test Fixtures (`tests/unit/conftest.py`)

### Strategy Mocks

```python
@pytest.fixture
def mock_ddpm_strategy():
    """DDPM strategy with mocked noise scheduler."""

@pytest.fixture
def mock_rflow_strategy():
    """RFlow strategy with mocked velocity prediction."""
```

### Trainer Mocks

```python
@pytest.fixture
def minimal_trainer_config():
    """Minimal config dict for creating test trainers."""

@pytest.fixture
def mock_trainer(minimal_trainer_config, mock_model):
    """Fully mocked trainer for unit testing."""
```

### Metric Mocks

```python
@pytest.fixture
def mock_lpips():
    """Mocked LPIPS network that returns 0.1."""
```

---

## Integration Test Fixtures (`tests/integration/conftest.py`)

### File System

```python
@pytest.fixture
def tmp_run_dir(tmp_path):
    """Temporary directory for training runs.

    Creates structure:
        tmp_path/
        ├── runs/
        └── data/
    """

@pytest.fixture
def sample_nifti_files(tmp_run_dir):
    """Creates minimal NIfTI files for testing dataloaders."""
```

### Config Loading

```python
@pytest.fixture
def load_config():
    """Factory for loading Hydra configs.

    Usage:
        cfg = load_config("config.yaml", overrides=["model.image_size=64"])
    """

@pytest.fixture
def minimal_diffusion_config(load_config):
    """Pre-loaded minimal diffusion config (64px, 2 epochs)."""

@pytest.fixture
def minimal_vae_config(load_config):
    """Pre-loaded minimal VAE config (64px, 2 epochs)."""
```

### Real Components

```python
@pytest.fixture
def small_unet(device):
    """Small but real UNet for integration tests.

    Specs: 64px, 32 base channels, 2 attention levels
    """

@pytest.fixture
def small_vae(device):
    """Small but real VAE for integration tests.

    Specs: 64px, 4 latent channels, 32 base channels
    """
```

---

## E2E Test Fixtures (`tests/e2e/conftest.py`)

### CLI Helpers

```python
@pytest.fixture
def run_cli():
    """Factory for running CLI commands.

    Usage:
        result = run_cli(["python", "-m", "medgen.scripts.train", "--help"])
        assert result.returncode == 0
    """

@pytest.fixture
def capture_tensorboard():
    """Captures TensorBoard logs to dict for verification."""
```

### Golden Checkpoints

```python
@pytest.fixture
def golden_checkpoint_dir():
    """Path to tests/fixtures/golden_checkpoint/ directory."""

@pytest.fixture
def bravo_checkpoint(golden_checkpoint_dir):
    """Loads golden bravo checkpoint for testing."""

@pytest.fixture
def seg_checkpoint(golden_checkpoint_dir):
    """Loads golden segmentation checkpoint for testing."""
```

### Data Fixtures

```python
@pytest.fixture
def synthetic_dataset(tmp_path):
    """Creates synthetic NIfTI dataset for E2E tests.

    Creates 10 patients with all modalities.
    """
```

---

## Benchmark Fixtures (`tests/benchmarks/conftest.py`)

```python
@pytest.fixture
def benchmark_config():
    """Config optimized for benchmarking (no validation, minimal logging)."""

@pytest.fixture
def warmup_model(small_unet, device):
    """Pre-warmed model to exclude compilation time from benchmarks."""
```

---

## Auto-Use Fixtures

These fixtures run automatically without explicit use:

| Fixture | Location | Purpose |
|---------|----------|---------|
| `set_seed` | `tests/conftest.py` | Sets random seed before each test |
| `gpu_memory_cleanup` | `tests/integration/conftest.py` | Clears CUDA cache after GPU tests |
| `suppress_warnings` | `tests/unit/conftest.py` | Suppresses torch deprecation warnings |

---

## Creating New Fixtures

### Guidelines

1. **Scope appropriately**: Use `scope="session"` for expensive fixtures
2. **Document parameters**: Use docstrings with usage examples
3. **Minimize side effects**: Avoid modifying global state
4. **Cleanup resources**: Use `yield` for fixtures needing cleanup

### Example

```python
@pytest.fixture(scope="module")
def expensive_model(device):
    """Large model shared across module tests.

    Usage:
        def test_something(expensive_model):
            output = expensive_model(input_tensor)
    """
    model = create_large_model().to(device)
    model.eval()
    yield model
    # Cleanup
    del model
    torch.cuda.empty_cache()
```

### Factory Pattern

```python
@pytest.fixture
def make_trainer():
    """Factory for creating trainers with custom configs.

    Usage:
        trainer = make_trainer(mode="bravo", strategy="rflow")
    """
    def _make(mode="seg", strategy="ddpm", **kwargs):
        config = build_config(mode=mode, strategy=strategy, **kwargs)
        return DiffusionTrainer(config)
    return _make
```

---

## Fixture Dependencies

```
device
├── small_unet
├── small_vae
└── warmup_model

seed
└── set_seed (autouse)

tmp_path (pytest builtin)
├── tmp_run_dir
│   └── sample_nifti_files
└── synthetic_dataset

load_config
├── minimal_diffusion_config
└── minimal_vae_config
```
