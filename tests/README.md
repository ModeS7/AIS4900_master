# Test Suite

## Running Tests

```bash
# All tests (excluding slow and GPU)
pytest -m "not slow and not gpu"

# Unit tests only (fast)
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# E2E tests (slow, may require GPU)
pytest tests/e2e/ -v -m "e2e"

# With coverage
pytest --cov=src/medgen --cov-report=html

# Specific test file
pytest tests/unit/test_quality_metrics.py -v
```

## Test Markers

| Marker | Description | Skip with |
|--------|-------------|-----------|
| `@pytest.mark.slow` | Tests taking >10s | `-m "not slow"` |
| `@pytest.mark.gpu` | Requires CUDA | Auto-skipped if no GPU |
| `@pytest.mark.e2e` | End-to-end tests | `-m "not e2e"` |
| `@pytest.mark.baseline` | Baseline comparisons | `-m "not baseline"` |

## Directory Structure

```
tests/
├── conftest.py          # Global fixtures (device, seeds, factories)
├── utils.py             # Shared assertion utilities
├── unit/                # Fast, isolated tests with mocks
│   ├── conftest.py      # Unit-specific mock factories
│   └── test_*.py
├── integration/         # Tests with real components
│   ├── conftest.py      # Temp dirs, config loading
│   ├── baselines/       # JSON baseline files
│   └── test_*.py
├── e2e/                 # Full pipeline tests
│   ├── conftest.py      # CLI helpers, full configs
│   └── test_*.py
└── benchmarks/          # Performance benchmarks
    ├── test_training_benchmarks.py
    └── test_metric_benchmarks.py
```

## Writing Tests

### Naming Convention
```
test_<what>_<condition>_<expected>
```

Examples:
- `test_psnr_identical_images_returns_100`
- `test_generate_with_cfg_doubles_forward_calls`
- `test_3d_msssim_returns_float_not_none`

### Using Fixtures

```python
# Use factory fixtures for dynamic tensor sizes
def test_with_custom_batch(make_images):
    images = make_images(batch_size=8, channels=3, height=128, width=128)
    ...

# Use shared assertions
from tests.utils import assert_valid_metric, assert_tensor_shape

def test_metric_output(result):
    assert_valid_metric(result, name="my_metric", max_val=1.0)
```

## Test Categories

| Category | Location | Characteristics |
|----------|----------|-----------------|
| Unit | `tests/unit/` | Mocks, no I/O, <1s per test |
| Integration | `tests/integration/` | Real components, may use GPU |
| E2E | `tests/e2e/` | Full pipeline, CLI, checkpoints |
| Benchmarks | `tests/benchmarks/` | Performance regression |

## Edge Case Handling

Tests should verify behavior for edge cases:

- **Empty batches**: `torch.rand(0, 1, 64, 64)` - expect ValueError or graceful handling
- **NaN/Inf inputs**: Functions should return NaN or raise, not crash silently
- **Device mismatch**: CPU vs CUDA tensors should raise RuntimeError
- **Non-contiguous tensors**: Should work correctly (call `.contiguous()` if needed)

Example:
```python
def test_psnr_nan_input_returns_nan(self):
    """NaN inputs should propagate to output."""
    images = torch.tensor([[[[float('nan')]]]])
    result = compute_psnr(images, images)
    assert math.isnan(result) or result == 100.0
```

## Timeout Best Practices

All slow tests MUST have timeout markers to prevent CI hangs:

```python
@pytest.mark.timeout(30)  # 30 seconds max
@pytest.mark.slow
def test_full_epoch_training(self):
    ...
```

Recommended timeouts:
- Unit tests: 5-10 seconds
- Integration tests: 30-60 seconds
- E2E tests: 120-300 seconds
