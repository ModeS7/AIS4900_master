# Test Categories

## Overview

| Category | Location | Speed | GPU | Purpose |
|----------|----------|-------|-----|---------|
| Unit | `tests/unit/` | <1s per test | No | Isolated component testing |
| Integration | `tests/integration/` | 1-60s per test | Optional | Component interaction |
| E2E | `tests/e2e/` | 60-300s per test | Yes | Full pipeline verification |
| Benchmarks | `tests/benchmarks/` | Varies | Optional | Performance regression |

---

## Unit Tests (`tests/unit/`)

### Characteristics
- **Speed**: Each test completes in under 1 second
- **Isolation**: All external dependencies are mocked
- **No I/O**: No file system, network, or database access
- **No GPU**: Tests run on CPU only
- **Deterministic**: Fixed seeds ensure reproducibility

### When to Use
- Testing pure functions
- Verifying mathematical operations
- Checking edge cases and error handling
- Testing configuration parsing
- Verifying individual class methods

### Run Commands
```bash
pytest tests/unit -v                    # All unit tests
pytest tests/unit -v -k "quality"       # Filter by name
pytest tests/unit --tb=short            # Short traceback
```

---

## Integration Tests (`tests/integration/`)

### Characteristics
- **Speed**: 1-60 seconds per test
- **Real Components**: Uses actual PyTorch models (small configs)
- **GPU Optional**: Some tests require GPU, marked with `@pytest.mark.gpu`
- **Temp Files**: Uses pytest `tmp_path` fixture for file operations

### When to Use
- Testing trainer with real model
- Verifying checkpoint save/load
- Testing dataloader with small dataset
- Verifying config → model pipeline
- Testing strategy + mode combinations

### Run Commands
```bash
pytest tests/integration -v                         # All integration tests
pytest tests/integration -m "not slow and not gpu"  # Fast, CPU only
pytest tests/integration -m "baseline"              # Baseline comparisons
```

---

## E2E Tests (`tests/e2e/`)

### Characteristics
- **Speed**: 60-300 seconds per test
- **Full Pipeline**: Tests complete training → generation flow
- **GPU Required**: Most tests need CUDA
- **Real Data**: May use fixture data or generate synthetic data
- **CLI Testing**: Verifies command-line interface works correctly

### When to Use
- Testing full training runs (few epochs)
- Testing generation pipeline
- Testing CLI commands
- Verifying checkpoint compatibility
- Testing multi-GPU scenarios

### Run Commands
```bash
pytest tests/e2e -v -m "e2e"            # All E2E tests
pytest tests/e2e -v --timeout=300       # With timeout
```

---

## Benchmark Tests (`tests/benchmarks/`)

### Characteristics
- **Purpose**: Track performance over time
- **Metrics**: Training throughput, metric computation speed
- **Deterministic**: Fixed seeds and sizes for reproducibility
- **Regression Detection**: Compare against baseline performance

### When to Use
- Measuring training iterations per second
- Measuring metric computation time
- Detecting performance regressions
- Comparing optimization strategies

### Run Commands
```bash
pytest tests/benchmarks -v -m "benchmark"   # All benchmarks
pytest tests/benchmarks --benchmark-only    # Skip non-benchmark tests
```

---

## Choosing the Right Category

```
Is the code a pure function with no external dependencies?
├── YES → Unit test
└── NO
    Does it need real PyTorch models?
    ├── NO → Unit test (mock the model)
    └── YES
        Does it test a complete workflow (train → generate)?
        ├── YES → E2E test
        └── NO
            Is it measuring performance?
            ├── YES → Benchmark test
            └── NO → Integration test
```

---

## Pytest Markers Reference

| Marker | Description | Skip Command |
|--------|-------------|--------------|
| `@pytest.mark.slow` | Tests taking >10 seconds | `-m "not slow"` |
| `@pytest.mark.gpu` | Requires CUDA device | `-m "not gpu"` |
| `@pytest.mark.e2e` | End-to-end pipeline tests | `-m "not e2e"` |
| `@pytest.mark.baseline` | Baseline comparison tests | `-m "not baseline"` |
| `@pytest.mark.benchmark` | Performance benchmarks | `-m "not benchmark"` |

### Combining Markers

```bash
# Fast CPU-only tests (default for CI)
pytest -m "not slow and not gpu"

# GPU tests only
pytest -m "gpu"

# Everything except benchmarks
pytest -m "not benchmark"

# Slow integration tests
pytest -m "slow and not e2e"
```

---

## CI/CD Integration

### PR Checks (`test.yml`)
- Runs unit tests
- Runs non-GPU, non-slow integration tests
- Coverage threshold: 70%

### Nightly (`tests-nightly.yml`)
- Runs E2E tests
- Runs slow integration tests
- Runs baseline comparisons
- Runs GPU tests
- Creates GitHub issue on failure
