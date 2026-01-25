# Pitfall to Test Mapping

This document maps documented pitfalls (from `docs/common-pitfalls.md`) to their corresponding regression tests.

## Purpose

Each pitfall that caused a bug should have a regression test to prevent recurrence.
This mapping ensures traceability between bugs and their test coverage.

## Mapping

| Pitfall # | Description | Test Class | Test File |
|-----------|-------------|------------|-----------|
| #39 | Empty validation spurious best checkpoint | `TestEmptyValidationRegression` | `test_regression_bugs.py` |
| #40 | Mode embedding requires homogeneous batches | `TestModeEmbeddingBatchRegression` | `test_regression_bugs.py` |
| #41 | BF16 precision in loss computation | `TestBF16PrecisionBug` | `test_regression_bugs.py` |
| #42 | Validation RNG divergence | `TestRNGDivergenceBug` | `test_regression_bugs.py` |
| #43 | RFlow generation timestep scaling | `TestRFlowGenerationScalingRegression` | `test_regression_bugs.py` |
| #44 | Mode embedding batch validation | `TestModeEmbeddingBatchRegression` | `test_regression_bugs.py` |
| #45 | Timestep jitter normalization | `TestTimestepJitterRegression` | `test_regression_bugs.py` |
| #47 | Euler integration sign (addition not subtraction) | `TestEulerIntegrationSignRegression` | `test_regression_bugs.py` |
| #48 | GroupedBatchSampler for mode embedding | `TestGroupedBatchSamplerRegression` | `test_regression_bugs.py` |
| #49 | FP32 clamping before BF16 cast | `TestFP32ClampingRegression` | `test_regression_bugs.py` |
| #50 | Checkpoint loading device mismatch | `TestCheckpointDeviceRegression` | `test_regression_bugs.py` |
| #51 | Gradient accumulation with mixed precision | `TestGradientAccumulationRegression` | `test_regression_bugs.py` |
| #52 | Scheduler step timing | `TestSchedulerStepTimingRegression` | `test_regression_bugs.py` |
| - | MS-SSIM 3D returns None | `TestMSSSIM3DRegression` | `test_regression_bugs.py` |
| - | size_bins not passed through pipeline | `TestSizeBinsRegression` | `test_regression_bugs.py` |

## Test Coverage by Category

### Precision & Numerics
- **#41**: BF16 loss computation - Always use `.float()` before loss
- **#49**: FP32 clamping - Clamp in FP32, then cast

### RNG & Reproducibility
- **#42**: Validation RNG - Save/restore RNG state around validation

### RFlow Generation
- **#43**: Timestep scaling - Scale [0,1] to [0, num_train_timesteps]
- **#45**: Timestep jitter - Normalize before jitter, clamp, then scale back
- **#47**: Euler integration - Use ADDITION (x + dt*v), not subtraction

### Mode Embedding
- **#40, #44, #48**: Homogeneous batches - Use GroupedBatchSampler

### Checkpoint & State
- **#39**: Empty validation - Check val_loss > 0 before marking best
- **#50**: Device mismatch - Use map_location when loading

### Training Loop
- **#51**: Gradient accumulation - Divide loss by accumulation_steps
- **#52**: Scheduler timing - Call scheduler.step() after optimizer.step()

## Running Regression Tests

```bash
# Run all regression tests
pytest tests/integration/test_regression_bugs.py -v

# Run specific pitfall test
pytest tests/integration/test_regression_bugs.py::TestBF16PrecisionBug -v

# Run with coverage
pytest tests/integration/test_regression_bugs.py --cov=src/medgen -v
```

## Adding New Regression Tests

When fixing a bug:

1. Add entry to `docs/common-pitfalls.md` with pitfall number
2. Create test class in `test_regression_bugs.py` with docstring explaining:
   - **BUG**: What went wrong
   - **FIX**: How it was fixed
3. Add mapping to this file
4. Verify test fails without fix, passes with fix
