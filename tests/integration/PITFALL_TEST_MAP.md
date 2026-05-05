# Pitfall to Test Mapping

This document maps regression tests in `test_regression_bugs.py` to the bug
they protect against. The test class names and the source-file comments
(e.g. `# Pitfall #41:`) are the ground truth — historically these were
keyed to numbers in `docs/common-pitfalls.md`, but the pitfalls doc has
been re-numbered since the test file was written, so a few of the
numbers in the test file's comments no longer line up with the current
pitfall numbering.

The **test classes themselves** are stable. The **descriptions** are
verbatim from the test file's class docstrings / source comments.

## Mapping

| Pitfall # in test file | Description (from test source) | Test Class |
|---|---|---|
| #41 | BF16 Precision in Loss Computation — always `.float()` before loss | `TestBF16PrecisionBug` |
| #42 | Validation RNG Divergence — save/restore RNG state around validation | `TestRNGDivergenceBug` |
| #43 | RFlow Generation Timestep Scaling — scale [0,1] to [0, num_train_timesteps] | `TestRFlowGenerationScalingRegression` |
| #45 | Timestep Jitter Normalization — normalize before jitter, clamp, scale back | `TestTimestepJitterRegression` |
| #47 | Euler Integration Sign — use ADDITION (x + dt·v), not subtraction | `TestEulerIntegrationSignRegression` |
| #48 | GroupedBatchSampler for Mode Embedding | `TestGroupedBatchSamplerRegression` |
| #49 | FP32 Clamping Before BF16 Cast | `TestFP32ClampingRegression` |
| #50 | Checkpoint Loading Device Mismatch | `TestCheckpointDeviceRegression` |
| #51 | Gradient Accumulation with Mixed Precision | `TestGradientAccumulationRegression` |
| #52 | Scheduler Step Timing — call `scheduler.step()` after `optimizer.step()` | `TestSchedulerStepTimingRegression` |
| #39 (current) | Empty Validation Spurious Best Checkpoint | `TestEmptyValidationRegression` |
| #40 (current) | Mode Embedding Requires Same-Modality Batches | `TestModeEmbeddingBatchRegression` |
| (no #) | MS-SSIM 3D returns None | `TestMSSSIM3DRegression` |
| (no #) | size_bins not passed through pipeline | `TestSizeBinsRegression` |

**Note on numbering drift**: Pitfalls #43, #45, #47-52 in the test file's
source comments refer to the historical numbering when the tests were
written. The current `docs/common-pitfalls.md` has those topics under
different numbers (or, in the case of #43, the slot is now intentionally
skipped). The test classes still cover the original bug behaviors —
verified to still match current source code as of May 2026. To find the
current pitfall # for a given topic, search `docs/common-pitfalls.md` by
description rather than by number.

All 14 test classes were verified to exist in
`tests/integration/test_regression_bugs.py` (as of May 2026).

## Test Coverage by Category

### Precision & Numerics
- **`TestBF16PrecisionBug`**: BF16 loss computation - Always use `.float()` before loss
- **`TestFP32ClampingRegression`**: FP32 clamping - Clamp in FP32, then cast

### RNG & Reproducibility
- **`TestRNGDivergenceBug`**: Validation RNG - Save/restore RNG state around validation

### RFlow Generation
- **`TestRFlowGenerationScalingRegression`**: Timestep scaling - Scale [0,1] to [0, num_train_timesteps]
- **`TestTimestepJitterRegression`**: Timestep jitter - Normalize before jitter, clamp, then scale back
- **`TestEulerIntegrationSignRegression`**: Euler integration - Use ADDITION (x + dt·v), not subtraction

### Mode Embedding
- **`TestModeEmbeddingBatchRegression`**, **`TestGroupedBatchSamplerRegression`**: Homogeneous batches - Use GroupedBatchSampler

### Checkpoint & State
- **`TestEmptyValidationRegression`**: Empty validation - default fallback prevents spurious "best"
- **`TestCheckpointDeviceRegression`**: Device mismatch - Use map_location when loading

### Training Loop
- **`TestGradientAccumulationRegression`**: Gradient accumulation - Divide loss by accumulation_steps
- **`TestSchedulerStepTimingRegression`**: Scheduler timing - Call scheduler.step() after optimizer.step()

### Misc
- **`TestMSSSIM3DRegression`**: MS-SSIM 3D returns None
- **`TestSizeBinsRegression`**: size_bins not passed through pipeline

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

1. Add entry to `docs/common-pitfalls.md` with a topic-descriptive heading
   (the entry's numeric position will shift over time — that's fine; tests
   should reference the topic, not the number).
2. Create test class in `test_regression_bugs.py` with docstring explaining:
   - **BUG**: What went wrong
   - **FIX**: How it was fixed
3. Add an entry to this file referencing the test class name and a
   topical description.
4. Verify test fails without fix, passes with fix.
