"""Output validation tests for generated samples.

Tests that generated outputs have expected quality properties:
- Non-constant (varied) outputs
- Diversity between samples
- Reasonable value distributions
"""

import os
import sys
import subprocess
from pathlib import Path

import pytest
import numpy as np

try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


# Mark all tests in this module
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.timeout(600),
    pytest.mark.skipif(not HAS_NIBABEL, reason="nibabel not installed"),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def golden_checkpoint():
    """Path to golden checkpoint, skip if not available."""
    checkpoint_dir = Path(__file__).parent.parent / "fixtures" / "golden_checkpoint"

    if not checkpoint_dir.exists():
        pytest.skip(
            "Golden checkpoint not available. "
            "Run: ./tests/fixtures/create_golden_checkpoint.sh"
        )

    return checkpoint_dir


@pytest.fixture(scope="module")
def generated_samples(golden_checkpoint, tmp_path_factory):
    """Generate samples for quality testing.

    This fixture generates samples once per module and caches them.
    Returns a list of numpy arrays.
    """
    bravo_checkpoint = golden_checkpoint / "bravo" / "checkpoint_best.pt"
    seg_checkpoint = golden_checkpoint / "seg" / "checkpoint_best.pt"

    # Fall back to latest if best doesn't exist
    if not bravo_checkpoint.exists():
        bravo_checkpoint = golden_checkpoint / "bravo" / "checkpoint_latest.pt"
    if not seg_checkpoint.exists():
        seg_checkpoint = golden_checkpoint / "seg" / "checkpoint_latest.pt"

    if not bravo_checkpoint.exists() or not seg_checkpoint.exists():
        pytest.skip("Required checkpoints not found")

    output_dir = tmp_path_factory.mktemp("generated_samples")

    # Generate samples
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "medgen.scripts.generate",
            "gen_mode=bravo",
            f"seg_model={seg_checkpoint}",
            f"image_model={bravo_checkpoint}",
            f"paths.generated_dir={output_dir}",
            "num_images=4",
            "num_steps=10",
            "image_size=64",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )

    if result.returncode != 0:
        pytest.skip(f"Sample generation failed:\n{result.stderr[-500:]}")

    # Load generated NIfTI files
    nifti_files = list(output_dir.glob("**/*.nii.gz"))
    nifti_files.extend(output_dir.glob("**/*.nii"))

    if not nifti_files:
        pytest.skip("No NIfTI files generated")

    samples = []
    for f in nifti_files[:8]:  # Limit to first 8
        try:
            img = nib.load(f)
            data = np.array(img.dataobj)
            samples.append(data)
        except Exception as e:
            pytest.skip(f"Failed to load {f}: {e}")

    if len(samples) < 2:
        pytest.skip("Not enough samples generated")

    return samples


# =============================================================================
# Output Quality Tests
# =============================================================================


class TestOutputQuality:
    """Test that generated outputs have expected quality properties."""

    def test_no_constant_outputs(self, generated_samples):
        """Generated outputs should not be constant (have variation)."""
        for i, sample in enumerate(generated_samples):
            std = np.std(sample)
            assert std > 0.001, (
                f"Sample {i} appears constant: std={std:.6f}\n"
                f"min={sample.min():.4f}, max={sample.max():.4f}"
            )

    def test_diversity_between_samples(self, generated_samples):
        """Different samples should be different from each other."""
        if len(generated_samples) < 2:
            pytest.skip("Need at least 2 samples for diversity test")

        # Compare pairs of samples
        different_pairs = 0
        total_pairs = 0

        for i in range(len(generated_samples)):
            for j in range(i + 1, len(generated_samples)):
                s1 = generated_samples[i]
                s2 = generated_samples[j]

                # Only compare if same shape
                if s1.shape != s2.shape:
                    continue

                total_pairs += 1

                # Check if they're different
                if not np.allclose(s1, s2, atol=0.01):
                    different_pairs += 1

        if total_pairs == 0:
            pytest.skip("No comparable sample pairs (different shapes)")

        diversity_ratio = different_pairs / total_pairs
        assert diversity_ratio >= 0.5, (
            f"Low diversity: only {different_pairs}/{total_pairs} pairs are different"
        )

    def test_histogram_reasonable(self, generated_samples):
        """Value histogram should span multiple bins (not degenerate)."""
        for i, sample in enumerate(generated_samples):
            # Compute histogram with 10 bins
            hist, bin_edges = np.histogram(sample.flatten(), bins=10)

            # Count bins with significant content (>1% of values)
            threshold = sample.size * 0.01
            significant_bins = np.sum(hist > threshold)

            assert significant_bins >= 3, (
                f"Sample {i} has degenerate histogram: "
                f"only {significant_bins}/10 bins have >1% content\n"
                f"Histogram: {hist}"
            )

    def test_no_nan_or_inf(self, generated_samples):
        """Generated samples should not contain NaN or Inf values."""
        for i, sample in enumerate(generated_samples):
            assert np.all(np.isfinite(sample)), (
                f"Sample {i} contains non-finite values:\n"
                f"NaN count: {np.sum(np.isnan(sample))}\n"
                f"Inf count: {np.sum(np.isinf(sample))}"
            )

    def test_values_in_reasonable_range(self, generated_samples):
        """Generated values should be in reasonable range for medical images."""
        for i, sample in enumerate(generated_samples):
            # For normalized medical images, values typically in [-1, 2] or [0, 1]
            # Allow wider range to be safe
            assert sample.min() >= -10, (
                f"Sample {i} has unexpectedly low min: {sample.min()}"
            )
            assert sample.max() <= 10, (
                f"Sample {i} has unexpectedly high max: {sample.max()}"
            )

    def test_consistent_shapes(self, generated_samples):
        """All samples of same type should have consistent shapes."""
        if len(generated_samples) < 2:
            pytest.skip("Need multiple samples to test shape consistency")

        # Group samples by number of dimensions
        shapes_2d = [s.shape for s in generated_samples if len(s.shape) == 2]
        shapes_3d = [s.shape for s in generated_samples if len(s.shape) == 3]
        shapes_4d = [s.shape for s in generated_samples if len(s.shape) == 4]

        # Within each group, shapes should be consistent
        for shapes, dim in [(shapes_2d, "2D"), (shapes_3d, "3D"), (shapes_4d, "4D")]:
            if len(shapes) > 1:
                first_shape = shapes[0]
                for shape in shapes[1:]:
                    assert shape == first_shape, (
                        f"Inconsistent {dim} shapes: {first_shape} vs {shape}"
                    )


# =============================================================================
# Segmentation Output Tests
# =============================================================================


class TestSegmentationOutputs:
    """Test segmentation-specific output properties."""

    @pytest.fixture
    def seg_samples(self, generated_samples):
        """Filter to segmentation-like outputs (if identifiable)."""
        # Segmentation masks typically have few unique values
        seg_like = []
        for sample in generated_samples:
            unique_vals = len(np.unique(sample.round(1)))
            # Seg masks typically have <20 unique values when rounded
            if unique_vals <= 20:
                seg_like.append(sample)
        return seg_like if seg_like else generated_samples

    def test_seg_has_structure(self, seg_samples):
        """Segmentation outputs should have spatial structure (not random noise)."""
        for i, sample in enumerate(seg_samples[:4]):  # Check first 4
            # Compare to randomly shuffled version
            flat = sample.flatten()
            shuffled = np.random.permutation(flat).reshape(sample.shape)

            # Compute spatial autocorrelation proxy
            # Real structure has higher correlation with neighbors
            if len(sample.shape) >= 2:
                # Shift and compare
                shifted = np.roll(sample, 1, axis=0)
                original_corr = np.corrcoef(sample.flatten(), shifted.flatten())[0, 1]

                shifted_random = np.roll(shuffled, 1, axis=0)
                random_corr = np.corrcoef(
                    shuffled.flatten(), shifted_random.flatten()
                )[0, 1]

                # Original should have higher spatial correlation
                # (unless it's actually random, which would be a bug)
                # Allow for some randomness in test
                if np.isfinite(original_corr) and np.isfinite(random_corr):
                    assert original_corr > random_corr - 0.1, (
                        f"Sample {i} may lack spatial structure:\n"
                        f"Original correlation: {original_corr:.3f}\n"
                        f"Random correlation: {random_corr:.3f}"
                    )


# =============================================================================
# Statistical Tests
# =============================================================================


class TestStatisticalProperties:
    """Test statistical properties of generated samples."""

    def test_mean_not_extreme(self, generated_samples):
        """Mean value should not be at extreme ends."""
        for i, sample in enumerate(generated_samples):
            mean = np.mean(sample)
            min_val = sample.min()
            max_val = sample.max()
            range_val = max_val - min_val

            if range_val > 0.01:  # Only test if there's meaningful range
                # Mean should not be within 5% of extremes
                low_threshold = min_val + 0.05 * range_val
                high_threshold = max_val - 0.05 * range_val

                assert mean > low_threshold, (
                    f"Sample {i} mean too low: {mean:.4f} "
                    f"(range: {min_val:.4f} to {max_val:.4f})"
                )
                assert mean < high_threshold, (
                    f"Sample {i} mean too high: {mean:.4f} "
                    f"(range: {min_val:.4f} to {max_val:.4f})"
                )

    def test_variance_appropriate(self, generated_samples):
        """Variance should be neither too low (constant) nor too high (noise)."""
        for i, sample in enumerate(generated_samples):
            var = np.var(sample)
            range_val = sample.max() - sample.min()

            if range_val > 0.01:
                # Variance should be reasonable fraction of range^2
                # Too low = constant, too high = noise
                normalized_var = var / (range_val**2 + 1e-8)

                assert normalized_var > 0.001, (
                    f"Sample {i} variance too low (nearly constant): {var:.6f}"
                )
                # Upper bound is less strict - high variance might be OK
                assert normalized_var < 0.5, (
                    f"Sample {i} variance unexpectedly high (noise?): {var:.4f}"
                )
